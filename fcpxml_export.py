"""
FCPXML Export for Doza Assist.
Generates Final Cut Pro X compatible XML with actual cuts on the timeline.

Export modes:
  1. "cuts"    — Pre-cut timeline with each clip as an edit referencing the source media
  2. "markers" — Markers on a gap (legacy, for reference)
  3. "both"    — Cuts on timeline + markers for context

When imported into FCPX, the editor gets:
  - An Event with the source media
  - A Project with each clip placed on the timeline in order
  - Keyword ranges on the source for browser filtering
  - Chapter markers for quick navigation
"""

import os
import re
import math
import uuid
from fractions import Fraction
from urllib.parse import quote

# NOTE: deliberately NOT imported from exporters.xml_text — the exporters
# package __init__ eagerly loads the router/exporters, which import THIS
# module for VIDEO_EXTS; importing the package from here is a cycle.
# Keep this tiny scrub in sync with exporters/xml_text.py.
_XML_ILLEGAL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def scrub_xml_text(text):
    """Drop XML-1.0-illegal control characters (see exporters/xml_text.py)."""
    if not text:
        return "" if text is None else str(text)
    return _XML_ILLEGAL_RE.sub("", str(text))


# One source of truth for "does this extension carry video" — previously
# duplicated (and drifted: .m4v counted as video in the Premiere exporter
# but not here) across five sites.
VIDEO_EXTS = ('.mp4', '.mov', '.m4v', '.mxf', '.avi', '.mkv', '.mts', '.m2ts',
              '.ts')


def _timebase(framerate=23.976):
    """Return (timebase, frame_dur) for a framerate. Fractional NTSC rates use a
    1001 frame duration; integer rates reduce to 1/N. Unknown rates fall back to
    23.976 (24000/1001)."""
    table = {
        23.976: (24000, 1001),
        24.0:   (24, 1),
        25.0:   (25, 1),
        29.97:  (30000, 1001),
        30.0:   (30, 1),
        48.0:   (48, 1),
        50.0:   (50, 1),
        59.94:  (60000, 1001),
        60.0:   (60, 1),
        100.0:  (100, 1),
        119.88: (120000, 1001),
        120.0:  (120, 1),
    }
    return table.get(framerate, (24000, 1001))


def seconds_to_frames(seconds, framerate=23.976):
    """Snap a time in seconds to the nearest whole frame on the timebase grid."""
    timebase, frame_dur = _timebase(framerate)
    return round(seconds * timebase / frame_dur)


def frames_to_fcpxml_time(frames, framerate=23.976):
    """Render a whole-frame count as an FCPXML rational time, e.g. '48048/24000s'.
    The numerator is always a multiple of frame_dur, so the value lands exactly on
    the format's frame grid — which is what FCP requires for a valid edit."""
    timebase, frame_dur = _timebase(framerate)
    return f"{frames * frame_dur}/{timebase}s"


def seconds_to_fcpxml_time(seconds, framerate=23.976):
    """
    Convert seconds to FCPXML rational time format.
    FCPXML uses rational numbers like '48048/24000s' for frame-accurate timing.
    """
    return frames_to_fcpxml_time(seconds_to_frames(seconds, framerate), framerate)


def get_frame_duration(framerate=23.976):
    """Get the frame duration string for FCPXML, e.g. '1001/24000s' or '1/50s'.

    Derived from the single _timebase() table (one frame = frames_to_fcpxml_time(1))
    so the format's frameDuration can never drift from the grid the edit times are
    snapped to — a denominator mismatch there makes FCP reject clips as invalid edits.
    """
    return frames_to_fcpxml_time(1, framerate)


# FCPXML marker colors
MARKER_COLORS = {
    'blue': 'Blue',
    'green': 'Green',
    'purple': 'Purple',
    'red': 'Red',
    'orange': 'Orange',
    'yellow': 'Yellow',
    'cyan': 'Cyan',
    'pink': 'Pink',
}


def _probe_audio_decl(source_path):
    """Probe the source's audio for FCPXML declaration.

    Returns ``(asset_audio_attrs, clip_audio_attr, dialogue_srcch)``:
      - asset_audio_attrs: e.g. ``' audioSources="1" audioChannels="4" audioRate="48000"'``
      - clip_audio_attr:   ``' audioRole="dialogue"'``
      - dialogue_srcch:    list of 1-based source channels to route as the
                           dialogue (the detected speech track[s] of a
                           multi-mono source), or ``None`` — meaning emit the
                           compact ``<asset-clip>`` and let the asset carry it.
    or ``('', '', None)`` when the source has no detectable audio.

    Resolve maps FCPXML clip audio from these DECLARATIONS, not from the
    media file's track table — an asset with a bare ``hasAudio="1"`` and an
    asset-clip with no ``audioRole`` imports the picture SILENT (the round-
    trip writer doesn't hit this because it reuses FCP's original asset,
    which already carries these attributes).

    We declare the source's REAL audio layout — ``audioSources`` = number of
    audio streams, ``audioChannels`` = total channels across them (a stereo
    file → 1/2, a 4-mono-track MXF → 4/4). Declaring only the first stream's
    channels imported just track 1, which is silent when the editor mixed
    "all tracks" (lav on an unknown track). Bringing every track guarantees
    the dialogue is present on the timeline; the editor solos/mutes the rest.

    Imported lazily: the exporters package __init__ eagerly loads this
    module for VIDEO_EXTS, so a module-level import would be a cycle.
    """
    if not source_path:
        return '', '', None
    try:
        from exporters.media_probe import (
            get_audio_layout, get_audio_sample_rate, detect_dialogue_channels)
        layout = get_audio_layout(source_path)
        if not layout or layout[1] < 1:
            return '', '', None
        n_sources, n_channels = layout
        rate = get_audio_sample_rate(source_path) or 48000
        # A single media file is ONE audio source with N channels — matches
        # FCP's own exports, which always emit audioSources="1".
        asset_attrs = (f' audioSources="1" audioChannels="{n_channels}"'
                       f' audioRate="{rate}"')
        clip_attr = ' audioRole="dialogue"'
        # Multi-mono broadcast source (e.g. 4 discrete tracks, lav on one, the
        # rest silent scratch): detect the speech-bearing track(s) so the spine
        # routes ONLY them. Resolve honors srcCh on a connected <audio> element
        # inside <clip><video> — NOT audioRole/audio-channel-source on an
        # <asset-clip> (verified by reverse-engineering Resolve's own FCPXML).
        # Returns 1-based source channels, or None -> compact asset-clip form.
        dialogue_srcch = None
        if n_sources > 1:
            active = detect_dialogue_channels(source_path)
            if active:
                dialogue_srcch = [idx + 1 for idx in active]
        return asset_attrs, clip_attr, dialogue_srcch
    except Exception:
        return '', '', None


def _spine_clip(clip_name, offset_str, dur_str, src_start_str, tc_format,
                anchored_xml, clip_audio_attr, dialogue_srcch,
                asset_start_str, media_dur_str):
    """One spine edit on the shared asset ``r2``.

    With a detected dialogue channel, emit Resolve's connected-clip form: a
    ``<clip>`` windowing the edit, holding a ``<video>`` over the asset's full
    span with a nested ``<audio srcCh="N">`` that routes ONLY the speech
    track(s). This is the form Resolve honors for source-channel selection — a
    flat ``<asset-clip>`` with ``audioRole``/``audio-channel-source`` is
    ignored and defaults to embedded channel 1 (verified by reverse-
    engineering Resolve's own FCPXML export of a 4-mono MXF). Without a detected
    channel (single-stream / normal media) emit the compact ``<asset-clip>``.

    ``anchored_xml`` is the clip's keyword/chapter-marker children (already
    indented); they follow the ``<video>`` per the content model (marker items
    after anchorable items), which also keeps the doc DTD-valid for FCP.
    """
    if dialogue_srcch:
        audio = ''.join(
            f'\n                                <audio lane="-1" ref="r2" '
            f'srcCh="{ch}" offset="{asset_start_str}" '
            f'duration="{media_dur_str}" start="{asset_start_str}"/>'
            for ch in dialogue_srcch)
        return (
            f'                        <clip name="{clip_name}" '
            f'offset="{offset_str}" duration="{dur_str}" start="{src_start_str}" '
            f'format="r1" tcFormat="{tc_format}" enabled="1">'
            f'\n                            <video ref="r2" '
            f'offset="{asset_start_str}" duration="{media_dur_str}" '
            f'start="{asset_start_str}">'
            f'{audio}'
            f'\n                            </video>'
            f'{anchored_xml}'
            f'\n                        </clip>'
        )
    return (
        f'                        <asset-clip name="{clip_name}" ref="r2" '
        f'offset="{offset_str}" duration="{dur_str}" start="{src_start_str}" '
        f'format="r1" tcFormat="{tc_format}"{clip_audio_attr}>'
        f'{anchored_xml}'
        f'\n                        </asset-clip>'
    )


def generate_fcpxml(markers, project_name="Interview", framerate=23.976,
                    source_path=None, media_duration=None, mode="cuts",
                    width=1920, height=1080, start_tc_frames=0, tc_format="NDF"):
    """
    Generate an FCPXML file.

    Args:
        markers: list of dicts with start, end, text, note, color, category
        project_name: str
        framerate: float
        source_path: str — path to the source media file (enables cut mode)
        media_duration: float — total duration of the source media in seconds
        mode: "cuts" | "markers" | "both"

    Returns:
        str: Complete FCPXML content
    """
    # If no source path, fall back to markers-only mode
    if not source_path or not os.path.exists(source_path):
        mode = "markers"

    if mode == "markers":
        return _generate_markers_only(markers, project_name, framerate, width, height)

    return _generate_cuts_timeline(markers, project_name, framerate,
                                   source_path, media_duration, mode, width, height,
                                   start_tc_frames, tc_format=tc_format)


def _generate_cuts_timeline(markers, project_name, framerate, source_path,
                            media_duration, mode, width=1920, height=1080,
                            start_tc_frames=0,
                            tc_format="NDF"):
    """Generate FCPXML with actual cuts on the timeline referencing source media.

    ``start_tc_frames`` is the media's embedded start timecode in whole frames.
    Final Cut keys an asset's source timecode off the media's real timecode, so
    every source-side time (the asset ``start`` and each clip/keyword ``start``)
    must be expressed relative to it. Cameras like DJI and Sony stamp
    time-of-day timecode; exporting 0-based edits against such media makes FCP
    reject every clip with "Invalid edit with no respective media."
    """
    frame_dur = get_frame_duration(framerate)
    safe_name = _escape_xml(project_name)
    uid = f"doza-{uuid.uuid4().hex[:8]}"

    # Sort markers by start time
    markers = sorted(markers, key=lambda m: m['start'])

    # Media duration fallback
    if not media_duration and markers:
        media_duration = max(m['end'] for m in markers) + 10.0
    elif not media_duration:
        media_duration = 60.0

    # The asset advertises media for the range [0, media_frames]. Every edit
    # below is frame-snapped and clamped to this same grid so FCP can always
    # resolve it — an edit that reaches past the asset is rejected on import as
    # "Invalid edit with no respective media."
    # FLOOR the media duration to whole frames: round() can declare one
    # more frame than the file has (container duration >= stream duration),
    # and the declared range is what FCP's import-time edit validation
    # checks — the final clip of a full-length export is the one at risk.
    _tb, _fd = _timebase(framerate)
    media_frames = int(media_duration * _tb / _fd)
    media_dur_str = frames_to_fcpxml_time(media_frames, framerate)

    # File reference — use file:// URL for the source media
    file_url = 'file://' + quote(source_path, safe='/')
    ext = os.path.splitext(source_path)[1].lower()

    # Determine if video or audio-only
    is_video = ext in VIDEO_EXTS

    # Declare the source audio so Resolve routes it onto the timeline. Without
    # these the clips import SILENT (Resolve maps FCPXML audio from the
    # declarations, not the file's track table).
    asset_audio_attrs, clip_audio_attr, dialogue_srcch = _probe_audio_decl(source_path)
    seq_audio_attrs = ' audioLayout="stereo" audioRate="48k"' if asset_audio_attrs else ''
    # Asset source-TC origin (= embedded start TC). Computed before the spine
    # loop because the connected-clip audio routing references it per edit.
    asset_start_str = frames_to_fcpxml_time(start_tc_frames, framerate) if start_tc_frames else "0/1s"

    # Build the spine — each marker becomes an asset-clip on the timeline
    spine_clips = []
    offset_frames = 0

    for i, m in enumerate(markers):
        # Snap in/out to whole frames first, then derive duration as (out - in).
        # Rounding start and duration independently can push start + duration a
        # frame past the source out point (and past the asset), which FCP
        # rejects. Clamp into [0, media_frames] so the edit never references
        # media the asset doesn't have.
        start_f = max(0, min(seconds_to_frames(m['start'], framerate), media_frames))
        end_f = max(start_f, min(seconds_to_frames(m['end'], framerate), media_frames))
        dur_f = end_f - start_f
        if dur_f <= 0:
            continue

        offset_str = frames_to_fcpxml_time(offset_frames, framerate)
        # Source-side times are absolute in the asset's timecode space:
        # embedded start TC + the in-point offset into the media.
        src_start_str = frames_to_fcpxml_time(start_tc_frames + start_f, framerate)
        dur_str = frames_to_fcpxml_time(dur_f, framerate)

        # Truncate the RAW text first, then escape. Escaping inflates quotes
        # and ampersands into multi-char entities (&quot; &apos; &amp;), so
        # slicing the ESCAPED string can cut an entity in half and produce
        # XML that FCP rejects with "EntityRef: expecting ';'".
        clip_name = _escape_xml((m.get('text') or f'Clip {i+1}')[:80])
        note = _escape_xml(m.get('note', ''))
        category = _escape_xml(m.get('category', 'Clip'))
        speaker = (m.get('speaker') or '').strip()

        # Optional marker inside the clip
        marker_xml = ''
        if mode == "both":
            marker_note = f"{note} [{category}]"
            if speaker:
                marker_note = f"{marker_note} — {_escape_xml(speaker)}"
            marker_xml = (
                f'\n                            <chapter-marker start="{src_start_str}" '
                f'duration="{frame_dur}" value="{clip_name}" '
                f'note="{marker_note}"/>'
            )

        # Keyword for the clip's category
        keyword_xml = (
            f'\n                            <keyword start="{src_start_str}" '
            f'duration="{dur_str}" value="{category}"/>'
        )
        # Second keyword carrying the speaker (when known). Editors can filter
        # the clip pool by speaker inside the NLE without us having to rename
        # the asset-clip itself.
        if speaker:
            keyword_xml += (
                f'\n                            <keyword start="{src_start_str}" '
                f'duration="{dur_str}" value="Speaker: {_escape_xml(speaker)}"/>'
            )

        spine_clips.append(_spine_clip(
            clip_name, offset_str, dur_str, src_start_str, tc_format,
            f'{keyword_xml}{marker_xml}', clip_audio_attr, dialogue_srcch,
            asset_start_str, media_dur_str))

        # Accumulate the timeline offset in whole frames so each clip butts
        # exactly against the previous one — summing rounded seconds drifts and
        # leaves sub-frame gaps/overlaps on the spine.
        offset_frames += dur_f

    spine_block = '\n'.join(spine_clips)

    timeline_dur_str = frames_to_fcpxml_time(offset_frames or media_frames, framerate)

    # The asset's source-timecode origin = the media's embedded start TC. FCP
    # validates every clip start against [asset.start, asset.start+duration].
    asset_start_str = frames_to_fcpxml_time(start_tc_frames, framerate) if start_tc_frames else "0/1s"

    fcpxml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>

<fcpxml version="1.11">
    <resources>
        <format id="r1"{_format_name_attr(width, height, framerate)} frameDuration="{frame_dur}" width="{width}" height="{height}" colorSpace="1-1-1 (Rec. 709)"/>
        <asset id="r2" name="{_escape_xml(os.path.basename(source_path))}" start="{asset_start_str}" duration="{media_dur_str}" hasVideo="{1 if is_video else 0}" hasAudio="1"{asset_audio_attrs} format="r1">
            <media-rep kind="original-media" src="{file_url}"/>
        </asset>
    </resources>
    <library>
        <event name="{safe_name}">
            <project name="{safe_name} - Selects" uid="{uid}">
                <sequence format="r1" duration="{timeline_dur_str}" tcStart="0/1s" tcFormat="NDF"{seq_audio_attrs}>
                    <spine>
{spine_block}
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>"""

    return fcpxml


def generate_story_fcpxml(markers, project_name="Interview", story_title="Story",
                          framerate=23.976, source_path=None, media_duration=None,
                          width=1920, height=1080, start_tc_frames=0, tc_format="NDF"):
    """
    Generate FCPXML for a Story Builder sequence.
    Creates a single timeline with clips in narrative order as actual edits.
    """
    if not source_path or not os.path.exists(source_path):
        return _generate_markers_only(markers, f"{project_name} - {story_title}", framerate, width, height)

    frame_dur = get_frame_duration(framerate)
    safe_name = _escape_xml(project_name)
    safe_title = _escape_xml(story_title)
    uid = f"doza-story-{uuid.uuid4().hex[:8]}"

    markers = [m for _, m in sorted(enumerate(markers), key=lambda iv: iv[1].get('_order', iv[0]))]

    if not media_duration and markers:
        media_duration = max(m['end'] for m in markers) + 10.0
    elif not media_duration:
        media_duration = 60.0

    # The asset advertises media for the range [0, media_frames]. Every edit
    # below is frame-snapped and clamped to this same grid so FCP can always
    # resolve it — an edit that reaches past the asset is rejected on import as
    # "Invalid edit with no respective media."
    _tb, _fd = _timebase(framerate)
    media_frames = int(media_duration * _tb / _fd)  # floor — see above
    media_dur_str = frames_to_fcpxml_time(media_frames, framerate)

    file_url = 'file://' + quote(source_path, safe='/')
    ext = os.path.splitext(source_path)[1].lower()
    is_video = ext in VIDEO_EXTS

    # Declare source audio so Resolve routes it (else clips import silent) —
    # see generate_fcpxml.
    asset_audio_attrs, clip_audio_attr, dialogue_srcch = _probe_audio_decl(source_path)
    seq_audio_attrs = ' audioLayout="stereo" audioRate="48k"' if asset_audio_attrs else ''
    # Asset source-TC origin (= embedded start TC). Computed before the spine
    # loop because the connected-clip audio routing references it per edit.
    asset_start_str = frames_to_fcpxml_time(start_tc_frames, framerate) if start_tc_frames else "0/1s"

    spine_clips = []
    offset_frames = 0

    for i, m in enumerate(markers):
        # Snap in/out to whole frames first, then derive duration as (out - in).
        # Rounding start and duration independently can push start + duration a
        # frame past the source out point (and past the asset), which FCP
        # rejects. Clamp into [0, media_frames] so the edit never references
        # media the asset doesn't have.
        start_f = max(0, min(seconds_to_frames(m['start'], framerate), media_frames))
        end_f = max(start_f, min(seconds_to_frames(m['end'], framerate), media_frames))
        dur_f = end_f - start_f
        if dur_f <= 0:
            continue

        offset_str = frames_to_fcpxml_time(offset_frames, framerate)
        # Source-side times are absolute in the asset's timecode space:
        # embedded start TC + the in-point offset into the media.
        src_start_str = frames_to_fcpxml_time(start_tc_frames + start_f, framerate)
        dur_str = frames_to_fcpxml_time(dur_f, framerate)

        # Raw-truncate THEN escape — see the entity-slicing note in
        # generate_fcpxml above.
        clip_name = _escape_xml((m.get('text') or f'Clip {i+1}')[:80])
        note = _escape_xml(m.get('note', ''))
        speaker = (m.get('speaker') or '').strip()
        marker_note = note
        if speaker:
            marker_note = f"{note} — {_escape_xml(speaker)}" if note else _escape_xml(speaker)

        marker_xml = (
            f'\n                            <chapter-marker start="{src_start_str}" '
            f'duration="{frame_dur}" value="{clip_name}" '
            f'note="{marker_note}"/>'
        )
        speaker_kw = ''
        if speaker:
            speaker_kw = (
                f'\n                            <keyword start="{src_start_str}" '
                f'duration="{dur_str}" value="Speaker: {_escape_xml(speaker)}"/>'
            )

        spine_clips.append(_spine_clip(
            clip_name, offset_str, dur_str, src_start_str, tc_format,
            f'{speaker_kw}{marker_xml}', clip_audio_attr, dialogue_srcch,
            asset_start_str, media_dur_str))

        # Accumulate the timeline offset in whole frames so each clip butts
        # exactly against the previous one — summing rounded seconds drifts and
        # leaves sub-frame gaps/overlaps on the spine.
        offset_frames += dur_f

    spine_block = '\n'.join(spine_clips)

    timeline_dur_str = frames_to_fcpxml_time(offset_frames or media_frames, framerate)

    # The asset's source-timecode origin = the media's embedded start TC. FCP
    # validates every clip start against [asset.start, asset.start+duration].
    asset_start_str = frames_to_fcpxml_time(start_tc_frames, framerate) if start_tc_frames else "0/1s"

    fcpxml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>

<fcpxml version="1.11">
    <resources>
        <format id="r1"{_format_name_attr(width, height, framerate)} frameDuration="{frame_dur}" width="{width}" height="{height}" colorSpace="1-1-1 (Rec. 709)"/>
        <asset id="r2" name="{_escape_xml(os.path.basename(source_path))}" start="{asset_start_str}" duration="{media_dur_str}" hasVideo="{1 if is_video else 0}" hasAudio="1"{asset_audio_attrs} format="r1">
            <media-rep kind="original-media" src="{file_url}"/>
        </asset>
    </resources>
    <library>
        <event name="{safe_name}">
            <project name="{safe_title}" uid="{uid}">
                <sequence format="r1" duration="{timeline_dur_str}" tcStart="0/1s" tcFormat="NDF"{seq_audio_attrs}>
                    <spine>
{spine_block}
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>"""

    return fcpxml


def _generate_markers_only(markers, project_name, framerate, width=1920, height=1080):
    """Legacy marker-only export (no source media reference)."""
    frame_dur = get_frame_duration(framerate)

    if markers:
        total_duration = max(m['end'] for m in markers) + 10.0
    else:
        total_duration = 60.0

    total_dur_str = seconds_to_fcpxml_time(total_duration, framerate)

    markers_xml = []
    for i, m in enumerate(markers):
        start_time = seconds_to_fcpxml_time(m['start'], framerate)
        duration = m['end'] - m['start']
        dur_str = seconds_to_fcpxml_time(max(duration, 1.0 / framerate), framerate)

        color = MARKER_COLORS.get(m.get('color', 'blue'), 'Blue')
        raw_name = m.get('text') or f'Marker {i+1}'
        note = _escape_xml(m.get('note', ''))
        category = _escape_xml(m.get('category', 'Marker'))
        speaker = (m.get('speaker') or '').strip()

        # Raw-truncate THEN escape — see the entity-slicing note in
        # generate_fcpxml above. (category was also previously unescaped here.)
        display_name = _escape_xml(
            raw_name[:80] + '...' if len(raw_name) > 80 else raw_name)
        marker_note = f"{note} [{category}]"
        if speaker:
            marker_note = f"{marker_note} — {_escape_xml(speaker)}"

        markers_xml.append(
            f'                        <chapter-marker start="{start_time}" '
            f'duration="{dur_str}" value="{display_name}" '
            f'note="{marker_note}"/>'
        )

    markers_block = '\n'.join(markers_xml)

    fcpxml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>

<fcpxml version="1.11">
    <resources>
        <format id="r1"{_format_name_attr(width, height, framerate)} frameDuration="{frame_dur}" width="{width}" height="{height}" colorSpace="1-1-1 (Rec. 709)"/>
    </resources>
    <library>
        <event name="{_escape_xml(project_name)} Markers">
            <project name="{_escape_xml(project_name)}" uid="doza-{re.sub(r'[^a-z0-9]+', '-', project_name.lower()).strip('-') or 'project'}">
                <sequence format="r1" duration="{total_dur_str}" tcStart="0/1s" tcFormat="NDF">
                    <spine>
                        <gap name="Gap" offset="0/1s" duration="{total_dur_str}" start="0/1s">
{markers_block}
                        </gap>
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>"""

    return fcpxml


def _format_name_attr(width, height, framerate):
    """' name="…"' when FCP defines a name for this combo, else ''."""
    name = _format_name(width, height, framerate)
    return f' name="{name}"' if name else ''


def _format_name(width, height, framerate):
    """Get the exact FCPX format name string.

    FCPX uses specific format names like:
      FFVideoFormat1080p2398
      FFVideoFormat3840x2160p2398
      FFVideoFormat720p25

    For resolutions > 1080p, width is included.
    """
    rate = _framerate_label(framerate)
    # Only fabricate names for combos FCP actually defines; for anything
    # else omit the attribute-value (caller drops name=) — formats are fully
    # specified by frameDuration/width/height and a bogus name like
    # FFVideoFormat600p120 invites importer strictness issues.
    if height in (720, 1080):
        return f"FFVideoFormat{height}p{rate}"
    if height > 1080:
        return f"FFVideoFormat{width}x{height}p{rate}"
    return ""


def _framerate_label(framerate):
    """Get FCPX format label for framerate."""
    labels = {
        23.976: "2398",
        24.0: "24",
        25.0: "25",
        29.97: "2997",
        30.0: "30",
        48.0: "48",
        50.0: "50",
        59.94: "5994",
        60.0: "60",
        100.0: "100",
        119.88: "11988",
        120.0: "120",
    }
    return labels.get(framerate, "2398")


def _escape_xml(text):
    """Escape special XML characters (after scrubbing XML-illegal bytes)."""
    if not text:
        return ""
    # C0 control chars are illegal in XML 1.0 even escaped — a single stray
    # byte from a bad encoding corrupts the whole export for FCP.
    text = scrub_xml_text(text)
    return (str(text)
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&apos;'))
