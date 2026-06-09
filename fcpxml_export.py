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
import math
import re
import uuid
from fractions import Fraction
from urllib.parse import quote


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


def generate_fcpxml(markers, project_name="Interview", framerate=23.976,
                    source_path=None, media_duration=None, mode="cuts",
                    width=1920, height=1080, start_tc_frames=0):
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
                                   start_tc_frames)


def _generate_cuts_timeline(markers, project_name, framerate, source_path,
                            media_duration, mode, width=1920, height=1080,
                            start_tc_frames=0):
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
    # below is clamped and frame-snapped against this same grid so FCP can always
    # resolve it — an edit that reaches past the asset is rejected on import as
    # "Invalid edit with no respective media."
    media_frames = seconds_to_frames(media_duration, framerate)
    media_dur_str = frames_to_fcpxml_time(media_frames, framerate)

    # File reference — use file:// URL for the source media
    file_url = 'file://' + quote(source_path, safe='/')
    ext = os.path.splitext(source_path)[1].lower()

    # Determine if video or audio-only
    is_video = ext in ('.mp4', '.mov', '.mxf', '.avi', '.mkv')

    # Build the spine — each marker becomes an asset-clip on the timeline
    spine_clips = []
    offset_frames = 0

    for i, m in enumerate(markers):
        # Snap in/out to whole frames first, then derive duration as (out - in).
        # Rounding start and duration independently can make start + duration land
        # a frame past the source out point — and past the asset — which FCP
        # rejects. Clamp the range into [0, media_frames] so the edit never
        # references media the asset doesn't have.
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

        clip_name = _escape_xml(m.get('text', f'Clip {i+1}'))[:80]
        note = _escape_xml(m.get('note', ''))
        category = _escape_xml(m.get('category', 'Clip'))

        # Optional marker inside the clip
        marker_xml = ''
        if mode == "both":
            marker_xml = (
                f'\n                            <chapter-marker start="{src_start_str}" '
                f'duration="{frame_dur}" value="{clip_name}" '
                f'note="{note} [{category}]"/>'
            )

        # Keyword for the clip
        keyword_xml = (
            f'\n                            <keyword start="{src_start_str}" '
            f'duration="{dur_str}" value="{category}"/>'
        )

        spine_clips.append(
            f'                        <asset-clip name="{clip_name}" ref="r2" '
            f'offset="{offset_str}" duration="{dur_str}" start="{src_start_str}" '
            f'format="r1" tcFormat="NDF">'
            f'{keyword_xml}{marker_xml}'
            f'\n                        </asset-clip>'
        )

        # Accumulate the timeline offset in whole frames so each clip butts
        # exactly against the previous one — summing rounded seconds drifts and
        # leaves sub-frame gaps/overlaps on the spine.
        offset_frames += dur_f

    spine_block = '\n'.join(spine_clips)

    timeline_dur_str = frames_to_fcpxml_time(offset_frames or media_frames, framerate)

    # The asset's source timecode origin = the media's embedded start TC. FCP
    # validates every clip's start against [asset.start, asset.start+duration].
    asset_start_str = frames_to_fcpxml_time(start_tc_frames, framerate) if start_tc_frames else "0/1s"

    fcpxml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>

<fcpxml version="1.11">
    <resources>
        <format id="r1" name="{_format_name(width, height, framerate)}" frameDuration="{frame_dur}" width="{width}" height="{height}" colorSpace="1-1-1 (Rec. 709)"/>
        <asset id="r2" name="{_escape_xml(os.path.basename(source_path))}" start="{asset_start_str}" duration="{media_dur_str}" hasVideo="{1 if is_video else 0}" hasAudio="1" format="r1">
            <media-rep kind="original-media" src="{file_url}"/>
        </asset>
    </resources>
    <library>
        <event name="{safe_name}">
            <project name="{safe_name} - Selects" uid="{uid}">
                <sequence format="r1" duration="{timeline_dur_str}" tcStart="0/1s" tcFormat="NDF">
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
                          width=1920, height=1080, start_tc_frames=0):
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

    # Stable order: _order when stamped, original position otherwise.
    # (list.index() as the fallback was O(n²) and aliased equal-dict
    # duplicates to the first occurrence, interleaving wrongly with real
    # _order values when a caller passed a mixed list.)
    markers = [m for _, m in sorted(
        enumerate(markers), key=lambda iv: iv[1].get('_order', iv[0]),
    )]

    if not media_duration and markers:
        media_duration = max(m['end'] for m in markers) + 10.0
    elif not media_duration:
        media_duration = 60.0

    # See _generate_cuts_timeline: edits are frame-snapped and clamped to the
    # asset's [0, media_frames] range so FCP can always resolve them.
    media_frames = seconds_to_frames(media_duration, framerate)
    media_dur_str = frames_to_fcpxml_time(media_frames, framerate)

    file_url = 'file://' + quote(source_path, safe='/')
    ext = os.path.splitext(source_path)[1].lower()
    is_video = ext in ('.mp4', '.mov', '.mxf', '.avi', '.mkv')

    spine_clips = []
    offset_frames = 0

    for i, m in enumerate(markers):
        start_f = max(0, min(seconds_to_frames(m['start'], framerate), media_frames))
        end_f = max(start_f, min(seconds_to_frames(m['end'], framerate), media_frames))
        dur_f = end_f - start_f
        if dur_f <= 0:
            continue

        offset_str = frames_to_fcpxml_time(offset_frames, framerate)
        # Absolute source time = embedded start TC + in-point into the media.
        src_start_str = frames_to_fcpxml_time(start_tc_frames + start_f, framerate)
        dur_str = frames_to_fcpxml_time(dur_f, framerate)

        clip_name = _escape_xml(m.get('text', f'Clip {i+1}'))[:80]
        note = _escape_xml(m.get('note', ''))

        marker_xml = (
            f'\n                            <chapter-marker start="{src_start_str}" '
            f'duration="{frame_dur}" value="{clip_name}" '
            f'note="{note}"/>'
        )

        spine_clips.append(
            f'                        <asset-clip name="{clip_name}" ref="r2" '
            f'offset="{offset_str}" duration="{dur_str}" start="{src_start_str}" '
            f'format="r1" tcFormat="NDF">{marker_xml}'
            f'\n                        </asset-clip>'
        )

        offset_frames += dur_f

    spine_block = '\n'.join(spine_clips)

    timeline_dur_str = frames_to_fcpxml_time(offset_frames or media_frames, framerate)

    asset_start_str = frames_to_fcpxml_time(start_tc_frames, framerate) if start_tc_frames else "0/1s"

    fcpxml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>

<fcpxml version="1.11">
    <resources>
        <format id="r1" name="{_format_name(width, height, framerate)}" frameDuration="{frame_dur}" width="{width}" height="{height}" colorSpace="1-1-1 (Rec. 709)"/>
        <asset id="r2" name="{_escape_xml(os.path.basename(source_path))}" start="{asset_start_str}" duration="{media_dur_str}" hasVideo="{1 if is_video else 0}" hasAudio="1" format="r1">
            <media-rep kind="original-media" src="{file_url}"/>
        </asset>
    </resources>
    <library>
        <event name="{safe_name}">
            <project name="{safe_title}" uid="{uid}">
                <sequence format="r1" duration="{timeline_dur_str}" tcStart="0/1s" tcFormat="NDF">
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
        name = _escape_xml(m.get('text', f'Marker {i+1}'))
        note = _escape_xml(m.get('note', ''))
        category = m.get('category', 'Marker')

        display_name = name[:80] + '...' if len(name) > 80 else name

        markers_xml.append(
            f'                        <chapter-marker start="{start_time}" '
            f'duration="{dur_str}" value="{display_name}" '
            f'note="{note} [{category}]"/>'
        )

    markers_block = '\n'.join(markers_xml)

    # uid must be XML-attribute safe. This was the one place project_name was
    # interpolated unescaped — a name like "Tom & Jerry" emitted a raw '&'
    # and FCP rejected the whole file at parse. The uid is an opaque token,
    # so reduce the name to [a-z0-9-] rather than entity-escaping it.
    uid_token = re.sub(r'[^a-z0-9]+', '-', project_name.lower()).strip('-') or 'project'

    fcpxml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>

<fcpxml version="1.11">
    <resources>
        <format id="r1" name="{_format_name(width, height, framerate)}" frameDuration="{frame_dur}" width="{width}" height="{height}" colorSpace="1-1-1 (Rec. 709)"/>
    </resources>
    <library>
        <event name="{_escape_xml(project_name)} Markers">
            <project name="{_escape_xml(project_name)}" uid="doza-{uid_token}">
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


def _format_name(width, height, framerate):
    """Get the exact FCPX format name string.

    FCPX uses specific format names like:
      FFVideoFormat1080p2398
      FFVideoFormat3840x2160p2398
      FFVideoFormat720p25

    For resolutions > 1080p, width is included.
    """
    rate = _framerate_label(framerate)
    if height <= 1080:
        return f"FFVideoFormat{height}p{rate}"
    else:
        return f"FFVideoFormat{width}x{height}p{rate}"


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
        120.0: "120",
    }
    return labels.get(framerate, "2398")


def _escape_xml(text):
    """Escape special XML characters."""
    if not text:
        return ""
    return (str(text)
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&apos;'))
