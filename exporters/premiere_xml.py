"""
Premiere XML (Final Cut Pro 7 XML, "xmeml v5") exporter.

This is the format Adobe recommends for getting third-party cuts into
Premiere Pro. Schema highlights:

  - Root: <xmeml version="5">
  - Time is integer frames, not rational fractions
  - <rate><timebase> is the rounded integer (24, 25, 30) and
    <ntsc>TRUE</ntsc> is set for fractional rates (23.976, 29.97, 59.94)
  - File references use file://localhost<absolute_path>

Frame math: for non-NTSC rates frames = round(seconds * timebase). For NTSC
rates Premiere expects integer frames at the rounded timebase (24/30/60),
so we use frames = round(seconds * actual_fps), then map onto that timebase.
This matches what FCP7 itself produced and what Premiere has accepted for
years.
"""

import os
import urllib.parse

from exporters.xml_text import scrub_xml_text
from fcpxml_export import VIDEO_EXTS
from exporters.media_probe import (
    get_audio_channels, get_audio_layout, detect_dialogue_channels,
    has_video_stream)
import xml.etree.ElementTree as ET
from xml.dom import minidom

from .base import BaseExporter, ExportResult

# (timebase, ntsc, actual_fps_for_frame_math)
_RATE_TABLE = {
    23.976: (24, True,  24000.0 / 1001.0),
    24.0:   (24, False, 24.0),
    25.0:   (25, False, 25.0),
    29.97:  (30, True,  30000.0 / 1001.0),
    30.0:   (30, False, 30.0),
    48.0:   (48, False, 48.0),
    50.0:   (50, False, 50.0),
    59.94:  (60, True,  60000.0 / 1001.0),
    60.0:   (60, False, 60.0),
    100.0:  (100, False, 100.0),
    119.88: (120, True,  120000.0 / 1001.0),
    120.0:  (120, False, 120.0),
}


def _rate_for(framerate: float):
    return _RATE_TABLE.get(framerate, _RATE_TABLE[23.976])


def _seconds_to_frames(seconds: float, framerate: float) -> int:
    if seconds < 0:
        seconds = 0.0
    _, _, actual_fps = _rate_for(framerate)
    return int(round(seconds * actual_fps))


def _add_rate(parent: ET.Element, framerate: float) -> None:
    timebase, ntsc, _ = _rate_for(framerate)
    rate = ET.SubElement(parent, "rate")
    ET.SubElement(rate, "timebase").text = str(timebase)
    ET.SubElement(rate, "ntsc").text = "TRUE" if ntsc else "FALSE"


def _file_url(source_path: str) -> str:
    if not source_path:
        return ""
    abs_path = os.path.abspath(source_path)
    return "file://localhost" + urllib.parse.quote(abs_path)


def _build_file_element(file_id: str, source_path: str, framerate: float,
                        width: int, height: int, media_duration_frames: int,
                        has_video: bool, has_audio: bool,
                        audio_channels: int = 2) -> ET.Element:
    file_el = ET.Element("file", id=file_id)
    ET.SubElement(file_el, "name").text = os.path.basename(source_path) if source_path else "Source"
    ET.SubElement(file_el, "pathurl").text = _file_url(source_path)
    _add_rate(file_el, framerate)
    ET.SubElement(file_el, "duration").text = str(media_duration_frames)

    media = ET.SubElement(file_el, "media")
    if has_video:
        video = ET.SubElement(media, "video")
        sample = ET.SubElement(video, "samplecharacteristics")
        _add_rate(sample, framerate)
        ET.SubElement(sample, "width").text = str(width)
        ET.SubElement(sample, "height").text = str(height)
    if has_audio:
        audio = ET.SubElement(media, "audio")
        sample = ET.SubElement(audio, "samplecharacteristics")
        ET.SubElement(sample, "depth").text = "16"
        ET.SubElement(sample, "samplerate").text = "48000"
        ET.SubElement(audio, "channelcount").text = str(audio_channels)

    return file_el


def _add_clipitem(track: ET.Element, *, clip_id: str, name: str, file_ref_id: str,
                  source_in: int, source_out: int, record_in: int, record_out: int,
                  framerate: float, media_type: str, masterclip_id: str,
                  reuse_file: bool, file_element: ET.Element | None,
                  comment: str = "", audio_channel: int | None = None) -> None:
    """media_type is 'video' or 'audio' — affects which track this lands on.

    ``audio_channel`` (1-based) is required for audio clipitems and IGNORED
    for video. FCP7 XML uses a ``<sourcetrack>`` element to tell the NLE
    which channel of the source file each audio clipitem should pull from
    — without it, Resolve (and Premiere) place the clip on the track but
    play silence because they don't know which source stream is wired
    through. The index is the file's FLAT audio track number across all
    streams (see _resolve_audio_wiring), so a sequence A1 may pull source
    track 2 when that's where the dialogue lives.
    """
    clipitem = ET.SubElement(track, "clipitem", id=clip_id)
    ET.SubElement(clipitem, "name").text = name
    ET.SubElement(clipitem, "enabled").text = "TRUE"
    ET.SubElement(clipitem, "duration").text = str(source_out - source_in)
    _add_rate(clipitem, framerate)
    ET.SubElement(clipitem, "start").text = str(record_in)
    ET.SubElement(clipitem, "end").text = str(record_out)
    ET.SubElement(clipitem, "in").text = str(source_in)
    ET.SubElement(clipitem, "out").text = str(source_out)
    ET.SubElement(clipitem, "masterclipid").text = masterclip_id

    if reuse_file:
        ET.SubElement(clipitem, "file", id=file_ref_id)
    else:
        # First reference: emit the full <file> definition.
        clipitem.append(file_element)

    if media_type == "audio" and audio_channel is not None:
        sourcetrack = ET.SubElement(clipitem, "sourcetrack")
        ET.SubElement(sourcetrack, "mediatype").text = "audio"
        ET.SubElement(sourcetrack, "trackindex").text = str(audio_channel)

    if comment:
        comments = ET.SubElement(clipitem, "comments")
        ET.SubElement(comments, "mastercomment1").text = comment


def _prettify(root: ET.Element) -> str:
    raw = ET.tostring(root, encoding="utf-8")
    parsed = minidom.parseString(raw)
    pretty = parsed.toprettyxml(indent="\t", encoding="UTF-8").decode("utf-8")
    # minidom prefers <?xml ... encoding="UTF-8"?>; FCP7 XML wants the DOCTYPE too.
    lines = pretty.splitlines()
    if lines and lines[0].startswith("<?xml"):
        lines.insert(1, "<!DOCTYPE xmeml>")
    return "\n".join(lines) + "\n"


def _is_video_source(source_path: str) -> bool:
    if not source_path:
        return False
    ext = os.path.splitext(source_path)[1].lower()
    return ext in VIDEO_EXTS


def _resolve_has_video(source_path: str) -> bool:
    """Whether the sequence should carry video clipitems for this source.

    Decided by probing for a REAL video stream (attached-picture cover art
    excluded) — the extension alone mislabels an audio-only .mov/.mp4 (AAC
    podcast export, multi-mono field-recorder QuickTime) as video, emitting
    <video> clipitems against a file with no picture, which Premiere shows
    as a black/offline video component on V1 for every select. The extension
    heuristic remains only as the fallback when the probe can't answer
    (missing file / no ffprobe), mirroring fcpxml_export._resolve_is_video.
    The probe reads stream headers only (no decode), so it adds no
    meaningful latency to the export path.
    """
    probed = has_video_stream(source_path) if source_path else None
    if probed is not None:
        return probed
    return _is_video_source(source_path)


def _resolve_audio_wiring(source_path: str) -> tuple[list, int]:
    """``(source track indices to wire, total source channels)`` for a file.

    FCP7 XML numbers a file's audio FLAT across all streams — a stereo file
    is tracks 1-2, a 4-mono-track broadcast MXF tracks 1-4 — so each returned
    index lands directly in ``<sourcetrack><trackindex>``. The old two-track
    cap probed only the first stream, so the lav mic on track 2+ of a
    multi-mono MXF was never referenced and Premiere played camera scratch
    or silence (the 1.0.21/1.0.22 layout fixes had landed on FCPXML only).

    Multi-mono (multi-STREAM) sources route ONLY the detected speech-bearing
    track(s), loudest first — same "Speech tracks only" policy as the FCPXML
    exporter, so the primary dialogue lands on A1. When loudness can't be
    measured, every track is wired: extra scratch tracks are mutable, a
    missing dialogue track is not.

    Single-STREAM multichannel sources (a camcorder 5.1 mix, a 16/32-channel
    Dante WAV-in-MOV master) keep the historical 2-track cap: dialogue on
    such mixes lives on the front pair, and wiring all N channels would
    stack LFE/surrounds as timeline tracks routed straight into the stereo
    master bus. Falls back to the historical first-stream probe (mono → one
    track, otherwise stereo) when the layout probe fails.
    """
    layout = get_audio_layout(source_path)
    if layout is None:
        # Probe FAILED (missing file / no ffprobe) — historical fallback.
        channels = min(get_audio_channels(source_path) or 2, 2)
        return list(range(1, channels + 1)), channels
    n_streams, total_channels = layout
    if total_channels < 1:
        # Probed clean with NO audio streams (silent B-roll): wire nothing —
        # audio clipitems against a silent file import as offline audio.
        return [], 0
    if n_streams > 1:
        active = detect_dialogue_channels(source_path)  # 0-based, loudest first
        if active:
            return [idx + 1 for idx in active], total_channels
        # Unmeasurable multi-mono: wire every track rather than risk
        # dropping the one carrying the dialogue.
        return list(range(1, total_channels + 1)), total_channels
    # Single stream: cap at the front pair (channels 1-2).
    return list(range(1, min(total_channels, 2) + 1)), total_channels


def _build_sequence(
    *,
    sequence_name: str,
    markers: list,
    source_path: str,
    media_duration: float | None,
    framerate: float,
    width: int,
    height: int,
    export_mode: str = "cuts",
    audio_channels: tuple = (1, 2),  # 1-based source track indices (A1..An)
    file_channelcount: int = 0,      # total channels declared on <file> (0 = len(audio_channels))
) -> ET.Element:
    sequence_name = scrub_xml_text(sequence_name)
    has_video = _resolve_has_video(source_path)
    # Transcribed sources always have audio (gated at import); an empty
    # wiring only happens when the probe POSITIVELY found no audio streams.
    has_audio = bool(audio_channels)

    timebase, _, _ = _rate_for(framerate)

    if not media_duration:
        if markers:
            media_duration = max(float(m.get("end") or 0) for m in markers) + 10.0
        else:
            media_duration = 60.0
    media_duration_frames = _seconds_to_frames(media_duration, framerate)

    xmeml = ET.Element("xmeml", version="5")
    sequence = ET.SubElement(xmeml, "sequence", id="sequence-1")
    ET.SubElement(sequence, "name").text = sequence_name
    ET.SubElement(sequence, "duration").text = str(media_duration_frames)
    _add_rate(sequence, framerate)

    media = ET.SubElement(sequence, "media")
    video = ET.SubElement(media, "video")
    video_format = ET.SubElement(video, "format")
    sample = ET.SubElement(video_format, "samplecharacteristics")
    _add_rate(sample, framerate)
    ET.SubElement(sample, "width").text = str(width)
    ET.SubElement(sample, "height").text = str(height)
    video_track = ET.SubElement(video, "track")

    audio = ET.SubElement(media, "audio")
    ET.SubElement(audio, "numOutputChannels").text = "2"  # stereo master bus
    # One sequence track per routed source channel (A1..An); each track's
    # clipitems carry <sourcetrack><trackindex> = the real source track.
    audio_tracks = [ET.SubElement(audio, "track") for _ in audio_channels]

    file_id = "file-1"
    masterclip_id = "masterclip-1"
    file_element = _build_file_element(
        file_id, source_path, framerate, width, height,
        media_duration_frames, has_video, has_audio,
        audio_channels=file_channelcount or len(audio_channels),
    )

    timeline_offset_frames = 0
    clip_index = 0
    file_emitted = False
    for m in markers:
        try:
            src_in_s = float(m.get("start") or 0)
            src_out_s = float(m.get("end") or 0)
        except (TypeError, ValueError):
            continue
        if src_out_s <= src_in_s:
            continue

        clip_index += 1
        src_in_f = _seconds_to_frames(src_in_s, framerate)
        src_out_f = _seconds_to_frames(src_out_s, framerate)
        # Clamp into the declared media range, mirroring the FCPXML
        # generator — an <out> past <file><duration> relies on importer
        # leniency.
        if media_duration_frames > 0:
            src_in_f = max(0, min(src_in_f, media_duration_frames))
            src_out_f = max(src_in_f, min(src_out_f, media_duration_frames))
        dur_f = src_out_f - src_in_f
        if dur_f <= 0:
            continue
        rec_in_f = timeline_offset_frames
        rec_out_f = rec_in_f + dur_f

        clip_name = scrub_xml_text(m.get("text") or f"Clip {clip_index}")[:80]
        comment = scrub_xml_text((m.get("note") or "").strip())

        if export_mode == "markers":
            # Sequence-level markers at the source position (the single
            # full-length clip below starts at 0, so timeline == source).
            marker_el = ET.SubElement(sequence, "marker")
            ET.SubElement(marker_el, "comment").text = comment
            ET.SubElement(marker_el, "name").text = clip_name
            ET.SubElement(marker_el, "in").text = str(src_in_f)
            ET.SubElement(marker_el, "out").text = str(src_out_f)
            continue

        if has_video:
            _add_clipitem(
                video_track,
                clip_id=f"clipitem-v-{clip_index}",
                name=clip_name,
                file_ref_id=file_id,
                source_in=src_in_f, source_out=src_out_f,
                record_in=rec_in_f, record_out=rec_out_f,
                framerate=framerate,
                media_type="video",
                masterclip_id=masterclip_id,
                reuse_file=file_emitted,
                file_element=None if file_emitted else file_element,
                comment=comment,
            )
            file_emitted = True

        for a_index, (audio_track, src_ch) in enumerate(
                zip(audio_tracks, audio_channels), start=1):
            _add_clipitem(
                audio_track,
                clip_id=f"clipitem-a{a_index}-{clip_index}",
                name=clip_name,
                file_ref_id=file_id,
                source_in=src_in_f, source_out=src_out_f,
                record_in=rec_in_f, record_out=rec_out_f,
                framerate=framerate,
                media_type="audio",
                masterclip_id=masterclip_id,
                reuse_file=file_emitted,
                file_element=None if file_emitted else file_element,
                comment=comment,
                audio_channel=src_ch,
            )
            file_emitted = True

        if export_mode == "both":
            # Cuts + Markers: a sequence-level marker spanning the cut's
            # RECORD range, so Premiere shows a named marker over every clip
            # (the FCPXML side does the same via per-clip chapter-markers).
            marker_el = ET.SubElement(sequence, "marker")
            ET.SubElement(marker_el, "comment").text = comment
            ET.SubElement(marker_el, "name").text = clip_name
            ET.SubElement(marker_el, "in").text = str(rec_in_f)
            ET.SubElement(marker_el, "out").text = str(rec_out_f)

        timeline_offset_frames = rec_out_f

    if export_mode == "markers" and media_duration_frames > 0:
        # One full-length clip under the markers (mirrors the FCPXML
        # markers-only mode: full asset on the timeline, annotations on top).
        full_name = scrub_xml_text(sequence_name)[:80]
        if has_video:
            _add_clipitem(
                video_track,
                clip_id="clipitem-v-full",
                name=full_name,
                file_ref_id=file_id,
                source_in=0, source_out=media_duration_frames,
                record_in=0, record_out=media_duration_frames,
                framerate=framerate,
                media_type="video",
                masterclip_id=masterclip_id,
                reuse_file=file_emitted,
                file_element=None if file_emitted else file_element,
            )
            file_emitted = True
        for a_index, (audio_track, src_ch) in enumerate(
                zip(audio_tracks, audio_channels), start=1):
            _add_clipitem(
                audio_track,
                clip_id=f"clipitem-a{a_index}-full",
                name=full_name,
                file_ref_id=file_id,
                source_in=0, source_out=media_duration_frames,
                record_in=0, record_out=media_duration_frames,
                framerate=framerate,
                media_type="audio",
                masterclip_id=masterclip_id,
                reuse_file=file_emitted,
                file_element=None if file_emitted else file_element,
                audio_channel=src_ch,
            )
            file_emitted = True
        timeline_offset_frames = media_duration_frames

    # Sequence duration should match the assembled timeline if we have clips.
    if timeline_offset_frames > 0:
        sequence.find("duration").text = str(timeline_offset_frames)

    return xmeml


def _sanitize_for_filename(name: str) -> str:
    import re
    return re.sub(r"[^\w\- ]", "_", name).strip()


_PREMIERE_WARNINGS = [
    "Premiere XML uses absolute file paths. Move the source media and the link will break.",
]


class PremiereXMLExporter(BaseExporter):
    format_name = "Premiere XML"
    file_extension = ".xml"
    platform_name = "Premiere Pro"

    def export_markers(
        self,
        markers,
        *,
        project_name,
        source_path,
        media_duration,
        framerate,
        width,
        height,
        export_type,
        exports_dir,
        export_mode="cuts",
        total_clips=0,
        start_tc_frames=0,
        tc_format="NDF",  # interface parity; this target renders its own TC convention  # accepted for interface parity; Premiere uses 0-based file in/out
    ) -> ExportResult:
        # One sequence track per routed source channel (multi-mono sources
        # bring the real dialogue track(s), mono sources a single track).
        audio_channels, file_channelcount = _resolve_audio_wiring(source_path)
        if export_type == "labels" and len(markers) == 1:
            suffix = (markers[0].get("text") or "Clip")[:40].strip()
        elif export_type == "labels":
            total = total_clips or len(markers)
            suffix = "All Clips" if len(markers) >= total else f"{len(markers)} Clips"
        elif export_type == "social":
            suffix = "Social Clips"
        elif export_type == "story":
            suffix = "Story Beats"
        elif export_type == "soundbites":
            suffix = "Soundbites"
        elif export_type == "all":
            suffix = "Full Export"
        else:
            suffix = export_type

        ordered = sorted(
            (m for m in markers if (m.get("end") or 0) > (m.get("start") or 0)),
            key=lambda m: float(m.get("start") or 0),
        )

        sequence_name = f"{project_name.strip()} - {suffix.strip()}"
        root = _build_sequence(
            sequence_name=sequence_name,
            markers=ordered,
            source_path=source_path or "",
            media_duration=media_duration,
            framerate=framerate,
            width=width,
            height=height,
            export_mode=(export_mode if export_mode in ("markers", "both") else "cuts"),
            audio_channels=audio_channels,
            file_channelcount=file_channelcount,
        )
        content = _prettify(root)

        filename = (
            f"{_sanitize_for_filename(project_name)} - {_sanitize_for_filename(suffix)}{self.file_extension}"
            .replace("/", "-")
        )
        file_path = os.path.join(exports_dir, filename)
        os.makedirs(exports_dir, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return ExportResult(
            file_path=file_path,
            filename=filename,
            format_name=self.format_name,
            platform_name=self.platform_name,
            warnings=list(_PREMIERE_WARNINGS),
        )

    def export_story(
        self,
        markers,
        *,
        project_name,
        story_title,
        source_path,
        media_duration,
        framerate,
        width,
        height,
        exports_dir,
        start_tc_frames=0,
        tc_format="NDF",  # interface parity; this target renders its own TC convention  # accepted for interface parity; Premiere uses 0-based file in/out
    ) -> ExportResult:
        # Same audio probe as export_markers — the story path used to skip it
        # and always built A1+A2, giving mono sources a phantom A2 clipitem.
        audio_channels, file_channelcount = _resolve_audio_wiring(source_path)
        ordered = sorted(
            (m for m in markers if (m.get("end") or 0) > (m.get("start") or 0)),
            key=lambda m: m.get("_order", 0),
        )

        sequence_name = f"{project_name.strip()} - {story_title.strip()}"
        root = _build_sequence(
            sequence_name=sequence_name,
            markers=ordered,
            source_path=source_path or "",
            media_duration=media_duration,
            framerate=framerate,
            width=width,
            height=height,
            audio_channels=audio_channels,
            file_channelcount=file_channelcount,
        )
        content = _prettify(root)

        filename = (
            f"{_sanitize_for_filename(project_name)} - {_sanitize_for_filename(story_title)}{self.file_extension}"
            .replace("/", "-")
        )
        file_path = os.path.join(exports_dir, filename)
        os.makedirs(exports_dir, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return ExportResult(
            file_path=file_path,
            filename=filename,
            format_name=self.format_name,
            platform_name=self.platform_name,
            warnings=list(_PREMIERE_WARNINGS),
        )
