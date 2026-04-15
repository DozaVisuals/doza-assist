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
    59.94:  (60, True,  60000.0 / 1001.0),
    60.0:   (60, False, 60.0),
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
                        has_video: bool, has_audio: bool) -> ET.Element:
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
        ET.SubElement(audio, "channelcount").text = "2"

    return file_el


def _add_clipitem(track: ET.Element, *, clip_id: str, name: str, file_ref_id: str,
                  source_in: int, source_out: int, record_in: int, record_out: int,
                  framerate: float, media_type: str, masterclip_id: str,
                  reuse_file: bool, file_element: ET.Element | None,
                  comment: str = "") -> None:
    """media_type is 'video' or 'audio' — affects which track this lands on."""
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
    return ext in (".mp4", ".mov", ".mxf", ".avi", ".mkv", ".m4v")


def _build_sequence(
    *,
    sequence_name: str,
    markers: list,
    source_path: str,
    media_duration: float | None,
    framerate: float,
    width: int,
    height: int,
) -> ET.Element:
    has_video = _is_video_source(source_path)
    has_audio = True  # transcribed sources always have audio

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
    ET.SubElement(audio, "numOutputChannels").text = "2"
    audio_track_1 = ET.SubElement(audio, "track")
    audio_track_2 = ET.SubElement(audio, "track")

    file_id = "file-1"
    masterclip_id = "masterclip-1"
    file_element = _build_file_element(
        file_id, source_path, framerate, width, height,
        media_duration_frames, has_video, has_audio,
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
        dur_f = src_out_f - src_in_f
        if dur_f <= 0:
            continue
        rec_in_f = timeline_offset_frames
        rec_out_f = rec_in_f + dur_f

        clip_name = (m.get("text") or f"Clip {clip_index}")[:80]
        comment = (m.get("note") or "").strip()

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

        for ch_index, audio_track in enumerate((audio_track_1, audio_track_2), start=1):
            _add_clipitem(
                audio_track,
                clip_id=f"clipitem-a{ch_index}-{clip_index}",
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
            )
            file_emitted = True

        timeline_offset_frames = rec_out_f

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
    ) -> ExportResult:
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
        )
        content = _prettify(root)

        filename = (
            f"{_sanitize_for_filename(project_name)} - {_sanitize_for_filename(suffix)}{self.file_extension}"
            .replace("/", "-")
        )
        file_path = os.path.join(exports_dir, filename)
        os.makedirs(exports_dir, exist_ok=True)
        with open(file_path, "w") as f:
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
    ) -> ExportResult:
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
        )
        content = _prettify(root)

        filename = (
            f"{_sanitize_for_filename(project_name)} - {_sanitize_for_filename(story_title)}{self.file_extension}"
            .replace("/", "-")
        )
        file_path = os.path.join(exports_dir, filename)
        os.makedirs(exports_dir, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)

        return ExportResult(
            file_path=file_path,
            filename=filename,
            format_name=self.format_name,
            platform_name=self.platform_name,
            warnings=list(_PREMIERE_WARNINGS),
        )
