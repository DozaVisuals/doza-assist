"""
EDL (CMX 3600) exporter for DaVinci Resolve.

EDL is a plain-text exchange format every NLE understands. Resolve in
particular reconstructs cuts cleanly when the source media is in the
project bin and the EDL references a matching reel name.

Format limitations surfaced as warnings to the user:
  - No multicam relationships
  - No color labels or rich notes (notes go in * COMMENT lines only)
  - Reel name truncated to 32 chars
"""

import os
import re

from .base import BaseExporter, ExportResult

def _timebase(framerate: float) -> int:
    """Return the integer timebase for a given framerate (23.976 -> 24, 29.97 -> 30)."""
    return int(round(framerate + 0.001))  # +0.001 nudges 23.976 to 24 cleanly


def _seconds_to_timecode(seconds: float, framerate: float) -> str:
    """
    HH:MM:SS:FF strict 8-character format, non-drop frame.

    CMX 3600 stores integer frames at the integer timebase (24, 25, 30). For
    NTSC-rate content (23.976, 29.97, 59.94) we use the same integer timebase
    so the timecode walks consistently — Resolve treats this as 1:1 frame
    mapping with the source media on import.
    """
    if seconds < 0:
        seconds = 0.0
    fps_int = _timebase(framerate)
    total_frames = int(round(seconds * fps_int))
    frames = total_frames % fps_int
    total_seconds = total_frames // fps_int
    secs = total_seconds % 60
    mins = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{mins:02d}:{secs:02d}:{frames:02d}"


def _sanitize_reel_name(source_path: str) -> str:
    """Derive an EDL-safe reel name from the source filename."""
    if not source_path:
        return "AX"
    base = os.path.splitext(os.path.basename(source_path))[0]
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", base).upper()
    cleaned = cleaned.strip("_") or "AX"
    return cleaned[:32]


def _sanitize_for_filename(name: str) -> str:
    return re.sub(r"[^\w\- ]", "_", name).strip()


def _build_edl(
    title: str,
    markers: list,
    source_path: str,
    framerate: float,
    sequential_record: bool,
) -> str:
    """
    Render the EDL text.

    sequential_record=True -> record TC starts at 01:00:00:00 and accumulates
                              clip durations (used for story sequences and
                              clip-style exports).
    sequential_record=False -> record TC mirrors source TC + 1h offset (used
                               when exporting markers that should keep their
                               original positions).
    """
    reel = _sanitize_reel_name(source_path)
    clip_basename = os.path.basename(source_path) if source_path else "Source"

    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]

    record_offset = 3600.0  # 01:00:00:00
    edit_num = 0
    for m in markers:
        try:
            src_in = float(m.get("start", 0) or 0)
            src_out = float(m.get("end", 0) or 0)
        except (TypeError, ValueError):
            continue
        dur = src_out - src_in
        if dur <= 0:
            continue
        edit_num += 1

        rec_in = record_offset
        rec_out = record_offset + dur
        if sequential_record:
            record_offset += dur

        src_in_tc = _seconds_to_timecode(src_in, framerate)
        src_out_tc = _seconds_to_timecode(src_out, framerate)
        rec_in_tc = _seconds_to_timecode(rec_in, framerate)
        rec_out_tc = _seconds_to_timecode(rec_out, framerate)

        lines.append(
            f"{edit_num:03d}  {reel:<8} AA/V  C        "
            f"{src_in_tc} {src_out_tc} {rec_in_tc} {rec_out_tc}"
        )
        clip_name = (m.get("text") or f"Clip {edit_num}").strip()
        lines.append(f"* FROM CLIP NAME: {clip_basename}")
        if clip_name:
            lines.append(f"* CLIP NAME: {clip_name}")
        note = (m.get("note") or "").strip()
        if note:
            # EDL comments should be single-line; collapse newlines.
            note = " ".join(note.split())
            lines.append(f"* COMMENT: {note}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


_EDL_WARNINGS = [
    "EDL does not preserve multicam, color labels, or rich notes.",
    "Import the source media into your Resolve project bin before importing the EDL.",
]


class EDLExporter(BaseExporter):
    format_name = "EDL"
    file_extension = ".edl"
    platform_name = "DaVinci Resolve"

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

        # Sort by start time so the EDL is monotonic.
        ordered = sorted(
            (m for m in markers if (m.get("end") or 0) > (m.get("start") or 0)),
            key=lambda m: float(m.get("start") or 0),
        )

        title = f"{project_name.strip()} - {suffix.strip()}"
        content = _build_edl(
            title=title,
            markers=ordered,
            source_path=source_path or "",
            framerate=framerate,
            sequential_record=True,
        )

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
            warnings=list(_EDL_WARNINGS),
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
        # Story markers may carry an _order field; preserve it the way the
        # FCPXML story exporter does.
        ordered = sorted(
            (m for m in markers if (m.get("end") or 0) > (m.get("start") or 0)),
            key=lambda m: m.get("_order", 0),
        )

        title = f"{project_name.strip()} - {story_title.strip()}"
        content = _build_edl(
            title=title,
            markers=ordered,
            source_path=source_path or "",
            framerate=framerate,
            sequential_record=True,
        )

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
            warnings=list(_EDL_WARNINGS),
        )
