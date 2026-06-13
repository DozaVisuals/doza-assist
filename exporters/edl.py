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

from fcpxml_export import VIDEO_EXTS
import re

from .base import BaseExporter, ExportResult

def _timebase(framerate: float) -> int:
    """Return the integer timebase for a given framerate (23.976 -> 24, 29.97 -> 30)."""
    return int(round(framerate + 0.001))  # +0.001 nudges 23.976 to 24 cleanly


# True frame cadence for the NTSC family. The frame INDEX of a moment t
# seconds into the media is round(t * actual_fps); the integer timebase is
# only how that index is rendered as HH:MM:SS:FF (non-drop).
_NTSC_ACTUAL_FPS = {
    23.976: 24000.0 / 1001.0,
    29.97: 30000.0 / 1001.0,
    59.94: 60000.0 / 1001.0,
}


def _actual_fps(framerate: float) -> float:
    return _NTSC_ACTUAL_FPS.get(framerate, float(framerate))


def _seconds_to_frames(seconds: float, framerate: float) -> int:
    """Media frame index of the moment ``seconds`` into the file.

    Must be computed at the media's ACTUAL rate: counting at the integer
    rate for NTSC media (the old behavior, seconds*24 for 23.976) overshot
    the real frame index by 1/1000 — ~86 frames (3.6s) of drift per hour,
    so selects late in a long interview landed seconds off in Resolve.
    This mirrors the (correct) Premiere exporter's actual_fps handling.
    """
    if seconds < 0:
        seconds = 0.0
    return int(round(seconds * _actual_fps(framerate)))


def _frames_to_timecode(total_frames: int, fps_int: int) -> str:
    """Render a whole-frame count as strict 8-char non-drop HH:MM:SS:FF
    at the integer timebase (24, 25, 30) — CMX 3600's TC convention."""
    if total_frames < 0:
        total_frames = 0
    frames = total_frames % fps_int
    total_seconds = total_frames // fps_int
    secs = total_seconds % 60
    mins = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{mins:02d}:{secs:02d}:{frames:02d}"


def _seconds_to_timecode(seconds: float, framerate: float) -> str:
    """Convenience: media moment -> NDF timecode string."""
    return _frames_to_timecode(
        _seconds_to_frames(seconds, framerate), _timebase(framerate),
    )


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
    start_tc_frames: int = 0,
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

    # EDL is line-oriented: a newline inside any interpolated text splits a
    # comment/header across lines and strict CMX parsers reject the file.
    # Collapse all whitespace runs, mirroring the note handling below.
    title = " ".join((title or "").split())
    clip_basename = " ".join(clip_basename.split())

    # Audio-only sources must not claim a video component: Resolve shows
    # offline video for AA/V events whose media has no picture.
    ext = os.path.splitext(source_path)[1].lower() if source_path else ""
    channel = "AA/V " if ext in VIDEO_EXTS else "AA   "

    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]

    # All arithmetic in whole frames: source positions are converted once
    # (at the actual NTSC rate — see _seconds_to_frames), and record TC is
    # accumulated in frames from exactly one TC-hour so the conventional
    # 01:00:00:00 record start holds on every framerate.
    fps_int = _timebase(framerate)
    hour_frames = 3600 * fps_int  # 01:00:00:00 on the TC grid
    record_offset_frames = hour_frames
    edit_num = 0
    for m in markers:
        try:
            src_in = float(m.get("start", 0) or 0)
            src_out = float(m.get("end", 0) or 0)
        except (TypeError, ValueError):
            continue
        src_in_f = _seconds_to_frames(src_in, framerate)
        src_out_f = _seconds_to_frames(src_out, framerate)
        dur_f = src_out_f - src_in_f
        if dur_f <= 0:
            continue
        edit_num += 1

        if sequential_record:
            rec_in_f = record_offset_frames
            rec_out_f = record_offset_frames + dur_f
            record_offset_frames += dur_f
        else:
            # Record mirrors source TC + 1 hour (markers keep positions).
            rec_in_f = hour_frames + src_in_f
            rec_out_f = hour_frames + src_out_f

        src_in_tc = _frames_to_timecode(start_tc_frames + src_in_f, fps_int)
        src_out_tc = _frames_to_timecode(start_tc_frames + src_out_f, fps_int)
        rec_in_tc = _frames_to_timecode(rec_in_f, fps_int)
        rec_out_tc = _frames_to_timecode(rec_out_f, fps_int)

        lines.append(
            f"{edit_num:03d}  {reel:<8} {channel} C        "
            f"{src_in_tc} {src_out_tc} {rec_in_tc} {rec_out_tc}"
        )
        clip_name = " ".join((m.get("text") or f"Clip {edit_num}").split())
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
        start_tc_frames=0,
        tc_format="NDF",  # interface parity; this target renders its own TC convention  # embedded source TC — added to source in/out (Resolve conform)
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
            start_tc_frames=start_tc_frames,
        )

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
        start_tc_frames=0,
        tc_format="NDF",  # interface parity; this target renders its own TC convention  # embedded source TC — added to source in/out (Resolve conform)
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
            start_tc_frames=start_tc_frames,
        )

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
            warnings=list(_EDL_WARNINGS),
        )
