"""
EDL (CMX 3600) exporter for DaVinci Resolve.

EDL is a plain-text exchange format every NLE understands. Resolve in
particular reconstructs cuts cleanly when the source media is in the
project bin and the EDL references a matching reel name.

Format limitations surfaced as warnings to the user:
  - No multicam relationships
  - No rich notes (notes go in * COMMENT lines only; markers become
    * LOC: locator lines, which Resolve imports as timeline markers)
  - Reel name truncated to 32 chars
"""

import os

from fcpxml_export import VIDEO_EXTS
from exporters.media_probe import frames_to_timecode_label, has_video_stream
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
    at the integer timebase (24, 25, 30) — CMX 3600's TC convention.

    Hours wrap modulo 24: timecode is a 24h clock, so a 23:30:00:00
    time-of-day reel plus a 40-minute select must render 00:10:00:00,
    not the 24:10:00:00 some EDL parsers reject outright."""
    if total_frames < 0:
        total_frames = 0
    frames = total_frames % fps_int
    total_seconds = total_frames // fps_int
    secs = total_seconds % 60
    mins = (total_seconds // 60) % 60
    hours = (total_seconds // 3600) % 24
    return f"{hours:02d}:{mins:02d}:{secs:02d}:{frames:02d}"


def _render_timecode(total_frames: int, framerate: float, drop: bool) -> str:
    """One TC label for the EDL: NDF at the integer base, or SMPTE
    drop-frame (';' separators, skips re-inserted) when the file is
    declared FCM: DROP FRAME. Hours wrap modulo 24 on both paths."""
    fps_int = _timebase(framerate)
    if drop:
        # Frame count of 24h of drop-frame TC: 2 frames dropped per minute
        # (×fps/30) except every tenth minute — wrap before re-labeling.
        day_frames = 24 * 3600 * fps_int - 2 * (fps_int // 30) * (24 * 60 - 24 * 6)
        return frames_to_timecode_label(total_frames % day_frames, framerate, drop=True)
    return _frames_to_timecode(total_frames, fps_int)


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


# Locator colors shared by Resolve's and Avid's EDL marker dialects; anything
# outside the intersection (e.g. our 'orange' label) falls back to BLUE
# rather than risking a color word a parser treats as part of the name.
_LOC_COLORS = {"blue", "cyan", "green", "yellow", "red", "pink", "purple"}


def _loc_color(color) -> str:
    c = (color or "").strip().lower()
    return c.upper() if c in _LOC_COLORS else "BLUE"


def _build_edl(
    title: str,
    markers: list,
    source_path: str,
    framerate: float,
    sequential_record: bool,
    start_tc_frames: int = 0,
    tc_format: str = "NDF",
    export_mode: str = "cuts",
    media_duration: float | None = None,
) -> tuple[str, list]:
    """
    Render the EDL text. Returns ``(content, warnings)``.

    sequential_record=True -> record TC starts at 01:00:00:00 and accumulates
                              clip durations (used for story sequences and
                              clip-style exports).
    sequential_record=False -> record TC mirrors source TC + 1h offset (used
                               when exporting markers that should keep their
                               original positions).

    export_mode: "cuts" (default) emits one event per marker; "markers"
    emits ONE full-length event with a ``* LOC:`` locator line per marker
    (Resolve imports locators as timeline markers, so 'Markers Only' lands
    as annotations instead of a re-cut timeline); "both" emits the cut
    events AND a locator at each cut's record-in.

    tc_format="DF" declares FCM: DROP FRAME and renders every TC column
    with ';' separators so labels match the camera's drop-frame TC —
    frame indices were always consistent, but the old NDF-only render
    drifted the LABELS 108 frames per TC-hour off the media's own.
    """
    reel = _sanitize_reel_name(source_path)
    clip_basename = os.path.basename(source_path) if source_path else "Source"

    # EDL is line-oriented: a newline inside any interpolated text splits a
    # comment/header across lines and strict CMX parsers reject the file.
    # Collapse all whitespace runs, mirroring the note handling below.
    title = " ".join((title or "").split())
    clip_basename = " ".join(clip_basename.split())

    # Audio-only sources must not claim a video component: Resolve shows
    # offline video for AA/V events whose media has no picture. Decided by
    # probing for a REAL video stream — the extension alone mislabels an
    # audio-only .mov/.mp4 (AAC podcast export, field-recorder QuickTime)
    # as video. Extension heuristic only when the probe can't answer
    # (missing file / no ffprobe), mirroring fcpxml_export._resolve_is_video.
    has_video = has_video_stream(source_path) if source_path else None
    if has_video is None:
        ext = os.path.splitext(source_path)[1].lower() if source_path else ""
        has_video = ext in VIDEO_EXTS
    channel = "AA/V " if has_video else "AA   "

    # Drop-frame only exists in the 30-frame family (29.97/59.94); a DF tag
    # on any other rate is a corrupt probe — keep the safe NDF declaration.
    drop = tc_format == "DF" and _timebase(framerate) % 30 == 0
    warnings: list = []

    lines = [f"TITLE: {title}",
             "FCM: DROP FRAME" if drop else "FCM: NON-DROP FRAME", ""]

    # All arithmetic in whole frames: source positions are converted once
    # (at the actual NTSC rate — see _seconds_to_frames), and record TC is
    # accumulated in frames from exactly one TC-hour so the conventional
    # 01:00:00:00 record start holds on every framerate.
    fps_int = _timebase(framerate)
    hour_frames = 3600 * fps_int  # 01:00:00:00 on the TC grid
    if drop:
        # Drop-frame labels skip 2 frame NUMBERS per minute (×fps/30) except
        # every tenth minute, so the frame COUNT whose DF label reads
        # 01:00:00;00 is 108 short of 3600×fps — without this the record
        # column would start at 01:00:03;18 instead of the one-hour mark.
        hour_frames -= 2 * (fps_int // 30) * (60 - 6)
    mode = export_mode if export_mode in ("markers", "both") else "cuts"

    # Validate up front so the event count is known before any line is
    # rendered — the event-number field width must be uniform per file.
    events = []  # (marker, src_in_f, src_out_f)
    for m in markers:
        try:
            src_in = float(m.get("start", 0) or 0)
            src_out = float(m.get("end", 0) or 0)
        except (TypeError, ValueError):
            continue
        src_in_f = _seconds_to_frames(src_in, framerate)
        src_out_f = _seconds_to_frames(src_out, framerate)
        if src_out_f - src_in_f <= 0:
            continue
        events.append((m, src_in_f, src_out_f))

    if mode == "markers":
        # One event carrying the whole source, locators on top — the marker
        # positions ride the record TC (source position + the 1h start).
        if not media_duration:
            if events:
                media_duration = max(
                    float(m.get("end") or 0) for m, _, _ in events) + 10.0
            else:
                media_duration = 60.0
        media_frames = max(1, _seconds_to_frames(media_duration, framerate))
        lines.append(
            f"001  {reel:<8} {channel} C        "
            f"{_render_timecode(start_tc_frames, framerate, drop)} "
            f"{_render_timecode(start_tc_frames + media_frames, framerate, drop)} "
            f"{_render_timecode(hour_frames, framerate, drop)} "
            f"{_render_timecode(hour_frames + media_frames, framerate, drop)}"
        )
        lines.append(f"* FROM CLIP NAME: {clip_basename}")
        for m, src_in_f, _src_out_f in events:
            loc_tc = _render_timecode(hour_frames + src_in_f, framerate, drop)
            loc_name = " ".join((m.get("text") or "Marker").split())
            lines.append(f"* LOC: {loc_tc} {_loc_color(m.get('color'))} {loc_name}")
        lines.append("")
        return "\n".join(lines).rstrip() + "\n", warnings

    # CMX 3600 caps a reel at 999 events. Erroring would strand a long
    # Full Transcript export, and wrapping modulo 999 duplicates event
    # numbers (breaking match-back and dupe detection), so past the cap we
    # widen the number field for the WHOLE file (0001…1005) — the precedent
    # Premiere Pro set (it zero-pads all its event numbers wide), keeping
    # every line's columns aligned for tokenizing parsers like Resolve —
    # and surface the strict-CMX risk as a warning instead of writing a
    # silently malformed file.
    num_width = 3 if len(events) <= 999 else len(str(len(events)))
    if len(events) > 999:
        # ASCII only: warnings ride the latin-1-strict X-Export-Warnings
        # HTTP header on the legacy attachment path (see app.py), where a
        # single em-dash kills the whole response mid-header.
        warnings.append(
            f"EDL contains {len(events)} events - beyond the CMX 3600 limit "
            "of 999. Resolve and Premiere import the full list, but strict "
            "CMX-only tools may truncate events past 999."
        )

    record_offset_frames = hour_frames
    edit_num = 0
    for m, src_in_f, src_out_f in events:
        dur_f = src_out_f - src_in_f
        edit_num += 1

        if sequential_record:
            rec_in_f = record_offset_frames
            rec_out_f = record_offset_frames + dur_f
            record_offset_frames += dur_f
        else:
            # Record mirrors source TC + 1 hour (markers keep positions).
            rec_in_f = hour_frames + src_in_f
            rec_out_f = hour_frames + src_out_f

        src_in_tc = _render_timecode(start_tc_frames + src_in_f, framerate, drop)
        src_out_tc = _render_timecode(start_tc_frames + src_out_f, framerate, drop)
        rec_in_tc = _render_timecode(rec_in_f, framerate, drop)
        rec_out_tc = _render_timecode(rec_out_f, framerate, drop)

        lines.append(
            f"{edit_num:0{num_width}d}  {reel:<8} {channel} C        "
            f"{src_in_tc} {src_out_tc} {rec_in_tc} {rec_out_tc}"
        )
        clip_name = " ".join((m.get("text") or f"Clip {edit_num}").split())
        lines.append(f"* FROM CLIP NAME: {clip_basename}")
        if clip_name:
            lines.append(f"* CLIP NAME: {clip_name}")
        if mode == "both":
            # Cuts + Markers: a locator at the cut's record-in mirrors the
            # FCPXML side's per-clip chapter-markers.
            lines.append(
                f"* LOC: {rec_in_tc} {_loc_color(m.get('color'))} "
                f"{clip_name or f'Marker {edit_num}'}"
            )
        note = (m.get("note") or "").strip()
        if note:
            # EDL comments should be single-line; collapse newlines.
            note = " ".join(note.split())
            lines.append(f"* COMMENT: {note}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n", warnings


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
        tc_format="NDF",  # "DF" renders FCM: DROP FRAME + ';' TC labels
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
        content, extra_warnings = _build_edl(
            title=title,
            markers=ordered,
            source_path=source_path or "",
            framerate=framerate,
            sequential_record=True,
            start_tc_frames=start_tc_frames,
            tc_format=tc_format,
            export_mode=export_mode,
            media_duration=media_duration,
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
            warnings=list(_EDL_WARNINGS) + extra_warnings,
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
        tc_format="NDF",  # "DF" renders FCM: DROP FRAME + ';' TC labels
    ) -> ExportResult:
        # Story markers may carry an _order field; preserve it the way the
        # FCPXML story exporter does.
        ordered = sorted(
            (m for m in markers if (m.get("end") or 0) > (m.get("start") or 0)),
            key=lambda m: m.get("_order", 0),
        )

        title = f"{project_name.strip()} - {story_title.strip()}"
        content, extra_warnings = _build_edl(
            title=title,
            markers=ordered,
            source_path=source_path or "",
            framerate=framerate,
            sequential_record=True,
            start_tc_frames=start_tc_frames,
            tc_format=tc_format,
            media_duration=media_duration,
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
            warnings=list(_EDL_WARNINGS) + extra_warnings,
        )
