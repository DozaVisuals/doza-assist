"""Render an FCPXML sequence's dialogue down to a single timeline-space WAV.

Used by the ingest path when a spine has multiple audio sources (mixed
mc-clip / sync-clip containers, or mc-clips referencing different multicams).
The transcription pipeline takes one audio file and emits timestamps against
it, so we compose a WAV whose time axis matches the sequence timeline: source
audio for each unmuted spine segment is trimmed to its in-segment range and
placed at that segment's timeline offset, with silence everywhere else.

Muted segments (per :class:`SegmentAudioSource.is_muted`) are deliberately
silent — this matches what FCP plays on the timeline.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from fractions import Fraction
from typing import List, Optional, Tuple

from .parser import ParsedFCPXML, SpineSegment


_log = logging.getLogger(__name__)


class TimelineAudioError(RuntimeError):
    """Raised when the timeline audio renderer cannot produce a WAV."""


def _find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    for candidate in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if os.path.isfile(candidate):
            return candidate
    return "ffmpeg"


def _find_ffprobe() -> Optional[str]:
    # Bundled binary first: packaged installs may have NO ffprobe on PATH
    # (same resolution order as exporters.media_probe). Returns None when
    # unavailable so callers degrade to not probing.
    bundled_dir = os.environ.get("DOZA_FFMPEG_DIR")
    if bundled_dir:
        candidate = os.path.join(bundled_dir, "ffprobe")
        if os.path.isfile(candidate):
            return candidate
    path = shutil.which("ffprobe")
    if path:
        return path
    for candidate in ("/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe"):
        if os.path.isfile(candidate):
            return candidate
    return None


def _input_has_audio_stream(path: str) -> bool:
    """True unless ffprobe positively reports zero audio streams.

    Feeding an audio-less file (video-only B-roll whose asset over-declared
    audio) into the amix filtergraph aborts the WHOLE render on a missing
    ``[n:a]`` stream, so those inputs must be dropped to silence instead.
    Errs on the side of keeping the input: no ffprobe, a probe error, or an
    unreadable file all return True, degrading to the status quo where ffmpeg
    itself surfaces the real problem.
    """
    ffprobe = _find_ffprobe()
    if not ffprobe:
        return True
    try:
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
        return True
    if probe.returncode != 0:
        return True  # unreadable is not the same as audio-less
    return bool(probe.stdout.strip())


def _segment_source_window(seg: SpineSegment) -> Tuple[Fraction, Fraction]:
    """Return (source_start_seconds, duration_seconds) for the segment's audio.

    Container time ranges ``[seg.start, seg.start + seg.duration)``. Within the
    container, the audio asset-clip is positioned at ``angle_offset`` with its
    own ``angle_start``. With timecode origins factored in, the mapping is:

      source_seek = (seg.start - container_tc_start)        # zero-based container time
                    - angle_offset                          # asset-clip's position in container
                    + (angle_start - asset_start)           # zero-based seek into source media

    For tcStart-zero multicams and sync-clips, ``container_tc_start`` and
    ``asset_start`` are both zero, so this collapses to the original
    ``seg.start - angle_offset + angle_start`` formula. For jam-synced /
    time-of-day TC multicams (Panasonic, RED, ARRI, Sony FX), subtracting both
    is what brings the seek position back inside the actual media file.
    """
    src = seg.audio_source
    assert src is not None, "segment audio_source must be resolved before rendering"
    source_start = (
        (seg.start_fraction - src.container_tc_start_fraction)
        - src.angle_offset_fraction
        + (src.angle_start_fraction - src.asset_start_fraction)
    )
    return source_start, seg.duration_fraction


def _fraction_to_ms(f: Fraction) -> int:
    # Round to the nearest millisecond. Sub-millisecond drift is acceptable for
    # transcript-grade timestamps — one frame @ 23.976 fps is already ~42ms.
    return int(round(float(f) * 1000))


def _fraction_to_seconds_str(f: Fraction) -> str:
    # ffmpeg filter arguments accept decimal seconds. Use 6 decimals
    # (microsecond precision) — well below a frame.
    return f"{float(f):.6f}"


def plan_render(parsed: ParsedFCPXML) -> List[dict]:
    """Return the render plan. Exposed for testing.

    Each entry: {segment_index, input_path, source_start_seconds,
    duration_seconds, timeline_offset_ms}. One entry per segment — except a
    segmented multicam angle (``seg.audio_parts`` > 1: many stop-start camera
    files under one mc-clip), which emits one entry per part so the composed
    WAV plays every file at its true timeline position; the gaps between
    files stay silence.
    """
    plan: List[dict] = []
    for i, seg in enumerate(parsed.spine_segments):
        if seg.audio_source is None or seg.audio_source.is_muted:
            continue
        if seg.duration_fraction <= 0:
            continue
        parts = getattr(seg, "audio_parts", None) or []
        if len(parts) > 1:
            # Per-part windows, intersected with the segment's zero-based
            # container window. Mirrors _segment_source_window's mapping with
            # the part's own offset/in-point:
            #   source_seek = (win_lo - part_offset) + (part_start - asset_start)
            #   delay       = seg.offset + (win_lo - segment_window_lo)
            tc = seg.audio_source.container_tc_start_fraction
            seg_win_lo = seg.start_fraction - tc
            seg_win_hi = seg_win_lo + seg.duration_fraction
            for part in parts:
                if part.is_muted or part.part_duration_fraction <= 0:
                    continue
                win_lo = max(part.angle_offset_fraction, seg_win_lo)
                win_hi = min(
                    part.angle_offset_fraction + part.part_duration_fraction,
                    seg_win_hi,
                )
                if win_hi <= win_lo:
                    continue
                source_start = (
                    (win_lo - part.angle_offset_fraction)
                    + (part.angle_start_fraction - part.asset_start_fraction)
                )
                if source_start < 0:
                    source_start = Fraction(0)
                delay = seg.offset_fraction + (win_lo - seg_win_lo)
                plan.append({
                    "segment_index": i,
                    "input_path": part.path,
                    "source_start_seconds": float(source_start),
                    "duration_seconds": float(win_hi - win_lo),
                    "timeline_offset_ms": _fraction_to_ms(delay),
                    "source_start_fraction": source_start,
                    "duration_fraction": win_hi - win_lo,
                    "timeline_offset_fraction": delay,
                })
            continue
        source_start, duration = _segment_source_window(seg)
        # Guard against negative trim starts (malformed input — fall back to 0).
        if source_start < 0:
            source_start = Fraction(0)
        plan.append({
            "segment_index": i,
            "input_path": seg.audio_source.path,
            "source_start_seconds": float(source_start),
            "duration_seconds": float(duration),
            "timeline_offset_ms": _fraction_to_ms(seg.offset_fraction),
            "source_start_fraction": source_start,
            "duration_fraction": duration,
            "timeline_offset_fraction": seg.offset_fraction,
        })
    return plan


def build_ffmpeg_command(
    parsed: ParsedFCPXML,
    output_path: str,
    *,
    sample_rate: int = 16000,
    ffmpeg_bin: Optional[str] = None,
    plan: Optional[List[dict]] = None,
) -> List[str]:
    """Build the ffmpeg argv for the given parsed FCPXML.

    Structure:
      - Input 0 is an ``anullsrc`` sized to the sequence duration; sets the
        output length so muted gaps become silence.
      - Inputs 1..N are the per-segment source files (duplicates allowed when
        a single file is referenced by multiple segments).
      - A ``filter_complex`` trims each segment's source range, resamples to
        the target sample rate and mono, delays it to the timeline offset,
        and amixes everything against the null base.

    ``plan`` lets :func:`render_timeline_audio` pass a filtered render plan
    (missing/audio-less inputs dropped to silence); defaults to the full
    :func:`plan_render` output.
    """
    if plan is None:
        plan = plan_render(parsed)
    if not plan:
        raise TimelineAudioError("no unmuted spine segments with resolvable audio")

    seq_secs = _fraction_to_seconds_str(parsed.timeline_duration_fraction)

    argv: List[str] = [ffmpeg_bin or _find_ffmpeg(), "-y", "-nostdin"]

    # Input 0: silent base of exactly sequence duration.
    argv += [
        "-f", "lavfi",
        "-i", f"anullsrc=r={sample_rate}:cl=mono:d={seq_secs}",
    ]

    # Inputs 1..N: segment sources.
    for item in plan:
        argv += ["-i", item["input_path"]]

    # Filter chain.
    filter_parts: List[str] = []
    label_out_idx: List[str] = ["[0:a]"]  # base silence first
    for n, item in enumerate(plan, start=1):
        src_start = _fraction_to_seconds_str(item["source_start_fraction"])
        dur = _fraction_to_seconds_str(item["duration_fraction"])
        delay_ms = item["timeline_offset_ms"]
        lbl = f"[s{n}]"
        filter_parts.append(
            f"[{n}:a]atrim=start={src_start}:duration={dur},"
            f"asetpts=PTS-STARTPTS,"
            f"aformat=sample_rates={sample_rate}:channel_layouts=mono,"
            f"adelay={delay_ms}{lbl}"
        )
        label_out_idx.append(lbl)

    mix_inputs = "".join(label_out_idx)
    filter_parts.append(
        f"{mix_inputs}amix=inputs={len(label_out_idx)}:"
        f"duration=first:normalize=0:dropout_transition=0[out]"
    )
    argv += ["-filter_complex", ";".join(filter_parts)]

    argv += [
        "-map", "[out]",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        output_path,
    ]
    return argv


def render_timeline_audio(
    parsed: ParsedFCPXML,
    output_path: str,
    *,
    sample_rate: int = 16000,
    ffmpeg_bin: Optional[str] = None,
    skip_missing: bool = False,
) -> str:
    """Compose the sequence's dialogue to a single WAV at ``output_path``.

    Verifies every referenced source exists before shelling out; raises
    :class:`TimelineAudioError` if any segment source is missing (typical when
    an edit drive is unmounted) or if ffmpeg fails. With ``skip_missing`` the
    offline segments render as silence instead — the My Style path's
    partial-import promise (skip the clip, log it, continue); the project
    ingest path keeps the strict default because it pre-validates with
    drive-mount hints. Inputs that exist but carry no audio stream are always
    dropped to silence (see :func:`_input_has_audio_stream`).
    """
    plan = plan_render(parsed)
    if not plan:
        raise TimelineAudioError("no unmuted spine segments with resolvable audio")

    missing = [item["input_path"] for item in plan
               if not os.path.exists(item["input_path"])]
    if missing:
        # Deduplicate while preserving order for a stable error message.
        seen = set()
        unique = [m for m in missing if not (m in seen or seen.add(m))]
        if not skip_missing:
            raise TimelineAudioError(
                "missing audio source(s): " + ", ".join(unique)
            )
        _log.warning(
            "timeline render: skipping offline source(s), their segments "
            "become silence: %s", ", ".join(unique),
        )
        missing_set = set(unique)
        plan = [item for item in plan if item["input_path"] not in missing_set]

    audioless = {p for p in {item["input_path"] for item in plan}
                 if not _input_has_audio_stream(p)}
    if audioless:
        _log.warning(
            "timeline render: skipping audio-less source(s), their segments "
            "become silence: %s", ", ".join(sorted(audioless)),
        )
        plan = [item for item in plan if item["input_path"] not in audioless]

    if not plan:
        raise TimelineAudioError(
            "no renderable audio: every unmuted segment's source is missing "
            "or has no audio stream"
        )

    argv = build_ffmpeg_command(
        parsed, output_path,
        sample_rate=sample_rate, ffmpeg_bin=ffmpeg_bin, plan=plan,
    )
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        raise TimelineAudioError(
            f"ffmpeg timeline render failed: {result.stderr[-1000:]}"
        )
    return output_path
