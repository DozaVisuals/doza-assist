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

import os
import shutil
import subprocess
from fractions import Fraction
from typing import List, Optional, Tuple

from .parser import ParsedFCPXML, SpineSegment


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


def _segment_source_window(seg: SpineSegment) -> Tuple[Fraction, Fraction]:
    """Return (source_start_seconds, duration_seconds) for the segment's audio.

    Container time ranges ``[seg.start, seg.start + seg.duration)``. Within the
    container, the audio asset-clip is positioned at ``angle_offset`` with its
    own ``angle_start``. Mapping from container time C to source time is
    ``C - angle_offset + angle_start``.
    """
    src = seg.audio_source
    assert src is not None, "segment audio_source must be resolved before rendering"
    source_start = seg.start_fraction - src.angle_offset_fraction + src.angle_start_fraction
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
    """Return the per-segment render plan. Exposed for testing.

    Each entry: {segment_index, input_path, source_start_seconds,
    duration_seconds, timeline_offset_ms}.
    """
    plan: List[dict] = []
    for i, seg in enumerate(parsed.spine_segments):
        if seg.audio_source is None or seg.audio_source.is_muted:
            continue
        if seg.duration_fraction <= 0:
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
    """
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
) -> str:
    """Compose the sequence's dialogue to a single WAV at ``output_path``.

    Verifies every referenced source exists before shelling out; raises
    :class:`TimelineAudioError` if any segment source is missing (typical when
    an edit drive is unmounted) or if ffmpeg fails.
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
        raise TimelineAudioError(
            "missing audio source(s): " + ", ".join(unique)
        )

    argv = build_ffmpeg_command(
        parsed, output_path,
        sample_rate=sample_rate, ffmpeg_bin=ffmpeg_bin,
    )
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        raise TimelineAudioError(
            f"ffmpeg timeline render failed: {result.stderr[-1000:]}"
        )
    return output_path
