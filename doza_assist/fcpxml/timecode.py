"""FCPXML rational-time utilities and timeline translation.

FCPXML encodes every time value as a rational number with an 's' suffix, e.g.
``1017016/24000s`` or ``28626s``. All math here is done with :class:`fractions.Fraction`
to stay frame-exact; callers can drop to ``float`` at the edges for display or
for passing to the transcription pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Optional


def parse_rational(value: Optional[str]) -> Fraction:
    """Parse an FCPXML rational timecode string into a :class:`Fraction`.

    Accepts ``"1017016/24000s"``, ``"28626s"``, ``"0s"``, or ``None`` (→ 0).
    """
    if value is None:
        return Fraction(0)
    v = value.strip()
    if not v:
        return Fraction(0)
    if v.endswith("s"):
        v = v[:-1]
    if "/" in v:
        num, _, den = v.partition("/")
        return Fraction(int(num), int(den))
    return Fraction(int(v))


def rational_to_seconds(value: Optional[str]) -> float:
    return float(parse_rational(value))


def seconds_to_rational(seconds: float | Fraction, frame_duration: Fraction) -> str:
    """Convert seconds to a rational string, snapped to the frame grid.

    ``frame_duration`` is a :class:`Fraction` in seconds, e.g. ``1001/24000`` for
    23.976 fps. The output numerator is always a multiple of
    ``frame_duration.numerator``, which is how FCPXML expresses frame-aligned
    times (see the sequence ``frameDuration`` attribute).
    """
    fd = Fraction(frame_duration)
    if fd <= 0:
        raise ValueError(f"frame_duration must be positive, got {fd}")
    frames = round(Fraction(seconds) / fd)
    num = frames * fd.numerator
    den = fd.denominator
    if num == 0:
        return "0s"
    return f"{num}/{den}s"


@dataclass(frozen=True)
class _SegmentView:
    """Minimal spine-segment shape needed for timeline translation.

    Mirrors the public :class:`doza_assist.fcpxml.parser.SpineSegment` but only
    the fields this module reads, kept here to avoid a circular import.
    """

    offset: Fraction
    start: Fraction
    duration: Fraction


def _coerce_segment(seg) -> _SegmentView:
    # Accept either a SpineSegment dataclass or a dict (from deserialized metadata).
    if hasattr(seg, "offset_fraction"):
        return _SegmentView(seg.offset_fraction, seg.start_fraction, seg.duration_fraction)
    if isinstance(seg, dict):
        return _SegmentView(
            Fraction(seg["offset_fraction"]) if "offset_fraction" in seg else parse_rational(seg.get("offset")),
            Fraction(seg["start_fraction"]) if "start_fraction" in seg else parse_rational(seg.get("start")),
            Fraction(seg["duration_fraction"]) if "duration_fraction" in seg else parse_rational(seg.get("duration")),
        )
    raise TypeError(f"cannot coerce {type(seg).__name__} into a spine segment")


def audio_source_to_timeline(
    source_seconds: float | Fraction,
    spine_segments: Iterable,
    *,
    audio_angle_offset: Fraction = Fraction(0),
    audio_angle_start: Fraction = Fraction(0),
) -> Optional[Fraction]:
    """Translate an audio-file-relative timecode into a timeline-relative one.

    Each spine segment plays container-internal times ``[start, start+duration)``
    at timeline offsets ``[offset, offset+duration)``. Within the container, the
    active audio angle maps container time → audio source time via
    ``source = container_time - audio_angle_offset + audio_angle_start``.

    So to find where source time ``T`` lands on the timeline, we solve:

        container_time = T + audio_angle_offset - audio_angle_start
        timeline       = segment.offset + (container_time - segment.start)

    for the one segment whose container range contains ``container_time``.
    Returns ``None`` if the source time falls in a gap between segments (i.e.
    that audio is not used anywhere on the timeline).
    """
    t = Fraction(source_seconds)
    container_time = t + Fraction(audio_angle_offset) - Fraction(audio_angle_start)
    for raw in spine_segments:
        seg = _coerce_segment(raw)
        if seg.start <= container_time < seg.start + seg.duration:
            return seg.offset + (container_time - seg.start)
    return None
