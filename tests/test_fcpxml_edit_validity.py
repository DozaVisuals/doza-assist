"""Regression tests for FCPXML edit validity (issue: DJI footage import failure).

FCP rejects an FCPXML on import with "Invalid edit with no respective media"
when any ``<asset-clip>`` on the spine references a source range that the
referenced ``<asset>`` does not have. Three ways the cuts/story exporters used
to produce such edits:

  1. Rounding ``start`` and ``duration`` independently, so ``start + duration``
     could land one frame past the source out point — and past the asset — on
     fractional NTSC rates (23.976 / 29.97 / 59.94).
  2. Not clamping marker out points to the media duration, so an AI-placed
     timestamp beyond the end of the clip produced an edit into nonexistent
     media.
  3. Summing rounded seconds for the timeline offset, drifting the spine into
     sub-frame gaps/overlaps.

These tests assert the invariants directly so the failure can never silently
return. They need a source file on disk (any bytes) so the exporter takes the
"cuts" path rather than markers-only.
"""

import os
import re
import sys
from fractions import Fraction

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fcpxml_export import generate_fcpxml, generate_story_fcpxml  # noqa: E402

ALL_RATES = [23.976, 24.0, 25.0, 29.97, 30.0, 59.94, 60.0]


@pytest.fixture
def source(tmp_path):
    p = tmp_path / "DJI 20000101162808 0033 D.MP4"
    p.write_bytes(b"\x00")
    return str(p)


def _rat(s):
    s = s.strip().rstrip("s")
    if "/" in s:
        n, d = s.split("/")
        return Fraction(int(n), int(d))
    return Fraction(int(s))


def _audit(xml):
    """Return (frame_duration, asset_duration, [(offset, start, duration), ...])."""
    fd = _rat(re.search(r'frameDuration="([^"]+)"', xml).group(1))
    asset = _rat(re.search(r'<asset id="r2"[^>]*duration="([^"]+)"', xml).group(1))
    clips = []
    for m in re.finditer(r"<asset-clip [^>]*?>", xml):
        tag = m.group(0)
        if 'ref="r2"' not in tag:
            continue
        clips.append((
            _rat(re.search(r' offset="([^"]+)"', tag).group(1)),
            _rat(re.search(r' start="([^"]+)"', tag).group(1)),
            _rat(re.search(r' duration="([^"]+)"', tag).group(1)),
        ))
    return fd, asset, clips


# Last clip deliberately ends exactly at a media duration that is not frame
# aligned — the case that overshot on 29.97 before the fix.
EDGE_MARKERS = [
    {"start": 1.5,  "end": 12.3,    "text": "a", "category": "x"},
    {"start": 23.4, "end": 38.1,    "text": "b", "category": "x"},
    {"start": 50.0, "end": 87.3331, "text": "c", "category": "x"},
]
EDGE_MEDIA = 87.3331


@pytest.mark.parametrize("fr", ALL_RATES)
def test_no_edit_exceeds_asset(source, fr):
    xml = generate_fcpxml(EDGE_MARKERS, "T", framerate=fr, source_path=source,
                          media_duration=EDGE_MEDIA, mode="cuts")
    _, asset, clips = _audit(xml)
    assert clips
    for off, start, dur in clips:
        assert start + dur <= asset, (
            f"fr={fr}: edit out {float(start + dur)} > asset {float(asset)}"
        )


@pytest.mark.parametrize("fr", ALL_RATES)
def test_edits_are_frame_aligned(source, fr):
    xml = generate_fcpxml(EDGE_MARKERS, "T", framerate=fr, source_path=source,
                          media_duration=EDGE_MEDIA, mode="cuts")
    fd, _, clips = _audit(xml)
    for off, start, dur in clips:
        assert (off / fd).denominator == 1, f"fr={fr}: offset off-grid"
        assert (start / fd).denominator == 1, f"fr={fr}: start off-grid"
        assert (dur / fd).denominator == 1, f"fr={fr}: duration off-grid"


@pytest.mark.parametrize("fr", ALL_RATES)
def test_spine_is_contiguous(source, fr):
    xml = generate_fcpxml(EDGE_MARKERS, "T", framerate=fr, source_path=source,
                          media_duration=EDGE_MEDIA, mode="cuts")
    _, _, clips = _audit(xml)
    prev_end = None
    for off, start, dur in clips:
        if prev_end is not None:
            assert off == prev_end, (
                f"fr={fr}: gap/overlap — offset {float(off)} != prev end {float(prev_end)}"
            )
        prev_end = off + dur


def test_marker_overshoot_is_clamped(source):
    # Marker out point (25s) runs past the media (20s); the edit must be clamped
    # to the media end, never reference beyond it.
    markers = [
        {"start": 1.0, "end": 8.0,  "text": "a", "category": "x"},
        {"start": 9.0, "end": 25.0, "text": "b", "category": "x"},
    ]
    xml = generate_fcpxml(markers, "T", framerate=30.0, source_path=source,
                          media_duration=20.0, mode="cuts")
    _, asset, clips = _audit(xml)
    assert len(clips) == 2
    last_off, last_start, last_dur = clips[-1]
    assert last_start + last_dur <= asset
    assert last_start + last_dur == asset  # clamped right to the media end


def test_clip_entirely_past_media_is_dropped(source):
    markers = [
        {"start": 1.0,  "end": 8.0,  "text": "a", "category": "x"},
        {"start": 50.0, "end": 60.0, "text": "b", "category": "x"},  # past 20s media
    ]
    xml = generate_fcpxml(markers, "T", framerate=30.0, source_path=source,
                          media_duration=20.0, mode="cuts")
    _, _, clips = _audit(xml)
    assert len(clips) == 1


@pytest.mark.parametrize("fr", ALL_RATES)
def test_story_export_edits_valid(source, fr):
    xml = generate_story_fcpxml(EDGE_MARKERS, "T", story_title="S", framerate=fr,
                                source_path=source, media_duration=EDGE_MEDIA)
    _, asset, clips = _audit(xml)
    assert clips
    for off, start, dur in clips:
        assert start + dur <= asset, f"fr={fr}: story edit exceeds asset"
