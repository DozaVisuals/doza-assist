"""Tests for high-frame-rate support (50 / 48 / 100 / 120 fps).

A real 50fps clip (very common on DJI and PAL-region cameras) used to snap to
59.94 and export on the wrong frame grid. These rates are now first-class across
the frame-rate detector and every exporter that maps a rate to a timebase.
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exporters.media_probe import STANDARD_FRAMERATES, snap_framerate  # noqa: E402
from exporters.premiere_xml import _rate_for  # noqa: E402
from exporters.edl import _seconds_to_timecode  # noqa: E402
from fcpxml_export import (  # noqa: E402
    generate_fcpxml, get_frame_duration, _timebase, _framerate_label,
)

HFR = [48.0, 50.0, 100.0, 120.0]


# ── frame-rate detection / snapping ──────────────────────────────────

@pytest.mark.parametrize("rate", HFR)
def test_new_rates_are_supported(rate):
    assert rate in STANDARD_FRAMERATES


def test_exact_rates_snap_to_themselves():
    for rate in STANDARD_FRAMERATES:
        assert snap_framerate(rate) == rate


def test_fifty_fps_no_longer_snaps_to_5994():
    # The regression: 50fps must resolve to 50.0, not the old nearest (59.94).
    assert snap_framerate(50.0) == 50.0
    assert snap_framerate(49.95) == 50.0


def test_ntsc_pulldown_rates_still_snap():
    assert snap_framerate(48000 / 1001) == 48.0      # 47.952 -> 48
    assert snap_framerate(50.0) == 50.0
    # 119.88 (NTSC 120 — iPhone/action-cam slo-mo) is first-class now:
    # snapping it to 120.0 put frame indexes on a 1.001x-wrong grid.
    assert snap_framerate(120000 / 1001) == 119.88


# ── FCPXML ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rate,expected", [
    (48.0, "1/48s"), (50.0, "1/50s"), (100.0, "1/100s"), (120.0, "1/120s"),
])
def test_fcpxml_frame_duration(rate, expected):
    assert get_frame_duration(rate) == expected


def test_get_frame_duration_matches_timebase_table():
    # get_frame_duration is derived from _timebase, so it can never drift.
    for rate in STANDARD_FRAMERATES:
        tb, fd = _timebase(rate)
        assert get_frame_duration(rate) == f"{fd}/{tb}s"


def test_fcpxml_50fps_uses_50_grid_not_5994(tmp_path):
    source = tmp_path / "clip.MP4"
    source.write_bytes(b"\x00")
    markers = [{"start": 1.0, "end": 5.0, "text": "a", "category": "x"}]
    xml = generate_fcpxml(markers, "T", framerate=50.0, source_path=str(source),
                          media_duration=10.0, mode="cuts")
    assert 'frameDuration="1/50s"' in xml
    assert "60000s" not in xml  # the 59.94 grid must not appear
    assert _framerate_label(50.0) == "50"
    assert "p50" in xml  # FFVideoFormat...p50
    # Every non-zero rational time is on the /50 grid (the asset's start="0/1s"
    # zero point is exempt — 0/1 is just zero).
    for num, denom in re.findall(r'(?:duration|start|offset)="(\d+)/(\d+)s"', xml):
        if num != "0":
            assert denom == "50"


# ── Premiere XML ─────────────────────────────────────────────────────

@pytest.mark.parametrize("rate,timebase", [
    (48.0, 48), (50.0, 50), (100.0, 100), (120.0, 120),
])
def test_premiere_rate_table(rate, timebase):
    tb, ntsc, actual = _rate_for(rate)
    assert tb == timebase
    assert ntsc is False          # these are true integer rates, not NTSC
    assert actual == rate


# ── EDL (derives timebase generically) ───────────────────────────────

def test_edl_50fps_timecode_grid():
    # 1.5s at 50fps = 75 frames = 00:00:01:25.
    assert _seconds_to_timecode(1.5, 50.0) == "00:00:01:25"
    # Frame field must never reach the timebase.
    assert _seconds_to_timecode(0.98, 50.0) == "00:00:00:49"


# ── MPEG-2 sources: ffprobe appends a trailing csv field ─────────────────────

class _Probe:
    def __init__(self, stdout):
        self.returncode = 0
        self.stdout = stdout
        self.stderr = ""


@pytest.mark.parametrize("stdout,expected", [
    ("25/1,\n", 25.0),                          # XDCAM-style MPEG-2 MXF: side data adds a trailing field
    ("30000/1001,\n30000/1001,\n", 29.97),      # MPEG-TS: listed twice (program + top level) with the field
    ("24000/1001\n", 23.976),                   # ProRes / H.264: no trailing field, unchanged
])
def test_framerate_probe_ignores_trailing_csv_field(monkeypatch, tmp_path, stdout, expected):
    from exporters import media_probe
    media = tmp_path / "clip.mxf"
    media.write_bytes(b"\0" * 64)
    monkeypatch.setattr(media_probe, "_ffprobe_path", lambda: "/usr/bin/true", raising=False)
    monkeypatch.setattr(media_probe.subprocess, "run", lambda *a, **k: _Probe(stdout))
    assert media_probe.get_video_framerate(str(media)) == expected
