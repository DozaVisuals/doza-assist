"""Regression tests for FCPXML export honoring the media's embedded start timecode.

Root cause of a DJI import failure: cameras stamp time-of-day timecode, but Doza
exported `asset start="0/1s"` with 0-based `asset-clip` starts. Final Cut keys an
asset's source timecode off the media's real timecode, so every edit fell outside
the asset's [start, start+duration] range and FCP rejected it with "Invalid edit
with no respective media."

The fix: read the media's embedded start timecode and express the asset `start`
and every source-side clip/keyword `start` relative to it.
"""

import os
import re
import sys
from fractions import Fraction

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exporters.media_probe import timecode_to_frames, get_video_start_timecode_frames  # noqa: E402
from fcpxml_export import generate_fcpxml, generate_story_fcpxml, seconds_to_frames  # noqa: E402


def _rat(s):
    s = s.strip().rstrip("s")
    if "/" in s:
        n, d = s.split("/")
        return Fraction(int(n), int(d))
    return Fraction(int(s))


def _asset_start_dur(xml):
    m = re.search(r'<asset id="r2"[^>]*start="([^"]+)"[^>]*duration="([^"]+)"', xml)
    return _rat(m.group(1)), _rat(m.group(2))


def _clips(xml):
    out = []
    for m in re.finditer(r"<asset-clip [^>]*?>", xml):
        tag = m.group(0)
        if 'ref="r2"' not in tag:
            continue
        out.append((
            _rat(re.search(r' start="([^"]+)"', tag).group(1)),
            _rat(re.search(r' duration="([^"]+)"', tag).group(1)),
        ))
    return out


# ── timecode parsing ─────────────────────────────────────────────────

class TestTimecodeToFrames:
    def test_zero(self):
        assert timecode_to_frames("00:00:00:00", 29.97) == 0

    def test_one_hour_at_25(self):
        assert timecode_to_frames("01:00:00:00", 25.0) == 3600 * 25

    def test_dji_time_of_day_2997(self):
        # 14:09:42:00 counted at the nominal 30 grid.
        assert timecode_to_frames("14:09:42:00", 29.97) == (14 * 3600 + 9 * 60 + 42) * 30

    def test_frames_field_2398(self):
        assert timecode_to_frames("00:00:01:12", 23.976) == 24 + 12

    def test_drop_frame_drops_two_per_minute(self):
        # 00:01:00;02 — nominal 1802 minus the 2 dropped frames at minute one.
        assert timecode_to_frames("00:01:00;02", 29.97) == 1800

    def test_non_timecode_returns_none(self):
        assert timecode_to_frames("not-a-tc", 29.97) is None
        assert timecode_to_frames("1.5", 29.97) is None
        assert timecode_to_frames("", 29.97) is None


class TestStartTimecodeProbe:
    def test_missing_file_returns_zero(self):
        assert get_video_start_timecode_frames("/no/such/file.mp4", 29.97) == 0

    def test_empty_path_returns_zero(self):
        assert get_video_start_timecode_frames("", 29.97) == 0


# ── export respects the embedded timecode ────────────────────────────

@pytest.fixture
def source(tmp_path):
    p = tmp_path / "DJI_20260410140942_0017_D.mp4"
    p.write_bytes(b"\x00")
    return str(p)


def test_zero_tc_keeps_legacy_asset_start(source):
    xml = generate_fcpxml(
        [{"start": 1.0, "end": 5.0, "text": "a", "category": "x"}],
        "T", framerate=29.97, source_path=source, media_duration=60.0,
        mode="cuts", start_tc_frames=0,
    )
    start, _ = _asset_start_dur(xml)
    assert start == 0  # asset start="0/1s"


def test_asset_start_equals_embedded_tc(source):
    tc = timecode_to_frames("14:09:42:00", 29.97)  # 1,529,460 frames
    xml = generate_fcpxml(
        [{"start": 1.0, "end": 5.0, "text": "a", "category": "x"}],
        "T", framerate=29.97, source_path=source, media_duration=60.0,
        mode="cuts", start_tc_frames=tc,
    )
    start, _ = _asset_start_dur(xml)
    # asset.start (seconds) == tc frames * frameDuration (1001/30000)
    assert start == Fraction(tc * 1001, 30000)


def test_every_clip_within_asset_timecode_range(source):
    """Reconstructs the reported DJI clip (29.97, time-of-day TC). Every
    asset-clip start must be >= asset.start and end <= asset.start+duration."""
    tc = timecode_to_frames("14:09:42:00", 29.97)
    markers = [
        {"start": 13.01, "end": 19.99, "text": "purple", "category": "purple"},
        {"start": 56.0,  "end": 59.0,  "text": "purple", "category": "purple"},
        {"start": 111.0, "end": 114.0, "text": "purple", "category": "purple"},
        {"start": 151.0, "end": 154.0, "text": "purple", "category": "purple"},
        {"start": 165.0, "end": 171.0, "text": "purple", "category": "purple"},
        {"start": 171.0, "end": 173.0, "text": "purple", "category": "purple"},
    ]
    xml = generate_fcpxml(markers, "DJI", framerate=29.97, source_path=source,
                          media_duration=177.74, mode="cuts", start_tc_frames=tc)
    a_start, a_dur = _asset_start_dur(xml)
    a_end = a_start + a_dur
    clips = _clips(xml)
    assert len(clips) == 6
    fd = Fraction(1001, 30000)
    for s, d in clips:
        assert s >= a_start, f"clip start {s} before asset start {a_start}"
        assert s + d <= a_end, f"clip end {s + d} past asset end {a_end}"
        assert (s / fd).denominator == 1  # frame-aligned


def test_clip_start_is_tc_plus_inpoint(source):
    tc = timecode_to_frames("10:00:00:00", 30.0)
    xml = generate_fcpxml(
        [{"start": 2.0, "end": 6.0, "text": "a", "category": "x"}],
        "T", framerate=30.0, source_path=source, media_duration=60.0,
        mode="cuts", start_tc_frames=tc,
    )
    a_start, _ = _asset_start_dur(xml)
    clip_start = _clips(xml)[0][0]
    inpoint_frames = seconds_to_frames(2.0, 30.0)  # 60
    # clip source start = asset start + in-point
    assert clip_start - a_start == Fraction(inpoint_frames, 30)


def test_story_export_also_offsets_by_tc(source):
    tc = timecode_to_frames("08:30:00:00", 25.0)
    xml = generate_story_fcpxml(
        [{"start": 5.0, "end": 10.0, "text": "beat", "_order": 0}],
        "T", story_title="S", framerate=25.0, source_path=source,
        media_duration=120.0, start_tc_frames=tc,
    )
    a_start, a_dur = _asset_start_dur(xml)
    assert a_start == Fraction(tc, 25)
    s, d = _clips(xml)[0]
    assert s >= a_start
    assert s + d <= a_start + a_dur


# ── tmcd gating (Sony XAVC-S regression) ─────────────────────────────
#
# Sony MP4s carry a `timecode` metadata TAG plus an `rtmd` data track but NO
# `tmcd` track. FCP keys source timecode off the tmcd track only, so honoring
# the tag put every exported clip hours outside the media ("Invalid edit with
# no respective media" on import — Ella_trustees.MP4, 2026-06-10). The probe
# must honor embedded TC only when a real tmcd stream exists.

import json as _json
import subprocess as _subprocess
from unittest import mock

from exporters import media_probe as _mp


def _fake_probe_result(payload):
    return _subprocess.CompletedProcess(
        args=[], returncode=0, stdout=_json.dumps(payload), stderr="")


def _run_gated_probe(monkeypatch, payload, framerate=23.976):
    monkeypatch.setattr(_mp, "_find_ffprobe", lambda: "/fake/ffprobe")
    monkeypatch.setattr(_mp.os.path, "exists", lambda p: True)
    monkeypatch.setattr(
        _mp.subprocess, "run", lambda *a, **k: _fake_probe_result(payload))
    return _mp.get_video_start_timecode_frames("/fake/clip.mp4", framerate)


class TestTmcdGating:
    SONY_MP4 = {
        "streams": [
            {"codec_type": "video", "codec_tag_string": "avc1"},
            {"codec_type": "audio", "codec_tag_string": "twos"},
            {"codec_type": "data", "codec_tag_string": "rtmd"},
        ],
        "format": {"tags": {"timecode": "05:26:30:20"}},
    }
    DJI_MOV = {
        "streams": [
            {"codec_type": "video", "codec_tag_string": "hvc1"},
            {"codec_type": "audio", "codec_tag_string": "mp4a"},
            {"codec_type": "data", "codec_tag_string": "tmcd",
             "tags": {"timecode": "14:23:07:12"}},
        ],
        "format": {"tags": {}},
    }

    def test_sony_mp4_tag_without_tmcd_returns_zero(self, monkeypatch):
        assert _run_gated_probe(monkeypatch, self.SONY_MP4) == 0

    def test_mov_with_tmcd_track_honors_timecode(self, monkeypatch):
        fr = 29.97
        expected = timecode_to_frames("14:23:07:12", fr)
        assert expected and _run_gated_probe(monkeypatch, self.DJI_MOV, fr) == expected

    def test_tmcd_without_own_tag_falls_back_to_format_tag(self, monkeypatch):
        payload = {
            "streams": [
                {"codec_type": "data", "codec_tag_string": "tmcd"},
            ],
            "format": {"tags": {"timecode": "01:00:00:00"}},
        }
        assert _run_gated_probe(monkeypatch, payload, 25.0) == 25 * 3600

    def test_no_timecode_anywhere_returns_zero(self, monkeypatch):
        payload = {"streams": [{"codec_type": "video", "codec_tag_string": "avc1"}],
                   "format": {"tags": {}}}
        assert _run_gated_probe(monkeypatch, payload) == 0

    def test_malformed_probe_json_returns_zero(self, monkeypatch):
        monkeypatch.setattr(_mp, "_find_ffprobe", lambda: "/fake/ffprobe")
        monkeypatch.setattr(_mp.os.path, "exists", lambda p: True)
        monkeypatch.setattr(
            _mp.subprocess, "run",
            lambda *a, **k: _subprocess.CompletedProcess([], 0, "not json", ""))
        assert _mp.get_video_start_timecode_frames("/fake/x.mp4", 23.976) == 0
