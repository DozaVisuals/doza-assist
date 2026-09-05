"""Drop-frame flag must follow the timebase, not the separator character.

A camera tag of "HH:MM:SS;FF" says drop-frame, but only the fractional NTSC
rates (29.97, 59.94, 119.88) have a drop-frame count. Media tagged with ';'
at 24/25 fps, or NTSC media probed at a forced export rate of 24, used to
come back "DF" and FCP rejected the resulting asset-clips."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exporters import media_probe as mp  # noqa: E402


@pytest.mark.parametrize("rate,expected", [
    (23.976, False), (24.0, False), (25.0, False), (29.97, True), (30.0, False),
    (48.0, False), (50.0, False), (59.94, True), (60.0, False), (119.88, True),
    (120.0, False), (0, False), (None, False), ("x", False),
])
def test_is_drop_frame_rate_table(rate, expected):
    assert mp.is_drop_frame_rate(rate) is expected


def _probe_with(monkeypatch, tmp_path, tag):
    class _Result:
        returncode = 0
        stdout = json.dumps({
            "streams": [{"codec_type": "data", "codec_tag_string": "tmcd",
                         "tags": {"timecode": tag}}],
            "format": {"tags": {}},
        })
    f = tmp_path / "x.mov"
    f.write_bytes(b"x")
    monkeypatch.setattr(mp.subprocess, "run", lambda *a, **k: _Result())
    return str(f)


def test_semicolon_tag_at_2997_is_drop(monkeypatch, tmp_path):
    f = _probe_with(monkeypatch, tmp_path, "01:00:00;02")
    tc = mp.get_video_start_timecode(f, 29.97)
    assert tc["drop"] is True and tc["frames"] > 0
    assert mp.get_video_start_timecode_info(f, 29.97)[1] == "DF"


def test_semicolon_tag_at_23976_is_not_drop(monkeypatch, tmp_path):
    f = _probe_with(monkeypatch, tmp_path, "01:00:00;02")
    tc = mp.get_video_start_timecode(f, 23.976)
    assert tc["drop"] is False and tc["frames"] > 0
    assert mp.get_video_start_timecode_info(f, 23.976)[1] == "NDF"


def test_ntsc_tag_probed_at_forced_24_is_not_drop(monkeypatch, tmp_path):
    # The Collection page used to force 24 fps; a 29.97 DF tag with FF < 24
    # still parsed and kept its DF flag against a 1/24s format.
    f = _probe_with(monkeypatch, tmp_path, "01:00:00;02")
    assert mp.get_video_start_timecode_info(f, 24.0)[1] == "NDF"


def test_integer_30_with_semicolon_is_not_drop(monkeypatch, tmp_path):
    f = _probe_with(monkeypatch, tmp_path, "01:00:00;02")
    assert mp.get_video_start_timecode_info(f, 30.0)[1] == "NDF"


def test_colon_tag_at_2997_is_ndf(monkeypatch, tmp_path):
    f = _probe_with(monkeypatch, tmp_path, "01:00:00:02")
    assert mp.get_video_start_timecode_info(f, 29.97)[1] == "NDF"
