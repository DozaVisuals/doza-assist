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


# ── MXF embedded timecode (broadcast/camera record TC) ───────────────
#
# MXF carries SMPTE-12M timecode in structural metadata — ffprobe surfaces it
# as a format-level (or data-stream) `timecode` tag, NOT a `tmcd` track. FCP
# and Resolve both anchor an MXF asset to that embedded TC, so honoring it is
# what keeps a real-world master (e.g. Clip0024-003.MXF, start 00:54:44:12)
# from exporting 0-based edits that land outside the asset and import offline /
# "Invalid edit with no respective media" in both NLEs.

def _run_probe(monkeypatch, payload, path, framerate=23.976):
    monkeypatch.setattr(_mp, "_find_ffprobe", lambda: "/fake/ffprobe")
    monkeypatch.setattr(_mp.os.path, "exists", lambda p: True)
    monkeypatch.setattr(
        _mp.subprocess, "run", lambda *a, **k: _fake_probe_result(payload))
    return _mp.get_video_start_timecode_frames(path, framerate)


class TestMxfTimecode:
    # h264 + 4×pcm_s24le + smpte_436m_anc, TC in the FORMAT tag. No tmcd track.
    MXF_036M = {
        "streams": [
            {"codec_type": "video", "codec_tag_string": ""},
            {"codec_type": "audio", "codec_tag_string": ""},
            {"codec_type": "audio", "codec_tag_string": ""},
            {"codec_type": "data", "codec_tag_string": ""},
        ],
        "format": {"format_name": "mxf", "tags": {"timecode": "00:54:44:12"}},
    }

    def test_mxf_format_tag_honored_via_format_name(self, monkeypatch):
        expected = timecode_to_frames("00:54:44:12", 23.976)  # 78828
        assert expected == 78828
        assert _run_probe(monkeypatch, self.MXF_036M, "/fake/clip.mxf") == expected

    def test_mxf_honored_via_extension_when_format_name_absent(self, monkeypatch):
        # Some builds/probes omit format_name; the .mxf extension still gates in.
        payload = {"streams": [{"codec_type": "video", "codec_tag_string": ""}],
                   "format": {"tags": {"timecode": "01:00:00:00"}}}
        assert _run_probe(monkeypatch, payload, "/fake/clip.MXF", 25.0) == 25 * 3600

    def test_mxf_zero_tc_tag_returns_zero(self, monkeypatch):
        payload = {"streams": [{"codec_type": "video", "codec_tag_string": ""}],
                   "format": {"format_name": "mxf", "tags": {"timecode": "00:00:00:00"}}}
        # A genuine 00:00:00:00 MXF tag -> 0 frames (legacy asset start="0/1s").
        assert _run_probe(monkeypatch, payload, "/fake/clip.mxf") == 0

    def test_mxf_no_timecode_tag_returns_zero(self, monkeypatch):
        payload = {"streams": [{"codec_type": "video", "codec_tag_string": ""}],
                   "format": {"format_name": "mxf", "tags": {}}}
        assert _run_probe(monkeypatch, payload, "/fake/clip.mxf") == 0

    def test_mp4_format_tag_still_ignored_with_format_name(self, monkeypatch):
        # Adding format=format_name to the probe must NOT regress the Sony
        # XAVC-S exclusion: an MP4's bare timecode tag stays unhonored.
        payload = {
            "streams": [
                {"codec_type": "video", "codec_tag_string": "avc1"},
                {"codec_type": "data", "codec_tag_string": "rtmd"},
            ],
            "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                       "tags": {"timecode": "05:26:30:20"}},
        }
        assert _run_probe(monkeypatch, payload, "/fake/clip.mp4") == 0

    def test_mpegts_format_tag_not_honored(self, monkeypatch):
        # The MXF widening is `tmcd_streams or is_mxf` — it must NOT start
        # honoring a bare format timecode tag on OTHER non-tmcd containers.
        # MPEG-TS (.ts/.m2ts/.mts) is freshly ingestable and surfaces format
        # metadata oddly (the double-count history), so pin it to 0.
        payload = {
            "streams": [
                {"codec_type": "video", "codec_tag_string": ""},
                {"codec_type": "audio", "codec_tag_string": ""},
            ],
            "format": {"format_name": "mpegts", "tags": {"timecode": "05:26:30:20"}},
        }
        assert _run_probe(monkeypatch, payload, "/fake/clip.ts") == 0


# ── Audio declaration (Resolve imports clips with audio, not silent) ─────────
#
# The from-scratch FCPXML exporter used to write a bare `<asset hasAudio="1">`
# with no audioSources/audioChannels/audioRate and asset-clips with no
# audioRole. Resolve maps FCPXML clip audio from those DECLARATIONS (not the
# file's track table), so the timeline imported SILENT. The fix probes the
# source audio and declares it; these pin that contract.

class TestFcpxmlAudioDecl:
    def _gen(self, monkeypatch, source, layout, rate, dialogue=None,
             fn=generate_fcpxml, **kw):
        # layout = (num_audio_streams, total_channels) | None
        # dialogue = list of 0-based speech-bearing stream indices | None
        monkeypatch.setattr(_mp, "get_audio_layout", lambda p: layout)
        monkeypatch.setattr(_mp, "get_audio_sample_rate", lambda p: rate)
        monkeypatch.setattr(_mp, "detect_dialogue_channels",
                            lambda p, *a, **k: dialogue)
        markers = [{"start": 1.0, "end": 5.0, "text": "a", "category": "Soundbite"}]
        if fn is generate_fcpxml:
            return fn(markers, "T", framerate=23.976, source_path=source,
                     media_duration=60.0, mode="cuts", **kw)
        return fn(markers, "T", story_title="S", framerate=23.976,
                  source_path=source, media_duration=60.0, **kw)

    def test_asset_declares_one_source_total_channels(self, monkeypatch, source):
        # A single media file is ONE source with N channels (FCP convention).
        xml = self._gen(monkeypatch, source, (4, 4), 48000, dialogue=[1])
        asset = re.search(r'<asset id="r2"[^>]*>', xml).group(0)
        assert 'audioSources="1"' in asset
        assert 'audioChannels="4"' in asset
        assert 'audioRate="48000"' in asset

    def test_multimono_routes_detected_channel_via_connected_audio(self, monkeypatch, source):
        # lav detected on stream index 1 -> Resolve-honored connected-clip
        # form (<clip><video><audio srcCh="2">), NOT a flat <asset-clip>.
        xml = self._gen(monkeypatch, source, (4, 4), 48000, dialogue=[1])
        assert '<clip ' in xml and '<video ref="r2"' in xml
        assert '<asset-clip' not in xml
        audios = re.findall(r'<audio [^>]*/>', xml)
        assert audios and all('srcCh="2"' in a for a in audios)
        # the silent tracks (1, 3, 4) are NOT routed, and the discredited
        # audio-channel-source element is gone.
        assert 'srcCh="1"' not in xml and 'srcCh="3"' not in xml and 'srcCh="4"' not in xml
        assert '<audio-channel-source' not in xml

    def test_two_live_mics_route_both(self, monkeypatch, source):
        xml = self._gen(monkeypatch, source, (4, 4), 48000, dialogue=[1, 2])
        assert 'srcCh="2"' in xml and 'srcCh="3"' in xml

    def test_multimono_no_detection_uses_asset_clip(self, monkeypatch, source):
        # Detection found nothing -> compact asset-clip + audioRole (no guess).
        xml = self._gen(monkeypatch, source, (4, 4), 48000, dialogue=None)
        assert '<asset-clip ' in xml and 'audioRole="dialogue"' in xml
        assert '<clip ' not in xml and '<audio ' not in xml

    def test_single_stream_uses_asset_clip(self, monkeypatch, source):
        # Stereo single stream -> 1/2, asset-clip + audioRole, no routing.
        xml = self._gen(monkeypatch, source, (1, 2), 44100, dialogue=None)
        asset = re.search(r'<asset id="r2"[^>]*>', xml).group(0)
        assert 'audioSources="1"' in asset and 'audioChannels="2"' in asset
        assert 'audioRate="44100"' in asset
        assert '<asset-clip ' in xml and '<audio ' not in xml

    def test_no_audio_decl_when_source_has_none(self, monkeypatch, source):
        xml = self._gen(monkeypatch, source, None, None)
        assert 'audioSources=' not in xml
        assert 'audioRole=' not in xml
        assert 'audioLayout=' not in xml
        assert '<audio ' not in xml

    def test_story_export_routes_dialogue(self, monkeypatch, source):
        xml = self._gen(monkeypatch, source, (4, 4), 48000, dialogue=[1],
                        fn=generate_story_fcpxml)
        assert '<clip ' in xml and '<video ref="r2"' in xml
        assert re.search(r'<audio [^>]*srcCh="2"', xml)

    def test_audio_only_multimono_uses_asset_clip_not_video(self, monkeypatch, tmp_path):
        # An audio-only source (hasVideo="0", e.g. a multi-stream .m4a) must NOT
        # emit a <video> on a video-less asset, even with a detected channel —
        # the connected-clip form is video-only; fall back to <asset-clip>.
        snd = tmp_path / "poly.m4a"
        snd.write_bytes(b"\x00")
        xml = self._gen(monkeypatch, str(snd), (4, 4), 48000, dialogue=[1])
        assert 'hasVideo="0"' in xml
        assert '<video ref="r2"' not in xml
        assert '<clip ' not in xml
        assert '<asset-clip ' in xml and 'audioRole="dialogue"' in xml


# ── get_audio_layout (multi-stream count + MPEG-TS dedup) ────────────────────

class TestGetAudioLayout:
    def _layout(self, monkeypatch, stdout):
        monkeypatch.setattr(_mp, "_find_ffprobe", lambda: "/fake/ffprobe")
        monkeypatch.setattr(_mp.os.path, "exists", lambda p: True)
        monkeypatch.setattr(_mp.subprocess, "run",
            lambda *a, **k: _subprocess.CompletedProcess([], 0, stdout, ""))
        return _mp.get_audio_layout("/fake/x")

    def test_stereo_single_stream(self, monkeypatch):
        assert self._layout(monkeypatch, "0,2\n") == (1, 2)

    def test_mono_single_stream(self, monkeypatch):
        assert self._layout(monkeypatch, "0,1\n") == (1, 1)

    def test_four_mono_tracks(self, monkeypatch):
        assert self._layout(monkeypatch, "1,1\n2,1\n3,1\n4,1\n") == (4, 4)

    def test_mpegts_double_listing_deduped(self, monkeypatch):
        # MPEG-TS lists each stream twice (same index) — must NOT double-count.
        assert self._layout(monkeypatch, "0,2\n\n0,2\n") == (1, 2)
        # 4-mono TS doubled stays 4/4, not 8/8.
        assert self._layout(monkeypatch, "1,1\n1,1\n2,1\n2,1\n3,1\n3,1\n4,1\n4,1\n") == (4, 4)

    def test_no_audio_returns_none(self, monkeypatch):
        assert self._layout(monkeypatch, "") is None

    def test_na_channels_skipped(self, monkeypatch):
        assert self._layout(monkeypatch, "0,N/A\n1,1\n") == (1, 1)


# ── detect_dialogue_channels (loudness-based speech-track picker) ────────────

class TestDetectDialogueChannels:
    def _detect(self, monkeypatch, layout, levels, **kw):
        # levels = {stream_idx: mean_db}; missing idx -> None (probe failed)
        monkeypatch.setattr(_mp.os.path, "exists", lambda p: True)
        monkeypatch.setattr(_mp, "get_audio_layout", lambda p: layout)
        monkeypatch.setattr(_mp, "_find_ffmpeg", lambda: "/fake/ffmpeg")
        monkeypatch.setattr(_mp, "get_media_duration", lambda p: 600.0)
        monkeypatch.setattr(_mp, "_mean_volume_db",
                            lambda ff, p, i, s, g: levels.get(i))
        return _mp.detect_dialogue_channels("/fake/x.mxf", **kw)

    def test_single_loud_track_wins(self, monkeypatch):
        # lav on stream 1; 0/2/3 are silent scratch -> only [1]
        assert self._detect(monkeypatch, (4, 4),
                            {0: -91.0, 1: -24.0, 2: -91.0, 3: -90.0}) == [1]

    def test_two_live_mics_both_returned_loudest_first(self, monkeypatch):
        assert self._detect(monkeypatch, (4, 4),
                            {0: -91.0, 1: -30.0, 2: -24.0, 3: -91.0}) == [2, 1]

    def test_quiet_track_beyond_rel_db_dropped(self, monkeypatch):
        # -55 is 31 dB below the -24 lead (>25) -> excluded
        assert self._detect(monkeypatch, (4, 4),
                            {0: -24.0, 1: -55.0, 2: -91.0, 3: -91.0}) == [0]

    def test_all_silent_returns_none(self, monkeypatch):
        assert self._detect(monkeypatch, (4, 4),
                            {0: -91.0, 1: -92.0, 2: -91.0, 3: -90.0}) is None

    def test_single_stream_returns_none(self, monkeypatch):
        assert self._detect(monkeypatch, (1, 1), {0: -24.0}) is None

    def test_multichannel_streams_skipped(self, monkeypatch):
        # 2 streams / 4 channels (stereo pairs) — stream->srcCh mapping is not
        # 1:1, so we don't guess; returns None.
        assert self._detect(monkeypatch, (2, 4), {0: -24.0, 1: -91.0}) is None
