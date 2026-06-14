"""Select audio channel for transcription.

Files often carry the camera mic on one track and a lav on another. The
``audio_channel`` selection ('all' default, or a 0-based track index) threads
from project creation → transcribe → extract_audio → the ffmpeg ``-map``,
without changing the default "All" behaviour.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import transcribe as transcribe_module
from transcribe import (
    _audio_stream_plan,
    count_audio_streams,
    normalize_audio_channel,
)
from exporters import media_probe


class _Result:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def _plan_with(monkeypatch, probe_stdout, channel=None):
    monkeypatch.setattr(media_probe, "_find_ffprobe", lambda: "/fake/ffprobe")
    monkeypatch.setattr(transcribe_module.subprocess, "run",
                        lambda *a, **k: _Result(stdout=probe_stdout))
    return _audio_stream_plan("/fake/input.mov", channel)


class TestNormalize:
    def test_values(self):
        assert normalize_audio_channel(None) is None
        assert normalize_audio_channel('all') is None
        assert normalize_audio_channel('') is None
        assert normalize_audio_channel('0') == 0
        assert normalize_audio_channel('2') == 2
        assert normalize_audio_channel(1) == 1
        assert normalize_audio_channel('lav') is None
        assert normalize_audio_channel(True) is None   # bool guard
        assert normalize_audio_channel(-1) is None


class TestAudioStreamPlanChannel:
    def test_all_default_unchanged_single(self, monkeypatch):
        # 1 stream, All → no extra args (historical behaviour).
        assert _plan_with(monkeypatch, "0\n") == (1, [])

    def test_all_default_unchanged_multi_mixes(self, monkeypatch):
        # 2 streams, All → amix (historical multi-mic behaviour).
        count, args = _plan_with(monkeypatch, "0\n1\n")
        assert count == 2 and "amix=inputs=2" in args[1]

    def test_channel_0_maps_first_track(self, monkeypatch):
        count, args = _plan_with(monkeypatch, "0\n1\n", channel=0)
        assert count == 2
        assert args == ['-map', '0:a:0']

    def test_channel_1_maps_second_track(self, monkeypatch):
        count, args = _plan_with(monkeypatch, "0\n1\n", channel=1)
        assert args == ['-map', '0:a:1']

    def test_out_of_range_channel_falls_back_to_all(self, monkeypatch):
        # Channel 3 on a 2-track file → safe fallback to All (amix), never fail.
        count, args = _plan_with(monkeypatch, "0\n1\n", channel=5)
        assert count == 2 and "amix" in args[1]

    def test_channel_on_single_track_maps_it(self, monkeypatch):
        count, args = _plan_with(monkeypatch, "0\n", channel=0)
        assert args == ['-map', '0:a:0']

    def test_no_audio_zero(self, monkeypatch):
        assert _plan_with(monkeypatch, "", channel=1) == (0, [])


class TestCountAudioStreams:
    def test_counts_distinct(self, monkeypatch):
        monkeypatch.setattr(media_probe, "_find_ffprobe", lambda: "/fake/ffprobe")
        monkeypatch.setattr(transcribe_module.subprocess, "run",
                            lambda *a, **k: _Result(stdout="0\n1\n2\n"))
        assert count_audio_streams("/x.mov") == 3

    def test_program_container_collapses(self, monkeypatch):
        # MPEG-TS lists each stream twice → 1 (not multi-mic).
        monkeypatch.setattr(media_probe, "_find_ffprobe", lambda: "/fake/ffprobe")
        monkeypatch.setattr(transcribe_module.subprocess, "run",
                            lambda *a, **k: _Result(stdout="1\n\n1\n"))
        assert count_audio_streams("/x.ts") == 1

    def test_fail_open_no_ffprobe(self, monkeypatch):
        monkeypatch.setattr(media_probe, "_find_ffprobe", lambda: None)
        assert count_audio_streams("/x.mov") == 1


class TestExtractAudioChannel:
    """The ffmpeg invocation must carry the channel's -map; the cache must
    re-extract when the channel changes."""

    def _capture_ffmpeg(self, monkeypatch, probe_stdout, channel):
        calls = {}
        monkeypatch.setattr(media_probe, "_find_ffprobe", lambda: "/fake/ffprobe")
        monkeypatch.setattr(transcribe_module, "_find_ffmpeg", lambda: "/fake/ffmpeg")

        def _run(cmd, *a, **k):
            if cmd and cmd[0] == "/fake/ffprobe":
                return _Result(stdout=probe_stdout)
            calls['ffmpeg_cmd'] = cmd
            return _Result(returncode=1, stderr="banner\nstub: stop before write")

        monkeypatch.setattr(transcribe_module.subprocess, "run", _run)
        return calls

    def test_channel_reaches_ffmpeg_map(self, monkeypatch, tmp_path):
        src = tmp_path / "interview.mov"
        src.write_bytes(b"\x00" * 64)
        calls = self._capture_ffmpeg(monkeypatch, "0\n1\n", channel=1)
        with pytest.raises(RuntimeError):  # stub ffmpeg returns rc=1
            transcribe_module.extract_audio(str(src), project_dir=str(tmp_path),
                                            audio_channel='1')
        cmd = calls['ffmpeg_cmd']
        assert '-map' in cmd and '0:a:1' in cmd
        assert 'amix=inputs=2:normalize=0[aout]' not in ' '.join(cmd)

    def test_all_default_uses_amix(self, monkeypatch, tmp_path):
        src = tmp_path / "interview.mov"
        src.write_bytes(b"\x00" * 64)
        calls = self._capture_ffmpeg(monkeypatch, "0\n1\n", channel=None)
        with pytest.raises(RuntimeError):
            transcribe_module.extract_audio(str(src), project_dir=str(tmp_path),
                                            audio_channel='all')
        assert 'amix=inputs=2:normalize=0[aout]' in ' '.join(calls['ffmpeg_cmd'])

    def test_cache_invalidates_on_channel_change(self, monkeypatch, tmp_path):
        from transcribe import _cached_audio_valid, _audio_meta_path, _EXTRACT_RECIPE
        src = tmp_path / "src.mov"
        src.write_bytes(b"\x00" * 4096)
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 200000)
        st = os.stat(str(src))
        # Sidecar recorded for channel 0.
        Path(_audio_meta_path(str(audio))).write_text(json.dumps({
            'recipe': _EXTRACT_RECIPE, 'channel': 0,
            'source_size': st.st_size, 'source_mtime': int(st.st_mtime),
            'wav_duration': 6.0,
        }))
        assert _cached_audio_valid(str(audio), str(src), channel=0) is True
        assert _cached_audio_valid(str(audio), str(src), channel=1) is False
        assert _cached_audio_valid(str(audio), str(src), channel=None) is False


_FULL_FFMPEG = shutil.which("ffmpeg")
_VENDOR_BIN = Path(__file__).resolve().parents[2] / "wrapper" / "vendor" / "ffmpeg" / "bin"


@pytest.mark.skipif(_FULL_FFMPEG is None,
                    reason="needs a full ffmpeg on PATH to synthesize a 2-track fixture")
class TestEndToEndTwoTrack:
    def _make_two_track(self, path):
        # Track 0 = 440 Hz tone, track 1 = 880 Hz tone, distinct so a
        # selected channel is verifiable by content if needed.
        proc = subprocess.run(
            [_FULL_FFMPEG, "-y", "-v", "error",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1.2",
             "-f", "lavfi", "-i", "sine=frequency=880:duration=1.2",
             "-map", "0:a", "-map", "1:a", "-c:a", "aac", str(path)],
            capture_output=True, text=True, timeout=30)
        return proc.returncode == 0 and path.exists()

    def test_count_and_channel_extraction(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DOZA_TRIAL", raising=False)
        src = tmp_path / "two_track.mp4"
        if not self._make_two_track(src):
            pytest.skip("PATH ffmpeg cannot build the fixture")
        if (_VENDOR_BIN / "ffmpeg").is_file() and (_VENDOR_BIN / "ffprobe").is_file():
            monkeypatch.setenv("DOZA_FFMPEG_DIR", str(_VENDOR_BIN))
        assert count_audio_streams(str(src)) == 2
        # Channel 1 extracts cleanly to a non-empty 16k mono WAV.
        out = transcribe_module.extract_audio(
            str(src), project_dir=str(tmp_path), audio_channel='1')
        assert os.path.getsize(out) > 32000
        # Switching to All re-extracts (different sidecar channel) — no crash.
        out2 = transcribe_module.extract_audio(
            str(src), project_dir=str(tmp_path), audio_channel='all')
        assert os.path.getsize(out2) > 32000


class TestAppPlumbing:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DOZA_DATA_DIR', str(tmp_path))
        import importlib, app as app_module
        importlib.reload(app_module)
        app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
        Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
        app_module.app.config['TESTING'] = True
        self.app_module = app_module
        return app_module.app.test_client()

    def test_create_stores_audio_channel(self, client, tmp_path):
        src = tmp_path / 'a.wav'
        src.write_bytes(b'RIFF' + b'\x00' * 64)
        pid = self.app_module.create_project_from_path(
            str(src), project_name='X', audio_channel='1')
        meta = json.loads((Path(self.app_module.app.config['PROJECTS_DIR']) / pid /
                           'meta.json').read_text())
        assert meta['audio_channel'] == '1'

    def test_create_default_is_all(self, client, tmp_path):
        src = tmp_path / 'a.wav'
        src.write_bytes(b'RIFF' + b'\x00' * 64)
        pid = self.app_module.create_project_from_path(str(src), project_name='X')
        meta = json.loads((Path(self.app_module.app.config['PROJECTS_DIR']) / pid /
                           'meta.json').read_text())
        assert meta['audio_channel'] == 'all'

    def test_create_clamps_garbage_channel(self, client, tmp_path):
        src = tmp_path / 'a.wav'
        src.write_bytes(b'RIFF' + b'\x00' * 64)
        pid = self.app_module.create_project_from_path(
            str(src), project_name='X', audio_channel='lav')
        meta = json.loads((Path(self.app_module.app.config['PROJECTS_DIR']) / pid /
                           'meta.json').read_text())
        assert meta['audio_channel'] == 'all'

    def test_probe_endpoint_missing_file_fails_open(self, client):
        r = client.post('/probe-audio-tracks', json={'path': '/nope/x.mov'})
        assert r.status_code == 200
        assert r.get_json() == {'count': 1}

    def test_probe_endpoint_counts(self, client, tmp_path, monkeypatch):
        f = tmp_path / 'clip.mov'
        f.write_bytes(b'\x00' * 64)
        monkeypatch.setattr('transcribe.count_audio_streams', lambda p: 3)
        r = client.post('/probe-audio-tracks', json={'path': str(f)})
        assert r.get_json() == {'count': 3}
