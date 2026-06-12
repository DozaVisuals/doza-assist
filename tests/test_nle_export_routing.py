"""NLE export routing + MPEG-TS media advisory (tester field reports).

Bug A: Final Cut Pro, Premiere and DaVinci Resolve cannot decode MPEG-TS
media — newsroom MAMs hand out ``.ts`` files and TS content misnamed
``.mp4`` (the Mimir shape), so exports referencing such sources import as
offline/unsupported with no explanation. The backend now probes the
project's source container (``get_media_container_format``) and attaches a
``media_warning`` to export-success JSON so the UI can explain and suggest
the lossless re-wrap.

Bug B's platform-gate fix lives in templates/project.html (asset-clip
FCPXML imports honor the editing-platform selector; multicam/sync-clip
containers stay FCP-only). These tests pin the new probe helper, the
warning helper's fail-open contract, and the /export/send-to-nle response
field.
"""

import json
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module
from exporters import media_probe

# Fixture synthesis needs a full ffmpeg (aac encoder + mpegts muxer); the
# bundled slim LGPL build is probe-only. Mirror test_mpegts_probe.py.
_FULL_FFMPEG = shutil.which("ffmpeg") or next(
    (p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg")
     if os.path.isfile(p)), None)
_VENDOR_BIN = Path(__file__).resolve().parents[2] / "wrapper" / "vendor" / "ffmpeg" / "bin"
_HAVE_FFPROBE = (_VENDOR_BIN / "ffprobe").is_file() or bool(media_probe._find_ffprobe())


def _make_ts_fixture(path):
    """Synthesize a ~1.2 s aac-in-mpegts file (the newsroom/Mimir shape)."""
    proc = subprocess.run(
        [_FULL_FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1.2:sample_rate=48000",
         "-c:a", "aac", "-b:a", "96k", "-f", "mpegts", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    return proc.returncode == 0 and path.exists()


def _make_wav_fixture(path):
    """A real (tiny) PCM wav — no ffmpeg needed."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    return path.exists()


def _use_best_ffprobe(monkeypatch):
    """Prefer the bundled slim LGPL ffprobe (the exact field configuration)."""
    if (_VENDOR_BIN / "ffprobe").is_file():
        monkeypatch.setenv("DOZA_FFMPEG_DIR", str(_VENDOR_BIN))
    else:
        monkeypatch.delenv("DOZA_FFMPEG_DIR", raising=False)


@pytest.fixture
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def _make_project(pid, **extra):
    pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    pdir.mkdir(parents=True)
    meta = {'id': pid, 'name': 'P', 'status': 'uploaded'}
    meta.update(extra)
    (pdir / 'meta.json').write_text(json.dumps(meta))
    return pdir


# ── get_media_container_format ──────────────────────────────────────────────

@pytest.mark.skipif(_FULL_FFMPEG is None,
                    reason="needs a full ffmpeg to synthesize the TS fixture")
@pytest.mark.skipif(not _HAVE_FFPROBE, reason="no ffprobe available to probe")
class TestGetMediaContainerFormat:
    def test_ts_misnamed_mp4_probes_as_mpegts(self, tmp_path, monkeypatch):
        src = tmp_path / "broadcast.mp4"  # TS content misnamed .mp4
        if not _make_ts_fixture(src):
            pytest.skip("ffmpeg cannot encode aac/mpegts")
        _use_best_ffprobe(monkeypatch)
        fmt = media_probe.get_media_container_format(str(src))
        assert fmt and "mpegts" in fmt

    def test_wav_probes_as_wav_not_mpegts(self, tmp_path, monkeypatch):
        src = tmp_path / "audio.wav"
        assert _make_wav_fixture(src)
        _use_best_ffprobe(monkeypatch)
        fmt = media_probe.get_media_container_format(str(src))
        assert fmt and "wav" in fmt
        assert "mpegts" not in fmt


class TestGetMediaContainerFormatFailOpen:
    def test_missing_or_empty_path_returns_none(self):
        assert media_probe.get_media_container_format("/no/such/file.ts") is None
        assert media_probe.get_media_container_format("") is None
        assert media_probe.get_media_container_format(None) is None

    def test_no_ffprobe_returns_none(self, tmp_path, monkeypatch):
        f = tmp_path / "clip.ts"
        f.write_bytes(b"\x47" * 188)
        monkeypatch.setattr(media_probe, "_find_ffprobe", lambda: None)
        assert media_probe.get_media_container_format(str(f)) is None

    def test_probe_exception_returns_none(self, tmp_path, monkeypatch):
        f = tmp_path / "clip.ts"
        f.write_bytes(b"\x47" * 188)
        monkeypatch.setattr(media_probe, "_find_ffprobe", lambda: "/fake/ffprobe")

        def _boom(*a, **k):
            raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=10)

        monkeypatch.setattr(media_probe.subprocess, "run", _boom)
        assert media_probe.get_media_container_format(str(f)) is None


# ── _mpegts_media_warning ───────────────────────────────────────────────────

class TestMpegtsMediaWarning:
    """Probe-independent contract tests (the probe itself is mocked)."""

    def _mock_format(self, monkeypatch, fmt):
        # app.py imports the probe by name; patch the bound module global.
        monkeypatch.setattr(app_module, 'get_media_container_format',
                            lambda path: fmt)

    def test_ts_source_warns_with_rewrap_hint(self, monkeypatch):
        self._mock_format(monkeypatch, 'mpegts')
        msg = app_module._mpegts_media_warning(
            {'source_path': '/media/newsroom-clip.mp4'})
        assert msg and 'MPEG-TS' in msg
        assert 'newsroom-clip.mp4' in msg
        assert 'ffmpeg -i in.ts -c copy out.mp4' in msg
        # No NLE given -> name all three readers.
        assert 'Final Cut Pro or DaVinci Resolve' in msg
        assert msg.startswith('Heads-up:')

    def test_premiere_gets_no_warning(self, monkeypatch):
        """Premiere Pro reads MPEG-TS natively — warning suppressed
        entirely (field report: the toast was crying wolf on every
        Premiere export of newsroom TS media)."""
        import app as app_module
        monkeypatch.setattr(app_module, 'get_media_container_format',
                            lambda p: 'mpegts')
        proj = {'source_path': '/tmp/x.ts'}
        assert app_module._mpegts_media_warning(proj, nle='premiere') is None

    def test_nle_arg_names_the_target_editor(self, monkeypatch):
        self._mock_format(monkeypatch, 'mpegts')
        msg = app_module._mpegts_media_warning(
            {'source_path': '/media/x.ts'}, nle='resolve')
        assert msg and 'DaVinci Resolve' in msg and 'offline or unsupported' in msg

    def test_clean_container_returns_none(self, monkeypatch):
        self._mock_format(monkeypatch, 'wav')
        assert app_module._mpegts_media_warning(
            {'source_path': '/media/x.wav'}) is None
        self._mock_format(monkeypatch, 'mov,mp4,m4a,3gp,3g2,mj2')
        assert app_module._mpegts_media_warning(
            {'source_path': '/media/x.mp4'}) is None

    def test_probe_none_fails_open(self, monkeypatch):
        self._mock_format(monkeypatch, None)
        assert app_module._mpegts_media_warning(
            {'source_path': '/media/x.ts'}) is None

    def test_probe_exception_fails_open(self, monkeypatch):
        def _boom(path):
            raise RuntimeError("probe exploded")
        monkeypatch.setattr(app_module, 'get_media_container_format', _boom)
        assert app_module._mpegts_media_warning(
            {'source_path': '/media/x.ts'}) is None

    def test_no_source_path_returns_none(self):
        assert app_module._mpegts_media_warning({}) is None

    @pytest.mark.skipif(_FULL_FFMPEG is None,
                        reason="needs a full ffmpeg to synthesize the TS fixture")
    @pytest.mark.skipif(not _HAVE_FFPROBE, reason="no ffprobe available to probe")
    def test_real_ts_fixture_end_to_end(self, tmp_path, monkeypatch):
        src = tmp_path / "real-broadcast.mp4"
        if not _make_ts_fixture(src):
            pytest.skip("ffmpeg cannot encode aac/mpegts")
        _use_best_ffprobe(monkeypatch)
        msg = app_module._mpegts_media_warning({'source_path': str(src)})
        assert msg and 'MPEG-TS' in msg

    @pytest.mark.skipif(not _HAVE_FFPROBE, reason="no ffprobe available to probe")
    def test_real_wav_fixture_returns_none(self, tmp_path, monkeypatch):
        src = tmp_path / "audio.wav"
        assert _make_wav_fixture(src)
        _use_best_ffprobe(monkeypatch)
        assert app_module._mpegts_media_warning({'source_path': str(src)}) is None

    def test_missing_file_returns_none(self):
        assert app_module._mpegts_media_warning(
            {'source_path': '/no/such/file.ts'}) is None


# ── /export/send-to-nle carries the advisory ────────────────────────────────

class TestSendToNleMediaWarning:
    def _stub_export_plumbing(self, monkeypatch):
        monkeypatch.setattr(app_module, '_find_nle_app_path',
                            lambda nle: '/Applications/Fake.app')
        monkeypatch.setattr(app_module, '_hand_file_to_nle',
                            lambda *a, **k: ('resolve', {}))
        stub = SimpleNamespace(file_path='/tmp/out.fcpxml',
                               filename='out.fcpxml', format_name='FCPXML')
        monkeypatch.setattr(
            app_module, '_build_nle_export',
            lambda project, body, force_platform=None: (stub, None))

    @pytest.mark.skipif(_FULL_FFMPEG is None,
                        reason="needs a full ffmpeg to synthesize the TS fixture")
    @pytest.mark.skipif(not _HAVE_FFPROBE, reason="no ffprobe available to probe")
    def test_ts_source_response_carries_media_warning(self, client, tmp_path,
                                                      monkeypatch):
        src = tmp_path / "newsroom.mp4"
        if not _make_ts_fixture(src):
            pytest.skip("ffmpeg cannot encode aac/mpegts")
        _use_best_ffprobe(monkeypatch)
        _make_project('p1', source_path=str(src))
        self._stub_export_plumbing(monkeypatch)

        res = client.post('/export/send-to-nle',
                          json={'project_id': 'p1', 'nle': 'resolve'})
        assert res.status_code == 200
        data = res.get_json()
        assert data['status'] == 'ok'
        assert 'MPEG-TS' in data.get('media_warning', '')
        assert 'DaVinci Resolve' in data['media_warning']

    @pytest.mark.skipif(not _HAVE_FFPROBE, reason="no ffprobe available to probe")
    def test_clean_source_response_has_no_media_warning(self, client, tmp_path,
                                                        monkeypatch):
        src = tmp_path / "audio.wav"
        assert _make_wav_fixture(src)
        _use_best_ffprobe(monkeypatch)
        _make_project('p2', source_path=str(src))
        self._stub_export_plumbing(monkeypatch)

        res = client.post('/export/send-to-nle',
                          json={'project_id': 'p2', 'nle': 'resolve'})
        assert res.status_code == 200
        data = res.get_json()
        assert data['status'] == 'ok'
        assert 'media_warning' not in data
