"""Regression tests for the MPEG-TS ffprobe double-listing bug.

ffprobe prints each stream once per enclosing section: containers with
programs (MPEG-TS broadcast files — newsroom systems like Mimir hand these
out misnamed ``.mp4``) list every stream under its program AND in the
top-level stream list. Three consequences covered here:

  - ``_audio_stream_plan`` counted the duplicated rows, doubled a one-audio
    TS to 2 streams, and emitted an amix map referencing the nonexistent
    ``[0:a:1]`` — ffmpeg exits with "Stream specifier matches no streams"
    and transcription dies in extraction (direct-1.0.12 field bug).
  - ``media_probe`` csv parsers (resolution / framerate / channels /
    duration) choked on the doubled rows and silently degraded to their
    fallbacks for every TS source.
  - the extraction RuntimeError carried ``stderr[:500]`` — exactly the
    ~3 KB ffmpeg banner head — hiding the actual error from user reports.

Program containers additionally OPT OUT of the multi-stream amix mixdown:
TS multi-audio is alternate services/languages (mixing them transcribes
unrelated speech over each other), unlike the multi-mic MXF/MOV camera case
the mixdown exists for.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import transcribe as transcribe_module
from transcribe import _audio_stream_plan, _ffmpeg_error_excerpt
from exporters import media_probe


class _Result:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def _plan_with(monkeypatch, probe_stdout, returncode=0):
    monkeypatch.setattr(media_probe, "_find_ffprobe", lambda: "/fake/ffprobe")
    monkeypatch.setattr(transcribe_module.subprocess, "run",
                        lambda *a, **k: _Result(stdout=probe_stdout,
                                                returncode=returncode))
    return _audio_stream_plan("/fake/input.mp4")


class TestAudioStreamPlan:
    def test_ts_duplicated_single_stream_counts_once(self, monkeypatch):
        """The field bug: one audio stream listed under the TS program and
        again at top level must plan as ONE stream — no amix map."""
        count, args = _plan_with(monkeypatch, "1\n\n1\n")
        assert count == 1
        assert args == []

    def test_ts_multi_audio_uses_default_pick_not_amix(self, monkeypatch):
        """Program containers never mix: two TS audio streams are alternate
        services/languages, so ffmpeg's default best-stream pick applies."""
        count, args = _plan_with(monkeypatch, "1\n2\n\n1\n2\n")
        assert count == 1
        assert args == []

    def test_ts_stream_shared_across_programs_counts_once(self, monkeypatch):
        # One audio stream carried by two services: three listings total.
        count, args = _plan_with(monkeypatch, "1\n\n1\n\n1\n")
        assert count == 1
        assert args == []

    def test_multitrack_camera_file_still_mixes(self, monkeypatch):
        """No duplication (MOV/MXF have no programs) — the multi-mic
        mixdown must keep working."""
        count, args = _plan_with(monkeypatch, "1\n2\n")
        assert count == 2
        assert "amix=inputs=2" in args[1]
        assert "[0:a:0][0:a:1]" in args[1]

    def test_plain_mp4_single_stream(self, monkeypatch):
        count, args = _plan_with(monkeypatch, "0\n")
        assert count == 1
        assert args == []

    def test_no_audio_stream(self, monkeypatch):
        count, args = _plan_with(monkeypatch, "")
        assert count == 0
        assert args == []


class TestAudioStreamPlanFailOpen:
    """Probing trouble must never block extraction: fail open to 1 stream
    (no extra args) so ffmpeg's default selection still runs."""

    def test_no_ffprobe_resolved(self, monkeypatch):
        monkeypatch.setattr(media_probe, "_find_ffprobe", lambda: None)
        assert _audio_stream_plan("/fake/input.mp4") == (1, [])

    def test_ffprobe_nonzero_exit(self, monkeypatch):
        count, args = _plan_with(monkeypatch, "", returncode=1)
        assert (count, args) == (1, [])

    def test_ffprobe_timeout(self, monkeypatch):
        monkeypatch.setattr(media_probe, "_find_ffprobe", lambda: "/fake/ffprobe")

        def _boom(*a, **k):
            raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=10)

        monkeypatch.setattr(transcribe_module.subprocess, "run", _boom)
        assert _audio_stream_plan("/fake/input.mp4") == (1, [])


_BANNER = (
    "ffmpeg version 8.1.1 Copyright (c) 2000-2026 the FFmpeg developers\n"
    "  built with Apple clang version 21.0.0 (clang-2100.0.123.102)\n"
    "  configuration: --prefix=/usr/local --bindir=/x/bin " + "-" * 2000 + "\n"
    "  libavutil      60. 26.101 / 60. 26.101\n"
    "  libavcodec     62. 28.101 / 62. 28.101\n"
    "  libavformat    62. 12.101 / 62. 12.101\n"
)

_FIELD_ERROR = (
    "Input #0, mpegts, from 'x.mp4':\n"
    "[fc#0 @ 0x1] Stream specifier ':a:1' in filtergraph description "
    "[0:a:0][0:a:1]amix=inputs=2:normalize=0[aout] matches no streams.\n"
    "Error binding filtergraph inputs/outputs: Invalid argument\n"
)


class TestFfmpegErrorExcerpt:
    def test_error_survives_banner_strip(self):
        excerpt = _ffmpeg_error_excerpt(_BANNER + _FIELD_ERROR)
        assert "matches no streams" in excerpt
        assert "configuration:" not in excerpt
        assert len(excerpt) <= 500

    def test_pure_banner_falls_back_to_raw_tail(self):
        excerpt = _ffmpeg_error_excerpt(_BANNER)
        assert excerpt  # never empty when stderr had content

    def test_empty_and_none(self):
        assert _ffmpeg_error_excerpt("") == ""
        assert _ffmpeg_error_excerpt(None) == ""


class TestExtractAudioErrorSurface:
    """The call site, not just the helper: a failing ffmpeg run must raise
    the banner-stripped excerpt AND print the full tail to stderr (which the
    wrapper pipes into the user's server.log)."""

    def test_failure_message_carries_error_not_banner(self, tmp_path,
                                                      monkeypatch, capfd):
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"\x00" * 64)
        monkeypatch.setattr(transcribe_module, "_find_ffmpeg",
                            lambda: "/fake/ffmpeg")
        # No ffprobe -> stream plan fails open (1, []) without subprocess.
        monkeypatch.setattr(media_probe, "_find_ffprobe", lambda: None)
        monkeypatch.setattr(
            transcribe_module.subprocess, "run",
            lambda *a, **k: _Result(returncode=234,
                                    stderr=_BANNER + _FIELD_ERROR))
        with pytest.raises(RuntimeError) as excinfo:
            transcribe_module.extract_audio(str(src),
                                            project_dir=str(tmp_path))
        msg = str(excinfo.value)
        assert "ffmpeg audio extraction failed" in msg
        assert "matches no streams" in msg
        assert "configuration:" not in msg
        # Support artifact: the full tail goes to stderr -> server.log.
        err = capfd.readouterr().err
        assert "[extract_audio] ffmpeg rc=234" in err
        assert "matches no streams" in err


class TestMediaProbeTsDuplication:
    """Doubled csv rows must parse as their first row, not the fallback."""

    def _probe_with(self, monkeypatch, tmp_path, stdout, by_arg=None):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"\x00" * 64)
        monkeypatch.setattr(media_probe, "_find_ffprobe", lambda: "/fake/ffprobe")

        def _run(cmd, *a, **k):
            if by_arg:
                for needle, out in by_arg.items():
                    if any(needle in str(part) for part in cmd):
                        return _Result(stdout=out)
            return _Result(stdout=stdout)

        monkeypatch.setattr(media_probe.subprocess, "run", _run)
        return str(f)

    def test_resolution(self, monkeypatch, tmp_path):
        f = self._probe_with(monkeypatch, tmp_path, "1280,720\n\n1280,720\n")
        assert media_probe.get_video_resolution(f) == (1280, 720)

    def test_resolution_slim_build_ts_dims_unparsed(self, monkeypatch, tmp_path):
        """The bundled LGPL ffprobe has no video decoders/parsers, so TS
        dimensions probe as 0,0 (doubled). That must keep hitting the
        documented 1920x1080 fallback, not crash."""
        f = self._probe_with(monkeypatch, tmp_path, "0,0\n\n0,0\n")
        assert media_probe.get_video_resolution(f) == (1920, 1080)

    def test_framerate(self, monkeypatch, tmp_path):
        f = self._probe_with(monkeypatch, tmp_path, "25/1\n\n25/1\n")
        assert media_probe.get_video_framerate(f) == 25.0

    def test_channels(self, monkeypatch, tmp_path):
        f = self._probe_with(monkeypatch, tmp_path, "2\n\n2\n")
        assert media_probe.get_audio_channels(f) == 2

    def test_duration_stream_row_doubled(self, monkeypatch, tmp_path):
        """Dispatch on the probe: the V:0 stream probe doubles on TS, the
        format=duration fallback never does. The (shorter) video-stream
        duration must now win — the documented clamp bound."""
        f = self._probe_with(
            monkeypatch, tmp_path, "",
            by_arg={
                "stream=duration": "10.000000\n\n10.000000\n",
                "format=duration": "10.221333\n",
            })
        assert media_probe.get_media_duration(f) == 10.0


# ── End-to-end: a real MPEG-TS misnamed .mp4 through extract_audio ─────────

_VENDOR_BIN = Path(__file__).resolve().parents[2] / "wrapper" / "vendor" / "ffmpeg" / "bin"
_FULL_FFMPEG = shutil.which("ffmpeg")  # fixture synthesis needs an aac encoder


def _make_ts_fixture(path):
    """Synthesize a ~1.2 s aac-in-mpegts file named .mp4 (the Mimir shape)."""
    proc = subprocess.run(
        [_FULL_FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1.2:sample_rate=48000",
         "-c:a", "aac", "-b:a", "96k", "-f", "mpegts", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    return proc.returncode == 0 and path.exists()


@pytest.mark.skipif(_FULL_FFMPEG is None,
                    reason="needs a full ffmpeg on PATH to synthesize the TS fixture")
class TestExtractAudioFromTs:
    def test_ts_misnamed_mp4_extracts(self, tmp_path, monkeypatch):
        src = tmp_path / "bddf3382-cafe-4a1d-ab32-8d6d64adfb0f.mp4"
        if not _make_ts_fixture(src):
            pytest.skip("PATH ffmpeg cannot encode aac/mpegts")
        # Prefer the exact field configuration: the bundled slim LGPL
        # binaries. Fall back to PATH when the vendor tree is absent.
        if (_VENDOR_BIN / "ffmpeg").is_file() and (_VENDOR_BIN / "ffprobe").is_file():
            monkeypatch.setenv("DOZA_FFMPEG_DIR", str(_VENDOR_BIN))
        else:
            monkeypatch.delenv("DOZA_FFMPEG_DIR", raising=False)
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        out = transcribe_module.extract_audio(str(src), project_dir=str(project_dir))
        assert out == str(project_dir / "audio.wav")
        # 16 kHz mono s16: ~1.15 s of samples ≥ the 0.1 s empty-audio gate
        # (the TS mux trims the 1.2 s sine slightly).
        assert os.path.getsize(out) > 32000
