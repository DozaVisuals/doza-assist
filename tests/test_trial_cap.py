"""Trial-cap fail-safes (FxFactory channel).

The review of the v3.5.12 fx-core merge flagged two trial gaps:
  - DOZA_TRIAL_MAX_SECONDS parsing crashed on malformed values and accepted
    inflated ones (an env-var route around the cap)
  - /media/audio served a full-length timeline_audio.wav ahead of the
    capped trial WAV
(The channel-neutral extract_audio same-path guard is covered in
test_v3512_port.py so it travels with the direct branch too.)
"""

import json
import os
import struct
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transcribe import _trial_max_seconds


class TestTrialMaxParsing:
    @pytest.mark.parametrize("raw,expected", [
        (None, 120),          # unset
        ("", 120),            # empty
        ("120", 120),         # normal
        ("90", 90),
        ("120.5", 120),       # float-ish string must not crash
        ("abc", 120),         # garbage must not crash
        ("-5", 1),            # negative clamps up (never an uncapped -t)
        ("0", 1),
        ("999999", 600),      # inflated value can't disable the cap
    ])
    def test_fail_safe_parse(self, monkeypatch, raw, expected):
        if raw is None:
            monkeypatch.delenv("DOZA_TRIAL_MAX_SECONDS", raising=False)
        else:
            monkeypatch.setenv("DOZA_TRIAL_MAX_SECONDS", raw)
        assert _trial_max_seconds() == expected


def _write_16k_mono_wav(path, seconds=2.0):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setframerate(16000)
        wf.setsampwidth(2)
        n = int(seconds * 16000)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))


class TestMediaAudioTrialOrdering:
    def test_trial_wav_beats_timeline_wav(self, tmp_path):
        import app as app_module
        app_module.app.config["PROJECTS_DIR"] = str(tmp_path / "projects")
        app_module.app.config["TESTING"] = True
        client = app_module.app.test_client()

        pid = "trial-order"
        pdir = Path(app_module.app.config["PROJECTS_DIR"]) / pid
        pdir.mkdir(parents=True)
        (pdir / "meta.json").write_text(json.dumps(
            {"id": pid, "name": "P", "status": "transcribed"}))
        _write_16k_mono_wav(pdir / "timeline_audio.wav", seconds=4.0)
        _write_16k_mono_wav(pdir / "audio_trial.wav", seconds=1.0)

        resp = client.get(f"/project/{pid}/media/audio")
        assert resp.status_code == 200
        trial_size = (pdir / "audio_trial.wav").stat().st_size
        assert int(resp.headers["Content-Length"]) == trial_size

    def test_timeline_wav_serves_when_no_trial_artifact(self, tmp_path):
        import app as app_module
        app_module.app.config["PROJECTS_DIR"] = str(tmp_path / "projects")
        app_module.app.config["TESTING"] = True
        client = app_module.app.test_client()

        pid = "timeline-only"
        pdir = Path(app_module.app.config["PROJECTS_DIR"]) / pid
        pdir.mkdir(parents=True)
        (pdir / "meta.json").write_text(json.dumps(
            {"id": pid, "name": "P", "status": "transcribed"}))
        _write_16k_mono_wav(pdir / "timeline_audio.wav", seconds=4.0)

        resp = client.get(f"/project/{pid}/media/audio")
        assert resp.status_code == 200
        tl_size = (pdir / "timeline_audio.wav").stat().st_size
        assert int(resp.headers["Content-Length"]) == tl_size
