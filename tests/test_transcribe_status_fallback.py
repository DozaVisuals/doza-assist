"""The transcribe status stream remembers the engine across events and
records a Parakeet to Whisper fallback so the UI can say why a fast job
turned slow. Previously each event replaced the snapshot and nothing
downstream rendered the engine at all."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module  # noqa: E402


@pytest.fixture
def projects_dir(tmp_path):
    app_module.app.config["PROJECTS_DIR"] = str(tmp_path / "projects")
    Path(app_module.app.config["PROJECTS_DIR"]).mkdir(parents=True)
    yield tmp_path / "projects"
    app_module._transcribe_jobs.pop("p1", None)


def _status(projects_dir):
    return json.loads((projects_dir / "p1" / "transcribe_status.json").read_text())


def test_parakeet_to_whisper_sets_fallback_from(projects_dir):
    w = app_module._make_transcribe_progress_writer("p1")
    w({"phase": "load_model", "pct": 5, "engine": "parakeet-mlx"})
    w({"phase": "load_audio", "pct": 8, "engine": "parakeet-mlx", "audio_sec": 129600})
    w({"phase": "queued", "pct": 0})  # memory gate event carries no engine
    snap = app_module._transcribe_jobs["p1"]
    assert snap["engine"] == "parakeet-mlx" and snap["fallback_from"] is None
    assert snap["audio_sec"] == 129600  # remembered across the engine-less event

    # WhisperX is announced first and is not bundled, then plain Whisper.
    w({"phase": "load_model", "pct": 5, "engine": "whisperx", "slow_mode": True})
    w({"phase": "load_model", "pct": 5, "engine": "whisper", "slow_mode": True})
    w({"phase": "transcribing", "pct": 12, "engine": "whisper", "audio_sec": 129600})
    snap = app_module._transcribe_jobs["p1"]
    assert snap["engine"] == "whisper"
    assert snap["fallback_from"] == "parakeet-mlx"
    assert _status(projects_dir)["fallback_from"] == "parakeet-mlx"


def test_no_fallback_when_whisper_was_the_first_engine(projects_dir):
    w = app_module._make_transcribe_progress_writer("p1")
    w({"phase": "load_model", "pct": 5, "engine": "whisper", "slow_mode": True})
    w({"phase": "transcribing", "pct": 20, "engine": "whisper"})
    assert app_module._transcribe_jobs["p1"]["fallback_from"] is None


def test_parakeet_only_run_never_reports_fallback(projects_dir):
    w = app_module._make_transcribe_progress_writer("p1")
    w({"phase": "load_model", "pct": 5, "engine": "parakeet-mlx"})
    w({"phase": "transcribing", "pct": 50, "engine": "parakeet-mlx", "audio_sec": 600})
    snap = app_module._transcribe_jobs["p1"]
    assert snap["fallback_from"] is None and snap["engine"] == "parakeet-mlx"
