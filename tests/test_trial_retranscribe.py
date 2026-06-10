"""Retranscribe hardening + share removal (direct channel).

Ported from the FX-channel suite (2026-06-10). The direct channel never
sets DOZA_TRIAL / truncated_for_trial and ships no trial banner, so the
banner tests stay FX-only; what's covered here is the channel-shared
behavior: /retranscribe validates the source BEFORE destroying state and
clears the previous run's job state, and the share/review routes fail
closed (404) — share is not part of the commercial channels.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as app_module  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    app_module.app.config['UPLOAD_FOLDER'] = str(tmp_path / 'uploads')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    Path(app_module.app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


@pytest.fixture
def trial_project(client):
    """A transcribed project with the artifacts a previous run leaves behind."""
    pid = "trial-retranscribe-test"
    project_dir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    project_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": pid,
        "name": "Trial Retranscribe Test",
        "status": "transcribed",
        "language": "en",
        "source_path": str(project_dir / "source.mp4"),
        "transcript": {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "hello", "speaker": "A"},
                {"start": 5.0, "end": 10.0, "text": "world", "speaker": "A"},
            ],
            "language": "en",
        },
    }
    (project_dir / "meta.json").write_text(json.dumps(meta))
    # Artifacts a transcription leaves behind.
    (project_dir / "audio.wav").write_bytes(b"RIFFfake")
    (project_dir / "paragraph_index.json").write_text("{}")
    (project_dir / "segment_vectors.json").write_text("{}")
    return pid, project_dir


def test_retranscribe_clears_previous_state(client, trial_project):
    pid, project_dir = trial_project
    # The route validates the source before destroying anything.
    Path(json.loads((project_dir / "meta.json").read_text())['source_path']).write_bytes(b"fake")
    resp = client.post(f'/project/{pid}/retranscribe', json={})
    assert resp.status_code == 200
    assert resp.get_json()['status'] == 'cleared'

    meta = json.loads((project_dir / "meta.json").read_text())
    assert meta['transcript'] is None
    assert meta['status'] == 'uploaded'        # routes back through /transcribe
    # No orphaned artifacts or stale caches.
    assert not (project_dir / "audio.wav").exists()
    assert not (project_dir / "paragraph_index.json").exists()
    assert not (project_dir / "segment_vectors.json").exists()


def test_retranscribe_validates_source_before_destroying(client, trial_project):
    """Missing source file → 404 error and the existing transcript SURVIVES."""
    pid, project_dir = trial_project
    resp = client.post(f'/project/{pid}/retranscribe', json={})
    assert resp.status_code == 404
    assert 'Source file not found' in resp.get_json()['error']
    meta = json.loads((project_dir / "meta.json").read_text())
    assert meta['transcript'] is not None
    assert (project_dir / "audio.wav").exists()


def test_retranscribe_refuses_while_job_running(client, trial_project):
    """A live transcription job blocks retranscribe with a 409."""
    pid, project_dir = trial_project
    Path(json.loads((project_dir / "meta.json").read_text())['source_path']).write_bytes(b"fake")
    app_module._transcribe_jobs[pid] = {"phase": "transcribing"}
    try:
        resp = client.post(f'/project/{pid}/retranscribe', json={})
        assert resp.status_code == 409
        assert 'already running' in resp.get_json()['error']
        meta = json.loads((project_dir / "meta.json").read_text())
        assert meta['transcript'] is not None
    finally:
        app_module._transcribe_jobs.pop(pid, None)


def test_share_routes_fail_closed(client, trial_project):
    """Share is not part of the commercial channels — stale links 404."""
    pid, _ = trial_project
    assert client.get(f'/share/{pid}').status_code == 404
    assert client.get(f'/review/{pid}').status_code == 404
    assert client.get(f'/project/{pid}/share-settings').status_code == 404


def test_no_share_button_in_project_page(client, trial_project):
    """The editor page ships no Share entry point at all."""
    pid, _ = trial_project
    html = client.get(f'/project/{pid}').get_data(as_text=True)
    assert 'copyShareLink' not in html
    assert 'generateShareLink' not in html
    assert '>Share</button>' not in html
