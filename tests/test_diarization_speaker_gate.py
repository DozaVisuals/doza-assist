"""Tests for the server-side gate on OSS speaker-rename endpoints.

When a project has been diarized, the canonical rename surface is the
diarization Speakers sidebar (writes to ``meta["speaker_names"]`` while
leaving the raw SPEAKER_NN labels on each segment intact). The OSS
``/project/<id>/update-speakers`` endpoint collapses segments by raw
label, which silently destroys diarization state when multiple renames
land on the same string. We hit this in production once: five
SPEAKER_NN labels turned into a single "Chris" across 1167 segments
after the front-end lockout raced with a user click.

These tests lock in the server-side gate so even if the JS lockout
fails the bad write cannot reach disk. Both ``/update-speakers`` and
``/update-speaker-range`` return 409 on diarized projects.
"""

import json
import os
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve()
CORE_DIR = HERE.parent.parent
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import app as app_module  # noqa: E402


@pytest.fixture
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    app_module.app.config['UPLOAD_FOLDER'] = str(tmp_path / 'uploads')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    Path(app_module.app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def _seed_project(diarization_status):
    """Write a minimal meta.json to disk with the given diarization status."""
    pid = "gate-test"
    project_dir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    project_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": pid,
        "name": "Gate Test",
        "transcript": {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "hello",  "speaker": "SPEAKER_00"},
                {"start": 5.0, "end": 10.0, "text": "world", "speaker": "SPEAKER_01"},
            ],
            "language": "en",
        },
    }
    if diarization_status is not None:
        meta["diarization"] = {"status": diarization_status}
    (project_dir / "meta.json").write_text(json.dumps(meta))
    return pid, project_dir


def _read_segments(project_dir):
    meta = json.loads((project_dir / "meta.json").read_text())
    return meta.get("transcript", {}).get("segments", [])


def test_update_speakers_gated_when_diarization_done(client):
    pid, project_dir = _seed_project(diarization_status="done")
    resp = client.post(
        f'/project/{pid}/update-speakers',
        json={'mapping': {'SPEAKER_00': 'Chris'}},
    )
    assert resp.status_code == 409, (
        f"Expected 409 on diarized project, got {resp.status_code}: {resp.get_data(as_text=True)}"
    )
    body = resp.get_json() or {}
    assert body.get('error') == 'diarization_active'
    assert 'Speakers sidebar' in (body.get('message') or '')
    # Segments must be untouched: SPEAKER_00 and SPEAKER_01 still on disk.
    segs = _read_segments(project_dir)
    assert [s['speaker'] for s in segs] == ['SPEAKER_00', 'SPEAKER_01']


def test_update_speakers_allowed_when_diarization_absent(client):
    """Non-diarized projects keep the legacy OSS rename behavior."""
    pid, project_dir = _seed_project(diarization_status=None)
    resp = client.post(
        f'/project/{pid}/update-speakers',
        json={'mapping': {'SPEAKER_00': 'Chris'}},
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    segs = _read_segments(project_dir)
    assert segs[0]['speaker'] == 'Chris'
    assert segs[1]['speaker'] == 'SPEAKER_01'


def test_update_speakers_allowed_when_diarization_in_progress(client):
    """While diarization is queued or running the user still has the OSS UI
    available — only ``done`` triggers the lockout."""
    pid, project_dir = _seed_project(diarization_status='running')
    resp = client.post(
        f'/project/{pid}/update-speakers',
        json={'mapping': {'SPEAKER_00': 'Chris'}},
    )
    assert resp.status_code == 200
    assert _read_segments(project_dir)[0]['speaker'] == 'Chris'


def test_update_speaker_range_gated_when_diarization_done(client):
    pid, project_dir = _seed_project(diarization_status='done')
    resp = client.post(
        f'/project/{pid}/update-speaker-range',
        json={'start': 0.0, 'end': 5.0, 'speaker': 'Chris'},
    )
    assert resp.status_code == 409
    body = resp.get_json() or {}
    assert body.get('error') == 'diarization_active'
    segs = _read_segments(project_dir)
    assert segs[0]['speaker'] == 'SPEAKER_00'
