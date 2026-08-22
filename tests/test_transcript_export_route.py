"""/project/<id>/transcript-export must serve FRESH meta.json state.

Field bug (2026-08-19, Butch Robinson): txt/srt/json downloads were built
from the page's render-time PROJECT snapshot, so speaker renames (and the
diarization worker's labels) that landed in meta.json after render leaked
raw SPEAKER_NN labels until the user happened to reload. The download
handlers now re-fetch this route first; it must always read current
meta.json state, never a cached copy.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module


@pytest.fixture
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def _write_meta(pid, meta):
    pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    pdir.mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(pdir / 'meta.json', 'w'))


def _meta(pid, speaker_names):
    return {
        'id': pid, 'name': pid, 'status': 'transcribed',
        'source_path': f'/nonexistent/{pid}.mp3',
        'speaker_names': speaker_names,
        'transcript': {'language': 'de', 'segments': [
            {'start': 0.0, 'end': 2.0, 'text': 'hallo', 'speaker': 'SPEAKER_00',
             'start_formatted': '00:00:00.000', 'end_formatted': '00:00:02.000',
             'words': []}]},
    }


def test_returns_transcript_and_speaker_names(client):
    _write_meta('p1', _meta('p1', {'SPEAKER_00': 'Anna Schmidt'}))
    resp = client.get('/project/p1/transcript-export')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['speaker_names'] == {'SPEAKER_00': 'Anna Schmidt'}
    assert data['transcript']['segments'][0]['text'] == 'hallo'
    assert data['transcript']['segments'][0]['speaker'] == 'SPEAKER_00'


def test_serves_fresh_state_after_rename(client):
    # The exact stale-snapshot scenario: state changes AFTER the first read
    # (a rename, or the diarization worker writing labels) — the route must
    # serve the new meta.json, not any cached copy.
    _write_meta('p2', _meta('p2', {}))
    first = client.get('/project/p2/transcript-export').get_json()
    assert first['speaker_names'] == {}

    _write_meta('p2', _meta('p2', {'SPEAKER_00': 'Butch'}))
    second = client.get('/project/p2/transcript-export').get_json()
    assert second['speaker_names'] == {'SPEAKER_00': 'Butch'}


def test_missing_project_404s(client):
    resp = client.get('/project/nope/transcript-export')
    assert resp.status_code == 404


def test_absent_fields_default_to_empty_objects(client):
    _write_meta('p3', {'id': 'p3', 'name': 'p3', 'status': 'uploaded',
                       'source_path': '/nonexistent/p3.mp3'})
    resp = client.get('/project/p3/transcript-export')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['transcript'] == {}
    assert data['speaker_names'] == {}
