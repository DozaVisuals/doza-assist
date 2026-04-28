"""Tests for the /analyze transcript-hash cache.

Re-running analysis on the same transcript should be instant (no AI call).
Editing the transcript invalidates the cache. force=true busts it.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module.app.config, '__setitem__', lambda *a, **k: None, raising=False)
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    app_module.app.config['UPLOAD_FOLDER'] = str(tmp_path / 'uploads')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    Path(app_module.app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


@pytest.fixture
def project_with_transcript(client):
    pid = "cache-test"
    project_dir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    project_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": pid,
        "name": "Cache Test",
        "transcript": {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "hello", "speaker": "A"},
                {"start": 5.0, "end": 10.0, "text": "world", "speaker": "A"},
            ],
            "language": "en",
        },
    }
    (project_dir / "meta.json").write_text(json.dumps(meta))
    return pid


def test_first_call_runs_analysis(client, project_with_transcript, monkeypatch):
    calls = {'count': 0}

    def _fake_analyze(*a, **k):
        calls['count'] += 1
        return {'summary': 'fresh', 'story_beats': []}

    monkeypatch.setattr('ai_analysis.analyze_transcript', _fake_analyze)
    monkeypatch.setattr('ai_analysis.generate_segment_vectors', lambda *a, **k: [])

    resp = client.post(f'/project/{project_with_transcript}/analyze', json={'type': 'all'})
    assert resp.status_code == 200
    assert calls['count'] == 1
    assert resp.get_json()['status'] == 'analyzed'


def test_second_call_hits_cache(client, project_with_transcript, monkeypatch):
    calls = {'count': 0}

    def _fake_analyze(*a, **k):
        calls['count'] += 1
        return {'summary': 'fresh', 'story_beats': []}

    monkeypatch.setattr('ai_analysis.analyze_transcript', _fake_analyze)
    monkeypatch.setattr('ai_analysis.generate_segment_vectors', lambda *a, **k: [])

    client.post(f'/project/{project_with_transcript}/analyze', json={'type': 'all'})
    resp = client.post(f'/project/{project_with_transcript}/analyze', json={'type': 'all'})
    assert resp.status_code == 200
    assert calls['count'] == 1  # the second call did NOT trigger another analysis
    assert resp.get_json()['status'] == 'cached'


def test_force_bypasses_cache(client, project_with_transcript, monkeypatch):
    calls = {'count': 0}

    def _fake_analyze(*a, **k):
        calls['count'] += 1
        return {'summary': f"run-{calls['count']}", 'story_beats': []}

    monkeypatch.setattr('ai_analysis.analyze_transcript', _fake_analyze)
    monkeypatch.setattr('ai_analysis.generate_segment_vectors', lambda *a, **k: [])

    client.post(f'/project/{project_with_transcript}/analyze', json={'type': 'all'})
    resp = client.post(
        f'/project/{project_with_transcript}/analyze',
        json={'type': 'all', 'force': True},
    )
    assert resp.status_code == 200
    assert calls['count'] == 2  # force=True ran a second analysis


def test_different_analysis_types_cached_separately(client, project_with_transcript, monkeypatch):
    calls = {'count': 0}

    def _fake_analyze(*a, **k):
        calls['count'] += 1
        return {'summary': k.get('analysis_type', '?'), 'story_beats': []}

    monkeypatch.setattr('ai_analysis.analyze_transcript', _fake_analyze)
    monkeypatch.setattr('ai_analysis.generate_segment_vectors', lambda *a, **k: [])

    client.post(f'/project/{project_with_transcript}/analyze', json={'type': 'story'})
    client.post(f'/project/{project_with_transcript}/analyze', json={'type': 'social'})
    # Each type ran once; neither was a cache hit because the bucket is keyed
    # by (hash, type).
    assert calls['count'] == 2

    # Now repeat both — both should hit the cache.
    client.post(f'/project/{project_with_transcript}/analyze', json={'type': 'story'})
    client.post(f'/project/{project_with_transcript}/analyze', json={'type': 'social'})
    assert calls['count'] == 2


def test_transcript_edit_invalidates_cache(client, project_with_transcript, monkeypatch):
    calls = {'count': 0}

    def _fake_analyze(*a, **k):
        calls['count'] += 1
        return {'summary': f"r{calls['count']}", 'story_beats': []}

    monkeypatch.setattr('ai_analysis.analyze_transcript', _fake_analyze)
    monkeypatch.setattr('ai_analysis.generate_segment_vectors', lambda *a, **k: [])

    client.post(f'/project/{project_with_transcript}/analyze', json={'type': 'all'})

    # Mutate the transcript content. The hash should change → next analyze
    # call must miss the cache and re-run.
    meta_path = Path(app_module.app.config['PROJECTS_DIR']) / project_with_transcript / 'meta.json'
    meta = json.loads(meta_path.read_text())
    meta['transcript']['segments'][0]['text'] = 'changed'
    meta_path.write_text(json.dumps(meta))

    client.post(f'/project/{project_with_transcript}/analyze', json={'type': 'all'})
    assert calls['count'] == 2
