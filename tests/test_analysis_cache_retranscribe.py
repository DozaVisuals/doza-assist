"""Retranscribe vs the analysis cache (field bug 2026-09-07, PAC project).

Retranscribe cleared the analysis and deleted the segment vectors but kept
the analysis cache; an identical re-transcription then hit the cache, the
server restored the analysis and said 'cached', and the page (server-
rendered tab) stayed on the empty state — with the vectors gone for good.

Now: the vectors are stashed and restored when the transcript is
unchanged; when they are gone, /analyze runs the worker, which reuses the
cached analysis and rebuilds only the derived files.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import app as app_module  # noqa: E402


@pytest.fixture
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    app_module.app.config['UPLOAD_FOLDER'] = str(tmp_path / 'uploads')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    Path(app_module.app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


TRANSCRIPT = {
    'segments': [
        {'start': 0.0, 'end': 5.0, 'text': 'hello', 'speaker': 'A'},
        {'start': 5.0, 'end': 10.0, 'text': 'world', 'speaker': 'A'},
    ],
    'language': 'en',
}
BEATS = [{'start': '00:00:00', 'end': '00:00:05', 'label': 'Beat', 'description': 'stub beat'}]


def _make_project(pid, extra=None):
    d = Path(app_module.app.config['PROJECTS_DIR']) / pid
    d.mkdir(parents=True, exist_ok=True)
    meta = {'id': pid, 'name': 'Cache Test', 'transcript': TRANSCRIPT}
    meta.update(extra or {})
    (d / 'meta.json').write_text(json.dumps(meta))
    return d


def _wait(client, pid, **payload):
    resp = client.post(f'/project/{pid}/analyze', json=payload)
    t = app_module._analysis_threads.get(pid)
    if t is not None:
        t.join(timeout=10)
    return resp


def test_cache_hit_with_dropped_vectors_rebuilds_without_the_model(client, monkeypatch):
    pid = 'retx'
    h = app_module._transcript_hash(TRANSCRIPT)
    _make_project(pid, {
        'analysis': None,
        'derived_stale': True,
        'analysis_cache': {h: {'all': {'analysis': {'summary': 'cached', 'story_beats': BEATS},
                                        'cached_at': 'x'}}},
    })
    calls = {'analyze': 0, 'vectors': 0}

    def _no_model(*a, **k):
        calls['analyze'] += 1
        raise AssertionError('the model must not run: the analysis is cached')

    def _vectors(*a, **k):
        calls['vectors'] += 1
        return [{'seg_id': 'SEG001', 'timecode_in': '00:00:00', 'timecode_out': '00:00:05',
                 'thread_title': 'T', 'narrative_score': 'high', 'theme_tags': []}]
    monkeypatch.setattr('ai_analysis.analyze_transcript', _no_model)
    monkeypatch.setattr('ai_analysis.generate_segment_vectors', _vectors)

    resp = _wait(client, pid, type='all')
    assert resp.status_code == 200
    assert resp.get_json()['status'] == 'started'
    assert calls['analyze'] == 0 and calls['vectors'] == 1
    meta = json.loads((Path(app_module.app.config['PROJECTS_DIR']) / pid / 'meta.json').read_text())
    assert meta['analysis']['summary'] == 'cached'
    assert 'derived_stale' not in meta
    assert app_module.load_segment_vectors(pid)


def test_cache_hit_with_vectors_present_still_short_circuits(client, monkeypatch):
    pid = 'hit'
    h = app_module._transcript_hash(TRANSCRIPT)
    d = _make_project(pid, {
        'analysis_cache': {h: {'all': {'analysis': {'summary': 'cached', 'story_beats': BEATS},
                                        'cached_at': 'x'}}},
    })
    (d / 'segment_vectors.json').write_text('[{"seg_id": "SEG001"}]')
    monkeypatch.setattr('ai_analysis.analyze_transcript',
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError('no model')))
    resp = _wait(client, pid, type='all')
    assert resp.get_json()['status'] == 'cached'


def test_force_still_runs_the_model_even_when_dropped(client, monkeypatch):
    pid = 'force'
    h = app_module._transcript_hash(TRANSCRIPT)
    _make_project(pid, {
        'derived_stale': True,
        'analysis_cache': {h: {'all': {'analysis': {'summary': 'cached', 'story_beats': BEATS},
                                        'cached_at': 'x'}}},
    })
    calls = {'analyze': 0}

    def _fresh(*a, **k):
        calls['analyze'] += 1
        return {'summary': 'fresh', 'story_beats': BEATS}
    monkeypatch.setattr('ai_analysis.analyze_transcript', _fresh)
    monkeypatch.setattr('ai_analysis.generate_segment_vectors', lambda *a, **k: [])
    resp = _wait(client, pid, type='all', force=True)
    assert resp.get_json()['status'] == 'started' and calls['analyze'] == 1


def test_stash_and_restore_when_transcript_unchanged(client):
    pid = 'stash'
    d = _make_project(pid, {'prev_transcript_hash': app_module._transcript_hash(TRANSCRIPT)})
    (d / 'segment_vectors.json').write_text('[{"seg_id": "SEG001"}]')
    assert app_module._stash_segment_vectors(pid) is True
    assert not (d / 'segment_vectors.json').exists() and (d / 'segment_vectors.stale.json').exists()
    assert app_module._restore_stashed_vectors(pid, app_module._transcript_hash(TRANSCRIPT)) is True
    assert (d / 'segment_vectors.json').exists() and not (d / 'segment_vectors.stale.json').exists()
    meta = json.loads((d / 'meta.json').read_text())
    assert 'derived_stale' not in meta and 'prev_transcript_hash' not in meta


def test_stash_dropped_when_transcript_changed(client):
    pid = 'changed'
    d = _make_project(pid, {'prev_transcript_hash': 'old-hash'})
    (d / 'segment_vectors.json').write_text('[{"seg_id": "SEG001"}]')
    app_module._stash_segment_vectors(pid)
    assert app_module._restore_stashed_vectors(pid, app_module._transcript_hash(TRANSCRIPT)) is False
    assert not (d / 'segment_vectors.json').exists() and not (d / 'segment_vectors.stale.json').exists()
    meta = json.loads((d / 'meta.json').read_text())
    assert meta.get('derived_stale') is True and 'prev_transcript_hash' not in meta


def test_stash_is_a_noop_without_vectors(client):
    pid = 'none'
    _make_project(pid)
    assert app_module._stash_segment_vectors(pid) is False
