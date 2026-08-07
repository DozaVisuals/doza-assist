"""Analysis-freshness UX contract (FX tester round 2).

Re-clicking AI Analysis on an unchanged transcript used to silently
cache-hit: 3 seconds of fake "Analyzing…", a reload, identical content.
Now the server exposes analysis_fresh at render (subdued Re-run button +
confirm), the frontend sends force=true on a confirmed re-run, and the
backend honors force by bypassing the cache. These tests pin all three.
"""

import json
import os
import sys
import tempfile
import unittest.mock as mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module


@pytest.fixture()
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def _seed(tmp_path, pid, *, fresh_cache, with_analysis=True):
    transcript = {'segments': [
        {'start': 0.0, 'end': 2.0, 'text': 'hello there', 'speaker': 'A'},
        {'start': 2.0, 'end': 4.0, 'text': 'general kenobi', 'speaker': 'B'},
    ], 'language': 'en'}
    analysis = {'summary': 'x', 'story_beats': [{'title': 't'}]}
    meta = {'id': pid, 'name': 'Fresh Test', 'status': 'transcribed',
            'source_path': '/nonexistent.mov', 'transcript': transcript}
    if with_analysis:
        meta['analysis'] = analysis
        h = (app_module._transcript_hash(transcript)
             if fresh_cache else 'deadbeef' * 8)
        meta['analysis_cache'] = {
            h: {'all': {'analysis': analysis, 'cached_at': 'now'}}}
    d = tmp_path / pid
    d.mkdir()
    (d / 'meta.json').write_text(json.dumps(meta))
    return meta


# ── _analysis_is_fresh truth table ───────────────────────────────────

def test_fresh_when_cache_matches_current_transcript(tmp_path):
    meta = _seed(tmp_path, 'p1', fresh_cache=True)
    assert app_module._analysis_is_fresh(meta) is True


def test_stale_when_transcript_changed_since_cache(tmp_path):
    meta = _seed(tmp_path, 'p2', fresh_cache=False)
    assert app_module._analysis_is_fresh(meta) is False


def test_not_fresh_without_analysis(tmp_path):
    meta = _seed(tmp_path, 'p3', fresh_cache=True, with_analysis=False)
    assert app_module._analysis_is_fresh(meta) is False


def test_not_fresh_on_malformed_cache():
    assert app_module._analysis_is_fresh(
        {'analysis': {'x': 1}, 'transcript': {'segments': []},
         'analysis_cache': 'not-a-dict'}) is False
    assert app_module._analysis_is_fresh({}) is False


# ── /analyze force contract ──────────────────────────────────────────

def test_analyze_without_force_cache_hits(client, tmp_path):
    _seed(tmp_path, 'p4', fresh_cache=True)
    r = client.post('/project/p4/analyze', json={'type': 'all'})
    assert r.get_json()['status'] == 'cached'


def test_analyze_with_force_bypasses_cache(client, tmp_path):
    _seed(tmp_path, 'p5', fresh_cache=True)
    with mock.patch.object(app_module, '_run_analysis_worker') as worker, \
         mock.patch('ai_analysis._ollama_is_active', return_value=False):
        r = client.post('/project/p5/analyze',
                        json={'type': 'all', 'force': True})
    body = r.get_json()
    assert body['status'] != 'cached'
    assert worker.called or body['status'] in ('started', 'running')


# ── rendered button state ────────────────────────────────────────────

def test_fresh_project_renders_rerun_button(client, tmp_path):
    _seed(tmp_path, 'p6', fresh_cache=True)
    html = client.get('/project/p6').data.decode()
    assert 'Re-run Analysis' in html
    assert 'const ANALYSIS_FRESH = true;' in html


def test_stale_project_renders_primary_button(client, tmp_path):
    _seed(tmp_path, 'p7', fresh_cache=False)
    html = client.get('/project/p7').data.decode()
    assert 'AI Analysis' in html
    assert 'Re-run Analysis' not in html
    assert 'const ANALYSIS_FRESH = false;' in html
