"""Labels live sync: the open page picks up selects an assistant adds over
the local connector, and the page's own save never erases them.

- GET /labels answers {unchanged} for a matching rev without reading the
  project, and the sections plus a new rev after any write.
- POST /labels with seen_sync_ids keeps assistant selects the page never
  held; ones it held and dropped are real deletions.
- Older clients (no seen_sync_ids) keep today's replace-all behaviour.
"""
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module  # noqa: E402


@pytest.fixture
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def _seed(pid, sections):
    d = Path(app_module.app.config['PROJECTS_DIR']) / pid
    d.mkdir(parents=True, exist_ok=True)
    (d / 'meta.json').write_text(json.dumps({'id': pid, 'name': pid, 'labeled_sections': sections}))


AI = {'start': 10.0, 'end': 14.0, 'color': 'purple', 'text': 'ai pick', 'origin': 'ai',
      'author': 'AI assistant', 'sync_id': 'ai:abc123', 'comment': 'Great line'}
MINE = {'start': 1.0, 'end': 3.0, 'color': 'blue', 'text': 'my pick'}


def test_get_labels_rev_short_circuits_and_reports_changes(client):
    _seed('p1', [MINE])
    first = client.get('/project/p1/labels').get_json()
    assert first['labeled_sections'] == [MINE] and first['rev']
    again = client.get(f"/project/p1/labels?rev={first['rev']}").get_json()
    assert again == {'unchanged': True, 'rev': first['rev']}
    time.sleep(0.01)
    _seed('p1', [MINE, AI])                     # an assistant wrote a select
    changed = client.get(f"/project/p1/labels?rev={first['rev']}").get_json()
    assert changed['rev'] != first['rev']
    assert [s.get('sync_id') for s in changed['labeled_sections']] == [None, 'ai:abc123']
    assert client.get('/project/nope/labels').status_code == 404


def test_save_keeps_unseen_assistant_selects_and_honours_real_deletions(client):
    _seed('p2', [MINE, AI])
    # The page loaded before the assistant's select existed and posts only its own.
    r = client.post('/project/p2/labels', json={'color_labels': {}, 'labeled_sections': [MINE],
                                                'seen_sync_ids': []})
    assert r.status_code == 200
    stored = json.loads((Path(app_module.app.config['PROJECTS_DIR']) / 'p2' / 'meta.json').read_text())
    assert [s.get('sync_id') for s in stored['labeled_sections']] == [None, 'ai:abc123']
    assert stored['labeled_sections'][1]['author'] == 'AI assistant'
    # The page has seen it (live sync) and the editor deleted it: it stays gone.
    client.post('/project/p2/labels', json={'color_labels': {}, 'labeled_sections': [MINE],
                                            'seen_sync_ids': ['ai:abc123']})
    stored = json.loads((Path(app_module.app.config['PROJECTS_DIR']) / 'p2' / 'meta.json').read_text())
    assert stored['labeled_sections'] == [MINE]


def test_save_without_seen_ids_is_the_old_replace_all(client):
    _seed('p3', [MINE, AI])
    client.post('/project/p3/labels', json={'color_labels': {}, 'labeled_sections': [MINE]})
    stored = json.loads((Path(app_module.app.config['PROJECTS_DIR']) / 'p3' / 'meta.json').read_text())
    assert stored['labeled_sections'] == [MINE]
