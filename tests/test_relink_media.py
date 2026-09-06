"""POST /project/<id>/relink-media: update the stored source path only."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402


@pytest.fixture
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    app_module.app.config['EXPORTS_DIR'] = str(tmp_path / 'exports')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    Path(app_module.app.config['EXPORTS_DIR']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def _make_project(pid, source_path, **extra):
    pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    pdir.mkdir(parents=True)
    meta = {'id': pid, 'name': 'P', 'status': 'complete', 'source_path': source_path,
            'filepath': source_path, 'filename': os.path.basename(source_path),
            'transcript': {'duration': 100.0, 'segments': [{'start': 0, 'end': 1, 'text': 'hi'}]},
            'analysis': {'story_beats': [{'label': 'Beat'}]}, 'labeled_sections': [{'start': 1, 'end': 2}]}
    meta.update(extra)
    (pdir / 'meta.json').write_text(json.dumps(meta))


def _read_meta(pid):
    return json.loads((Path(app_module.app.config['PROJECTS_DIR']) / pid / 'meta.json').read_text())


def _durations(monkeypatch, table):
    monkeypatch.setattr(app_module, 'get_media_duration', lambda p: table.get(os.path.abspath(p)))


def test_relink_updates_path_and_touches_nothing_else(client, tmp_path, monkeypatch):
    old = tmp_path / 'gone.mov'
    new = tmp_path / 'moved.mov'
    new.write_bytes(b'x' * 2048)
    _make_project('p1', str(old))
    _durations(monkeypatch, {str(new): 100.4})
    res = client.post('/project/p1/relink-media', json={'source_path': str(new)})
    assert res.status_code == 200
    data = res.get_json()
    assert data['status'] == 'relinked' and data['filename'] == 'moved.mov' and data['length_mismatch'] is False
    meta = _read_meta('p1')
    assert meta['source_path'] == str(new) and meta['filepath'] == str(new)
    assert meta['filename'] == 'moved.mov' and meta['file_size'] == 2048 and meta['file_size_formatted'] == '2.0 KB'
    assert meta['transcript']['segments'][0]['text'] == 'hi'
    assert meta['analysis'] == {'story_beats': [{'label': 'Beat'}]}
    assert meta['labeled_sections'] == [{'start': 1, 'end': 2}]
    assert meta['status'] == 'complete'


def test_length_mismatch_warns_then_force_relinks(client, tmp_path, monkeypatch):
    old = tmp_path / 'gone.mov'
    new = tmp_path / 'other.mov'
    new.write_bytes(b'x')
    _make_project('p2', str(old))
    _durations(monkeypatch, {str(new): 130.0})  # original known from the transcript: 100 s
    res = client.post('/project/p2/relink-media', json={'source_path': str(new)})
    assert res.status_code == 409
    data = res.get_json()
    assert data['warning'] is True
    assert data['message'] == 'This file is a different length than the original. Timecodes may not line up.'
    assert data['old_duration'] == 100.0 and data['new_duration'] == 130.0
    assert _read_meta('p2')['source_path'] == str(old)  # unchanged
    res = client.post('/project/p2/relink-media', json={'source_path': str(new), 'force': True})
    assert res.status_code == 200 and res.get_json()['length_mismatch'] is True
    assert _read_meta('p2')['source_path'] == str(new)


def test_unknown_durations_do_not_block(client, tmp_path, monkeypatch):
    new = tmp_path / 'new.wav'
    new.write_bytes(b'x')
    _make_project('p3', str(tmp_path / 'gone.wav'), transcript={'segments': []})
    _durations(monkeypatch, {})
    assert client.post('/project/p3/relink-media', json={'source_path': str(new)}).status_code == 200


def test_original_still_present_is_probed_for_its_length(client, tmp_path, monkeypatch):
    old = tmp_path / 'still-here.mov'
    old.write_bytes(b'x')
    new = tmp_path / 'copy.mov'
    new.write_bytes(b'x')
    _make_project('p4', str(old))
    _durations(monkeypatch, {str(old): 250.0, str(new): 250.6})
    assert client.post('/project/p4/relink-media', json={'source_path': str(new)}).status_code == 200


def test_validation(client, tmp_path, monkeypatch):
    _make_project('p5', str(tmp_path / 'gone.mov'))
    _durations(monkeypatch, {})
    assert client.post('/project/p5/relink-media', json={}).status_code == 400
    assert client.post('/project/p5/relink-media', json={'source_path': str(tmp_path / 'missing.mov')}).status_code == 400
    bad = tmp_path / 'notes.txt'
    bad.write_text('x')
    assert client.post('/project/p5/relink-media', json={'source_path': str(bad)}).status_code == 400
    assert client.post('/project/nope/relink-media', json={'source_path': str(bad)}).status_code == 404
    assert _read_meta('p5')['source_path'] == str(tmp_path / 'gone.mov')
