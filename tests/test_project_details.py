"""PATCH /project/<id>/details: edit name, client, interviewer, subject and
speaker count after creation. Storage only; validation shared with /create."""

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


def _make_project(pid, **extra):
    pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    pdir.mkdir(parents=True)
    meta = {'id': pid, 'name': 'Old name', 'client_name': 'Acme', 'interviewer_name': 'Chris',
            'subject_name': 'Pedro', 'num_speakers': 2, 'language': 'en', 'status': 'complete',
            'transcript': {'segments': [{'start': 0, 'end': 1, 'text': 'hi'}]}}
    meta.update(extra)
    (pdir / 'meta.json').write_text(json.dumps(meta))
    return pdir


def _read_meta(pid):
    return json.loads((Path(app_module.app.config['PROJECTS_DIR']) / pid / 'meta.json').read_text())


def test_patch_updates_only_the_editable_fields(client):
    _make_project('p1')
    res = client.patch('/project/p1/details', json={
        'name': 'New name', 'client_name': 'Globex', 'interviewer_name': 'Dana',
        'subject_name': 'Maya', 'num_speakers': 3,
        'language': 'no', 'status': 'transcribing', 'transcript': None})
    assert res.status_code == 200
    data = res.get_json()
    assert sorted(data['changed']) == ['client_name', 'interviewer_name', 'name', 'num_speakers', 'subject_name']
    meta = _read_meta('p1')
    assert meta['name'] == 'New name' and meta['client_name'] == 'Globex'
    assert meta['interviewer_name'] == 'Dana' and meta['subject_name'] == 'Maya'
    assert meta['num_speakers'] == 3
    # not editable here: language stays under Retranscribe; nothing else moves
    assert meta['language'] == 'en' and meta['status'] == 'complete'
    assert meta['transcript']['segments'][0]['text'] == 'hi'


def test_patch_partial_body_touches_only_given_keys(client):
    _make_project('p2')
    res = client.patch('/project/p2/details', json={'client_name': ''})
    assert res.status_code == 200 and res.get_json()['changed'] == ['client_name']
    meta = _read_meta('p2')
    assert meta['client_name'] == '' and meta['name'] == 'Old name' and meta['num_speakers'] == 2


def test_patch_validation(client):
    _make_project('p3')
    assert client.patch('/project/p3/details', json={'name': '   '}).status_code == 400
    assert client.patch('/project/p3/details', json={'num_speakers': 'two'}).status_code == 400
    assert client.patch('/project/p3/details', json={'num_speakers': 0}).status_code == 400
    assert client.patch('/project/p3/details', json={'num_speakers': 13}).status_code == 400
    assert client.patch('/project/p3/details', json={'language': 'no'}).status_code == 400  # nothing editable
    assert client.patch('/project/nope/details', json={'name': 'x'}).status_code == 404
    assert _read_meta('p3')['name'] == 'Old name'


def test_no_change_is_a_quiet_success(client):
    _make_project('p4')
    res = client.patch('/project/p4/details', json={'name': 'Old name', 'num_speakers': 2})
    assert res.status_code == 200 and res.get_json()['changed'] == []


def test_create_and_edit_share_one_rule_set():
    d = app_module._normalize_project_details({'project_name': '', 'num_speakers': '4'})
    assert d['name'] is None and d['num_speakers'] == 4
    assert d['interviewer_name'] == 'Interviewer' and d['subject_name'] == 'Subject'
    with pytest.raises(ValueError):
        app_module._normalize_project_details({'num_speakers': 99})
    with pytest.raises(ValueError):
        app_module._normalize_project_details({'name': ''}, partial=True)
