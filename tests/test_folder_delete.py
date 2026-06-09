"""POST /folder/delete removes every project in a folder — and only that
folder — with the same path-confined rmtree as delete_project.

Backs the dashboard's per-folder "✕" control, which clears an entire
collection after an explicit confirmation. Folders are implicit (a folder
exists only while a project's meta.folder names it), so deleting all member
projects makes the folder disappear from the dashboard.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module  # noqa: E402


@pytest.fixture
def projects_dir(tmp_path):
    d = tmp_path / "projects"
    d.mkdir()
    app_module.app.config['PROJECTS_DIR'] = str(d)
    app_module.app.config['TESTING'] = True
    return d


@pytest.fixture
def client(projects_dir):
    return app_module.app.test_client()


def _mk_project(projects_dir, pid, folder):
    pdir = projects_dir / pid
    pdir.mkdir()
    (pdir / "meta.json").write_text(
        json.dumps({"name": pid, "folder": folder, "created_at": "2026-01-01"})
    )
    (pdir / "transcript.json").write_text("{}")  # prove the whole dir goes
    return pdir


def test_deletes_only_the_target_folder(client, projects_dir):
    a1 = _mk_project(projects_dir, "a1", "Alpha")
    a2 = _mk_project(projects_dir, "a2", "Alpha")
    b1 = _mk_project(projects_dir, "b1", "Beta")
    unfiled = _mk_project(projects_dir, "u1", "")

    res = client.post('/folder/delete', json={"folder": "Alpha"})
    assert res.status_code == 200
    body = res.get_json()
    assert body['deleted'] == 2
    assert body['folder'] == "Alpha"

    # Alpha's projects are gone; Beta and unfiled are untouched.
    assert not a1.exists() and not a2.exists()
    assert b1.exists() and unfiled.exists()
    # PROJECTS_DIR itself survives.
    assert os.path.isdir(str(projects_dir))


def test_blank_folder_name_is_rejected(client, projects_dir):
    keep = _mk_project(projects_dir, "u1", "")  # unfiled has folder == ""
    res = client.post('/folder/delete', json={"folder": "   "})
    assert res.status_code == 400
    # A blank name must NOT sweep the unfiled bucket.
    assert keep.exists()


def test_unknown_folder_is_a_noop(client, projects_dir):
    b1 = _mk_project(projects_dir, "b1", "Beta")
    res = client.post('/folder/delete', json={"folder": "Nope"})
    assert res.status_code == 200
    assert res.get_json()['deleted'] == 0
    assert b1.exists()


def test_never_touches_paths_outside_projects_dir(client, projects_dir, tmp_path):
    sentinel = tmp_path / "outside.txt"
    sentinel.write_text("keep me")
    _mk_project(projects_dir, "g1", "Gamma")
    client.post('/folder/delete', json={"folder": "Gamma"})
    assert sentinel.exists()
