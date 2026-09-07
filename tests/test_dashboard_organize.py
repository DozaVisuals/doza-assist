"""Dashboard drag-and-drop organizing (owner feature, 2026-08-07).

Pins the rendered contract: full-surface folder drop wiring, an Unfiled
drop zone that exists whenever folders exist (so drag-OUT is always
possible), and the mid-drag affordance CSS.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module


@pytest.fixture()
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def _seed(tmp_path, pid, folder=''):
    meta = {'id': pid, 'name': f'P {pid}', 'status': 'uploaded',
            'source_path': '/nonexistent.mov', 'filename': 'x.mov',
            'created_at': '2026-08-07T00:00:00'}
    if folder:
        meta['folder'] = folder
    d = tmp_path / pid
    d.mkdir()
    (d / 'meta.json').write_text(json.dumps(meta))


def test_dashboard_with_folder_renders_all_drop_machinery(client, tmp_path):
    _seed(tmp_path, 'pa', folder='Docs')
    _seed(tmp_path, 'pb')
    html = client.get('/').data.decode()
    assert 'wireFolderDropTarget' in html
    assert 'unfiledDropZone' in html            # drag-out target present
    assert 'is-dragging-project' in html        # mid-drag affordance CSS
    assert 'data-folder="Docs"' in html


def test_unfiled_zone_exists_even_when_everything_is_filed(client, tmp_path):
    # Drag-OUT must be possible when no project is currently unfiled.
    _seed(tmp_path, 'pc', folder='Docs')
    html = client.get('/').data.decode()
    assert 'unfiledDropZone' in html
    assert 'Drop here to remove from its folder' in html


def test_new_folder_never_nests_inside_the_unfiled_zone(client, tmp_path):
    """Regression (field report 2026-09-07): with no folders yet, New Folder
    inserted its group before the first ``.project-list`` — which lives
    INSIDE ``#unfiledDropZone`` — so a drop on the new folder bubbled to the
    Unfiled zone too and the "unfile" write landed last: the first project
    dragged in came straight back out and the folder vanished on reload.
    The insertion anchor is now the Unfiled zone itself, and a filed drop
    stops at the innermost target."""
    import re
    _seed(tmp_path, 'pa')
    _seed(tmp_path, 'pb')
    html = client.get('/').data.decode()
    assert 'function _newFolderAnchor' in html
    assert "document.getElementById('unfiledDropZone')" in html
    assert "querySelector('.project-list') || document.querySelector('.empty-state')" not in html
    assert re.search(r"stopPropagation\(\);\s*doMove\(projectId", html)
