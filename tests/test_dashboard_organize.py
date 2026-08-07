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
