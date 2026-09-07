"""AI Analysis tab (2026-09-07): story beats and social clips carry the same
Show text transcript disclosure as the soundbites, so every clip the page
can play can also be read."""
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


def test_every_analysis_row_has_show_text(client):
    pid = 'showtext'
    d = Path(app_module.app.config['PROJECTS_DIR']) / pid
    d.mkdir()
    (d / 'meta.json').write_text(json.dumps({
        'id': pid, 'name': 'Show text', 'filename': 'x.wav', 'filepath': str(d / 'x.wav'),
        'transcript': {'segments': [{'start': 0, 'end': 5, 'text': 'hello there', 'speaker': 'A'}], 'language': 'en'},
        'analysis': {
            'summary': 'S',
            'strongest_soundbites': [{'start': '00:00:00', 'end': '00:00:05', 'text': 'hello there', 'why': 'w', 'title': 'Hi'}],
            'story_beats': [{'order': 1, 'label': 'Open', 'description': 'd', 'start': '00:00:00', 'end': '00:00:05'},
                            {'order': 2, 'label': 'Close', 'description': 'd', 'start': '00:00:01', 'end': '00:00:05'}],
            'social_clips': [{'title': 'Reel', 'why': 'w', 'start': '00:00:00', 'end': '00:00:05', 'platform': 'TikTok'}],
        },
    }))
    r = client.get(f'/project/{pid}')
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    # one disclosure per row: 1 soundbite + 2 beats + 1 social
    assert html.count('<span class="clip-expand-label">Show text</span>') >= 4
    assert html.count('class="analysis-item-transcript"') >= 4
    assert 'function fillAnalysisTranscripts' in html
