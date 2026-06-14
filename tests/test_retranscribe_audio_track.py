"""Audio-track selection on retranscribe (1.0.19, tester: "no audio selector
even there is 4 tracks").

The audio-track picker existed only at project CREATION; a tester viewing an
already-transcribed multi-track source had no way to choose a track. The
Retranscribe modal now surfaces the picker and POSTs an ``audio_channel`` the
route already accepts. The DEFAULT for a multi-track source is the single
PRIMARY track — never "all tracks (mixed)" — because separate mono tracks are
discrete mics and mixing them corrupts per-voice separation.

These pin the backend plumbing (route stores the chosen channel) and guard the
frontend default against regressing to mix-by-default.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module

_PROJECT_HTML = Path(__file__).resolve().parents[1] / 'templates' / 'project.html'


@pytest.fixture
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def _make_project(pid, source_path):
    pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    pdir.mkdir(parents=True)
    meta = {'id': pid, 'name': 'P', 'status': 'transcribed',
            'language': 'no', 'source_path': source_path, 'audio_channel': 'all'}
    (pdir / 'meta.json').write_text(json.dumps(meta))
    return pdir


def _saved_meta(pid):
    p = Path(app_module.app.config['PROJECTS_DIR']) / pid / 'meta.json'
    return json.loads(p.read_text())


# ── Backend plumbing ─────────────────────────────────────────────────────────

class TestRetranscribeStoresAudioChannel:
    def test_specific_track_is_stored(self, client, tmp_path):
        src = tmp_path / 'multitrack.mxf'
        src.write_bytes(b'\x00')  # exists so the route's source check passes
        _make_project('p1', str(src))
        res = client.post('/project/p1/retranscribe',
                          json={'language': 'no', 'audio_channel': '1'})
        assert res.status_code == 200
        # _valid_audio_channel stores a valid index as its string form.
        assert _saved_meta('p1')['audio_channel'] == '1'

    def test_all_maps_to_mixed_sentinel(self, client, tmp_path):
        src = tmp_path / 'multitrack.mxf'
        src.write_bytes(b'\x00')
        _make_project('p2', str(src))
        res = client.post('/project/p2/retranscribe',
                          json={'language': 'no', 'audio_channel': 'all'})
        assert res.status_code == 200
        # 'all' is the stored "mix all tracks" sentinel.
        assert _saved_meta('p2')['audio_channel'] == 'all'

    def test_omitted_channel_leaves_existing(self, client, tmp_path):
        src = tmp_path / 'multitrack.mxf'
        src.write_bytes(b'\x00')
        _make_project('p3', str(src))
        res = client.post('/project/p3/retranscribe', json={'language': 'no'})
        assert res.status_code == 200
        # Untouched when not sent (existing 'all' preserved).
        assert _saved_meta('p3')['audio_channel'] == 'all'


# ── Frontend default guard (project.html) ────────────────────────────────────

class TestRetranscribeDefaultNotMixed:
    def test_template_defaults_to_single_primary(self):
        html = _PROJECT_HTML.read_text()
        assert 'function populateRetranscribeTracks' in html
        # Default is the single primary track ('0'), NOT 'all'.
        assert "const defaultVal = explicitIdx !== null ? explicitIdx : '0';" in html
        # The "all (mixed)" option is selected ONLY if defaultVal is 'all'
        # (which the line above never produces) — i.e. never mix-by-default.
        assert "value=\"all\"${defaultVal === 'all' ? ' selected' : ''}" in html
        # And the picker POSTs the chosen channel.
        assert 'body.audio_channel = audioChannel;' in html
