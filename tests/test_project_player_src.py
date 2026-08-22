"""Audio-only playback source must be drift-safe per source format.

Field bug (2026-08-21, 2.5-hr German interview MP3): transcript timestamps
are decode-exact (the engines run on the ffmpeg-extracted WAV) but the
<audio> element played the ORIGINAL VBR MP3, which Chromium seeks by
byte-estimate — click-to-play drifted tens of seconds deep into long files.

mp3-family (estimate-seeked VBR/ADTS) projects must point the player at
/media/audio (the extracted WAV). Sample-table formats (m4a) and PCM (wav)
keep the original /media source for full quality. The same gate must cover
the multi-project switch map (ESTIMATE_SEEKED_AUDIO) so cross-project jumps
don't reintroduce the drift.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module


@pytest.fixture
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def _seed(pid, ext):
    pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    pdir.mkdir(parents=True)
    json.dump({
        'id': pid, 'name': pid, 'status': 'transcribed',
        # Nonexistent on purpose: skips the start-TC / framerate ffprobe paths.
        'source_path': f'/nonexistent/{pid}-source{ext}',
        'transcript': {'language': 'de', 'segments': [
            {'start': 0.0, 'end': 2.0, 'text': 'hallo', 'speaker': 'SPEAKER_00',
             'start_formatted': '00:00:00.000', 'end_formatted': '00:00:02.000',
             'words': []}]},
    }, open(pdir / 'meta.json', 'w'))


def _page(client, pid):
    resp = client.get(f'/project/{pid}')
    assert resp.status_code == 200
    return resp.get_data(as_text=True)


def test_mp3_project_plays_extracted_wav(client):
    _seed('pmp3', '.mp3')
    html = _page(client, 'pmp3')
    assert 'src="/project/pmp3/media/audio"' in html


def test_aac_project_plays_extracted_wav(client):
    _seed('paac', '.aac')
    html = _page(client, 'paac')
    assert 'src="/project/paac/media/audio"' in html


def test_wav_project_keeps_original_source(client):
    _seed('pwav', '.wav')
    html = _page(client, 'pwav')
    assert 'src="/project/pwav/media"' in html
    assert 'src="/project/pwav/media/audio"' not in html


def test_m4a_project_keeps_original_source(client):
    # MP4 containers carry sample tables — seeking is already exact.
    _seed('pm4a', '.m4a')
    html = _page(client, 'pm4a')
    assert 'src="/project/pm4a/media"' in html
    assert 'src="/project/pm4a/media/audio"' not in html


def test_switch_map_marks_estimate_seeked_projects(client):
    _seed('pmp3', '.mp3')
    _seed('pwav', '.wav')
    html = _page(client, 'pmp3,pwav')
    assert '"pmp3": true' in html
    assert '"pwav": false' in html
    # Primary project is the mp3 — the preset element src follows the gate too.
    assert 'src="/project/pmp3/media/audio"' in html
