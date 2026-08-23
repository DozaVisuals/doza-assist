"""Direct-channel free trial: the in-app conversion surfaces.

The direct download now ships the same trial the FxFactory build has: the
wrapper sets DOZA_TRIAL=1 while no license key is activated, transcripts
are capped (transcribe.py trial path), and the UI sells the unlock. These
tests pin the template layer for the direct channel specifically:

  - trial active + trial-length transcript -> "Free trial" banner with the
    doza.ai purchase link (DOZA_BUY_URL overrides it per channel)
  - licensed (env absent) + trial-length transcript -> "Transcribe Again",
    no purchase pitch
  - dashboard row carries the Trial-length badge
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


def _seed(pid, truncated=True):
    pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    pdir.mkdir(parents=True)
    transcript = {'language': 'en', 'segments': [
        {'start': 0.0, 'end': 2.0, 'text': 'hello', 'speaker': 'SPEAKER_00',
         'start_formatted': '00:00:00.000', 'end_formatted': '00:00:02.000',
         'words': []}]}
    if truncated:
        transcript['truncated_for_trial'] = True
        transcript['trial_max_seconds'] = 120
    json.dump({'id': pid, 'name': pid, 'status': 'transcribed',
               'source_path': f'/nonexistent/{pid}.mp4',
               'filename': f'{pid}.mp4',
               'created_at': '2026-08-21T12:00:00',
               'transcript': transcript},
              open(pdir / 'meta.json', 'w'))


def test_trial_active_sells_with_direct_store_link(client, monkeypatch):
    monkeypatch.setenv('DOZA_TRIAL', '1')
    monkeypatch.delenv('DOZA_BUY_URL', raising=False)
    _seed('t1')
    html = client.get('/project/t1').get_data(as_text=True)
    assert 'Free trial:' in html
    assert 'only the first 2 minutes' in html
    assert 'href="https://doza.ai/buy"' in html
    # Selling, not re-running: the Transcribe Again BUTTON must not render
    # (the banner copy merely promises it after purchase).
    assert 'onclick="showRetranscribe()">Transcribe Again' not in html


def test_buy_url_is_channel_parametrized(client, monkeypatch):
    monkeypatch.setenv('DOZA_TRIAL', '1')
    monkeypatch.setenv('DOZA_BUY_URL', 'https://fxfactory.com/buy/dozaassist')
    _seed('t2')
    html = client.get('/project/t2').get_data(as_text=True)
    assert 'href="https://fxfactory.com/buy/dozaassist"' in html
    assert 'doza.ai/buy' not in html


def test_licensed_offers_transcribe_again_without_pitch(client, monkeypatch):
    monkeypatch.delenv('DOZA_TRIAL', raising=False)
    _seed('t3')
    html = client.get('/project/t3').get_data(as_text=True)
    assert 'Trial-length transcript:' in html
    assert 'onclick="showRetranscribe()">Transcribe Again' in html
    assert 'Get Full Access' not in html


def test_full_transcript_shows_no_trial_ui(client, monkeypatch):
    monkeypatch.setenv('DOZA_TRIAL', '1')
    _seed('t4', truncated=False)
    html = client.get('/project/t4').get_data(as_text=True)
    assert 'Free trial:' not in html
    assert 'Trial-length' not in html


def test_dashboard_row_badge(client, monkeypatch):
    monkeypatch.setenv('DOZA_TRIAL', '1')
    _seed('t5')
    html = client.get('/').get_data(as_text=True)
    assert 'status-trial' in html
    assert 'Trial-length' in html


def test_dashboard_trial_pill_only_while_trial_active(client, monkeypatch):
    monkeypatch.setenv('DOZA_TRIAL', '1')
    monkeypatch.delenv('DOZA_BUY_URL', raising=False)
    assert 'header-trial-pill' in client.get('/').get_data(as_text=True)
    monkeypatch.delenv('DOZA_TRIAL', raising=False)
    assert 'header-trial-pill' not in client.get('/').get_data(as_text=True)
