"""Import guardrail: warn before transcribing very long single files or
FCPXML imports whose source media was never synced in the NLE.

Both checks are advisory. The route answers 409 with the warnings and the
frontend re-posts with confirm_long_media to continue."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module  # noqa: E402


@pytest.fixture
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    with app_module.app.test_client() as c:
        yield c
    app_module._transcribe_jobs.clear()


def _make_project(pid, **extra):
    pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    pdir.mkdir(parents=True)
    meta = {'id': pid, 'name': 'P', 'status': 'uploaded'}
    meta.update(extra)
    (pdir / 'meta.json').write_text(json.dumps(meta))
    return pdir


class TestGuardRules:
    def test_english_under_eight_hours_passes(self, monkeypatch):
        monkeypatch.setattr(app_module, 'get_media_duration', lambda p: 7.9 * 3600)
        assert app_module._long_media_guard({}, '/x.wav', 'en') is None

    def test_english_over_eight_hours_warns(self, monkeypatch):
        monkeypatch.setattr(app_module, 'get_media_duration', lambda p: 9 * 3600 + 120)
        guard = app_module._long_media_guard({}, '/x.wav', 'en')
        assert guard['needs_long_media_confirm'] is True
        [w] = guard['warnings']
        assert w['kind'] == 'long_media' and w['threshold_hours'] == 8
        assert '9h 2m' in w['message']
        assert 'Collection' in w['message']
        assert 'Continue anyway' in guard['error']

    @pytest.mark.parametrize('language', ['no', 'de', 'auto'])
    def test_other_languages_use_four_hours(self, monkeypatch, language):
        monkeypatch.setattr(app_module, 'get_media_duration', lambda p: 4.5 * 3600)
        guard = app_module._long_media_guard({}, '/x.wav', language)
        assert guard and guard['warnings'][0]['threshold_hours'] == 4
        monkeypatch.setattr(app_module, 'get_media_duration', lambda p: 3.9 * 3600)
        assert app_module._long_media_guard({}, '/x.wav', language) is None

    def test_unknown_duration_never_blocks(self, monkeypatch):
        monkeypatch.setattr(app_module, 'get_media_duration', lambda p: None)
        assert app_module._long_media_guard({}, '/x.wav', 'en') is None

    def test_unsynced_fcpxml_sources_warn(self, monkeypatch):
        monkeypatch.setattr(app_module, 'get_media_duration', lambda p: 600.0)
        project = {'fcpxml_source': {'timeline_duration_seconds': 600.0,
                                     'source_audio_duration_seconds': 1500.0}}
        guard = app_module._long_media_guard(project, '/x.wav', 'en')
        [w] = guard['warnings']
        assert w['kind'] == 'unsynced_sources'
        assert 'inherits sync' in w['message'] and 'synced in the NLE first' in w['message']

    def test_synced_fcpxml_sources_pass(self, monkeypatch):
        monkeypatch.setattr(app_module, 'get_media_duration', lambda p: 600.0)
        project = {'fcpxml_source': {'timeline_duration_seconds': 600.0,
                                     'source_audio_duration_seconds': 1150.0}}
        assert app_module._long_media_guard(project, '/x.wav', 'en') is None

    def test_both_warnings_stack(self, monkeypatch):
        monkeypatch.setattr(app_module, 'get_media_duration', lambda p: 9 * 3600)
        project = {'fcpxml_source': {'timeline_duration_seconds': 9 * 3600,
                                     'source_audio_duration_seconds': 30 * 3600}}
        guard = app_module._long_media_guard(project, '/x.wav', 'en')
        assert [w['kind'] for w in guard['warnings']] == ['long_media', 'unsynced_sources']


class TestRoute:
    def _no_thread(self, monkeypatch):
        started = []

        class _T:
            daemon = False

            def __init__(self, *a, **k):
                started.append(k.get('target'))

            def start(self):
                pass

        monkeypatch.setattr(app_module.threading, 'Thread', _T)
        return started

    def test_route_answers_409_until_confirmed(self, client, monkeypatch, tmp_path):
        src = tmp_path / 'long.wav'
        src.write_bytes(b'RIFF' + b'\x00' * 64)
        _make_project('g1', language='en', source_path=str(src))
        monkeypatch.setattr(app_module, 'get_media_duration', lambda p: 10 * 3600)
        started = self._no_thread(monkeypatch)

        r = client.post('/project/g1/transcribe')
        assert r.status_code == 409
        body = r.get_json()
        assert body['needs_long_media_confirm'] is True
        assert body['warnings'][0]['kind'] == 'long_media'
        assert started == []  # nothing was launched

        r = client.post('/project/g1/transcribe', json={'confirm_long_media': True})
        assert r.status_code == 202
        assert started, 'the worker thread must start once confirmed'

    def test_route_passes_short_media_straight_through(self, client, monkeypatch, tmp_path):
        src = tmp_path / 'short.wav'
        src.write_bytes(b'RIFF' + b'\x00' * 64)
        _make_project('g2', language='en', source_path=str(src))
        monkeypatch.setattr(app_module, 'get_media_duration', lambda p: 1800.0)
        self._no_thread(monkeypatch)
        r = client.post('/project/g2/transcribe')
        assert r.status_code == 202


def test_summed_media_duration_dedupes_and_skips_unprobed(monkeypatch):
    from exporters import media_probe as mp
    durations = {'/a.mov': 3600.0, '/b.wav': 3500.0, '/c.mov': None}
    monkeypatch.setattr(mp, 'get_media_duration', lambda p: durations.get(p))
    assert mp.summed_media_duration(['/a.mov', '/a.mov', '/b.wav', '/c.mov', '']) == 7100.0
    assert mp.summed_media_duration(['/c.mov']) is None
    assert mp.summed_media_duration([]) is None
