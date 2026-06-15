"""Video-preview proxy route + per-track audio audition (1.0.19).

- /media/proxy builds a Chromium-playable mp4 on demand for browser-
  undecodable masters (MXF/AVC-Intra etc.), caches it per project, and
  reports building/done/error so the player can swap from the audio-only
  placeholder to real video. Exports always use the original file.
- /media/audio?track=N plays one source track on demand (the player's
  audition dropdown), independent of the transcribed track, via a per-track
  cache subdir so it never clobbers the transcribed audio.wav.

ffmpeg/extract are mocked — these pin the route/caching contract, not codecs.
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module
import transcribe


@pytest.fixture
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def _make_project(pid, source_path, audio_channel='all'):
    pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    pdir.mkdir(parents=True)
    json.dump({'id': pid, 'name': 'P', 'status': 'transcribed',
               'source_path': source_path, 'audio_channel': audio_channel},
              open(pdir / 'meta.json', 'w'))
    return pdir


def _wait_phase(client, pid, target, n=60):
    for _ in range(n):
        s = client.get(f'/project/{pid}/media/proxy/status').get_json()
        if s.get('phase') in target:
            return s
        time.sleep(0.05)
    return s


# ── Video proxy ──────────────────────────────────────────────────────────────

def test_proxy_builds_then_serves_cached(client, tmp_path, monkeypatch):
    src = tmp_path / 'master.mxf'
    src.write_bytes(b'\x00' * 4096)
    pdir = _make_project('p1', str(src))

    def fake_build(project_id, source_path, project_dir, audio_channel):
        out = app_module._proxy_path(project_dir)
        with open(out, 'wb') as f:
            f.write(b'FAKEMP4' * 500)
        st = os.stat(source_path)
        with open(app_module._proxy_meta_path(project_dir), 'w') as f:
            json.dump({'recipe': app_module._PROXY_RECIPE,
                       'source_size': st.st_size,
                       'source_mtime': int(st.st_mtime)}, f)
        with app_module._proxy_jobs_lock:
            app_module._proxy_jobs[project_id] = {'phase': 'done'}
    monkeypatch.setattr(app_module, '_build_preview_proxy', fake_build)

    r = client.get('/project/p1/media/proxy')
    assert r.status_code == 202
    assert r.get_json()['status'] == 'building'

    assert _wait_phase(client, 'p1', ('done',)).get('phase') == 'done'

    r2 = client.get('/project/p1/media/proxy')
    assert r2.status_code == 200
    assert r2.mimetype == 'video/mp4'
    assert r2.get_data() == b'FAKEMP4' * 500


def test_proxy_error_is_reported(client, tmp_path, monkeypatch):
    src = tmp_path / 'm.mxf'
    src.write_bytes(b'\x00' * 4096)
    _make_project('p2', str(src))

    def fail_build(project_id, *a, **k):
        with app_module._proxy_jobs_lock:
            app_module._proxy_jobs[project_id] = {'phase': 'error', 'error': 'boom'}
    monkeypatch.setattr(app_module, '_build_preview_proxy', fail_build)

    client.get('/project/p2/media/proxy')
    assert _wait_phase(client, 'p2', ('error', 'done')).get('phase') == 'error'


def test_proxy_stale_cache_rebuilds_on_source_change(client, tmp_path, monkeypatch):
    src = tmp_path / 'm.mxf'
    src.write_bytes(b'\x00' * 4096)
    pdir = _make_project('p4', str(src))
    # A proxy + meta recorded for a DIFFERENT (smaller) source size -> stale.
    with open(app_module._proxy_path(pdir), 'wb') as f:
        f.write(b'OLD' * 500)
    with open(app_module._proxy_meta_path(pdir), 'w') as f:
        json.dump({'recipe': app_module._PROXY_RECIPE, 'source_size': 1,
                   'source_mtime': 1}, f)
    monkeypatch.setattr(app_module, '_build_preview_proxy', lambda *a, **k: None)
    # Stale cache must NOT be served; route should kick a (no-op) build -> 202.
    r = client.get('/project/p4/media/proxy')
    assert r.status_code == 202


# ── Per-track audition ───────────────────────────────────────────────────────

def test_audition_serves_specific_track(client, tmp_path, monkeypatch):
    src = tmp_path / 'mt.mxf'
    src.write_bytes(b'\x00' * 4096)
    _make_project('p3', str(src), audio_channel='all')

    calls = {}
    def fake_extract(filepath, project_dir=None, audio_channel=None):
        calls['audio_channel'] = audio_channel
        calls['project_dir'] = project_dir
        out = os.path.join(project_dir, 'audio.wav')
        with open(out, 'wb') as f:
            f.write(b'TRACK1AUDIO' * 200)
        return out
    monkeypatch.setattr(transcribe, 'extract_audio', fake_extract)

    r = client.get('/project/p3/media/audio?track=1')
    assert r.status_code == 200
    assert calls.get('audio_channel') == '1', "must extract the auditioned track"
    assert '_audition_1' in calls.get('project_dir', ''), "track audition must use its own cache dir"
    assert r.get_data() == b'TRACK1AUDIO' * 200


def test_audition_does_not_touch_transcribed_wav(client, tmp_path, monkeypatch):
    # The transcribed audio.wav must be untouched by an audition request.
    src = tmp_path / 'mt.mxf'
    src.write_bytes(b'\x00' * 4096)
    pdir = _make_project('p5', str(src), audio_channel='0')
    transcribed = pdir / 'audio.wav'
    transcribed.write_bytes(b'TRANSCRIBED-TRACK-0' * 100)

    def fake_extract(filepath, project_dir=None, audio_channel=None):
        out = os.path.join(project_dir, 'audio.wav')
        with open(out, 'wb') as f:
            f.write(b'AUD' * 400)
        return out
    monkeypatch.setattr(transcribe, 'extract_audio', fake_extract)

    client.get('/project/p5/media/audio?track=2')
    # The project's transcribed audio.wav is unchanged (audition wrote to a subdir).
    assert transcribed.read_bytes() == b'TRANSCRIBED-TRACK-0' * 100
