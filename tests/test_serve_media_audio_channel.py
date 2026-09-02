"""Playback must serve the project's SELECTED audio track, not a stale mix.

Field bug (1.0.19 testing): a multi-track project showed meta.audio_channel='1'
but its cached audio.wav was the all-mix (sidecar channel=null). serve_media_audio
validated the cache WITHOUT the channel, so the all-mix passed as "valid" and
playback served every track at once while the transcript was a single track.

The fix passes the project's current channel to _cached_audio_valid, so a WAV
whose recorded channel differs from the selection is rejected and re-extracted
(self-healing). These tests pin both directions without needing a real
multi-track media file (extract_audio is spied).
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module
import transcribe
from transcribe import _EXTRACT_RECIPE


@pytest.fixture
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def _seed(pid, tmp_path, *, meta_channel, sidecar_channel):
    """A project whose audio.wav sidecar records `sidecar_channel` while the
    project meta selects `meta_channel`. Returns (project_dir, source_path)."""
    pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    pdir.mkdir(parents=True)
    src = tmp_path / f'{pid}-source.mxf'
    src.write_bytes(b'\x00' * 4096)  # stand-in source; extract_audio is spied
    st = src.stat()
    json.dump({'id': pid, 'name': 'P', 'status': 'transcribed',
               'source_path': str(src), 'audio_channel': meta_channel},
              open(pdir / 'meta.json', 'w'))
    wav = pdir / 'audio.wav'
    wav.write_bytes(b'MIXMIX' * 8000)  # >1024 bytes; this is the stale all-mix
    json.dump({'recipe': _EXTRACT_RECIPE, 'channel': sidecar_channel,
               'source_size': st.st_size, 'source_mtime': int(st.st_mtime),
               'wav_duration': 1.0},
              open(str(wav) + '.meta.json', 'w'))
    return pdir, src


def test_stale_mix_is_rejected_and_reextracted(client, tmp_path, monkeypatch):
    # The exact field state: meta selects track 2 ('1'), cached WAV is the
    # all-mix (sidecar channel=None).
    pdir, src = _seed('p1', tmp_path, meta_channel='1', sidecar_channel=None)

    calls = {}
    def _spy_extract(filepath, project_dir=None, audio_channel=None):
        calls['audio_channel'] = audio_channel
        out = os.path.join(project_dir, 'audio.wav')
        with open(out, 'wb') as f:
            f.write(b'TRACK2' * 8000)  # the re-extracted single track
        return out
    monkeypatch.setattr(transcribe, 'extract_audio', _spy_extract)

    resp = client.get('/project/p1/media/audio')
    assert resp.status_code == 200
    # Cache rejected (channel mismatch) -> re-extracted with the SELECTED track.
    assert calls.get('audio_channel') == '1', "must re-extract the selected track"
    assert resp.get_data() == b'TRACK2' * 8000, "must serve the re-extracted track, not the stale mix"


def test_matching_channel_serves_cache_without_reextract(client, tmp_path, monkeypatch):
    # Cached WAV already matches the selection (sidecar channel == 1) -> valid.
    _seed('p2', tmp_path, meta_channel='1', sidecar_channel=1)

    called = {'n': 0}
    def _spy_extract(*a, **k):
        called['n'] += 1
        raise AssertionError("must NOT re-extract when the cache matches")
    monkeypatch.setattr(transcribe, 'extract_audio', _spy_extract)

    resp = client.get('/project/p2/media/audio')
    assert resp.status_code == 200
    assert called['n'] == 0
    assert resp.get_data() == b'MIXMIX' * 8000  # served the existing cache


def test_all_selection_serves_mix_cache(client, tmp_path, monkeypatch):
    # Project genuinely on 'all' with an all-mix cache (sidecar None) -> valid,
    # no re-extract (the historical behaviour for mixed projects).
    _seed('p3', tmp_path, meta_channel='all', sidecar_channel=None)
    monkeypatch.setattr(transcribe, 'extract_audio',
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no re-extract for matching 'all'")))
    resp = client.get('/project/p3/media/audio')
    assert resp.status_code == 200
    assert resp.get_data() == b'MIXMIX' * 8000
