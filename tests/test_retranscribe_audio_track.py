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


# ── Empty-channel retranscribe recovery (don't strand the project) ───────────
#
# Retranscribing onto a silent/scratch track produced 0 segments -> status=
# 'error' with the transcript ALREADY destroyed, locking the user out. Now the
# route snapshots the working transcript and the worker rolls back to it.

class TestRetranscribeEmptyChannelRecovery:
    def _run_empty(self, monkeypatch, pid, with_backup):
        import transcribe as t
        monkeypatch.setattr(t, 'transcribe_file', lambda *a, **k: {'segments': []})
        if with_backup:
            with app_module._transcribe_jobs_lock:
                app_module._retranscribe_backups[pid] = {
                    'transcript': {'segments': [{'start': 0, 'end': 1, 'text': 'hi'}],
                                   'language': 'no'},
                    'analysis': None, 'client_selects': [], 'social_clips': [],
                    'detected_language': 'no', 'audio_channel': 'all',
                }
        # audio_channel=1 (0-based) -> the notice should say "Track 2"
        app_module._run_transcribe_job(pid, '/x.mxf', None, 'no', 'Int', 'Subj',
                                       audio_channel=1)

    def test_empty_retranscribe_restores_prior_transcript(self, client, tmp_path, monkeypatch):
        _make_project('r1', str(tmp_path / 'm.mxf'))
        self._run_empty(monkeypatch, 'r1', with_backup=True)
        meta = _saved_meta('r1')
        assert meta['status'] == 'transcribed'
        assert meta.get('transcript') and meta['transcript']['segments']
        assert meta.get('audio_channel') == 'all'  # rolled back to the working channel
        assert 'r1' not in app_module._retranscribe_backups  # snapshot consumed
        st = app_module._transcribe_jobs.get('r1', {})
        assert st.get('phase') == 'done'  # frontend reloads into the restored transcript
        assert 'Track 2' in (st.get('notice') or '')

    def test_empty_fresh_transcribe_still_errors(self, client, tmp_path, monkeypatch):
        _make_project('r2', str(tmp_path / 'm.mxf'))
        self._run_empty(monkeypatch, 'r2', with_backup=False)
        meta = _saved_meta('r2')
        assert meta['status'] == 'error'
        err = (meta.get('error') or '').lower()
        # An empty/silent fresh transcribe surfaces an actionable "no usable
        # speech / silent track" error (wording broadened when the guard grew
        # to also catch silent audio + hallucinated transcripts).
        assert 'speech' in err and 'silent' in err

    def test_engine_error_drops_backup(self, client, tmp_path, monkeypatch):
        # If the engine raises (not an empty result), the rollback snapshot
        # must NOT leak — else a later genuinely-empty run restores it stale.
        _make_project('r3', str(tmp_path / 'm.mxf'))
        with app_module._transcribe_jobs_lock:
            app_module._retranscribe_backups['r3'] = {
                'transcript': {'segments': [{'start': 0, 'end': 1, 'text': 'hi'}]},
            }
        import transcribe as t
        def boom(*a, **k):
            raise RuntimeError("engine crash")
        monkeypatch.setattr(t, 'transcribe_file', boom)
        app_module._run_transcribe_job('r3', '/x.mxf', None, 'no', 'Int', 'Subj',
                                       audio_channel=1)
        assert 'r3' not in app_module._retranscribe_backups  # finally popped it
        assert _saved_meta('r3')['status'] == 'error'


# ── Silent / degenerate (hallucinated) transcript guards ─────────────────────
#
# A silent / speech-empty extract (e.g. a camera proxy whose real mic is only
# on the master) does NOT yield zero segments — Whisper hallucinates a repeated
# subtitle-credit line. The old seg_count==0 guard missed it, stored garbage as
# 'transcribed', then a pyannote pass reported "no segments". The guard now also
# catches a degenerate (near-all-identical) transcript AND a measurably silent
# audio.wav, routing both through the same rollback-or-error path.

import wave as _wave
import struct as _struct


def _write_wav(path, samples, rate=16000):
    with _wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(_struct.pack('<%dh' % len(samples), *samples))


def _degenerate_result():
    # 30 identical "Teksting av …" lines — the reported hallucination shape.
    return {'segments': [
        {'start': i * 30, 'end': i * 30 + 30, 'text': 'Teksting av Nicolai Winther'}
        for i in range(30)
    ]}


class TestUnusableTranscriptGuards:
    def test_wav_peak_dbfs_silent_vs_loud(self, tmp_path):
        from transcribe import wav_peak_dbfs
        sil = tmp_path / 'sil.wav'
        _write_wav(sil, [0] * 16000)
        loud = tmp_path / 'loud.wav'
        _write_wav(loud, [16000, -16000] * 8000)
        assert wav_peak_dbfs(str(sil)) == float('-inf')
        assert wav_peak_dbfs(str(loud)) > -10.0

    def test_degenerate_transcript_fresh_errors(self, client, tmp_path, monkeypatch):
        import transcribe as t
        monkeypatch.setattr(t, 'transcribe_file', lambda *a, **k: _degenerate_result())
        _make_project('d1', str(tmp_path / 'm.mxf'))
        app_module._run_transcribe_job('d1', '/x.mxf', None, 'no', 'Int', 'Subj',
                                       audio_channel=0)
        meta = _saved_meta('d1')
        assert meta['status'] == 'error'
        err = (meta.get('error') or '').lower()
        assert 'speech' in err and 'silent' in err
        # The hallucinated transcript must NOT be saved.
        assert not (meta.get('transcript') or {}).get('segments')

    def test_degenerate_transcript_retranscribe_rolls_back(self, client, tmp_path, monkeypatch):
        import transcribe as t
        monkeypatch.setattr(t, 'transcribe_file', lambda *a, **k: _degenerate_result())
        _make_project('d2', str(tmp_path / 'm.mxf'))
        with app_module._transcribe_jobs_lock:
            app_module._retranscribe_backups['d2'] = {
                'transcript': {'segments': [{'start': 0, 'end': 1, 'text': 'real speech'}],
                               'language': 'no'},
                'analysis': None, 'client_selects': [], 'social_clips': [],
                'detected_language': 'no', 'audio_channel': 'all',
            }
        app_module._run_transcribe_job('d2', '/x.mxf', None, 'no', 'Int', 'Subj',
                                       audio_channel=1)
        meta = _saved_meta('d2')
        # Previous good transcript preserved, not the hallucination.
        assert meta['status'] == 'transcribed'
        assert meta['transcript']['segments'][0]['text'] == 'real speech'

    def test_silent_audio_with_distinct_text_errors(self, client, tmp_path, monkeypatch):
        # Whisper returned plausible DISTINCT text, but the extracted audio.wav
        # is silent -> still rejected by the peak gate (the degenerate check
        # alone would pass it).
        import transcribe as t
        pdir = _make_project('s1', str(tmp_path / 'm.mxf'))
        _write_wav(pdir / 'audio.wav', [0] * 16000)  # silent extract
        distinct = {'segments': [
            {'start': i, 'end': i + 1, 'text': f'line number {i}'} for i in range(8)
        ]}
        monkeypatch.setattr(t, 'transcribe_file', lambda *a, **k: distinct)
        app_module._run_transcribe_job('s1', '/x.mxf', None, 'no', 'Int', 'Subj',
                                       audio_channel=0)
        meta = _saved_meta('s1')
        assert meta['status'] == 'error'
        assert 'silent' in (meta.get('error') or '').lower()
