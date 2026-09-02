"""Regression tests for the v3.5.12 camera-media port (+ review fixes).

Covers the gaps the adversarial review flagged as unguarded:
  - the engine→status-writer progress chain (the dispatch blocker: engines
    receive the RAW event-dict callback, not the milestone helper)
  - the transcribe-status truth gate (backend-restart orphans)
  - _cached_audio_valid recipe/sidecar semantics
  - zero-segment honesty in the transcribe worker
  - _diarization_state status-file branch + stale-claim demotion
  - the rich TC probe (zero-vs-None, dual-tag prefer-nonzero, byte-compat
    export wrapper)
  - start_tc lazy-probe persistence shape
  - typed Ollama errors (insufficient_memory) + chunk-loop fast-abort /
    all-failed promotion with code
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module
import transcribe as transcribe_module
from transcribe import _EXTRACT_RECIPE, _audio_meta_path, _cached_audio_valid


@pytest.fixture
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def _make_project(pid, **extra):
    pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    pdir.mkdir(parents=True)
    meta = {'id': pid, 'name': 'P', 'status': 'uploaded'}
    meta.update(extra)
    (pdir / 'meta.json').write_text(json.dumps(meta))
    return pdir


# ── Progress dispatch (the review blocker) ─────────────────────────────────

class TestProgressDispatch:
    def test_engine_event_dicts_reach_the_callback(self, monkeypatch, tmp_path):
        """transcribe_file must hand engines the RAW event-dict callback.
        Passing the milestone helper raised a swallowed TypeError on every
        engine event — the bar parked at 5% for entire runs."""
        events = []

        def _fake_parakeet(audio_path, speaker_labels, progress_cb=None):
            # Engines call the callback with a complete EVENT DICT.
            progress_cb({'phase': 'transcribing', 'pct': 42,
                         'engine': 'parakeet-mlx', 'audio_sec': 600})
            return {'segments': [{'start': 0, 'end': 1, 'text': 'hi',
                                  'speaker': 'A'}],
                    'language': 'en', 'duration': 1, 'engine': 'parakeet-mlx'}

        monkeypatch.setattr(transcribe_module, '_transcribe_parakeet', _fake_parakeet)
        monkeypatch.setattr(transcribe_module, 'extract_audio',
                            lambda f, project_dir=None, **kw: str(tmp_path / 'a.wav'))
        result = transcribe_module.transcribe_file(
            str(tmp_path / 'src.mov'), project_dir=str(tmp_path),
            language='en', progress_cb=events.append,
        )
        assert result['segments']
        transcribing = [e for e in events
                        if e.get('phase') == 'transcribing' and e.get('pct') == 42]
        assert transcribing, f'engine event never reached the callback: {events}'
        assert transcribing[0]['audio_sec'] == 600


# ── Transcribe-status truth gate ───────────────────────────────────────────

class TestTranscribeTruthGate:
    def test_orphaned_disk_snapshot_becomes_recoverable_error(self, client):
        pid = 'orphan1'
        _make_project(pid, status='transcribing')
        stale_ts = (datetime.now() - timedelta(seconds=300)).isoformat()
        app_module._write_transcribe_status(pid, {
            'started_at': stale_ts, 'updated_at': stale_ts,
            'phase': 'transcribing', 'pct': 37,
        })
        body = client.get(f'/project/{pid}/transcribe/status').get_json()
        state = body['state']
        assert state['phase'] == 'error'
        assert state.get('recoverable') is True
        assert 'restarted' in state['error']
        # The project is re-armed so the transcribe card renders.
        meta = json.loads((Path(app_module.app.config['PROJECTS_DIR']) / pid / 'meta.json').read_text())
        assert meta['status'] == 'uploaded'

    def test_fresh_disk_snapshot_is_left_alone(self, client):
        pid = 'orphan2'
        _make_project(pid, status='transcribing')
        now = datetime.now().isoformat()
        app_module._write_transcribe_status(pid, {
            'started_at': now, 'updated_at': now,
            'phase': 'transcribing', 'pct': 12,
        })
        state = client.get(f'/project/{pid}/transcribe/status').get_json()['state']
        # Within the 120s grace window: do not declare it dead.
        assert state['phase'] == 'transcribing'


# ── Audio cache validation ─────────────────────────────────────────────────

class TestCachedAudioValid:
    def _wav(self, tmp_path, seconds=4.0):
        wav = tmp_path / 'audio.wav'
        wav.write_bytes(b'RIFF' + b'\x00' * int(seconds * 32000))
        return wav

    def _source(self, tmp_path):
        src = tmp_path / 'src.mov'
        src.write_bytes(b'fake camera bytes')
        return src

    def test_no_sidecar_invalid(self, tmp_path):
        wav, src = self._wav(tmp_path), self._source(tmp_path)
        assert _cached_audio_valid(str(wav), str(src)) is False

    def test_current_recipe_sidecar_valid(self, tmp_path):
        wav, src = self._wav(tmp_path), self._source(tmp_path)
        st = os.stat(src)
        Path(_audio_meta_path(str(wav))).write_text(json.dumps({
            'recipe': _EXTRACT_RECIPE, 'source_size': st.st_size,
            'source_mtime': int(st.st_mtime), 'wav_duration': 4.0,
        }))
        assert _cached_audio_valid(str(wav), str(src)) is True

    def test_old_recipe_invalid(self, tmp_path):
        wav, src = self._wav(tmp_path), self._source(tmp_path)
        st = os.stat(src)
        Path(_audio_meta_path(str(wav))).write_text(json.dumps({
            'recipe': 1, 'source_size': st.st_size,
            'source_mtime': int(st.st_mtime), 'wav_duration': 4.0,
        }))
        assert _cached_audio_valid(str(wav), str(src)) is False

    def test_truncated_since_write_invalid(self, tmp_path):
        wav, src = self._wav(tmp_path, seconds=1.0), self._source(tmp_path)
        st = os.stat(src)
        Path(_audio_meta_path(str(wav))).write_text(json.dumps({
            'recipe': _EXTRACT_RECIPE, 'source_size': st.st_size,
            'source_mtime': int(st.st_mtime), 'wav_duration': 4.0,
        }))
        assert _cached_audio_valid(str(wav), str(src)) is False


# ── extract_audio same-path guard ──────────────────────────────────────────

def _write_16k_mono_wav(path, seconds=2.0):
    import struct
    import wave
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setframerate(16000)
        wf.setsampwidth(2)
        n = int(seconds * 16000)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))


class TestSamePathGuard:
    def test_extract_audio_does_not_delete_its_own_input(self, tmp_path, monkeypatch):
        """My Style import: transcribe_file re-enters extract_audio with the
        already-extracted WAV as filepath. The stale sidecar (recorded from
        the ORIGINAL source) must not trigger the self-heal delete."""
        monkeypatch.delenv("DOZA_TRIAL", raising=False)
        audio = tmp_path / "audio.wav"
        _write_16k_mono_wav(audio)
        # Sidecar describing a DIFFERENT (original) source — guaranteed
        # mismatch if validated against the WAV itself.
        (tmp_path / "audio.wav.meta.json").write_text(json.dumps({
            "recipe": 999, "source_size": 1, "source_mtime": 1,
            "wav_duration": 999.0,
        }))
        result = transcribe_module.extract_audio(str(audio), project_dir=str(tmp_path))
        assert result == str(audio)
        assert audio.exists(), "extract_audio deleted its own input"

    def test_distinct_source_still_self_heals(self, tmp_path, monkeypatch):
        """The guard must not weaken the normal self-heal: a stale cache for
        a real (different) source is still discarded."""
        monkeypatch.delenv("DOZA_TRIAL", raising=False)
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 64)   # poisoned tiny cache
        (tmp_path / "audio.wav.meta.json").write_text(json.dumps({
            "recipe": 1, "source_size": 1, "source_mtime": 1,
            "wav_duration": 999.0,
        }))
        src = tmp_path / "source.wav"
        _write_16k_mono_wav(src, seconds=3.0)
        result = transcribe_module.extract_audio(str(src), project_dir=str(tmp_path))
        # Source is already 16k mono s16 → passthrough returns it, but the
        # poisoned cache must be GONE so nothing downstream can pick it up.
        assert result == str(src)
        assert not audio.exists()


# ── Zero-segment honesty (transcribe worker) ───────────────────────────────

class TestZeroSegmentWorker:
    def test_empty_transcript_is_error_and_skips_downstream(self, client, tmp_path, monkeypatch):
        pid = 'zeroseg'
        src = tmp_path / 'src.mov'
        src.write_bytes(b'x')
        pdir = _make_project(pid, source_path=str(src), language='en')
        monkeypatch.setattr('transcribe.transcribe_file',
                            lambda *a, **k: {'segments': [], 'duration': 0})
        monkeypatch.setattr(app_module, '_engine_available', lambda n: True)
        app_module._run_transcribe_job(pid, str(src), 2, 'en', 'I', 'S')
        meta = json.loads((pdir / 'meta.json').read_text())
        assert meta['status'] == 'error'
        # Empty/silent transcript surfaces an actionable "no usable speech /
        # silent track" error (wording broadened when the zero-segment guard
        # grew to also catch silent audio + hallucinated transcripts).
        assert 'speech' in meta['error'].lower() and 'silent' in meta['error'].lower()
        assert 'transcript' not in meta
        # No paragraph index was built from the empty result.
        assert not (pdir / 'paragraph_index.json').exists()
        # Terminal status file carries BOTH error keys.
        st = json.loads((pdir / 'transcribe_status.json').read_text())
        assert st['phase'] == 'error'
        assert st['error'] == st['message']


# ── _diarization_state ─────────────────────────────────────────────────────

class TestDiarizationState:
    def test_meta_done_short_circuits(self, client):
        pid = 'd1'
        _make_project(pid, diarization={'status': 'done'})
        proj = app_module.get_project(pid)
        assert app_module._diarization_state(pid, proj) == 'done'

    def test_status_file_done_detected(self, client, monkeypatch):
        pid = 'd2'
        pdir = _make_project(pid)
        # Fake the extension's read_status via a stub module.
        import types
        stub_worker = types.ModuleType('diarization.worker')
        stub_worker.read_status = lambda p: {'status': 'done',
                                             'updated_at': datetime.now().isoformat()}
        stub_pkg = types.ModuleType('diarization')
        stub_pkg.worker = stub_worker
        monkeypatch.setitem(sys.modules, 'diarization', stub_pkg)
        monkeypatch.setitem(sys.modules, 'diarization.worker', stub_worker)
        assert app_module._diarization_state(pid, app_module.get_project(pid)) == 'done'

    def test_stale_running_demoted_fresh_queued_kept(self, client, monkeypatch):
        import types
        stale = (datetime.now() - timedelta(seconds=1200)).isoformat()
        cases = [({'status': 'running', 'updated_at': stale}, 'not_started'),
                 ({'status': 'queued', 'updated_at': stale}, 'queued'),
                 ({'status': 'running',
                   'updated_at': datetime.now().isoformat()}, 'running')]
        for st, expected in cases:
            stub_worker = types.ModuleType('diarization.worker')
            stub_worker.read_status = lambda p, _st=st: dict(_st)
            stub_pkg = types.ModuleType('diarization')
            stub_pkg.worker = stub_worker
            monkeypatch.setitem(sys.modules, 'diarization', stub_pkg)
            monkeypatch.setitem(sys.modules, 'diarization.worker', stub_worker)
            pid = f'd3{expected}'
            _make_project(pid)
            assert app_module._diarization_state(pid, app_module.get_project(pid)) == expected

    def test_no_extension_is_unavailable(self, client, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def _no_diar(name, *a, **k):
            if name.startswith('diarization'):
                raise ImportError(name)
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, '__import__', _no_diar)
        monkeypatch.delitem(sys.modules, 'diarization', raising=False)
        monkeypatch.delitem(sys.modules, 'diarization.worker', raising=False)
        pid = 'd4'
        _make_project(pid)
        assert app_module._diarization_state(pid, app_module.get_project(pid)) == 'unavailable'


# ── Rich TC probe ──────────────────────────────────────────────────────────

class TestTcProbe:
    def _probe_with(self, monkeypatch, tmp_path, payload):
        from exporters import media_probe

        class _Result:
            returncode = 0
            stdout = json.dumps(payload)

        f = tmp_path / 'x.mov'
        f.write_bytes(b'x')
        monkeypatch.setattr(media_probe.subprocess, 'run', lambda *a, **k: _Result())
        return media_probe, str(f)

    def test_dual_tag_prefers_nonzero(self, monkeypatch, tmp_path):
        mp, f = self._probe_with(monkeypatch, tmp_path, {
            'streams': [{'codec_type': 'data', 'codec_tag_string': 'tmcd',
                         'tags': {'timecode': '00:00:00:00'}}],
            'format': {'tags': {'timecode': '14:23:11:05'}},
        })
        tc = mp.get_video_start_timecode(f, 25.0)
        assert tc['raw'] == '14:23:11:05' and tc['frames'] > 0

    def test_zero_tc_reported_not_absent(self, monkeypatch, tmp_path):
        mp, f = self._probe_with(monkeypatch, tmp_path, {
            'streams': [{'codec_type': 'data', 'codec_tag_string': 'tmcd',
                         'tags': {'timecode': '00:00:00;00'}}],
            'format': {'tags': {}},
        })
        tc = mp.get_video_start_timecode(f, 29.97)
        assert tc == {'frames': 0, 'drop': True, 'raw': '00:00:00;00', 'source': 'tmcd'}
        # Export wrapper stays byte-compatible: zero frames → NDF.
        assert mp.get_video_start_timecode_info(f, 29.97) == (0, 'NDF')

    def test_no_tmcd_tag_only_returns_none(self, monkeypatch, tmp_path):
        # Sony XAVC-S: tag without a tmcd stream must NOT be honored.
        mp, f = self._probe_with(monkeypatch, tmp_path, {
            'streams': [{'codec_type': 'video', 'codec_tag_string': 'avc1'}],
            'format': {'tags': {'timecode': '14:23:11:05'}},
        })
        assert mp.get_video_start_timecode(f, 25.0) is None

    def test_df_label_round_trip(self):
        from exporters.media_probe import timecode_to_frames, frames_to_timecode_label
        for tc in ['14:23:11;05', '00:10:00;00', '01:00:00;00']:
            frames = timecode_to_frames(tc, 29.97)
            assert frames_to_timecode_label(frames, 29.97, drop=True) == tc


# ── start_tc lazy-probe persistence ────────────────────────────────────────

class TestStartTcPersistence:
    def test_shape_and_probe_once(self, client, tmp_path, monkeypatch):
        pid = 'tcp1'
        src = tmp_path / 'cam.mov'
        src.write_bytes(b'x')
        pdir = _make_project(pid, status='transcribed', source_path=str(src),
                             transcript={'segments': [
                                 {'start': 0, 'end': 5, 'text': 'a', 'speaker': 'X',
                                  'start_formatted': '00:00:00.000',
                                  'end_formatted': '00:00:05.000'}]})
        calls = {'n': 0}

        def _probe(path, fps):
            calls['n'] += 1
            return {'frames': 1295280, 'drop': False,
                    'raw': '14:23:11:05', 'source': 'tmcd'}

        monkeypatch.setattr(app_module, 'get_video_start_timecode', _probe)
        monkeypatch.setattr(app_module, 'get_video_framerate', lambda p: 25.0)
        assert client.get(f'/project/{pid}').status_code == 200
        meta = json.loads((pdir / 'meta.json').read_text())
        assert meta['start_tc'] == {'frames': 1295280, 'fps': 25.0,
                                    'drop': False, 'raw': '14:23:11:05'}
        assert 'source_exists' not in meta
        assert client.get(f'/project/{pid}').status_code == 200
        assert calls['n'] == 1


# ── Typed Ollama errors + chunk loop ───────────────────────────────────────

class TestTypedAnalysisErrors:
    def test_insufficient_memory_code(self):
        from ai_providers import ProviderError
        from ai_providers.ollama_provider import _raise_ollama_error

        class _R:
            status_code = 500
            text = ''
            def json(self):
                return {'error': 'model requires more system memory (6.1 GiB) '
                                 'than is available (4.2 GiB)'}

        with pytest.raises(ProviderError) as exc:
            _raise_ollama_error(_R(), 'gemma4:e4b')
        assert exc.value.code == 'insufficient_memory'
        assert 'gemma4:e2b' in str(exc.value)

    def _long_transcript(self, minutes=20):
        def _fmt(sec):
            return f'{int(sec) // 3600:02d}:{int(sec) % 3600 // 60:02d}:{int(sec) % 60:02d}.000'
        segs = [{'start': i * 10.0, 'end': i * 10.0 + 9.0,
                 'start_formatted': _fmt(i * 10.0),
                 'end_formatted': _fmt(i * 10.0 + 9.0),
                 'text': f's{i}', 'speaker': 'A'} for i in range(minutes * 6)]
        return {'segments': segs, 'duration': segs[-1]['end']}

    def test_fatal_code_aborts_first_call(self, monkeypatch):
        import ai_analysis
        from ai_providers import ProviderError
        calls = {'n': 0}

        def _oom(*a, **k):
            calls['n'] += 1
            raise ProviderError('needs more memory', code='insufficient_memory')

        monkeypatch.setattr(ai_analysis, '_analyze_story', _oom)
        monkeypatch.setattr(ai_analysis, '_analyze_social', _oom)
        with pytest.raises(ProviderError, match='memory'):
            ai_analysis.analyze_transcript(self._long_transcript(), analysis_type='all')
        assert calls['n'] == 1

    def test_all_failed_promotes_message_and_code(self, monkeypatch):
        import ai_analysis
        from ai_providers import ProviderError

        def _boom(*a, **k):
            raise ProviderError('Ollama error (HTTP 500): model exploded',
                                code='server_error')

        monkeypatch.setattr(ai_analysis, '_analyze_story', _boom)
        monkeypatch.setattr(ai_analysis, '_analyze_social', _boom)
        with pytest.raises(ProviderError, match='model exploded') as exc:
            ai_analysis.analyze_transcript(self._long_transcript(), analysis_type='all')
        assert exc.value.code == 'server_error'
