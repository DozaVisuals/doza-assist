"""Regression tests for the issue #39 camera-media fix set (v3.5.12).

The reporter's profile: MXF (AVC) + MP4 camera originals, multi-stream
audio, embedded source timecode, two-speaker interviews, Whisper-on-CPU.
Each test class maps to one of the six audit fixes:

  F1  extract_audio hardening (poisoned-cache self-heal, multi-stream
      mixdown, empty-audio guard) + zero-segment honesty
  F3  real Whisper progress (tqdm shim accounting, status plumbing)
  F4  speaker reassignment (start-containment range matching)
  F5  analysis error surfacing (all-chunks-failed promotion, fast-abort,
      Ollama preflight)
  F6  source-timecode probe + label round-trips

Media fixtures are fabricated with ffmpeg lavfi — no camera files needed.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import transcribe as transcribe_module
from transcribe import extract_audio, _EXTRACT_RECIPE, _audio_meta_path

FFMPEG = shutil.which('ffmpeg') or (
    '/opt/homebrew/bin/ffmpeg' if os.path.isfile('/opt/homebrew/bin/ffmpeg') else None)
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason='ffmpeg not installed')


def _run_ffmpeg(args):
    result = subprocess.run([FFMPEG, '-y', *args], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-500:]


@pytest.fixture(scope='module')
def fixtures(tmp_path_factory):
    """Fabricated camera-style media (module-scoped — built once)."""
    if FFMPEG is None:
        pytest.skip('ffmpeg not installed')
    d = tmp_path_factory.mktemp('issue39media')

    two_stream = d / 'two_stream.mov'
    _run_ffmpeg([
        '-f', 'lavfi', '-i', 'anullsrc=channel_layout=mono:sample_rate=48000',
        '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000',
        '-t', '5', '-map', '0:a', '-map', '1:a', '-c:a', 'pcm_s16le',
        str(two_stream),
    ])

    no_audio = d / 'noaudio.mp4'
    _run_ffmpeg([
        '-f', 'lavfi', '-i', 'testsrc2=size=320x180:rate=25',
        '-t', '2', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        str(no_audio),
    ])

    tc_ndf = d / 'tc_ndf.mov'
    _run_ffmpeg([
        '-f', 'lavfi', '-i', 'testsrc2=size=320x180:rate=25',
        '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000',
        '-t', '2', '-timecode', '14:23:11:05',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'pcm_s16le',
        str(tc_ndf),
    ])

    plain = d / 'plain.mov'
    _run_ffmpeg([
        '-f', 'lavfi', '-i', 'sine=frequency=330:sample_rate=48000',
        '-t', '4', '-c:a', 'pcm_s16le', str(plain),
    ])

    return d


def _mean_volume_db(wav_path):
    result = subprocess.run(
        [FFMPEG, '-i', str(wav_path), '-af', 'volumedetect', '-f', 'null', '-'],
        capture_output=True, text=True,
    )
    for line in result.stderr.splitlines():
        if 'mean_volume' in line:
            return float(line.split('mean_volume:')[1].split('dB')[0].strip())
    return None


# ── F1: extract_audio hardening ────────────────────────────────────────────

@needs_ffmpeg
class TestExtractAudioSelfHeal:
    HEADER_ONLY_WAV = (
        b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00'
        b'\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
    )

    def test_poisoned_cache_without_sidecar_is_reextracted(self, fixtures, tmp_path):
        # A truncated WAV from <=3.5.11 has no sidecar — must be discarded.
        (tmp_path / 'audio.wav').write_bytes(self.HEADER_ONLY_WAV)
        out = extract_audio(str(fixtures / 'plain.mov'), project_dir=str(tmp_path))
        assert os.path.getsize(out) > 100_000  # 4s of 16kHz mono s16
        meta = json.loads(Path(_audio_meta_path(out)).read_text())
        assert meta['recipe'] == _EXTRACT_RECIPE

    def test_old_recipe_sidecar_invalidates_cache(self, fixtures, tmp_path):
        src = str(fixtures / 'plain.mov')
        out = extract_audio(src, project_dir=str(tmp_path))
        first_mtime = os.path.getmtime(out)
        # Rewrite the sidecar as recipe 1 — must trigger re-extraction.
        meta = json.loads(Path(_audio_meta_path(out)).read_text())
        meta['recipe'] = 1
        Path(_audio_meta_path(out)).write_text(json.dumps(meta))
        os.utime(out, (first_mtime - 100, first_mtime - 100))
        out2 = extract_audio(src, project_dir=str(tmp_path))
        assert os.path.getmtime(out2) != first_mtime - 100

    def test_valid_cache_is_reused(self, fixtures, tmp_path):
        src = str(fixtures / 'plain.mov')
        out = extract_audio(src, project_dir=str(tmp_path))
        stamp = os.path.getmtime(out) - 100
        os.utime(out, (stamp, stamp))
        out2 = extract_audio(src, project_dir=str(tmp_path))
        assert os.path.getmtime(out2) == stamp  # untouched = cache hit

    def test_multistream_mixdown_recovers_offchannel_mic(self, fixtures, tmp_path):
        # Stream 1 is digital silence (dead CH1), stream 2 carries the tone
        # ("the lav mic"). ffmpeg's default single-stream pick produced pure
        # silence here; the amix mixdown must keep the tone audible.
        out = extract_audio(str(fixtures / 'two_stream.mov'), project_dir=str(tmp_path))
        vol = _mean_volume_db(out)
        assert vol is not None and vol > -60.0, f'mixdown is silent ({vol} dB)'

    def test_empty_audio_source_raises_clear_error(self, fixtures, tmp_path):
        with pytest.raises(RuntimeError, match='no usable audio'):
            extract_audio(str(fixtures / 'noaudio.mp4'), project_dir=str(tmp_path))
        assert not (tmp_path / 'audio.wav').exists()


# ── F3: Whisper progress plumbing ──────────────────────────────────────────

class TestWhisperProgressShim:
    def test_production_shim_reports_monotonic_fractions(self):
        # Exercises the PRODUCTION class installed by the hook — not a
        # copy — so shim accounting can't regress behind a green suite.
        pytest.importorskip('whisper')
        assert transcribe_module._install_whisper_progress_hook() is True
        import importlib
        wt = importlib.import_module('whisper.transcribe')

        seen = []
        token = transcribe_module._whisper_progress_cb.set(seen.append)
        try:
            with wt.tqdm.tqdm(total=100, disable=True) as bar:
                for _ in range(4):
                    bar.update(30)
        finally:
            transcribe_module._whisper_progress_cb.reset(token)

        assert seen == [0.3, 0.6, 0.9, 1.0]

    def test_production_shim_silent_without_contextvar(self):
        pytest.importorskip('whisper')
        assert transcribe_module._install_whisper_progress_hook() is True
        import importlib
        wt = importlib.import_module('whisper.transcribe')
        # No callback set: updates must be a no-op, never an error.
        with wt.tqdm.tqdm(total=10, disable=True) as bar:
            bar.update(5)

    def test_hook_installs_against_real_whisper(self):
        pytest.importorskip('whisper')
        assert transcribe_module._install_whisper_progress_hook() is True
        # NB: must use import_module — whisper/__init__ rebinds the
        # `transcribe` attribute to the function, shadowing the module.
        import importlib
        wt = importlib.import_module('whisper.transcribe')
        assert getattr(wt, '_doza_progress_hooked', False) is True
        # Idempotent
        assert transcribe_module._install_whisper_progress_hook() is True


class TestTranscribeStatusEndpoint:
    @pytest.fixture
    def client(self, tmp_path):
        import app as app_module
        app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
        Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
        app_module.app.config['TESTING'] = True
        return app_module.app.test_client()

    def test_progress_merged_when_running(self, client, tmp_path):
        import app as app_module
        pid = 'tc-progress'
        pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
        pdir.mkdir(parents=True)
        (pdir / 'meta.json').write_text(json.dumps({'status': 'transcribing'}))
        writer = app_module._make_transcribe_progress_writer(pid)
        writer(0.42)
        assert app_module._claim_job(pid, 'transcribe')
        try:
            body = client.get(f'/project/{pid}/transcribe/status').get_json()
            assert body['running'] is True
            assert body['progress'] == 0.42
        finally:
            app_module._release_job(pid, 'transcribe')

    def test_clear_removes_progress_file(self, client):
        import app as app_module
        pid = 'tc-clear'
        Path(app_module.app.config['PROJECTS_DIR'], pid).mkdir(parents=True)
        app_module._make_transcribe_progress_writer(pid)(0.5)
        assert os.path.exists(app_module._transcribe_progress_path(pid))
        app_module._clear_transcribe_progress(pid)
        assert not os.path.exists(app_module._transcribe_progress_path(pid))


# ── F1 (route half): zero segments = error, not success ───────────────────

class TestZeroSegmentTranscript:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        import app as app_module
        app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
        Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
        app_module.app.config['TESTING'] = True
        return app_module.app.test_client()

    def test_empty_transcript_sets_error_status(self, client, tmp_path, monkeypatch):
        import app as app_module
        pid = 'zeroseg'
        pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
        pdir.mkdir(parents=True)
        src = tmp_path / 'src.mov'
        src.write_bytes(b'x')
        (pdir / 'meta.json').write_text(json.dumps({
            'status': 'uploaded', 'language': 'en', 'source_path': str(src),
        }))
        monkeypatch.setattr('transcribe.transcribe_file',
                            lambda *a, **k: {'segments': [], 'duration': 0})
        resp = client.post(f'/project/{pid}/transcribe')
        assert resp.status_code == 500
        assert 'no speech' in resp.get_json()['error']
        meta = json.loads((pdir / 'meta.json').read_text())
        assert meta['status'] == 'error'
        assert 'transcript' not in meta


# ── F4: speaker reassignment range matching ────────────────────────────────

class TestUpdateSpeakerRange:
    @pytest.fixture
    def client(self, tmp_path):
        import app as app_module
        app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
        Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
        app_module.app.config['TESTING'] = True
        return app_module.app.test_client()

    def test_segment_with_overshooting_end_is_included(self, client):
        # Whisper segment `end` pads past the last word's timestamp; the
        # paragraph range derived from word times must still capture it.
        import app as app_module
        pid = 'spkrange'
        pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
        pdir.mkdir(parents=True)
        (pdir / 'meta.json').write_text(json.dumps({
            'status': 'transcribed',
            'transcript': {'segments': [
                {'start': 0.0, 'end': 4.0, 'speaker': 'A', 'text': 'one'},
                {'start': 4.2, 'end': 9.7, 'speaker': 'A', 'text': 'two'},
            ]},
        }))
        # Paragraph's last WORD ends at 9.2 — old end-containment dropped
        # the second segment (9.7 > 9.2 + 0.1).
        resp = client.post(f'/project/{pid}/update-speaker-range', json={
            'start': 0.0, 'end': 9.2, 'speaker': 'B',
        })
        assert resp.get_json()['count'] == 2
        meta = json.loads((pdir / 'meta.json').read_text())
        assert [s['speaker'] for s in meta['transcript']['segments']] == ['B', 'B']

    def test_segment_starting_after_range_is_excluded(self, client):
        import app as app_module
        pid = 'spkrange2'
        pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
        pdir.mkdir(parents=True)
        (pdir / 'meta.json').write_text(json.dumps({
            'status': 'transcribed',
            'transcript': {'segments': [
                {'start': 0.0, 'end': 4.0, 'speaker': 'A', 'text': 'one'},
                {'start': 10.5, 'end': 14.0, 'speaker': 'A', 'text': 'next para'},
            ]},
        }))
        resp = client.post(f'/project/{pid}/update-speaker-range', json={
            'start': 0.0, 'end': 9.2, 'speaker': 'B',
        })
        assert resp.get_json()['count'] == 1
        meta = json.loads((pdir / 'meta.json').read_text())
        assert [s['speaker'] for s in meta['transcript']['segments']] == ['B', 'A']


class TestWhisperHonestyFlag:
    def test_multispeaker_request_gets_diarization_flag(self, monkeypatch, tmp_path):
        import types
        fake_result = {
            'segments': [{'start': 0, 'end': 1, 'text': 'hi', 'words': []}],
            'language': 'en',
        }

        class _FakeModel:
            def transcribe(self, path, **kw):
                return fake_result

        monkeypatch.setitem(transcribe_module._whisper_cache, 'turbo', _FakeModel())
        wav = tmp_path / 'a.wav'
        wav.write_bytes(b'RIFF')
        monkeypatch.setitem(sys.modules, 'whisper', types.ModuleType('whisper'))
        out = transcribe_module._transcribe_whisper(
            str(wav), {'SPEAKER_00': 'Chris'}, num_speakers=2, language='en')
        assert out['diarization'] == 'unavailable'
        assert 'reassign' in out['note']
        out1 = transcribe_module._transcribe_whisper(
            str(wav), {'SPEAKER_00': 'Chris'}, num_speakers=1, language='en')
        assert 'diarization' not in out1


# ── F5: analysis error surfacing ───────────────────────────────────────────

class TestAnalysisErrorSurfacing:
    def _long_transcript(self, minutes=20):
        segs = [
            {'start': i * 10.0, 'end': i * 10.0 + 9.0,
             'text': f'segment {i} text', 'speaker': 'A'}
            for i in range(minutes * 6)
        ]
        return {'segments': segs, 'duration': segs[-1]['end']}

    def test_all_chunks_failed_promotes_real_error(self, monkeypatch):
        import ai_analysis
        from ai_providers import ProviderError

        def _boom(*a, **k):
            raise ProviderError('Ollama error (HTTP 500): model exploded',
                                code='server_error')

        monkeypatch.setattr(ai_analysis, '_analyze_story', _boom)
        monkeypatch.setattr(ai_analysis, '_analyze_social', _boom)
        with pytest.raises(ProviderError, match='model exploded'):
            ai_analysis.analyze_transcript(self._long_transcript(),
                                           analysis_type='all')

    def test_fatal_code_aborts_immediately(self, monkeypatch):
        import ai_analysis
        from ai_providers import ProviderError
        calls = {'n': 0}

        def _refused(*a, **k):
            calls['n'] += 1
            raise ProviderError("Ollama isn't reachable", code='unreachable')

        monkeypatch.setattr(ai_analysis, '_analyze_story', _refused)
        monkeypatch.setattr(ai_analysis, '_analyze_social', _refused)
        with pytest.raises(ProviderError, match='reachable'):
            ai_analysis.analyze_transcript(self._long_transcript(),
                                           analysis_type='all')
        assert calls['n'] == 1  # no per-chunk timeout burn

    def test_unreachable_after_success_is_transient(self, monkeypatch):
        # Mid-run connection refusal = the bundled Ollama supervisor
        # bouncing after an OOM. Completed chunks must survive; the run
        # warns and continues instead of discarding everything.
        import ai_analysis
        from ai_providers import ProviderError
        calls = {'n': 0}

        def _story(*a, **k):
            calls['n'] += 1
            if calls['n'] == 2:
                raise ProviderError("Ollama isn't reachable", code='unreachable')
            return {'story_beats': [{'label': 'Beat', 'start': '00:00:05',
                                     'end': '00:00:15',
                                     'description': 'a beat'}],
                    'summary': 's', 'suggested_title': 't'}

        monkeypatch.setattr(ai_analysis, '_analyze_story', _story)
        monkeypatch.setattr(
            ai_analysis, '_synthesize_overall_summary',
            lambda *a, **k: {'summary': 's', 'suggested_title': 't'})
        result = ai_analysis.analyze_transcript(self._long_transcript(),
                                                analysis_type='story')
        assert result['story_beats']
        assert any('reachable' in w for w in result['analysis_warnings'])

    def test_two_consecutive_timeouts_abort(self, monkeypatch):
        import ai_analysis
        from ai_providers import ProviderError
        calls = {'n': 0}

        def _slow(*a, **k):
            calls['n'] += 1
            raise ProviderError('Ollama timed out mid-request', code='timeout')

        monkeypatch.setattr(ai_analysis, '_analyze_story', _slow)
        monkeypatch.setattr(ai_analysis, '_analyze_social', _slow)
        with pytest.raises(ProviderError, match='timed out'):
            ai_analysis.analyze_transcript(self._long_transcript(),
                                           analysis_type='all')
        assert calls['n'] == 2

    def test_partial_failure_keeps_warn_and_continue(self, monkeypatch):
        import ai_analysis
        from ai_providers import ProviderError
        calls = {'n': 0}

        def _flaky_story(*a, **k):
            calls['n'] += 1
            if calls['n'] == 1:
                raise ProviderError('Ollama error (HTTP 500): hiccup',
                                    code='server_error')
            return {'story_beats': [{'label': 'Beat', 'start': '00:00:05',
                                     'end': '00:00:15',
                                     'description': 'a beat'}],
                    'summary': 's', 'suggested_title': 't'}

        monkeypatch.setattr(ai_analysis, '_analyze_story', _flaky_story)
        monkeypatch.setattr(
            ai_analysis, '_synthesize_overall_summary',
            lambda *a, **k: {'summary': 's', 'suggested_title': 't'})
        result = ai_analysis.analyze_transcript(self._long_transcript(),
                                                analysis_type='story')
        assert any('hiccup' in w for w in result['analysis_warnings'])
        assert result['story_beats']


class TestAnalyzePreflight:
    @pytest.fixture
    def client(self, tmp_path):
        import app as app_module
        app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
        Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
        app_module.app.config['TESTING'] = True
        return app_module.app.test_client()

    def _project(self, pid):
        import app as app_module
        pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
        pdir.mkdir(parents=True)
        (pdir / 'meta.json').write_text(json.dumps({
            'name': 'P', 'status': 'transcribed',
            'transcript': {'segments': [
                {'start': 0, 'end': 5, 'text': 'a', 'speaker': 'X'}]},
        }))

    def test_unreachable_ollama_400s_fast(self, client, monkeypatch):
        import ai_providers

        class _Dead:
            def test_connection(self):
                return {'success': False, 'error': 'refused'}

        monkeypatch.setattr(ai_providers, 'get_active_provider',
                            lambda *a, **k: _Dead())
        monkeypatch.setattr('ai_analysis._ollama_is_active', lambda: True)
        self._project('pf1')
        resp = client.post('/project/pf1/analyze', json={'type': 'all'})
        assert resp.status_code == 400
        body = resp.get_json()
        assert body['code'] == 'unreachable'
        assert body['settings_url'] == '/settings'

    def test_no_models_400s_with_model_missing(self, client, monkeypatch):
        import ai_providers

        class _EmptyLibrary:
            def test_connection(self):
                return {'success': True, 'models': []}

        monkeypatch.setattr(ai_providers, 'get_active_provider',
                            lambda *a, **k: _EmptyLibrary())
        monkeypatch.setattr('ai_analysis._ollama_is_active', lambda: True)
        self._project('pf2')
        resp = client.post('/project/pf2/analyze', json={'type': 'all'})
        assert resp.status_code == 400
        assert resp.get_json()['code'] == 'model_missing'


class TestOllamaTypedErrors:
    def test_model_missing_code(self):
        from ai_providers import ProviderError
        from ai_providers.ollama_provider import _raise_ollama_error

        class _R:
            status_code = 404
            text = ''
            def json(self):
                return {'error': "model 'gemma4:e4b' not found"}

        with pytest.raises(ProviderError) as exc:
            _raise_ollama_error(_R(), 'gemma4:e4b')
        assert exc.value.code == 'model_missing'

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


# ── F6: source-timecode probe + labels ─────────────────────────────────────

class TestTimecodeLabels:
    def test_ndf_round_trip(self):
        from exporters.media_probe import timecode_to_frames, frames_to_timecode_label
        for tc, fps in [('14:23:11:05', 25.0), ('01:00:00:00', 24.0),
                        ('00:00:00:00', 23.976)]:
            frames = timecode_to_frames(tc, fps)
            assert frames_to_timecode_label(frames, fps, drop=False) == tc

    def test_df_round_trip(self):
        from exporters.media_probe import timecode_to_frames, frames_to_timecode_label
        for tc in ['14:23:11;05', '00:01:00;02', '00:10:00;00', '01:00:00;00']:
            frames = timecode_to_frames(tc, 29.97)
            assert frames_to_timecode_label(frames, 29.97, drop=True) == tc

    def test_zero_tc_is_not_absent(self, monkeypatch, tmp_path):
        # A legitimate 00:00:00:00 tag must return frames=0, not None —
        # the old falsy check treated it as "no timecode".
        from exporters import media_probe
        f = tmp_path / 'x.mov'
        f.write_bytes(b'x')

        class _Result:
            returncode = 0
            stdout = '00:00:00:00\n'

        monkeypatch.setattr(media_probe.subprocess, 'run',
                            lambda *a, **k: _Result())
        tc = media_probe.get_video_start_timecode(str(f), 25.0)
        assert tc == {'frames': 0, 'drop': False, 'raw': '00:00:00:00'}

    @needs_ffmpeg
    def test_probe_reads_embedded_ndf_tc(self, fixtures):
        from exporters.media_probe import get_video_start_timecode
        tc = get_video_start_timecode(str(fixtures / 'tc_ndf.mov'), 25.0)
        assert tc is not None
        assert tc['raw'].startswith('14:23:11')
        assert tc['drop'] is False
        assert tc['frames'] == ((14 * 3600 + 23 * 60 + 11) * 25) + 5

    @needs_ffmpeg
    def test_probe_returns_none_without_tc(self, fixtures):
        from exporters.media_probe import get_video_start_timecode
        assert get_video_start_timecode(str(fixtures / 'plain.mov'), 25.0) is None

    def test_dual_tag_prefers_nonzero(self, monkeypatch, tmp_path):
        # Camera files often carry a format-level 00:00:00:00 AND a tmcd
        # stream with the real TC. The nonzero one must win — it feeds
        # FCPXML asset.start (the v3.5.7 "Invalid edit" invariant).
        from exporters import media_probe
        f = tmp_path / 'x.mov'
        f.write_bytes(b'x')

        class _Result:
            returncode = 0
            stdout = '00:00:00:00\n14:23:11:05\n'

        monkeypatch.setattr(media_probe.subprocess, 'run',
                            lambda *a, **k: _Result())
        tc = media_probe.get_video_start_timecode(str(f), 25.0)
        assert tc['raw'] == '14:23:11:05'
        assert media_probe.get_video_start_timecode_frames(str(f), 25.0) == tc['frames']


# ── Review-driven gap tests ────────────────────────────────────────────────

class TestMediaAudioRoute:
    @pytest.fixture
    def client(self, tmp_path):
        import app as app_module
        app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
        Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
        app_module.app.config['TESTING'] = True
        return app_module.app.test_client()

    def _project_with_wav(self, pid, wav_bytes, source=None):
        import app as app_module
        pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
        pdir.mkdir(parents=True)
        (pdir / 'audio.wav').write_bytes(wav_bytes)
        meta = {'name': 'P', 'status': 'transcribed'}
        if source:
            meta['source_path'] = str(source)
        (pdir / 'meta.json').write_text(json.dumps(meta))
        return pdir

    def test_range_request_gets_206(self, client):
        # Safari requires Range support on <audio> sources — this route is
        # the automatic fallback player for camera formats.
        self._project_with_wav('rng', b'RIFF' + b'\x00' * 200000)
        resp = client.get('/project/rng/media/audio',
                          headers={'Range': 'bytes=0-99'})
        assert resp.status_code == 206
        assert resp.headers.get('Accept-Ranges') == 'bytes'
        assert 'no-cache' in resp.headers.get('Cache-Control', '')

    def test_poisoned_wav_not_served_when_source_present(self, client, tmp_path):
        # The fallback player must never hand out the truncated/silent WAV
        # cached by <=3.5.11 — with the source available it re-extracts.
        # Source here is a fake (extraction fails), so the route falls all
        # the way through to 404 instead of serving the poison.
        src = tmp_path / 'gone.mov'
        src.write_bytes(b'not real media')
        self._project_with_wav('poison', b'RIFF' + b'\x00' * 40, source=src)
        resp = client.get('/project/poison/media/audio')
        assert resp.status_code == 404

    def test_wav_served_when_source_missing(self, client, tmp_path):
        # Offline drive: no source to validate against — serving the
        # existing WAV beats serving nothing.
        self._project_with_wav('offline', b'RIFF' + b'\x00' * 200000,
                               source=tmp_path / 'unmounted.mov')
        resp = client.get('/project/offline/media/audio')
        assert resp.status_code == 200


class TestStaleIndexClearing:
    @pytest.fixture
    def client(self, tmp_path):
        import app as app_module
        app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
        Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
        app_module.app.config['TESTING'] = True
        return app_module.app.test_client()

    def test_new_transcript_clears_derived_indexes(self, client, tmp_path, monkeypatch):
        import app as app_module
        pid = 'staleidx'
        pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
        pdir.mkdir(parents=True)
        src = tmp_path / 'src.mov'
        src.write_bytes(b'x')
        (pdir / 'meta.json').write_text(json.dumps({
            'name': 'P', 'status': 'uploaded', 'language': 'en',
            'source_path': str(src),
        }))
        # Stale artifacts from the OLD (poisoned) transcript.
        (pdir / 'paragraph_index.json').write_text('{"stale": true}')
        (pdir / 'segment_vectors.json').write_text('[{"stale": true}]')
        monkeypatch.setattr(
            'transcribe.transcribe_file',
            lambda *a, **k: {'segments': [
                {'start': 0, 'end': 2, 'text': 'fresh words', 'speaker': 'A',
                 'start_formatted': '00:00:00.000', 'end_formatted': '00:00:02.000'}],
                'duration': 2, 'language': 'en'})
        resp = client.post(f'/project/{pid}/transcribe')
        assert resp.status_code == 200
        assert not (pdir / 'segment_vectors.json').exists()
        # paragraph_index was cleared, then rebuilt from the NEW transcript.
        if (pdir / 'paragraph_index.json').exists():
            rebuilt = (pdir / 'paragraph_index.json').read_text()
            assert 'stale' not in rebuilt


class TestStartTcPersistence:
    @pytest.fixture
    def client(self, tmp_path):
        import app as app_module
        app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
        Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
        app_module.app.config['TESTING'] = True
        return app_module.app.test_client()

    def test_probe_persists_shape_and_runs_once(self, client, tmp_path, monkeypatch):
        import app as app_module
        pid = 'tcpersist'
        pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
        pdir.mkdir(parents=True)
        src = tmp_path / 'cam.mov'
        src.write_bytes(b'x')
        (pdir / 'meta.json').write_text(json.dumps({
            'id': pid, 'name': 'P', 'status': 'transcribed',
            'source_path': str(src),
            'transcript': {'segments': [
                {'start': 0, 'end': 5, 'text': 'a', 'speaker': 'X',
                 'start_formatted': '00:00:00.000', 'end_formatted': '00:00:05.000'}]},
        }))
        calls = {'n': 0}

        def _probe(path, fps):
            calls['n'] += 1
            return {'frames': 1295280, 'drop': False, 'raw': '14:23:11:05'}

        monkeypatch.setattr(app_module, 'get_video_start_timecode', _probe)
        monkeypatch.setattr(app_module, 'get_video_framerate', lambda p: 25.0)

        assert client.get(f'/project/{pid}').status_code == 200
        meta = json.loads((pdir / 'meta.json').read_text())
        # Shape contract with the JS layer (TC_BY_PROJECT consumer).
        assert meta['start_tc'] == {'frames': 1295280, 'fps': 25.0,
                                    'drop': False, 'raw': '14:23:11:05'}
        # Transient view-loop keys must never leak into meta.json.
        assert 'source_exists' not in meta
        # Second view: probe must NOT run again.
        assert client.get(f'/project/{pid}').status_code == 200
        assert calls['n'] == 1
