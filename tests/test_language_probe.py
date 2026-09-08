"""Auto-detect routes English through Parakeet (1.1) via Whisper language
identification: the probe runner, the WAV reader, and the job's language
resolution with stubbed engines."""

import contextlib
import json
import os
import struct
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402
import language_probe as lp  # noqa: E402
import transcribe  # noqa: E402


def _wav(path, seconds=1.0, rate=16000, channels=1):
    n = int(seconds * rate)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b''.join(struct.pack('<' + 'h' * channels, *([1000] * channels)) for _ in range(n))
        w.writeframes(frames)
    return str(path)


def test_read_wav_float32_mono_and_stereo(tmp_path):
    mono = lp._read_wav_float32(_wav(tmp_path / 'm.wav', 0.5))
    assert mono.dtype.name == 'float32' and len(mono) == 8000 and abs(mono[0] - 1000 / 32768) < 1e-6
    stereo = lp._read_wav_float32(_wav(tmp_path / 's.wav', 0.5, channels=2))
    assert len(stereo) == 8000


def test_probe_reports_confident_language_only(tmp_path):
    wav = _wav(tmp_path / 'audio.wav')
    out = lp.probe_language(wav, None, lambda head: ('en', 0.97))
    assert (out['language'], out['probability'], out['error'], out['method']) == ('en', 0.97, None, 'whisper-lid')
    out = lp.probe_language(wav, None, lambda head: ('no', 0.88))
    assert out['language'] == 'no'
    out = lp.probe_language(wav, None, lambda head: ('de', 0.31))   # not confident: no verdict
    assert out['language'] is None and out['probability'] == 0.31
    out = lp.probe_language(wav, None, lambda head: ('', 0.9))
    assert out['language'] is None

    def boom(head):
        raise RuntimeError('no model')
    out = lp.probe_language(wav, None, boom)
    assert out['language'] is None and 'no model' in out['error']
    out = lp.probe_language(str(tmp_path / 'nope.wav'), None, lambda head: ('en', 1.0))
    assert out['error'] == 'no audio to probe'


def test_probe_trims_from_first_speech_and_removes_the_head(tmp_path, monkeypatch):
    wav = _wav(tmp_path / 'audio.wav')
    made = []
    got = {}

    def fake_trim(audio_path, ffmpeg, seconds=lp.PROBE_SECONDS, offset=0.0):
        got['offset'] = offset
        p = _wav(tmp_path / 'head.wav', 1.0)
        made.append(p)
        return p
    monkeypatch.setattr(lp, 'trim_head', fake_trim)
    monkeypatch.setattr(lp, 'first_speech_offset', lambda a, f=None, search_seconds=600: 42.5)
    seen = []
    out = lp.probe_language(wav, '/usr/bin/true', lambda head: seen.append(head) or ('en', 0.9))
    assert out['language'] == 'en' and out['offset'] == 42.5 and got['offset'] == 42.5
    assert seen == made and not os.path.exists(made[0])


def test_silent_openings_give_no_verdict(tmp_path, monkeypatch):
    wav = _wav(tmp_path / 'audio.wav')
    # the whole searched span is silent: nothing to listen to
    monkeypatch.setattr(lp, 'first_speech_offset', lambda a, f=None, search_seconds=600: -1.0)
    called = []
    out = lp.probe_language(wav, '/usr/bin/true', lambda head: called.append(1) or ('en', 0.99))
    assert out['language'] is None and 'no speech' in out['error'] and called == []
    # a window that is digitally silent is too quiet to judge
    monkeypatch.setattr(lp, 'first_speech_offset', lambda a, f=None, search_seconds=600: 0.0)
    silent = tmp_path / 'silent.wav'
    with wave.open(str(silent), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b'\0' * 32000)
    monkeypatch.setattr(lp, 'trim_head', lambda a, f, seconds=30, offset=0.0: str(silent))
    out = lp.probe_language(wav, '/usr/bin/true', lambda head: called.append(1) or ('en', 0.99))
    assert out['language'] is None and 'too quiet' in out['error'] and called == []
    assert out['peak_dbfs'] is None


def _wav_with_pattern(path, pattern, rate=16000):
    """pattern: list of (seconds, amplitude) stretches."""
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        for seconds, amp in pattern:
            w.writeframes(struct.pack('<h', amp) * int(seconds * rate))
    return str(path)


def test_first_speech_offset_in_pure_python(tmp_path):
    # opens on sound
    assert lp.first_speech_offset(_wav_with_pattern(tmp_path / 'a.wav', [(3, 3000)])) == 0.0
    # 8 s of silence, a 0.3 s click, then talking: the click is not a run
    off = lp.first_speech_offset(_wav_with_pattern(tmp_path / 'b.wav', [(8, 0), (0.3, 3000), (2, 0), (5, 3000)]))
    assert 10.0 <= off <= 10.5
    # room tone under the threshold counts as silence
    off = lp.first_speech_offset(_wav_with_pattern(tmp_path / 'c.wav', [(5, 200), (5, 4000)]))
    assert 4.8 <= off <= 5.2
    # nothing but silence in the searched span
    assert lp.first_speech_offset(_wav_with_pattern(tmp_path / 'd.wav', [(20, 0)])) == -1.0
    # a searched span shorter than the leading silence
    assert lp.first_speech_offset(_wav_with_pattern(tmp_path / 'e.wav', [(20, 0), (5, 4000)]), search_seconds=10) == -1.0
    # unreadable file: fall back to the top
    (tmp_path / 'x.wav').write_bytes(b'not a wav')
    assert lp.first_speech_offset(str(tmp_path / 'x.wav')) == 0.0


def test_peak_dbfs(tmp_path):
    assert lp.peak_dbfs(_wav(tmp_path / 'tone.wav', 0.2)) == pytest.approx(20 * __import__('math').log10(1000 / 32768), abs=0.01)


def test_load_lid_model_prefers_cached_then_smallest(monkeypatch):
    import types
    calls = []
    fake_whisper = types.SimpleNamespace(load_model=lambda name: calls.append(name) or f'model-{name}')
    monkeypatch.setitem(sys.modules, 'whisper', fake_whisper)
    cache = {'turbo': 'cached-turbo'}
    assert lp._load_lid_model(cache) == 'cached-turbo' and calls == []
    cache = {}
    assert lp._load_lid_model(cache) == 'model-base' and calls == ['base'] and cache['base'] == 'model-base'

    def flaky(name):
        calls.append(name)
        if name == 'base':
            raise OSError('not downloaded')
        return f'model-{name}'
    monkeypatch.setitem(sys.modules, 'whisper', types.SimpleNamespace(load_model=flaky))
    assert lp._load_lid_model({}) == 'model-small'


# ── the job ────────────────────────────────────────────────────────────────

@pytest.fixture
def projects(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    return Path(app_module.app.config['PROJECTS_DIR'])


def _make_project(projects, pid):
    pdir = projects / pid
    pdir.mkdir()
    (pdir / 'meta.json').write_text(json.dumps({'id': pid, 'name': pid, 'status': 'uploaded', 'language': 'auto'}))
    return pdir


def _run_job(monkeypatch, projects, pid, verdict, language='auto', whisper_installed=True):
    seen = {'identify_calls': 0}
    pdir = projects / pid

    def fake_extract(filepath, project_dir=None, audio_channel=None):
        return _wav(pdir / 'audio.wav')

    def fake_identify(head, cache=None):
        seen['identify_calls'] += 1
        return verdict

    def fake_transcribe_file(source_path, **kwargs):
        seen['language'] = kwargs.get('language')
        return {'segments': [{'start': 0, 'end': 1, 'text': 'hello there friend'}], 'duration': 1.0}

    monkeypatch.setattr(transcribe, 'extract_audio', fake_extract)
    monkeypatch.setattr(transcribe, '_find_ffmpeg', lambda: None)
    monkeypatch.setattr(transcribe, 'transcribe_file', fake_transcribe_file)
    monkeypatch.setattr(transcribe, 'wav_peak_dbfs', lambda p: -20.0, raising=False)
    monkeypatch.setattr(lp, 'whisper_available', lambda: whisper_installed)
    monkeypatch.setattr(lp, 'whisper_identify', fake_identify)
    monkeypatch.setattr(app_module, '_memory_heavy_stage', lambda *a, **k: contextlib.nullcontext())
    if hasattr(app_module, '_release_transcribe_caches_if_budgeted'):
        monkeypatch.setattr(app_module, '_release_transcribe_caches_if_budgeted', lambda *a, **k: None)
    app_module._run_transcribe_job(pid, str(pdir / 'source.wav'), 2, language, 'Interviewer', 'Subject')
    return seen


def _meta(projects, pid):
    return json.loads((projects / pid / 'meta.json').read_text())


def test_auto_english_goes_to_parakeet(projects, monkeypatch):
    _make_project(projects, 'en1')
    seen = _run_job(monkeypatch, projects, 'en1', ('en', 0.96))
    assert seen['language'] == 'en' and seen['identify_calls'] == 1
    meta = _meta(projects, 'en1')
    assert meta['detected_language'] == 'en' and meta['language_probe']['language'] == 'en'


def test_auto_other_language_is_named_for_whisper(projects, monkeypatch):
    _make_project(projects, 'no1')
    seen = _run_job(monkeypatch, projects, 'no1', ('no', 0.91))
    assert seen['language'] == 'no'                      # explicit, never 'en'
    assert _meta(projects, 'no1')['language_probe']['language'] == 'no'


def test_auto_unsure_stays_auto(projects, monkeypatch):
    _make_project(projects, 'x1')
    seen = _run_job(monkeypatch, projects, 'x1', ('de', 0.2))
    assert seen['language'] == 'auto'
    assert _meta(projects, 'x1')['language_probe']['language'] is None


def test_auto_without_whisper_keeps_the_old_behavior(projects, monkeypatch):
    _make_project(projects, 'w1')
    seen = _run_job(monkeypatch, projects, 'w1', ('en', 0.99), whisper_installed=False)
    assert seen['language'] == 'auto' and seen['identify_calls'] == 0
    assert _meta(projects, 'w1')['language_probe']['error'] == 'whisper not installed'


def test_explicit_language_skips_the_probe(projects, monkeypatch):
    _make_project(projects, 'de1')
    seen = _run_job(monkeypatch, projects, 'de1', ('en', 0.99), language='de')
    assert seen['language'] == 'de' and seen['identify_calls'] == 0
