"""Auto-detect routes English through Parakeet (1.1): the probe scorer, the
probe runner, and the job's language resolution with a stubbed engine."""

import contextlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402
import language_probe as lp  # noqa: E402
import transcribe  # noqa: E402

ENGLISH = ("So when we started the company we really wanted to help people who call in every day. "
           "The first thing we did was listen, and that changed how we think about the whole job.")
NORWEGIAN_THROUGH_ENGLISH_MODEL = ("yoy your day ah scully vee hard vert like ah stort prosjekt fun donder "
                                   "oxo scal vee gyorra dot ee morgen ah tack for hjelpen dinner")
GERMAN_THROUGH_ENGLISH_MODEL = ("vir haben das projekt im letzten yar angefangen und es var sehr shwear "
                                "aber die kollegen haben gearbeitet und yetzt ist es fertig")


def test_scorer_separates_english_from_other_languages():
    en_score, en_n = lp.english_score(ENGLISH)
    assert en_n >= lp.MIN_TOKENS and en_score >= 0.7
    assert lp.looks_english(ENGLISH)
    for other in (NORWEGIAN_THROUGH_ENGLISH_MODEL, GERMAN_THROUGH_ENGLISH_MODEL):
        score, n = lp.english_score(other)
        assert n >= lp.MIN_TOKENS
        assert score < lp.ENGLISH_THRESHOLD, (other, score)
        assert not lp.looks_english(other)


def test_scorer_needs_enough_words_and_survives_empty_text():
    assert lp.english_score('') == (0.0, 0)
    assert not lp.looks_english('the and to')            # too few tokens to judge
    assert not lp.looks_english(None)


def test_probe_returns_en_only_for_english_heads(tmp_path):
    wav = tmp_path / 'audio.wav'
    wav.write_bytes(b'RIFF' + b'\0' * 64)
    seen = []

    def fake_engine(head):
        seen.append(head)
        return {'segments': [{'text': ENGLISH}]}
    out = lp.probe_language(str(wav), None, fake_engine)   # ffmpeg None: no trim
    assert out['language'] == 'en' and out['tokens'] >= lp.MIN_TOKENS and out['error'] is None
    assert seen == [str(wav)]

    out = lp.probe_language(str(wav), None, lambda head: {'segments': [{'text': NORWEGIAN_THROUGH_ENGLISH_MODEL}]})
    assert out['language'] is None and out['score'] < lp.ENGLISH_THRESHOLD

    out = lp.probe_language(str(wav), None, lambda head: {'segments': []})
    assert out['language'] is None and out['tokens'] == 0

    def boom(head):
        raise RuntimeError('Parakeet missing')
    out = lp.probe_language(str(wav), None, boom)
    assert out['language'] is None and 'Parakeet missing' in out['error']

    out = lp.probe_language(str(tmp_path / 'nope.wav'), None, fake_engine)
    assert out['language'] is None and out['error'] == 'no audio to probe'


def test_probe_removes_its_trimmed_head(tmp_path, monkeypatch):
    wav = tmp_path / 'audio.wav'
    wav.write_bytes(b'RIFF' + b'\0' * 64)
    made = []

    def fake_trim(audio_path, ffmpeg, seconds=lp.PROBE_SECONDS):
        p = tmp_path / 'head.wav'
        p.write_bytes(b'x')
        made.append(str(p))
        return str(p)
    monkeypatch.setattr(lp, 'trim_head', fake_trim)
    out = lp.probe_language(str(wav), '/usr/bin/true', lambda head: {'segments': [{'text': ENGLISH}]})
    assert out['language'] == 'en' and made and not os.path.exists(made[0])


# ── the job: auto resolves to 'en' for English, stays 'auto' otherwise ──────

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


def _run_job_with(monkeypatch, projects, pid, head_text, language='auto'):
    seen = {}
    pdir = projects / pid
    wav = pdir / 'audio.wav'

    def fake_extract(filepath, project_dir=None, audio_channel=None):
        wav.write_bytes(b'RIFF' + b'\0' * 64)
        return str(wav)

    def fake_parakeet(head, speaker_labels=None, progress_cb=None):
        return {'segments': [{'text': head_text}]}

    def fake_transcribe_file(source_path, **kwargs):
        seen['language'] = kwargs.get('language')
        return {'segments': [{'start': 0, 'end': 1, 'text': 'hello there friend'}], 'duration': 1.0}

    monkeypatch.setattr(transcribe, 'extract_audio', fake_extract)
    monkeypatch.setattr(transcribe, '_transcribe_parakeet', fake_parakeet)
    monkeypatch.setattr(transcribe, '_find_ffmpeg', lambda: None)
    monkeypatch.setattr(transcribe, 'transcribe_file', fake_transcribe_file)
    monkeypatch.setattr(transcribe, 'wav_peak_dbfs', lambda p: -20.0, raising=False)
    monkeypatch.setattr(app_module, '_memory_heavy_stage', lambda *a, **k: contextlib.nullcontext())
    for name in ('_release_transcribe_caches_if_budgeted',):
        if hasattr(app_module, name):
            monkeypatch.setattr(app_module, name, lambda *a, **k: None)
    app_module._run_transcribe_job(pid, str(pdir / 'source.wav'), 2, language, 'Interviewer', 'Subject')
    return seen


def test_auto_becomes_en_for_english_audio(projects, monkeypatch):
    _make_project(projects, 'en1')
    seen = _run_job_with(monkeypatch, projects, 'en1', ENGLISH)
    assert seen['language'] == 'en'
    meta = json.loads((projects / 'en1' / 'meta.json').read_text())
    assert meta['detected_language'] == 'en'
    assert meta['language_probe']['language'] == 'en'


def test_auto_stays_auto_for_other_languages(projects, monkeypatch):
    _make_project(projects, 'no1')
    seen = _run_job_with(monkeypatch, projects, 'no1', NORWEGIAN_THROUGH_ENGLISH_MODEL)
    assert seen['language'] == 'auto'
    meta = json.loads((projects / 'no1' / 'meta.json').read_text())
    # the probe recorded its miss; the transcript's own detection may still
    # write detected_language later, that is the existing job behavior
    assert meta['language_probe']['language'] is None


def test_explicit_language_skips_the_probe(projects, monkeypatch):
    _make_project(projects, 'de1')
    called = []
    monkeypatch.setattr(transcribe, '_transcribe_parakeet', lambda *a, **k: called.append(1) or {'segments': []})
    seen = _run_job_with(monkeypatch, projects, 'de1', ENGLISH, language='de')
    assert seen['language'] == 'de' and called == []
