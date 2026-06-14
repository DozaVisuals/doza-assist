"""Output Language feature: resolver, directive, meta plumbing, bug fixes.

Spec invariants under test:
- ONE shared resolver; reads meta only (never the transcript blob).
- 'match' follows interview language; 'auto' resolves via detected_language;
  unavailable detection falls back to English. Collections default English.
- English resolution injects NOTHING (zero diff vs pre-feature prompts).
- detected_language is persisted to meta at transcription completion.
- 'auto' on a Whisper-less install fails early at the engine guard with a
  clear message (it used to die deep in transcribe_file).
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from doza_assist.output_language import (
    LANGUAGES,
    language_directive,
    language_name,
    resolve_collection_output_language,
    resolve_output_language,
)
import app as app_module
import transcribe as transcribe_module


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


class TestResolver:
    def test_match_follows_interview_language(self):
        assert resolve_output_language({'language': 'no'}) == 'no'
        assert resolve_output_language(
            {'language': 'no', 'output_language': 'match'}) == 'no'

    def test_match_auto_resolves_via_detected(self):
        meta = {'language': 'auto', 'detected_language': 'no'}
        assert resolve_output_language(meta) == 'no'

    def test_match_auto_missing_detection_falls_back_english(self):
        assert resolve_output_language({'language': 'auto'}) == 'en'
        assert resolve_output_language(
            {'language': 'auto', 'detected_language': ''}) == 'en'
        # Unknown detected code (engine drift) must not leak into prompts.
        assert resolve_output_language(
            {'language': 'auto', 'detected_language': 'xx'}) == 'en'

    def test_explicit_code_wins(self):
        meta = {'language': 'no', 'output_language': 'de'}
        assert resolve_output_language(meta) == 'de'

    def test_invalid_values_fall_back(self):
        assert resolve_output_language({}) == 'en'
        assert resolve_output_language({'output_language': 'klingon'}) == 'en'
        assert resolve_output_language({'language': 'xx'}) == 'en'
        assert resolve_output_language(None) == 'en'

    def test_collection_defaults_english(self):
        assert resolve_collection_output_language({}) == 'en'
        assert resolve_collection_output_language(None) == 'en'
        assert resolve_collection_output_language(
            {'output_language': 'no'}) == 'no'

    def test_detected_nynorsk_is_resolvable(self):
        # whisper can detect 'nn' which is outside the UI list but named.
        meta = {'language': 'auto', 'detected_language': 'nn'}
        assert resolve_output_language(meta) == 'nn'
        assert language_name('nn')


class TestDirective:
    def test_english_is_empty_string(self):
        assert language_directive('en') == ''
        assert language_directive('en', chat=True) == ''
        assert language_directive(None) == ''
        assert language_directive('') == ''
        assert language_directive('xx') == ''  # unknown -> no directive

    def test_norwegian_directive_contents(self):
        d = language_directive('no')
        assert 'Norwegian' in d
        assert 'Never translate verbatim quotes' in d
        assert 'speaker labels' in d
        assert d.startswith('\n\n')

    def test_chat_variant_adds_user_language_exception(self):
        d = language_directive('no', chat=True)
        assert 'reply in the language the user wrote in' in d
        assert 'reply in the language the user wrote in' not in language_directive('no')

    def test_every_canonical_language_has_a_directive_except_english(self):
        for code, name in LANGUAGES:
            d = language_directive(code)
            if code == 'en':
                assert d == ''
            else:
                assert name in d


class TestMetaPlumbing:
    def test_create_project_writes_output_language_default(self, client, tmp_path):
        src = tmp_path / 'a.wav'
        src.write_bytes(b'RIFF' + b'\x00' * 64)
        pid = app_module.create_project_from_path(str(src), project_name='X',
                                                  language='no')
        meta = json.loads((Path(app_module.app.config['PROJECTS_DIR']) / pid /
                           'meta.json').read_text())
        assert meta['output_language'] == 'match'
        assert meta['language'] == 'no'

    def test_create_project_clamps_invalid_output_language(self, client, tmp_path):
        src = tmp_path / 'a.wav'
        src.write_bytes(b'RIFF' + b'\x00' * 64)
        pid = app_module.create_project_from_path(
            str(src), project_name='X', output_language='klingon')
        meta = json.loads((Path(app_module.app.config['PROJECTS_DIR']) / pid /
                           'meta.json').read_text())
        assert meta['output_language'] == 'match'

    def test_output_language_endpoint_updates_and_hints(self, client):
        _make_project('p1', language='no', analysis={'summary': 'x'})
        r = client.post('/project/p1/output-language',
                        json={'output_language': 'de'})
        assert r.status_code == 200
        body = r.get_json()
        assert body['output_language'] == 'de'
        assert body['resolved'] == 'de'
        assert body['reanalyze_hint'] is True
        # Same value again: no change -> no hint.
        r2 = client.post('/project/p1/output-language',
                         json={'output_language': 'de'})
        assert r2.get_json()['reanalyze_hint'] is False

    def test_output_language_endpoint_no_analysis_no_hint(self, client):
        _make_project('p2', language='no')
        r = client.post('/project/p2/output-language',
                        json={'output_language': 'de'})
        assert r.get_json()['reanalyze_hint'] is False

    def test_api_languages_serves_canonical_list(self, client):
        r = client.get('/api/languages')
        langs = r.get_json()['languages']
        assert langs[0] == {'code': 'en', 'name': 'English'}
        assert {'code': 'no', 'name': 'Norwegian'} in langs
        assert len(langs) == len(LANGUAGES)


class TestDetectedLanguagePersistence:
    def test_run_transcribe_job_persists_detected_language(self, client, monkeypatch):
        _make_project('p3', language='auto', source_path='/tmp/x.wav')
        monkeypatch.setattr(
            'transcribe.transcribe_file',
            lambda *a, **k: {'segments': [{'start': 0, 'end': 1.0, 'text': 'hei'}],
                             'language': 'no', 'duration': 1.0, 'engine': 'whisper'})
        app_module._run_transcribe_job('p3', '/tmp/x.wav', 2, 'auto',
                                       'Interviewer', 'Subject')
        meta = json.loads((Path(app_module.app.config['PROJECTS_DIR']) / 'p3' /
                           'meta.json').read_text())
        assert meta['detected_language'] == 'no'
        assert resolve_output_language(meta) == 'no'


class TestEngineGuardAuto:
    def test_auto_without_whisper_fails_early_with_clear_message(self, client,
                                                                  monkeypatch, tmp_path):
        src = tmp_path / 'real.wav'
        src.write_bytes(b'RIFF' + b'\x00' * 64)
        _make_project('p4', language='auto', source_path=str(src))
        monkeypatch.setattr(app_module, '_engine_available',
                            lambda name: name != 'whisper')
        r = client.post('/project/p4/transcribe')
        assert r.status_code == 400
        body = r.get_json()
        assert body.get('needs_whisper_install') is True
        assert 'Auto-detect' in body['error']

    def test_english_still_allowed_without_whisper(self, client, monkeypatch,
                                                   tmp_path):
        src = tmp_path / 'real.wav'
        src.write_bytes(b'RIFF' + b'\x00' * 64)
        _make_project('p5', language='en', source_path=str(src))
        monkeypatch.setattr(app_module, '_engine_available',
                            lambda name: name != 'whisper')
        monkeypatch.setattr(app_module.threading, 'Thread',
                            lambda *a, **k: type('T', (), {'start': lambda s: None,
                                                           'daemon': False})())
        r = client.post('/project/p5/transcribe')
        assert r.status_code != 400 or 'whisper' not in (r.get_json() or {}).get('error', '')


class TestWhisperxLanguageFix:
    def test_whisperx_result_uses_captured_lang_code(self):
        # Source-level regression pin: the whisperx return dict must carry
        # the captured lang_code, not result.get('language') (which align()
        # destroys). Full-path test impossible without whisperx installed.
        import inspect
        src = inspect.getsource(transcribe_module._transcribe_whisperx)
        assert "'language': lang_code or 'en'" in src
        assert "result.get('language', 'en')" not in src


class TestDirectiveInjectionCoreSurfaces:
    """Gates 2+3 for the three CORE surfaces (Analysis, Chat, Story
    Builder): English/None resolution assembles byte-identical prompts with
    no directive text; Norwegian injects a directive naming Norwegian."""

    def _transcript(self):
        return {'segments': [{'start': 0.0, 'end': 2.0,
                              'start_formatted': '00:00:00,000',
                              'end_formatted': '00:00:02,000',
                              'text': 'hello there world', 'speaker': 'SPEAKER_00'}],
                'language': 'no', 'duration': 2.0, 'engine': 'whisper'}

    def test_chat_surface(self, monkeypatch):
        import ai_analysis
        captured = []
        monkeypatch.setattr(
            ai_analysis, '_call_ai_chat',
            lambda system_message, messages, num_ctx=32768:
                (captured.append(system_message) or 'a fine answer'))
        t = self._transcript()
        ai_analysis.chat_about_transcript(t, 'what is said here?')
        ai_analysis.chat_about_transcript(t, 'what is said here?',
                                          output_language='en')
        ai_analysis.chat_about_transcript(t, 'what is said here?',
                                          output_language='no')
        assert len(captured) >= 3
        assert 'OUTPUT LANGUAGE' not in captured[0]
        assert captured[0] == captured[1], 'English must be byte-identical to default'
        assert 'OUTPUT LANGUAGE' in captured[2]
        assert 'Norwegian' in captured[2]
        assert 'reply in the language the user wrote in' in captured[2]

    def test_analysis_surface(self, monkeypatch):
        import ai_analysis
        sys_prompts = []

        def fake_call(prompt, system_prompt="", task_type="analysis",
                      force_json=True, **kwargs):
            sys_prompts.append(system_prompt)
            return '{}'

        monkeypatch.setattr(ai_analysis, '_call_ai', fake_call)
        t = self._transcript()
        ai_analysis.analyze_transcript(t, analysis_type='all')
        baseline = list(sys_prompts)
        sys_prompts.clear()
        ai_analysis.analyze_transcript(t, analysis_type='all',
                                       output_language='en')
        english = list(sys_prompts)
        sys_prompts.clear()
        ai_analysis.analyze_transcript(t, analysis_type='all',
                                       output_language='no')
        norwegian = list(sys_prompts)
        assert baseline and norwegian
        assert all('OUTPUT LANGUAGE' not in s for s in baseline)
        assert english == baseline, 'English must be byte-identical to default'
        assert all('Norwegian' in s for s in norwegian), \
            'every analysis model call must carry the directive'

    def test_story_builder_surface(self, monkeypatch):
        import ai_analysis
        sys_prompts = []

        def fake_call(prompt, system_prompt="", task_type="analysis",
                      force_json=True, **kwargs):
            sys_prompts.append(system_prompt)
            return '{"story_title": "t", "clips": []}'

        def fake_json(system_prompt, user_prompt, timeout=None,
                      model_override=None, **kwargs):
            sys_prompts.append(system_prompt)
            return '{"story_title": "t", "clips": []}'

        monkeypatch.setattr(ai_analysis, '_call_ai', fake_call)
        monkeypatch.setattr(ai_analysis, '_call_ai_json', fake_json)
        t = self._transcript()
        ai_analysis.build_story(t, 'make a 1 minute cut')
        baseline = list(sys_prompts)
        sys_prompts.clear()
        ai_analysis.build_story(t, 'make a 1 minute cut', output_language='no')
        norwegian = list(sys_prompts)
        assert baseline and norwegian
        assert all('OUTPUT LANGUAGE' not in s for s in baseline)
        assert any('Norwegian' in s for s in norwegian)

    def test_title_anchor_skipped_cross_language(self):
        """7a: a marker whose title tokens appear elsewhere in the transcript
        but not in its window is dropped same-language, kept cross-language."""
        import ai_analysis
        # The clip window (start-5s .. end+5s) must NOT overlap the
        # harbor segment, or the title counts as anchored and is kept.
        segments = [
            {'start': 0.0, 'end': 10.0, 'text': 'we talked about the harbor'},
            {'start': 10.0, 'end': 30.0, 'text': 'nothing notable here'},
        ]
        text = ('Here is one:\n'
                '[CLIP: start=00:00:20 end=00:00:25 title="Harbor memories"]')
        kept_default = ai_analysis._validate_clip_markers_in_text(
            text, segments)
        kept_skip = ai_analysis._validate_clip_markers_in_text(
            text, segments, skip_title_anchor=True)
        assert '[CLIP:' not in kept_default
        assert '[CLIP:' in kept_skip


class TestCollectionLanguageInheritance:
    """Collections inherit the member projects' output language (the tester
    fix): a Norwegian folder must produce Norwegian collection prose, not
    English. Explicit collection setting still wins; no members → English."""

    def test_inherits_majority_member_language(self):
        members = [
            {'language': 'no'},
            {'language': 'auto', 'detected_language': 'no'},
            {'language': 'en'},
        ]
        assert resolve_collection_output_language({}, members) == 'no'

    def test_all_english_members_stay_english(self):
        members = [{'language': 'en'}, {'language': 'en'}]
        assert resolve_collection_output_language({}, members) == 'en'
        # English → empty directive → byte-identical prompts.
        assert language_directive(
            resolve_collection_output_language({}, members)) == ''

    def test_explicit_collection_setting_wins_over_members(self):
        members = [{'language': 'no'}, {'language': 'no'}]
        assert resolve_collection_output_language(
            {'output_language': 'de'}, members) == 'de'

    def test_explicit_match_falls_through_to_members(self):
        members = [{'language': 'sv'}]
        assert resolve_collection_output_language(
            {'output_language': 'match'}, members) == 'sv'

    def test_no_members_defaults_english(self):
        assert resolve_collection_output_language({}, []) == 'en'
        assert resolve_collection_output_language({}, None) == 'en'

    def test_tie_breaks_to_first_member_order(self):
        # one 'no', one 'de' — tie; first-seen ('no') wins deterministically.
        members = [{'language': 'no'}, {'language': 'de'}]
        assert resolve_collection_output_language({}, members) == 'no'

    def test_english_members_never_outvote_non_english(self):
        # Norwegian + two undetected/English members must stay Norwegian:
        # English is the fallback, not a content vote.
        members = [{'language': 'no'},
                   {'language': 'auto'},  # not yet detected → 'en'
                   {'language': 'en'}]
        assert resolve_collection_output_language({}, members) == 'no'

    def test_unknown_member_codes_do_not_force_english(self):
        members = [{'language': 'no'}, {'language': 'xx'}, {'language': 'zz'}]
        # 'xx'/'zz' resolve to 'en' (unknown) and are not counted → 'no' wins.
        assert resolve_collection_output_language({}, members) == 'no'
