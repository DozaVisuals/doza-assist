"""Regression pins for the 1.0.36 fixes (Chris's post-runtime-bump finds).

Root cause of the diffuse quality drift after the 0.31.1 runtime bump:
the gemma4 manifest bakes Google's loose sampling (temp 1, top_k 64,
top_p 0.95, min_p 0) and 0.31.x honors model-baked params for options we
leave unset, where 0.23.2 used Ollama's classic tight defaults. Every
piece of chat-quality tuning ran against the tight values — so they are
now pinned EXPLICITLY in every Ollama request and the app no longer
depends on either the runtime's or the manifest's opinion.
"""

import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ai_analysis  # noqa: E402
from ai_analysis import (  # noqa: E402
    _clean_chat_response,
    _story_clip_hygiene,
    parse_duration_bound_seconds,
    parse_target_duration_seconds,
)


class TestSamplerPin:
    def _captured_options(self, call):
        captured = {}

        class _Resp:
            status_code = 200
            def json(self):
                return {'message': {'content': 'ok'}, 'response': 'ok'}

        def fake_post(url, *, json=None, timeout=None, stream=False):
            captured['url'] = url
            captured['options'] = (json or {}).get('options') or {}
            return _Resp()

        from ai_providers import ollama_provider as op
        with patch.object(op, '_post_with_reconnect', side_effect=fake_post):
            call(op)
        return captured

    def test_chat_request_pins_classic_samplers(self):
        def call(op):
            p = op.OllamaProvider(base_url='http://127.0.0.1:1',
                                  model_resolver=lambda: 'gemma4:e4b')
            p.generate('sys', 'hello', task_type='chat')
        opts = self._captured_options(call)['options']
        assert opts['top_k'] == 40
        assert opts['top_p'] == 0.9
        assert opts['min_p'] == 0.05

    def test_stream_request_pins_classic_samplers(self):
        captured = {}

        class _Resp:
            status_code = 200
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def iter_lines(self):
                yield json.dumps({'message': {'content': 'x'}, 'done': True}).encode()

        def fake_post(url, *, json=None, timeout=None, stream=False):
            captured['options'] = (json or {}).get('options') or {}
            return _Resp()

        from ai_providers import ollama_provider as op
        with patch.object(op, '_post_with_reconnect', side_effect=fake_post):
            p = op.OllamaProvider(base_url='http://127.0.0.1:1',
                                  model_resolver=lambda: 'gemma4:e4b')
            list(p.generate_stream('sys', 'hello', task_type='chat'))
        assert captured['options']['top_k'] == 40
        assert captured['options']['top_p'] == 0.9
        assert captured['options']['min_p'] == 0.05


class TestShorthandUnits:
    def test_shorthand_targets_and_bounds(self):
        # The live bug: "make me a 60s story" parsed to nothing -> the
        # model freestyled a 7:23 build against a 60s ask.
        assert parse_target_duration_seconds('make me a 60s story') == 60.0
        assert parse_target_duration_seconds('build a 2min teaser') == 120.0
        assert parse_target_duration_seconds('make me a story under 90s') == 72.0
        assert parse_duration_bound_seconds('make me a story under 90s') == 90.0
        assert parse_target_duration_seconds('give me 2m of selects') == 120.0
        # Correction phrasing with a competing bare timecode.
        assert parse_target_duration_seconds(
            'that should be a 60 second story not 7:23') == 60.0

    def test_shorthand_does_not_widen_false_positives(self):
        for msg in ('give me a sec', 'whats this all about',
                    'she waited a moment', 'the 10am interview',
                    'find me a clip under 2 minutes'):
            assert parse_target_duration_seconds(msg) is None, msg


class TestStoryClipHygiene:
    def test_sliver_and_heavy_overlap_dropped(self):
        clips = [
            {'start_time': '00:00:12', 'end_time': '00:00:14', 'title': 'sliver'},
            {'start_time': '00:07:00', 'end_time': '00:08:37', 'title': 'keep A'},
            {'start_time': '00:06:38', 'end_time': '00:07:53', 'title': 'overlaps A'},
            {'start_time': '00:10:00', 'end_time': '00:10:30', 'title': 'keep B'},
        ]
        out = _story_clip_hygiene(clips)
        titles = [c['title'] for c in out]
        assert titles == ['keep A', 'keep B'], titles

    def test_unparseable_timecodes_kept(self):
        clips = [{'start_time': None, 'end_time': None, 'title': 'odd'}]
        assert _story_clip_hygiene(clips) == clips


class TestLatexArtifacts:
    def test_rightarrow_normalized(self):
        out = _clean_chat_response(
            'Motivation $\\rightarrow$ Conflict \\rightarrow Solution.')
        assert '→' in out and 'rightarrow' not in out and '$' not in out


class TestConversationalCardsEnforcement:
    def test_correction_message_with_cards_gets_duration_enforced(self):
        # "that should be a 60 second story not 7:23" classifies
        # conversational (no extractive verb) but the reply carries cards
        # — the parsed 60s target must be enforced on them.
        segs = []
        t = 0
        for i in range(40):
            segs.append({'start': t, 'end': t + 25,
                         'start_formatted': f'00:{t//60:02d}:{t%60:02d}.000',
                         'text': f'Segment {i} about the land and memory.',
                         'speaker': 'Mae'})
            t += 30
        reply = ('Suggested structure:\n'
                 '[CLIP: start=00:00:30 end=00:01:10 title="Beat one"]\n'
                 '[CLIP: start=00:03:00 end=00:04:00 title="Beat two"]\n'
                 '[CLIP: start=00:08:00 end=00:09:30 title="Beat three"]\n')
        with patch.object(ai_analysis, '_call_ai_chat', return_value=reply):
            out = ai_analysis.chat_about_transcript(
                {'segments': segs}, 'that should be a 60 second story not 7:23',
                history=[], project_name='T')
        import re
        total = 0
        for m in re.findall(r'\[CLIP:[^\]]*\]', out):
            s = re.search(r'start=([\d:]+)', m).group(1)
            e = re.search(r'end=([\d:]+)', m).group(1)
            def sec(tc):
                p = [int(x) for x in tc.split(':')]
                return p[0] * 3600 + p[1] * 60 + p[2] if len(p) == 3 else p[0] * 60 + p[1]
            total += sec(e) - sec(s)
        # A plain TARGET (not a bound) keeps the designed floor-break
        # semantics: trim stops rather than under-delivering below 0.8x.
        # The pin here is that enforcement RAN AT ALL on a conversational-
        # classified reply — 190s of cards must not ship untouched.
        assert total < 190, (total, out)
        assert out.count('[CLIP:') == 2, out
