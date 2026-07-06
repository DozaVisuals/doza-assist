"""Latency-path contracts from the 1.0.33 chat speed bundle.

Pins the mechanics the speed work relies on:
  - PREFIX-CACHE HYGIENE: the transcript message is byte-stable per
    project (per-query excerpts must never touch it — that broke Ollama's
    KV prefix reuse and forced a full transcript re-prefill every turn);
    excerpts ride the FINAL user turn ahead of the question + reminder.
  - SALVAGE DIET: with >=2 matched paragraphs the salvage extractor gets
    the excerpt menu, not a second copy of the whole transcript.
  - PRE-WARM: prewarm_chat_context sends the stable prefix with
    num_predict=1 and honors its cooldown.
  - REPLY BUDGET (flag-gated): conversational asks cap at 2048 only when
    DOZA_ADAPTIVE_REPLY_BUDGET=1.
  - STREAMING SYNTHESIS: the >60-min conversational stream yields real
    token events (it used to emit only progress + done).
"""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ai_analysis  # noqa: E402


def _segments(n=6, seg_len=25, gap=5, text='Mae talks about the chestnut trees.'):
    segs = []
    t = 0
    for i in range(n):
        segs.append({
            'start': t,
            'end': t + seg_len,
            'start_formatted': f'{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}.000',
            'text': f'{text} (part {i})',
            'speaker': 'Mae',
        })
        t += seg_len + gap
    return segs


def _build(message, excerpts_block, history=None):
    segs = _segments()
    return ai_analysis._build_chat_messages(
        message, history or [], 'Interview', segs,
        formatted='[00:00:00-00:00:25] Mae: hello.',
        analysis_block='', relevant_excerpts_block=excerpts_block,
        profile_id=None,
    )


class TestPrefixCacheHygiene:
    def test_transcript_message_stable_across_queries(self):
        _, msgs_a = _build('what about the trees?', 'RELEVANT EXCERPTS:\n[x] Mae: trees.')
        _, msgs_b = _build('how should I open?', '')
        # First user message = transcript message in both (no style block).
        assert msgs_a[0]['content'] == msgs_b[0]['content']
        assert 'RELEVANT EXCERPTS' not in msgs_a[0]['content']

    def test_excerpts_lead_the_final_turn(self):
        block = 'RELEVANT EXCERPTS:\n[00:00:10] Mae: the chestnut line.'
        _, msgs = _build('what about the trees?', block)
        final = msgs[-1]['content']
        assert final.index('RELEVANT EXCERPTS') == 0
        assert final.index('the chestnut line') < final.index('what about the trees?')
        assert 'FINAL REMINDER' in final
        assert final.index('what about the trees?') < final.index('FINAL REMINDER')

    def test_no_excerpts_keeps_final_turn_shape(self):
        _, msgs = _build('how should I open the piece?', '')
        assert msgs[-1]['content'] == (
            'how should I open the piece?\n\n' + ai_analysis._FINAL_REMINDER)


class TestSalvageDiet:
    def _run(self, matched):
        captured = {}

        def fake_call(system_message, messages, num_ctx=32768, **kwargs):
            captured['prompt'] = messages[0]['content']
            return ''  # force the deterministic fallback afterward

        segs = _segments()
        with patch.object(ai_analysis, '_call_ai_chat', side_effect=fake_call):
            ai_analysis._salvage_clips_if_missing(
                'prose with no markers',
                'FULL-TRANSCRIPT-SENTINEL ' * 50,
                segs,
                matched_paragraphs=matched,
                user_message='pull the tree moments',
            )
        return captured.get('prompt', '')

    def test_menu_replaces_transcript_when_matches_exist(self):
        matched = [
            {'start': 10.0, 'end': 40.0, 'text': 'the chestnut menu line one'},
            {'start': 60.0, 'end': 90.0, 'text': 'the chestnut menu line two'},
        ]
        prompt = self._run(matched)
        assert 'FULL-TRANSCRIPT-SENTINEL' not in prompt
        assert 'chestnut menu line one' in prompt

    def test_thin_matches_fall_back_to_full_transcript(self):
        matched = [{'start': 10.0, 'end': 40.0, 'text': 'only one line'}]
        prompt = self._run(matched)
        assert 'FULL-TRANSCRIPT-SENTINEL' in prompt


class TestPrewarm:
    def test_prewarm_sends_stable_prefix_with_one_token_budget(self):
        captured = {}

        def fake_call(system_message, messages, num_ctx=32768, **kwargs):
            captured['kwargs'] = kwargs
            captured['messages'] = messages
            return 'ok'

        ai_analysis._PREWARM_STATE.clear()
        transcript = {'segments': _segments()}
        with patch.object(ai_analysis, '_call_ai_chat', side_effect=fake_call), \
                patch.object(ai_analysis, '_ollama_is_active', return_value=True):
            assert ai_analysis.prewarm_chat_context(
                transcript, project_name='Warm Test') is True
        assert captured['kwargs'].get('num_predict') == 1
        assert captured['kwargs'].get('timing_tag') == 'prewarm'
        # Transcript rides the prefix; no FINAL REMINDER on the throwaway turn.
        joined = '\n'.join(m['content'] for m in captured['messages'])
        assert 'TRANSCRIPT:' in joined
        assert 'FINAL REMINDER' not in joined

    def test_prewarm_cooldown_reports_warm_without_second_call(self):
        # A cooldown hit must return True ("prefix is warm") with NO second
        # LLM call — returning False here made the app.py wrapper fall back
        # to the plain 2048-ctx warmup, which flipped the runner's context
        # size and destroyed the still-warm prefix (review defect #1).
        ai_analysis._PREWARM_STATE.clear()
        transcript = {'segments': _segments()}
        calls = []
        with patch.object(ai_analysis, '_call_ai_chat',
                          side_effect=lambda *a, **k: calls.append(1) or 'ok'), \
                patch.object(ai_analysis, '_ollama_is_active', return_value=True):
            assert ai_analysis.prewarm_chat_context(
                transcript, project_name='Cooldown Test') is True
            assert ai_analysis.prewarm_chat_context(
                transcript, project_name='Cooldown Test') is True
        assert len(calls) == 1

    def test_prewarm_language_change_busts_cooldown(self):
        # The language directive rides the system message — a different
        # output language is a different prefix and must re-warm.
        ai_analysis._PREWARM_STATE.clear()
        transcript = {'segments': _segments()}
        calls = []
        with patch.object(ai_analysis, '_call_ai_chat',
                          side_effect=lambda *a, **k: calls.append(1) or 'ok'), \
                patch.object(ai_analysis, '_ollama_is_active', return_value=True):
            ai_analysis.prewarm_chat_context(
                transcript, project_name='Lang Test')
            ai_analysis.prewarm_chat_context(
                transcript, project_name='Lang Test', output_language='no')
        assert len(calls) == 2

    def test_prewarm_skips_long_projects(self):
        ai_analysis._PREWARM_STATE.clear()
        segs = [{'start': 0, 'end': ai_analysis._LONG_CHAT_SECONDS + 100,
                 'text': 'long', 'speaker': 'Mae'}]
        with patch.object(ai_analysis, '_ollama_is_active', return_value=True):
            assert ai_analysis.prewarm_chat_context(
                {'segments': segs}, project_name='Long Test') is False


class TestReplyBudgetFlag:
    def test_flag_off_returns_empty(self):
        os.environ.pop('DOZA_ADAPTIVE_REPLY_BUDGET', None)
        assert ai_analysis._chat_reply_budget_kwargs('whats this all about') == {}

    def test_flag_on_caps_conversational_only(self):
        os.environ['DOZA_ADAPTIVE_REPLY_BUDGET'] = '1'
        try:
            assert ai_analysis._chat_reply_budget_kwargs(
                'whats this all about') == {'num_predict': 2048}
            # Extractive and content-lookup asks keep the full budget.
            assert ai_analysis._chat_reply_budget_kwargs(
                'find me the strongest moments') == {}
            assert ai_analysis._chat_reply_budget_kwargs(
                'what does she say about the river?') == {}
        finally:
            os.environ.pop('DOZA_ADAPTIVE_REPLY_BUDGET', None)


class TestStreamingSynthesis:
    def test_synthesis_stream_yields_tokens(self):
        def fake_stream(system_message, messages, num_ctx=32768, **kwargs):
            for piece in ('The story ', 'is about ', 'the river.'):
                yield piece

        segs = _segments()
        with patch.object(ai_analysis, '_call_ai_chat_stream',
                          side_effect=fake_stream):
            events = list(ai_analysis._chat_layer2_conversational_synthesis_stream(
                'whats the story here?', [], 'Long Interview', segs,
                None, None, None,
            ))
        kinds = [ev for ev, _ in events]
        assert 'token' in kinds
        assert kinds[-1] == 'done'
        done_payload = events[-1][1]
        assert 'the river' in done_payload

    def test_synthesis_stream_prompt_matches_blocking_variant(self):
        seen = {'stream': None, 'block': None}

        def fake_stream(system_message, messages, num_ctx=32768, **kwargs):
            seen['stream'] = (system_message, [m['content'] for m in messages])
            yield 'x'

        def fake_call(system_message, messages, num_ctx=32768, **kwargs):
            seen['block'] = (system_message, [m['content'] for m in messages])
            return 'x'

        segs = _segments()
        args = ('whats the story here?', [], 'Long Interview', segs,
                None, None, None)
        with patch.object(ai_analysis, '_call_ai_chat_stream',
                          side_effect=fake_stream):
            list(ai_analysis._chat_layer2_conversational_synthesis_stream(*args))
        with patch.object(ai_analysis, '_call_ai_chat', side_effect=fake_call):
            ai_analysis._chat_layer2_conversational_synthesis(*args)
        assert seen['stream'] == seen['block']
