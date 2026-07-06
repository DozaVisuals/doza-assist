"""Clip-count adherence: deterministic top-up + plural-ask minimum.

Live tester bug (screenshot, 2026-07): two chat turns on one interview —
  1. "What are the strongest emotional moments in this interview?" —
     the reply's prose said "I've pulled three moments for you" but ONE
     clip card rendered. Root cause: clip-seeking questions phrased as
     questions routed CONVERSATIONAL, so no salvage, no count
     enforcement, and no grounding ran — the card count was whatever the
     model happened to emit.
  2. "Give me 5 more" — THREE cards. Root cause: _enforce_clip_count was
     trim-only ("if len(clips) <= target: return text"), so model
     under-delivery passed through untouched.

These tests lock in the fix: clip-noun questions route extractive (the
routing itself is covered in test_chat_query_intent), explicit counts top
UP from the ranked pool, plural asks guarantee a minimum, moments already
shown earlier in the conversation are never re-issued, conversational
questions grow no cards, and a thin pool ships the honest count.
"""

import os
import re
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ai_analysis  # noqa: E402
from ai_analysis import (  # noqa: E402
    _enforce_clip_count,
    _exclude_shown_candidates,
    _history_clip_spans,
    _is_conversational_query,
    _layer2_final_top_k,
    _plural_clip_minimum,
    _tc_to_seconds,
)


def _marker_spans(text):
    """[(start_sec, end_sec)] for every fully-formed marker in ``text``."""
    return [
        (_tc_to_seconds(s), _tc_to_seconds(e))
        for s, e in re.findall(r'\[CLIP:[^\]]*?start=([\d:]+)[^\]]*?end=([\d:]+)', text)
    ]


def _assert_pairwise_disjoint(spans):
    for i, (s1, e1) in enumerate(spans):
        for s2, e2 in spans[i + 1:]:
            overlap = max(0.0, min(e1, e2) - max(s1, s2))
            shorter = min(e1 - s1, e2 - s2)
            assert overlap / max(1.0, shorter) <= 0.5, (
                f'spans overlap: ({s1},{e1}) vs ({s2},{e2})'
            )


class TestPluralClipMinimum:
    def test_plural_clip_nouns_return_the_minimum(self):
        assert _plural_clip_minimum("strongest emotional moments") == 3
        assert _plural_clip_minimum("show me the best quotes") == 3
        assert _plural_clip_minimum("any good soundbites?") == 3
        assert _plural_clip_minimum("what are the highlights?") == 3

    def test_singular_and_nounless_asks_return_none(self):
        assert _plural_clip_minimum("the best moment") is None
        assert _plural_clip_minimum("give me 5 more") is None
        assert _plural_clip_minimum("what's the emotional arc?") is None
        assert _plural_clip_minimum("") is None
        assert _plural_clip_minimum(None) is None


class TestHistoryClipSpans:
    def test_extracts_spans_from_assistant_turns(self):
        history = [
            {'role': 'user', 'content': 'strongest moments?'},
            {'role': 'assistant', 'content':
                'Here.\n[CLIP: start=00:01:00 end=00:01:30 title="A"]\nWhy A.'},
        ]
        assert _history_clip_spans(history) == [(60.0, 90.0, None)]

    def test_project_field_becomes_the_group(self):
        history = [{'role': 'assistant', 'content':
                    '[CLIP: start=00:02:00 end=00:02:30 project="Intervju B" '
                    'title="Stormen"]'}]
        assert _history_clip_spans(history) == [(120.0, 150.0, 'Intervju B')]

    def test_markerless_and_malformed_turns_are_skipped(self):
        history = [
            {'role': 'assistant', 'content': 'no markers here'},
            {'role': 'assistant', 'content': '[CLIP: title="no times"]'},
            'not a dict',
            {'role': 'assistant'},  # no content
        ]
        assert _history_clip_spans(history) == []
        assert _history_clip_spans(None) == []
        assert _history_clip_spans([]) == []


def _pool(n=6, dur=30, offset=100.0):
    """Ranked pool in the Layer-1 matched-paragraph shape."""
    return [
        {'start': offset + i * (dur + 10), 'end': offset + i * (dur + 10) + dur,
         'text': f'paragraph {i} about the emotional topic at hand'}
        for i in range(n)
    ]


class TestEnforceClipCountTopUp:
    ONE_MARKER = 'One pick.\n[CLIP: start=00:00:00 end=00:00:20 title="Opening"]'

    def test_under_delivery_tops_up_to_target(self):
        out = _enforce_clip_count(self.ONE_MARKER, 3, candidates=_pool())
        assert len(_marker_spans(out)) == 3
        # Prose and the model's own marker survive.
        assert out.startswith('One pick.')
        assert 'start=00:00:00 end=00:00:20' in out

    def test_topup_skips_overlaps_with_emitted_markers(self):
        text = '[CLIP: start=00:01:40 end=00:02:10 title="Already here"]'  # 100-130
        out = _enforce_clip_count(text, 3, candidates=_pool())
        spans = _marker_spans(out)
        assert len(spans) == 3
        assert spans.count((100.0, 130.0)) == 1  # pool duplicate not re-added
        _assert_pairwise_disjoint(spans)

    def test_topup_skips_history_spans(self):
        # First pool candidate (100-130) was already shown last turn — the
        # top-up must jump past it, never re-issuing a seen moment.
        out = _enforce_clip_count(
            self.ONE_MARKER, 3, candidates=_pool(),
            exclude_spans=[(100.0, 130.0, None)],
        )
        spans = _marker_spans(out)
        assert len(spans) == 3
        assert (100.0, 130.0) not in spans

    def test_pool_exhaustion_ships_the_honest_count(self):
        out = _enforce_clip_count(self.ONE_MARKER, 5, candidates=_pool(n=2))
        assert len(_marker_spans(out)) == 3  # 1 emitted + 2 available, no padding
        _assert_pairwise_disjoint(_marker_spans(out))

    def test_min_count_tops_up_but_never_trims(self):
        # Under: plural-ask floor fills to 3.
        out = _enforce_clip_count(
            self.ONE_MARKER, None, candidates=_pool(), min_count=3)
        assert len(_marker_spans(out)) == 3
        # Over: a model that volunteers five keeps all five.
        five = '\n'.join(
            f'[CLIP: start=00:{i:02d}:00 end=00:{i:02d}:20 title="M{i}"]'
            for i in range(5)
        )
        assert _enforce_clip_count(five, None, candidates=_pool(),
                                   min_count=3) == five

    def test_explicit_target_ignores_min_count(self):
        # target=1 with min_count=3: the explicit count owns the reply.
        three = '\n'.join(
            f'[CLIP: start=00:{i:02d}:00 end=00:{i:02d}:20 title="M{i}"]'
            for i in range(3)
        )
        out = _enforce_clip_count(three, 1, candidates=_pool(), min_count=3)
        assert len(_marker_spans(out)) == 1

    def test_trim_behavior_is_unchanged(self):
        three = (
            'Here are three.\n\n'
            '[CLIP: start=00:00:00 end=00:01:00 title="A"]\nWhy A.\n\n'
            '[CLIP: start=00:02:00 end=00:03:00 title="B"]\nWhy B.\n\n'
            '[CLIP: start=00:04:00 end=00:05:00 title="C"]\nWhy C.'
        )
        out = _enforce_clip_count(three, 1)
        assert out.count('[CLIP:') == 1
        assert 'Why A.' in out and 'Why B.' not in out

    def test_no_target_no_min_returns_unchanged(self):
        assert _enforce_clip_count(self.ONE_MARKER, None) == self.ONE_MARKER
        assert _enforce_clip_count(self.ONE_MARKER, 0) == self.ONE_MARKER
        assert _enforce_clip_count('', 3, candidates=_pool()) == ''

    def test_no_candidates_keeps_trim_only_shape(self):
        # Historical callers (two positional args) see byte-identical
        # behavior: under-delivery passes through when there is no pool.
        assert _enforce_clip_count(self.ONE_MARKER, 3) == self.ONE_MARKER

    def test_long_pool_spans_are_capped_to_clip_size(self):
        pool = [{'start': 200.0, 'end': 350.0, 'text': 'one very long story beat'}]
        out = _enforce_clip_count(self.ONE_MARKER, 2, candidates=pool)
        assert (200.0, 245.0) in _marker_spans(out)  # 150s span capped to 45s

    def test_group_key_keeps_per_project_timelines_apart(self):
        # Same LOCAL span on two projects is different footage: a marker on
        # project B must not block project A's candidate at equal times.
        text = ('[CLIP: start=00:01:40 end=00:02:10 project="B" '
                'title="B opening"]')
        pool = [
            {'start': 100.0, 'end': 130.0, 'project_name': 'A',
             'text': 'project A parallel moment'},
        ]
        out = _enforce_clip_count(text, 2, candidates=pool,
                                  group_key='project_name')
        assert len(_marker_spans(out)) == 2
        # Without group_key the equal span is treated as one timeline.
        out_ungrouped = _enforce_clip_count(text, 2, candidates=pool)
        assert len(_marker_spans(out_ungrouped)) == 1


class TestLayer2CountDerivation:
    def test_chat_side_count_parses_when_layer2_parser_misses(self):
        # "give me 5 more" has no clip noun — only the chat-side parser
        # reads it. Layer 2 must honor it.
        assert _layer2_final_top_k('give me 5 more') == 5

    def test_layer2_parser_still_wins_its_own_shapes(self):
        assert _layer2_final_top_k('find me 2 clips about the bakery') == 2
        assert _layer2_final_top_k('the best clip') == 1

    def test_no_count_keeps_default_top_k(self):
        assert _layer2_final_top_k('find the best moments') == \
            ai_analysis._CHAT_TOP_K_CLIPS
        assert _layer2_final_top_k("what's the emotional arc?") == \
            ai_analysis._CHAT_TOP_K_CLIPS


class TestExcludeShownCandidates:
    HISTORY = [{'role': 'assistant', 'content':
                '[CLIP: start=00:01:00 end=00:01:30 title="Seen"]'}]
    CANDS = [
        {'start_sec': 60.0, 'end_sec': 90.0, 'title': 'Seen', 'score': 9},
        {'start_sec': 300.0, 'end_sec': 330.0, 'title': 'Fresh', 'score': 8},
    ]

    def test_more_ask_drops_already_shown_candidates(self):
        out = _exclude_shown_candidates(self.CANDS, 'give me 5 more', self.HISTORY)
        assert [c['title'] for c in out] == ['Fresh']

    def test_fresh_ask_keeps_the_full_pool(self):
        # A re-asked fresh question may legitimately re-find the same
        # best moment — only "more"-style follow-ups exclude history.
        out = _exclude_shown_candidates(
            self.CANDS, 'find the best moments', self.HISTORY)
        assert out == self.CANDS

    def test_empty_history_keeps_the_full_pool(self):
        assert _exclude_shown_candidates(self.CANDS, 'give me 5 more', []) \
            == self.CANDS
        assert _exclude_shown_candidates(self.CANDS, 'give me 5 more', None) \
            == self.CANDS


# ── End-to-end: the exact screenshot cases through the real pipeline ───────

def _transcript(n=20, text='she gets emotional about topic {i} here'):
    segments = []
    for i in range(n):
        start = i * 30.0
        segments.append({
            'start': start, 'end': start + 30.0,
            'start_formatted':
                f'{int(start) // 3600:02d}:{(int(start) % 3600) // 60:02d}:{int(start) % 60:02d}',
            'text': text.format(i=i),
            'speaker': 'A',
        })
    return {'segments': segments}


# One high-narrative-score vector spanning the whole interview: the count
# top-up's fallback pool for keyword-less follow-ups ("give me 5 more").
def _vectors(tc_out='00:10:00'):
    return [{'timecode_in': '00:00:00', 'timecode_out': tc_out,
             'narrative_score': 'high', 'theme_tags': []}]


class TestScreenshotCase1PluralAsk:
    """Plural ask + model emitting one marker + prose → at least three
    cards after the deterministic top-up, on the extractive route."""

    MESSAGE = "What are the strongest emotional moments in this interview?"
    REPLY = (
        "I've pulled three moments for you.\n"
        '[CLIP: start=00:00:00 end=00:00:20 title="Opening emotional beat"]'
    )

    def test_routes_extractive(self):
        assert not _is_conversational_query(self.MESSAGE)

    def test_plural_ask_tops_up_to_the_minimum(self):
        with patch.object(ai_analysis, '_call_ai_chat', return_value=self.REPLY):
            out = ai_analysis.chat_about_transcript(_transcript(), self.MESSAGE)
        spans = _marker_spans(out)
        assert len(spans) >= 3, f'expected >=3 cards, got {len(spans)}: {out!r}'
        assert "I've pulled three moments for you." in out
        assert (0.0, 20.0) in spans  # the model's own pick survives
        _assert_pairwise_disjoint(spans)


class TestScreenshotCase2GiveMeFiveMore:
    """Explicit count follow-up: 5 asked, model emits 3, one clip already
    shown last turn → exactly five cards, all NEW moments."""

    MESSAGE = 'Give me 5 more'
    HISTORY = [
        {'role': 'user',
         'content': 'What are the strongest emotional moments in this interview?'},
        {'role': 'assistant',
         'content': ('Here is one.\n'
                     '[CLIP: start=00:00:00 end=00:00:30 title="Opening"]')},
    ]
    REPLY = (
        'Three more.\n'
        '[CLIP: start=00:01:00 end=00:01:30 title="Second"]\n'
        '[CLIP: start=00:03:00 end=00:03:30 title="Third"]\n'
        '[CLIP: start=00:05:00 end=00:05:30 title="Fourth"]'
    )

    def _run(self):
        with patch.object(ai_analysis, '_call_ai_chat', return_value=self.REPLY):
            return ai_analysis.chat_about_transcript(
                _transcript(), self.MESSAGE, history=self.HISTORY,
                segment_vectors=_vectors(),
            )

    def test_exactly_five_new_cards(self):
        out = self._run()
        spans = _marker_spans(out)
        assert len(spans) == 5, f'expected exactly 5 cards: {out!r}'
        # None may overlap the clip shown in the previous turn (0-30)...
        for s, e in spans:
            overlap = max(0.0, min(e, 30.0) - max(s, 0.0))
            assert overlap / max(1.0, min(e - s, 30.0)) <= 0.5, (
                f'({s},{e}) re-issues the history clip'
            )
        # ...nor each other.
        _assert_pairwise_disjoint(spans)

    def test_stream_path_matches(self):
        def fake_stream(system_message, messages, num_ctx=32768):
            yield self.REPLY[:15]
            yield self.REPLY[15:]

        with patch.object(ai_analysis, '_call_ai_chat_stream',
                          side_effect=fake_stream):
            events = list(ai_analysis.chat_about_transcript_stream(
                _transcript(), self.MESSAGE, history=self.HISTORY,
                segment_vectors=_vectors(),
            ))
        assert events[-1][0] == 'done'
        spans = _marker_spans(events[-1][1])
        assert len(spans) == 5, f'stream done payload: {events[-1][1]!r}'
        _assert_pairwise_disjoint(spans)

    def test_pool_exhaustion_ships_the_honest_count(self):
        # A transcript with NO carryover-matchable keywords and a single
        # non-adjacent high-score window → the fallback pool has exactly
        # ONE fresh candidate. 3 emitted + 1 topped up = 4, never padded
        # with overlaps (or adjacent continuations) to fake a 5.
        with patch.object(ai_analysis, '_call_ai_chat', return_value=self.REPLY):
            out = ai_analysis.chat_about_transcript(
                _transcript(text='she talks about topic {i} here'),
                self.MESSAGE, history=self.HISTORY,
                segment_vectors=[{'timecode_in': '00:02:00',
                                  'timecode_out': '00:02:30',
                                  'narrative_score': 'high', 'theme_tags': []}],
            )
        spans = _marker_spans(out)
        assert len(spans) == 4, f'expected the honest 4: {out!r}'
        _assert_pairwise_disjoint(spans)

    def test_adjacent_window_is_a_continuation_not_a_new_clip(self):
        # The only fallback window (30-60s) directly CONTINUES the clip
        # shown last turn (0-30s) — the adjacency rule refuses it, and the
        # reply ships the honest 3 the model emitted.
        with patch.object(ai_analysis, '_call_ai_chat', return_value=self.REPLY):
            out = ai_analysis.chat_about_transcript(
                _transcript(text='she talks about topic {i} here'),
                self.MESSAGE, history=self.HISTORY,
                segment_vectors=[{'timecode_in': '00:00:30',
                                  'timecode_out': '00:01:00',
                                  'narrative_score': 'high', 'theme_tags': []}],
            )
        spans = _marker_spans(out)
        assert len(spans) == 3, f'continuation must not pad the count: {out!r}'
        assert (30.0, 60.0) not in spans


class TestTopupPoolQuality:
    """H2/H13 — the keyword-less top-up pool must be PRECISE: one merged
    candidate per curated vector window (never raw 2-8s whisper slivers),
    ranked high-before-medium, with adjacency exclusion (no continuations
    of shown clips, no consecutive slices of one window) and theme
    carryover from the last clip-producing turn. A padded sliver card is
    worse than an honest under-count."""

    def test_fallback_is_one_candidate_per_window_ranked(self):
        segs = _transcript()['segments']
        vecs = [
            {'timecode_in': '00:01:00', 'timecode_out': '00:01:30',
             'narrative_score': 'medium', 'theme_tags': []},
            {'timecode_in': '00:05:00', 'timecode_out': '00:05:30',
             'narrative_score': 'high', 'theme_tags': []},
            {'timecode_in': '00:07:00', 'timecode_out': '00:07:40',
             'narrative_score': 'low', 'theme_tags': []},
        ]
        pool = ai_analysis._vector_window_candidates(segs, vecs)
        # One candidate per window, high ranked before medium, low absent.
        assert [(c['start'], c['end']) for c in pool] == \
            [(300.0, 330.0), (60.0, 90.0)]
        # The candidate carries transcript text so a real title derives.
        assert pool[0]['text']

    def test_no_consecutive_slices_of_one_window(self):
        # Two 30s shown clips + a fallback whose windows continue them:
        # the top-up appends nothing adjacent — no 4-second sliver chains.
        text = ('Two.\n'
                '[CLIP: start=00:02:00 end=00:02:20 title="A"]\n'
                '[CLIP: start=00:05:00 end=00:05:15 title="B"]')
        slivers = [
            {'start': 140.0, 'end': 144.0, 'text': 'sliver one'},
            {'start': 145.0, 'end': 149.0, 'text': 'sliver two'},
            {'start': 315.0, 'end': 319.0, 'text': 'sliver three'},
            {'start': 200.0, 'end': 230.0, 'text': 'a genuinely new moment'},
        ]
        out = _enforce_clip_count(text, 5, candidates=slivers)
        spans = _marker_spans(out)
        assert (200.0, 230.0) in spans
        assert len(spans) == 3  # 2 emitted + 1 genuine; slivers refused
        assert 'sliver' not in out

    def test_theme_carryover_from_last_clip_producing_turn(self):
        segs = [
            {'start': i * 30.0, 'end': i * 30.0 + 30.0,
             'text': ('she cries about the fire here' if i % 4 == 0
                      else 'neutral segment filler text')}
            for i in range(20)
        ]
        history = [
            {'role': 'user', 'content': 'strongest moments about the fire?'},
            {'role': 'assistant',
             'content': '[CLIP: start=00:00:00 end=00:00:30 title="Fire"]'},
        ]
        pool = ai_analysis._count_topup_pool(
            [], segs, None, message='give me 5 more', history=history)
        assert pool, 'carryover must seed a pool from the prior turn'
        assert any('fire' in (c.get('text') or '') for c in pool)

    def test_carryover_skips_turns_that_produced_no_clips(self):
        segs = [{'start': 0.0, 'end': 30.0, 'text': 'about the storm'}]
        history = [
            {'role': 'user', 'content': 'moments about the storm?'},
            {'role': 'assistant',
             'content': '[CLIP: start=00:00:00 end=00:00:30 title="S"]'},
            {'role': 'user', 'content': 'thanks, that was helpful'},
            {'role': 'assistant', 'content': 'Glad to help!'},
        ]
        assert ai_analysis._carryover_clip_queries(history) == \
            ['moments about the storm?']


class TestMoreClipsFollowUpRegex:
    """H3/H14 — _MORE_CLIPS_RE anchors to follow-up syntax. Bare content
    words must not silently drop a fresh ask's best candidates; the
    message lists are the verifiers' exact repros."""

    @pytest.mark.parametrize("message", [
        "any quotes about moving to new york?",
        "pull clips where they have different opinions",
        "clips of the new house tour",
        "what moments show her other siblings?",
        "find the moment she talks about her brother and everything else "
        "she lost",
        "her new job",
        "moments where she felt different from the others",
        "clips about her other siblings",
        "the new house",
        "find me 3 clips about moving to New York",
    ])
    def test_content_words_do_not_trigger_exclusion(self, message):
        assert not ai_analysis._MORE_CLIPS_RE.search(message.lower())

    @pytest.mark.parametrize("message", [
        "give me 5 more",
        "what other moments are there?",
        "show me others",
        "give me another",
        "one more",
        "a few more please",
        "more moments like that",
        "anything else?",
        "what else do you have?",
        "give me another 2 minutes of clips",
        "a couple extra options",
        "any others?",
        "more of those",
    ])
    def test_follow_up_shapes_trigger_exclusion(self, message):
        assert ai_analysis._MORE_CLIPS_RE.search(message.lower())

    def test_fresh_new_york_ask_keeps_its_best_candidate(self):
        # The H14 end-to-end shape: a prior generic turn showed a clip
        # overlapping the NY moment; the fresh themed ask must keep it.
        history = [{'role': 'assistant', 'content':
                    '[CLIP: start=00:01:40 end=00:02:08 title="Best"]'}]
        cands = [{'start_sec': 100.0, 'end_sec': 128.0, 'title': 'NY',
                  'score': 9}]
        out = _exclude_shown_candidates(
            cands, 'find me 3 clips about moving to New York', history)
        assert out == cands


class TestCountWinsOverJointDuration:
    """H5 — when BOTH a count and a duration parse ("give me 5 clips,
    2 minutes of selects"), the explicit count owns the card count and
    the duration degrades to a soft bound, on Layer 1 AND Layer 2."""

    MESSAGE = 'give me 5 clips, 2 minutes of selects'

    def test_both_parse(self):
        assert ai_analysis.parse_target_duration_seconds(self.MESSAGE) == 120.0
        assert ai_analysis._detect_explicit_clip_count(self.MESSAGE) == 5

    def test_layer1_count_owns_the_reply(self):
        reply = ('Here.\n'
                 '[CLIP: start=00:00:00 end=00:00:30 title="One"]')
        with patch.object(ai_analysis, '_call_ai_chat', return_value=reply):
            out = ai_analysis.chat_about_transcript(
                _transcript(text='selects about topic {i} emotional'),
                self.MESSAGE)
        spans = _marker_spans(out)
        assert len(spans) == 5, f'count=5 must win over duration hint: {out!r}'

    def test_layer2_final_top_k_uses_the_count(self):
        assert ai_analysis._layer2_explicit_count(self.MESSAGE) == 5
        assert ai_analysis._layer2_explicit_count(
            'find me 4 clips for a 90 second teaser') == 4
        # Count-less duration asks still defer to the duration hint.
        assert ai_analysis._layer2_explicit_count(
            'give me 6 minutes of the best moments') is None


class TestDurationTopupHistoryExclusion:
    """H6/H17 — the duration top-up must not re-issue (or continue)
    already-shown clips: _enforce_duration_target takes exclude_spans and
    the Layer-1 call sites thread _history_clip_spans through."""

    def test_enforce_duration_target_honors_exclude_spans(self):
        pool = [
            {'start': 0.0, 'end': 40.0, 'text': 'shown last turn already'},
            {'start': 100.0, 'end': 140.0, 'text': 'brand new footage one'},
            {'start': 200.0, 'end': 240.0, 'text': 'brand new footage two'},
            {'start': 300.0, 'end': 340.0, 'text': 'brand new footage x'},
        ]
        out = ai_analysis._enforce_duration_target(
            'no markers', 120.0, pool, exclude_spans=[(0.0, 40.0, None)])
        spans = _marker_spans(out)
        assert (0.0, 40.0) not in spans, 'shown clip re-issued'
        assert (100.0, 140.0) in spans

    def test_layer1_duration_ask_excludes_history(self):
        # "give me another 2 minutes of clips": extractive, duration=120,
        # and the pool's top span was shown last turn — the top-up must
        # skip it.
        message = 'give me another 2 minutes of clips'
        assert ai_analysis.parse_target_duration_seconds(message) == 120.0
        history = [{'role': 'assistant', 'content':
                    'Prev.\n[CLIP: start=00:00:00 end=00:00:30 title="Seen"]'}]
        reply = 'Fresh prose only, no markers.'
        with patch.object(ai_analysis, '_call_ai_chat', return_value=reply):
            out = ai_analysis.chat_about_transcript(
                _transcript(text='clips about topic {i} emotional'),
                message, history=history)
        for s, e in _marker_spans(out):
            overlap = max(0.0, min(e, 30.0) - max(s, 0.0))
            assert overlap / max(1.0, min(e - s, 30.0)) <= 0.5, (
                f'({s},{e}) re-issues the history clip'
            )


class TestReemittedHistoryClipsDoNotCount:
    """H10 — on "more"-style asks, markers the model re-emits from
    history are duplicates: they are removed (with their prose) and do
    not count toward the target, so "5 more" ships 5 genuinely new
    cards (pool permitting)."""

    def test_reemitted_markers_are_dropped_and_replaced(self):
        reply = (
            'Five more.\n'
            '[CLIP: start=00:00:00 end=00:00:30 title="Old1"]\nWhy old1.\n\n'
            '[CLIP: start=00:01:00 end=00:01:30 title="Old2"]\nWhy old2.\n\n'
            '[CLIP: start=00:05:00 end=00:05:30 title="New1"]\nWhy new1.'
        )
        pool = [
            {'start': 420.0 + i * 45, 'end': 450.0 + i * 45,
             'text': f'fresh material number {i} right here'}
            for i in range(8)
        ]
        out = _enforce_clip_count(
            reply, 5, candidates=pool,
            exclude_spans=[(0.0, 30.0, None), (60.0, 90.0, None)],
            drop_reemitted=True,
        )
        spans = _marker_spans(out)
        assert len(spans) == 5
        assert (0.0, 30.0) not in spans and (60.0, 90.0) not in spans
        # The dropped markers' prose went with them; the kept one stayed.
        assert 'Why old1.' not in out and 'Why old2.' not in out
        assert 'Why new1.' in out

    def test_all_repeats_reply_is_fully_replaced(self):
        # The trim-path variant: a model re-emitting >= target old clips
        # must not ship all-repeats.
        reply = '\n'.join(
            f'[CLIP: start=00:0{i}:00 end=00:0{i}:30 title="Old{i}"]'
            for i in range(2)
        )
        pool = [
            {'start': 400.0 + i * 45, 'end': 430.0 + i * 45,
             'text': f'new material {i} for the ask'}
            for i in range(4)
        ]
        out = _enforce_clip_count(
            reply, 2, candidates=pool,
            exclude_spans=[(0.0, 30.0, None), (60.0, 90.0, None)],
            drop_reemitted=True,
        )
        spans = _marker_spans(out)
        assert len(spans) == 2
        assert (0.0, 30.0) not in spans and (60.0, 90.0) not in spans

    def test_fresh_asks_may_repeat(self):
        # Without drop_reemitted (fresh, non-"more" ask) a legitimate
        # repeat survives — only the top-up avoids exclusions.
        reply = '[CLIP: start=00:00:00 end=00:00:30 title="Same"]'
        out = _enforce_clip_count(
            reply, 1, candidates=_pool(),
            exclude_spans=[(0.0, 30.0, None)],
        )
        assert (0.0, 30.0) in _marker_spans(out)


class TestMisparsedCountCanNeverFlood:
    """H11 — "give me 30 more seconds of selects" is a duration ask; it
    must never yield 30 cards, and the top-up goal is defensively capped
    regardless of what any future parser gap produces."""

    MESSAGE = 'give me 30 more seconds of selects'

    def test_routes_to_duration_not_count(self):
        assert ai_analysis._detect_explicit_clip_count(self.MESSAGE) is None
        assert ai_analysis.parse_target_duration_seconds(self.MESSAGE) == 30.0

    def test_end_to_end_never_thirty_cards(self):
        reply = 'Here.\n[CLIP: start=00:00:00 end=00:00:25 title="Pick"]'
        with patch.object(ai_analysis, '_call_ai_chat', return_value=reply):
            out = ai_analysis.chat_about_transcript(
                _transcript(text='selects about topic {i} emotional'),
                self.MESSAGE)
        spans = _marker_spans(out)
        assert 1 <= len(spans) <= 3, f'30s target grew {len(spans)} cards'
        total = sum(e - s for s, e in spans)
        assert total <= 1.25 * 30.0

    def test_topup_goal_is_capped(self):
        pool = [
            {'start': i * 100.0, 'end': i * 100.0 + 30.0,
             'text': f'candidate number {i} text'}
            for i in range(40)
        ]
        out = _enforce_clip_count('no markers', 30, candidates=pool)
        assert len(_marker_spans(out)) <= ai_analysis._COUNT_TOPUP_MAX


class TestCountStillTrimsAndConversationalStaysProse:
    def test_explicit_count_still_trims_over_delivery(self):
        reply = '\n'.join(
            f'[CLIP: start=00:0{i}:00 end=00:0{i}:20 title="M{i}"]'
            for i in range(4)
        )
        with patch.object(ai_analysis, '_call_ai_chat', return_value=reply):
            out = ai_analysis.chat_about_transcript(
                _transcript(), 'find 2 clips about the topic')
        assert len(_marker_spans(out)) == 2

    def test_conversational_question_gets_no_topup(self):
        reply = 'It is really a story about grief and repair.'
        with patch.object(ai_analysis, '_call_ai_chat', return_value=reply):
            out = ai_analysis.chat_about_transcript(
                _transcript(), 'what is this interview really about?',
                segment_vectors=_vectors(),
            )
        assert '[CLIP:' not in out

    def test_conversational_count_phrase_grows_no_cards(self):
        # "just one" parses count=1, but the message is discussion — the
        # trim-only shape applies and no card is invented.
        reply = 'Her name is Mae.'
        with patch.object(ai_analysis, '_call_ai_chat', return_value=reply):
            out = ai_analysis.chat_about_transcript(
                _transcript(), "just one thing — what's her name?",
                segment_vectors=_vectors(),
            )
        assert '[CLIP:' not in out
