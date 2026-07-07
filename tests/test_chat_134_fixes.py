"""Regression pins for the 1.0.34 fix bundle (Chris's live-testing finds).

1. Duration CEILINGS: "under 60 seconds" / "no more than 2 minutes" /
   "90 seconds max" parse to BOUND × 0.8, so the chat band (×1.25) tops
   out exactly at the ceiling and the story band (×1.15) under it. The
   live bug: "build me a story under 60 seconds" parsed to nothing and
   built 2:14.
2. (prompt-only — no code pin) length guidance loosened.
3. Marker-imitation rescue: heading line + bare timecode range + dangling
   note="…" (the model imitating the old compact-history shape) now
   yields ONE clean card carrying the real title and the note.
4. (frontend) unique per-render card ids — pinned by grepping the
   template source, same technique the canonicalization suite uses.
5. Prewarm invalidation after heavy calls.
"""

import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ai_analysis  # noqa: E402
from ai_analysis import (  # noqa: E402
    _absorb_stray_note_lines,
    _adopt_adjacent_marker_titles,
    _clean_chat_response,
    invalidate_prewarm,
    parse_target_duration_seconds,
)


class TestDurationCeilings:
    def test_the_live_bug_phrase(self):
        # 60s ceiling → 48s target; chat ceiling 1.25×48 = 60 exactly.
        assert parse_target_duration_seconds(
            'build me a story under 60 seconds about may') == 48.0

    def test_ceiling_phrasings(self):
        for msg, want in (
            ('build me a story under 60 seconds', 48.0),
            ('make a reel no more than 2 minutes', 96.0),
            ('keep it under a minute, a teaser', 48.0),
            ('give me a cut, 90 seconds max', 72.0),
            ('pull selects at most 3 minutes', 144.0),
            # Revision phrasings the review found missing:
            ('keep it under 2 minutes', 96.0),
            ('keep this under 2 minutes', 96.0),
            ('hold it under 2 minutes', 96.0),
            ('trim it to 90 seconds max', 72.0),
            ('cap it at 2 minutes', 96.0),
            ('make it under 2 minutes', 96.0),
        ):
            assert parse_target_duration_seconds(msg) == want, msg

    def test_unanchored_and_positional_bounds_ignored(self):
        for msg in (
            'she was under 2 minutes from disaster when it happened',
            'the moment under 2 minutes in',
            'under no circumstances cut the intro',
        ):
            assert parse_target_duration_seconds(msg) is None, msg

    def test_review_hardened_rejections(self):
        # Proximity idioms: 'within' is NOT a bound cue (it is positional).
        for msg in (
            'pull everything within 2 minutes of the crash',
            'show me what happens within 3 minutes of the start',
            'can you do it within 2 minutes',
            'a montage within 2:30',
        ):
            assert parse_target_duration_seconds(msg) is None, msg
        # Per-clip LENGTH FILTERS are not total-runtime ceilings.
        for msg in (
            'find me a clip under 2 minutes',
            'find clips shorter than 2 minutes',
            'the clip under 2 minutes is better',
            'find me something under 2 minutes',
        ):
            assert parse_target_duration_seconds(msg) is None, msg
        # Clause boundaries stop the verb lookback.
        assert parse_target_duration_seconds(
            'find the guy who ran a mile under 4 minutes') is None

    def test_bound_accessor_returns_raw_ceiling(self):
        from ai_analysis import parse_duration_bound_seconds
        assert parse_duration_bound_seconds(
            'build me a story under 60 seconds') == 60.0
        assert parse_duration_bound_seconds(
            'give me 2 minutes of the best moments') is None

    def test_hard_cap_trims_past_the_floor_escape(self):
        # 30s+45s markers against an 'under 60' ask: the old trim loop
        # broke at the floor and shipped 75s; the hard cap must win.
        from ai_analysis import _enforce_duration_target
        text = ('Intro.\n'
                '[CLIP: start=00:01:00 end=00:01:30 title="First"]\n'
                '[CLIP: start=00:03:00 end=00:03:45 title="Second"]')
        out = _enforce_duration_target(text, 48.0, [], hard_cap_seconds=60.0)
        import re as _re
        spans = [( _re.search(r'start=([\d:]+)', m).group(1),
                   _re.search(r'end=([\d:]+)', m).group(1))
                 for m in _re.findall(r'\[CLIP:[^\]]*\]', out)]
        total = sum(
            (int(e.split(':')[0]) * 3600 + int(e.split(':')[1]) * 60 + int(e.split(':')[2]))
            - (int(s.split(':')[0]) * 3600 + int(s.split(':')[1]) * 60 + int(s.split(':')[2]))
            for s, e in spans)
        assert total <= 60.0, out

    def test_hard_cap_shrinks_lone_oversized_marker(self):
        from ai_analysis import _enforce_duration_target
        text = '[CLIP: start=00:01:00 end=00:03:00 title="Long"]'
        out = _enforce_duration_target(text, 48.0, [], hard_cap_seconds=60.0)
        assert 'end=00:02:00' in out, out

    def test_existing_targets_unchanged(self):
        assert parse_target_duration_seconds(
            'give me 2 minutes of the best moments') == 120.0
        assert parse_target_duration_seconds('build me a 14 minute cut') == 840.0
        assert parse_target_duration_seconds(
            'give me a 15 to 20 minute video') == 1050.0
        assert parse_target_duration_seconds('give me a sec') is None

    def test_bound_span_consumed_no_competing_candidates(self):
        # "under 2 minutes of selects" would ALSO match the plain
        # number+unit pass as anchored ("of" follows) — the bound span
        # must be consumed first or the two anchored values compete and
        # the parser returns None.
        assert parse_target_duration_seconds(
            'keep it under 2 minutes of selects') == 96.0


class TestMarkerImitationRescue:
    SCREENSHOT_SHAPE = (
        'This needs to be a "how it\'s made" reel.\n\n'
        '‣ From flax to fiber: Ancient craft process\n'
        '(16:09 - 18:03)\n'
        'note="This segment provides a highly visual, educational hook."\n\n'
        'I recommend starting there.'
    )

    def test_screenshot_3_shape_end_to_end(self):
        out = _clean_chat_response(self.SCREENSHOT_SHAPE)
        assert 'Moment at' not in out
        assert 'title="From flax to fiber: Ancient craft process"' in out
        assert 'note="This segment provides a highly visual, educational hook."' in out
        assert not re.search(r'^\s*note=', out, re.MULTILINE)
        assert 'I recommend starting there.' in out

    def test_adoption_never_overwrites_real_titles(self):
        src = ('A heading line above\n'
               '[CLIP: start=00:01:00 end=00:01:30 title="Model wrote this"]')
        out = _adopt_adjacent_marker_titles(src)
        assert 'title="Model wrote this"' in out
        assert 'A heading line above' in out  # prose untouched

    def test_adoption_requires_title_ish_line(self):
        # A full prose sentence (ends with '.') must not be consumed.
        src = ('She explains the whole process here.\n'
               '[CLIP: start=00:01:00 end=00:01:30 title="Moment at 00:01:00"]')
        out = _adopt_adjacent_marker_titles(src)
        assert 'She explains the whole process here.' in out
        assert 'title="Moment at 00:01:00"' in out

    def test_stray_note_absorbed_into_noteless_marker(self):
        src = ('[CLIP: start=00:01:00 end=00:01:30 title="The turn"]\n'
               'note="Why this lands."')
        out = _absorb_stray_note_lines(src)
        assert out.count('note=') == 1
        assert 'note="Why this lands."' in out
        assert '\nnote=' not in out

    def test_stray_duplicate_note_dropped_differing_note_kept_as_prose(self):
        # An EXACT duplicate of the marker's note is junk — dropped. A
        # DIFFERING stray note is content: unwrapped to plain prose,
        # never silently deleted (review fix — move text, don't lose it).
        dup = ('[CLIP: start=00:01:00 end=00:01:30 title="The turn" note="Original."]\n'
               'note="Original."')
        out = _absorb_stray_note_lines(dup)
        assert out.count('Original.') == 1
        diff = ('[CLIP: start=00:01:00 end=00:01:30 title="The turn" note="Original."]\n'
                'note="A second thought."')
        out2 = _absorb_stray_note_lines(diff)
        assert 'note="Original."' in out2
        assert 'A second thought.' in out2
        assert 'note="A second thought."' not in out2  # unwrapped, not attribute

    def test_note_with_backslash_never_crashes(self):
        # Model-authored note text goes through a LITERAL lambda
        # replacement — a backslash group reference used to raise
        # re.error and take the whole chat turn down.
        src = ('[CLIP: start=00:01:00 end=00:01:30 title="A"]\n'
               'note="see clip \\1 above"')
        out = _absorb_stray_note_lines(src)
        assert 'see clip' in out

    def test_note_like_prose_untouched_without_marker(self):
        src = 'Take note="of nothing" — this is prose with no card above.'
        assert _absorb_stray_note_lines(src) == src


class TestUniqueCardIds:
    def test_template_mints_per_render_uids(self):
        here = os.path.dirname(os.path.abspath(__file__))
        tpl = open(os.path.join(here, '..', 'templates', 'project.html'),
                   encoding='utf-8').read()
        # The uid must include the render sequence — duplicate ids from
        # the same moment cited in two replies routed play-state updates
        # to the wrong card (the live "play button doesn't change" bug).
        assert '__ccCardSeq' in tpl
        assert re.search(r'cc_\$\{start\}_\$\{end\}_r\$\{window\.__ccCardSeq\}', tpl)


class TestPrewarmInvalidation:
    def test_invalidate_busts_cooldown(self):
        from unittest.mock import patch
        ai_analysis._PREWARM_STATE.clear()
        segs = [{'start': 0, 'end': 25,
                 'start_formatted': '00:00:00.000',
                 'text': 'hello world text', 'speaker': 'Mae'}] * 4
        transcript = {'segments': segs}
        calls = []
        with patch.object(ai_analysis, '_call_ai_chat',
                          side_effect=lambda *a, **k: calls.append(1) or 'ok'), \
                patch.object(ai_analysis, '_ollama_is_active', return_value=True):
            assert ai_analysis.prewarm_chat_context(
                transcript, project_name='Rewarm Test') is True
            # Cooldown hit — no second call.
            assert ai_analysis.prewarm_chat_context(
                transcript, project_name='Rewarm Test') is True
            assert len(calls) == 1
            # After a heavy call evicts the cache, invalidation must make
            # the next prewarm actually re-prefill.
            invalidate_prewarm('Rewarm Test')
            assert ai_analysis.prewarm_chat_context(
                transcript, project_name='Rewarm Test') is True
        assert len(calls) == 2
