"""Server-side [CLIP:] marker canonicalization — the 1.0.30 chat-UX bug
cluster (owner screenshot, 2026-07).

Three symptoms on one reply:

  1. Raw bracket debris rendered as prose between cards
     (`to communicate deep emotional weight."]`,
     `note="A moment of realization…"]`). Root cause: a marker whose
     attributes span a newline (or whose quoted values contain parens)
     was HALF-rewritten by the variant normalizer — its candidate regex
     stops at the first ')' or ']' even inside a quoted value — while
     the newline-bounded _CLIP_MARKER_RE gave the enforcement passes
     ZERO matches for the same marker. Two regexes, two behaviors,
     debris on screen.
  2. Junk sliver cards ("Tell me about that" 00:15–00:16, 1s) —
     interviewer-question fragments with no quality floor.
  3. "Identify the 8-10 strongest standalone soundbites…" produced 3
     cards — numeric count RANGES didn't parse, so the ask fell to the
     plural 3-minimum.

The fix: a tolerant reader (_canonicalize_clip_markers) re-serializes
every recoverable marker into ONE canonical single-line grammar before
any lossy pass touches it; a quality floor (_enforce_marker_quality_floor)
drops sliver/unparseable/empty-title cards WITH their prose; a residue
scrub (_scrub_clip_marker_residue) sweeps every remaining partial-marker
fragment; and _detect_explicit_clip_count learned count ranges
(midpoint, rounded up). These tests lock all of it in, end-to-end
against the exact screenshot shapes, plus the canonical grammar's
round-trip through every consumer regex (core enforcement, the
project.html renderer extracted from the template itself, and the
collection renderer/stash shapes).
"""

import os
import re
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ai_analysis  # noqa: E402
from ai_analysis import (  # noqa: E402
    _CANONICAL_CLIP_MARKER_RE,
    _CLIP_MARKER_RE,
    _canonicalize_clip_markers,
    _clean_chat_response,
    _detect_explicit_clip_count,
    _enforce_marker_quality_floor,
    _history_clip_spans,
    _scrub_clip_marker_residue,
    _seconds_to_tc,
    _tc_to_seconds,
    parse_target_duration_seconds,
)


def _marker_spans(text):
    return [
        (_tc_to_seconds(s), _tc_to_seconds(e))
        for s, e in re.findall(
            r'\[CLIP:[^\]]*?start=([\d:]+)[^\]]*?end=([\d:]+)', text)
    ]


def _assert_no_bracket_debris(text):
    """No partial-marker junk anywhere outside well-formed markers."""
    stripped = _CLIP_MARKER_RE.sub('', text)
    assert '[CLIP' not in stripped, f'unrecovered [CLIP fragment: {text!r}'
    assert '"]' not in stripped, f'orphan attribute tail: {text!r}'
    assert not re.search(r'^\s*\]\s*$', stripped, re.M), \
        f'stray lone ] line: {text!r}'
    assert not re.search(r'\b(?:note|title|start|end|project)\s*=\s*"', stripped), \
        f'orphan attr= fragment: {text!r}'


def _assert_all_canonical(text):
    for m in _CLIP_MARKER_RE.findall(text):
        assert _CANONICAL_CLIP_MARKER_RE.fullmatch(m), \
            f'non-canonical marker survived: {m!r}'


# ── The tolerant reader ─────────────────────────────────────────────────────

class TestCanonicalizeClipMarkers:
    def test_note_spanning_a_newline_is_recovered(self):
        # The exact screenshot shape: attributes split across lines used to
        # get ZERO matches from _CLIP_MARKER_RE (so enforcement never saw
        # the marker) while the UI half-rendered it.
        src = ('[CLIP: start=00:05:00 end=00:05:30 title="Owning the story"\n'
               'note="A moment\nof realization for the subject."]')
        out = _canonicalize_clip_markers(src)
        assert out == ('[CLIP: start=00:05:00 end=00:05:30 '
                       'title="Owning the story" '
                       'note="A moment of realization for the subject."]')

    def test_parens_inside_values_no_longer_truncate(self):
        # The variant normalizer's candidate regex stopped at the first ')'
        # INSIDE the quoted note, leaking the tail as prose (screenshot
        # debris: `to communicate deep emotional weight."]`).
        src = ('[CLIP: start=00:02:10 end=00:02:40 title="The reveal" '
               'note="She laughs (nervously) then goes quiet to communicate '
               'deep emotional weight."]')
        out = _clean_chat_response(src)
        assert out == src  # already canonical — byte-identical
        _assert_no_bracket_debris(out)

    def test_bracket_inside_note_is_normalized_to_parens(self):
        src = '[CLIP: start=00:01:00 end=00:01:30 title="A" note="he said [wow] there"]'
        out = _canonicalize_clip_markers(src)
        assert out == ('[CLIP: start=00:01:00 end=00:01:30 title="A" '
                       'note="he said (wow) there"]')
        _assert_all_canonical(out)

    def test_inner_double_quotes_become_single(self):
        # Single- and curly-quoted values may legitimately contain straight
        # double quotes — normalized to ' so the canonical double-quoted
        # wrapping can never be broken from inside.
        src = "[CLIP: start=00:01:00 end=00:01:30 title='A' note='she said \"run\"']"
        out = _canonicalize_clip_markers(src)
        assert 'note="she said \'run\'"' in out
        src2 = '[CLIP: start=00:01:00 end=00:01:30 title=“A "quote"” ]'
        assert 'title="A \'quote\'"' in _canonicalize_clip_markers(src2)

    def test_attribute_order_is_free_and_alt_keys_map(self):
        src = '[clip note="why" start_time=0:10 end_time=0:40 label="One"]'
        out = _canonicalize_clip_markers(src)
        assert out == '[CLIP: start=0:10 end=0:40 title="One" note="why"]'

    def test_project_field_is_preserved_in_canonical_position(self):
        src = '[CLIP: title="Andre" note="fordi" project="Intervju A" start=00:01:00 end=00:01:30]'
        out = _canonicalize_clip_markers(src)
        assert out == ('[CLIP: start=00:01:00 end=00:01:30 '
                       'project="Intervju A" title="Andre" note="fordi"]')

    def test_unrecoverable_fragment_is_removed(self):
        assert _canonicalize_clip_markers('[CLIP: title="Just an idea"]') == ''

    def test_idempotent_on_canonical_markers(self):
        src = ('Intro.\n[CLIP: start=00:01:00 end=00:01:30 title="A" '
               'note="with (parens) and \'quotes\'"]\nOutro.')
        assert _canonicalize_clip_markers(_canonicalize_clip_markers(src)) \
            == _canonicalize_clip_markers(src) == src

    def test_non_clip_brackets_are_untouched(self):
        src = 'Tags like [note] or [BEAT: hook] and [CLIPBOARD] survive. At [00:05:12].'
        assert _canonicalize_clip_markers(src) == src

    def test_prose_after_unclosed_marker_is_not_swallowed(self):
        # Attribute scanning must never cross a newline into prose that
        # happens to contain an '='.
        src = ('[CLIP: start=00:01:00 end=00:01:30 title="A"\n'
               'The pacing = great here, truly.')
        out = _canonicalize_clip_markers(src)
        assert 'The pacing = great here, truly.' in out
        assert '[CLIP: start=00:01:00 end=00:01:30 title="A"]' in out

    def test_multiline_marker_counts_for_enforcement_after_cleaning(self):
        # The root-cause assertion: after canonicalization the enforcement
        # regex sees exactly what the UI will render.
        src = ('[CLIP: start=00:05:00 end=00:05:30 title="T"\n'
               'note="A moment\nof realization."]')
        assert len(_CLIP_MARKER_RE.findall(src)) == 0       # the 1.0.30 hole
        cleaned = _clean_chat_response(src)
        assert len(_CLIP_MARKER_RE.findall(cleaned)) == 1   # closed


# ── Quality floor ───────────────────────────────────────────────────────────

class TestMarkerQualityFloor:
    def test_sliver_card_is_dropped_with_its_prose(self):
        src = ('Keep this intro.\n\n'
               '[CLIP: start=00:00:15 end=00:00:16 title="Tell me about that"]\n'
               'An interviewer fragment that must go with its card.\n\n'
               '[CLIP: start=00:01:00 end=00:01:30 title="Real moment"]\n'
               'Why it lands.')
        out = _clean_chat_response(src)
        assert 'Tell me about that' not in out
        assert 'interviewer fragment' not in out
        assert 'Keep this intro.' in out
        assert 'title="Real moment"' in out
        assert 'Why it lands.' in out

    def test_five_second_card_survives_the_floor(self):
        src = '[CLIP: start=00:00:10 end=00:00:15 title="Exactly five"]'
        assert _enforce_marker_quality_floor(src) == src

    def test_backwards_and_unparseable_ranges_are_dropped(self):
        assert _enforce_marker_quality_floor(
            '[CLIP: start=00:02:00 end=00:01:00 title="Backwards"]') == ''
        assert _enforce_marker_quality_floor(
            '[CLIP: start=nonsense end=alsono title="Junk"]') == ''

    def test_empty_title_is_dropped(self):
        assert _enforce_marker_quality_floor(
            '[CLIP: start=00:01:00 end=00:01:30 title=""]') == ''

    def test_mixed_line_keeps_the_healthy_marker(self):
        src = ('[CLIP: start=00:01:00 end=00:01:30 title="Good"] '
               '[CLIP: start=00:02:00 end=00:02:01 title="Sliver"]')
        out = _enforce_marker_quality_floor(src)
        assert 'title="Good"' in out
        assert 'title="Sliver"' not in out


# ── Residue scrub ───────────────────────────────────────────────────────────

class TestResidueScrub:
    def test_orphan_attr_tail_is_swept(self):
        src = 'Some prose.\nnote="A moment of realization…"]\nMore prose.'
        out = _scrub_clip_marker_residue(src)
        assert 'note=' not in out
        assert ']' not in out
        assert 'Some prose.' in out and 'More prose.' in out

    def test_half_note_tail_line_with_vocab_is_swept(self):
        # A "]-terminated line is only provably marker debris when it
        # carries marker-attribute vocabulary (attr=). Vocab-less lines
        # ending in "] are indistinguishable from legitimate JSON-style
        # content and must be KEPT (F7) — the source marker that used to
        # leak such tails is now fully recovered upstream by the
        # tolerant reader (see test_parens_inside_values_no_longer_truncate).
        src = 'Prose.\nnote="to communicate deep emotional weight."]\nAfter.'
        out = _scrub_clip_marker_residue(src)
        assert 'emotional weight' not in out
        assert 'Prose.' in out and 'After.' in out

    def test_vocabless_quote_bracket_line_is_kept(self):
        # F7: JSON-style answers end lines in "] with no marker
        # vocabulary — deleting them corrupts the answer.
        src = 'Here is JSON:\n"soundbites": [\n  "one",\n  "two"]\ndone.'
        assert _scrub_clip_marker_residue(src) == src

    def test_lone_bracket_lines_are_swept(self):
        assert _scrub_clip_marker_residue('A.\n]\nB.\n"]\nC.') == 'A.\nB.\nC.'

    def test_unclosed_clip_fragment_is_swept(self):
        out = _scrub_clip_marker_residue('Take [CLIP: start=00:01:00 end=\nNext line.')
        assert '[CLIP' not in out
        assert 'Next line.' in out

    def test_canonical_markers_and_legit_prose_survive(self):
        src = ('Real [note] here and a [BEAT: hook].\n'
               '[CLIP: start=00:01:00 end=00:01:30 title="A" note="ok (fine)"]\n'
               'Closing thought at 00:05:12.')
        assert _scrub_clip_marker_residue(src) == src


# ── Count ranges ────────────────────────────────────────────────────────────

class TestClipCountRanges:
    def test_screenshot_range_ask_parses_to_the_rounded_midpoint(self):
        assert _detect_explicit_clip_count(
            'Identify the 8-10 strongest standalone soundbites in this '
            'transcript. Strong means quotable, self-contained, and '
            'emotionally or narratively distinctive. Return as [CLIP:] '
            'markers with a one-line note explaining why each one lands.'
        ) == 9

    def test_digit_range_variants(self):
        assert _detect_explicit_clip_count('give me 8 to 10 clips') == 9
        assert _detect_explicit_clip_count('give me 8-10') == 9
        assert _detect_explicit_clip_count('2-3 strong quotes about the fire') == 3
        assert _detect_explicit_clip_count('pull 4–6 highlights') == 5

    def test_word_range(self):
        assert _detect_explicit_clip_count('pull eight to ten moments') == 9
        assert _detect_explicit_clip_count('give me eight to ten') == 9

    def test_duration_ranges_stay_durations(self):
        # Unit presence disambiguates — parse_target_duration_seconds owns
        # these, the count parser must not collide.
        assert _detect_explicit_clip_count('give me a 15 to 20 minute cut') is None
        assert parse_target_duration_seconds('give me a 15 to 20 minute cut') == 1050.0
        assert _detect_explicit_clip_count('8-10s clips please') is None
        assert _detect_explicit_clip_count('one to two minute clips') is None
        assert _detect_explicit_clip_count('give me 30 more seconds of selects') is None

    def test_backtracking_cannot_steal_duration_ranges(self):
        # F1: regex backtracking used to re-split the second number
        # ("8-10 second clips" → lo=8 hi=1 → count 5; "15 to 20 minutes"
        # → hi=2 → count 9), defeating the time-unit lookahead and
        # stealing ownership from parse_target_duration_seconds.
        assert _detect_explicit_clip_count('give me 8-10 second clips') is None
        assert _detect_explicit_clip_count('8-10 second clips') is None
        assert _detect_explicit_clip_count('8-10s clips') is None
        assert _detect_explicit_clip_count(
            '15 to 20 minutes of the best material') is None
        assert parse_target_duration_seconds(
            '15 to 20 minutes of the best material') == 1050.0
        assert _detect_explicit_clip_count(
            'give me 15 to 20 minutes of the best material') is None
        assert parse_target_duration_seconds(
            'give me 15 to 20 minutes of the best material') == 1050.0

    def test_hyphen_attached_word_unit_is_a_duration(self):
        # F1: "eight to ten-minute segments" attaches the unit with a
        # hyphen — the word-range lookahead must see through it.
        assert _detect_explicit_clip_count('eight to ten-minute segments') is None
        assert _detect_explicit_clip_count(
            'give me eight to ten-minute segments') is None
        assert parse_target_duration_seconds(
            'give me eight to ten-minute segments') == 540.0

    def test_plain_digit_range_with_noun_still_counts(self):
        assert _detect_explicit_clip_count('8-10 clips') == 9

    def test_range_with_ago_reference_is_not_a_count(self):
        assert _detect_explicit_clip_count(
            'a few moments ago you said 8 to 10 moments ago') is None

    def test_existing_single_count_shapes_are_unchanged(self):
        assert _detect_explicit_clip_count('give me 4 more') == 4
        assert _detect_explicit_clip_count('give me 5 more') == 5
        assert _detect_explicit_clip_count('find me 3') == 3
        assert _detect_explicit_clip_count('compare the two moments where she cries') == 2
        assert _detect_explicit_clip_count('give me 1 minute of selects') is None
        assert _detect_explicit_clip_count('what are the strongest emotional moments?') is None


# ── End-to-end: the exact 1.0.30 screenshot through the real pipeline ──────

def _transcript(n=24):
    segments = []
    for i in range(n):
        start = i * 30.0
        segments.append({
            'start': start, 'end': start + 30.0,
            'start_formatted': _seconds_to_tc(start),
            'text': f'a strong standalone soundbite about topic {i} with emotional weight',
            'speaker': 'A',
        })
    return {'segments': segments}


def _vectors(n=14):
    return [{'timecode_in': _seconds_to_tc(i * 50),
             'timecode_out': _seconds_to_tc(i * 50 + 30),
             'narrative_score': 'high', 'theme_tags': []} for i in range(n)]


SCREENSHOT_REPLY = (
    'Here are the strongest soundbites I found.\n\n'
    '[CLIP: start=00:02:10 end=00:02:40 title="The turning point" '
    'note="She stops performing (finally) and starts telling the truth '
    'to communicate deep emotional weight."]\n\n'
    '[CLIP: start=00:05:00 end=00:05:30 title="Owning the story"\n'
    'note="A moment\nof realization for the subject."]\n\n'
    '[CLIP: start=00:00:15 end=00:00:16 title="Tell me about that"]\n\n'
    'These carry the arc.'
)
SCREENSHOT_MESSAGE = (
    'Identify the 8-10 strongest standalone soundbites in this transcript. '
    'Strong means quotable, self-contained, and emotionally or narratively '
    'distinctive. Return as [CLIP:] markers with a one-line note explaining '
    'why each one lands.'
)


class TestScreenshotEndToEnd:
    def _run(self):
        with patch.object(ai_analysis, '_call_ai_chat',
                          return_value=SCREENSHOT_REPLY):
            return ai_analysis.chat_about_transcript(
                _transcript(), SCREENSHOT_MESSAGE,
                segment_vectors=_vectors(),
            )

    def test_range_ask_enforces_nine_canonical_cards(self):
        out = self._run()
        markers = _CLIP_MARKER_RE.findall(out)
        assert len(markers) == 9, f'expected 9 cards for 8-10: {out!r}'
        _assert_all_canonical(out)

    def test_zero_bracket_debris_and_sliver_replaced(self):
        out = self._run()
        _assert_no_bracket_debris(out)
        # The 1s interviewer fragment is gone; the two real picks survive
        # with their (recovered) notes.
        assert 'Tell me about that' not in out
        assert (135.0, 136.0) not in _marker_spans(out)
        assert 'title="The turning point"' in out
        assert 'note="A moment of realization for the subject."' in out

    def test_natural_flow_intro_and_outro_prose_survive(self):
        out = self._run()
        assert out.startswith('Here are the strongest soundbites I found.')
        assert 'These carry the arc.' in out

    def test_stream_path_matches(self):
        def fake_stream(system_message, messages, num_ctx=32768):
            yield SCREENSHOT_REPLY[:40]
            yield SCREENSHOT_REPLY[40:]

        with patch.object(ai_analysis, '_call_ai_chat_stream',
                          side_effect=fake_stream):
            events = list(ai_analysis.chat_about_transcript_stream(
                _transcript(), SCREENSHOT_MESSAGE,
                segment_vectors=_vectors(),
            ))
        assert events[-1][0] == 'done'
        out = events[-1][1]
        assert len(_CLIP_MARKER_RE.findall(out)) == 9
        _assert_no_bracket_debris(out)
        _assert_all_canonical(out)

    def test_give_me_four_more_still_yields_four(self):
        history = [
            {'role': 'user', 'content': SCREENSHOT_MESSAGE},
            {'role': 'assistant', 'content':
                'One.\n[CLIP: start=00:00:00 end=00:00:30 title="Shown"]'},
        ]
        reply = ('Two more.\n'
                 '[CLIP: start=00:02:00 end=00:02:30 title="A"]\n'
                 '[CLIP: start=00:04:00 end=00:04:30 title="B"]')
        with patch.object(ai_analysis, '_call_ai_chat', return_value=reply):
            out = ai_analysis.chat_about_transcript(
                _transcript(), 'give me 4 more', history=history,
                segment_vectors=_vectors(),
            )
        assert len(_CLIP_MARKER_RE.findall(out)) == 4, f'{out!r}'
        _assert_no_bracket_debris(out)


# ── Canonical grammar round-trip through every consumer ────────────────────

CANONICAL_SAMPLES = [
    '[CLIP: start=00:02:10 end=00:02:40 title="The turning point" '
    'note="She stops (finally) telling \'her\' truth."]',
    '[CLIP: start=0:10 end=0:40 title="MM:SS form"]',
    '[CLIP: start=00:01:00 end=00:01:30 project="Intervju A" '
    'title="Andre" note="fordi det treffer"]',
]


def _extract_project_html_regex():
    """Pull the renderChatReply card regex OUT of the shipped template so
    the test breaks if the two grammars ever drift apart."""
    path = os.path.join(os.path.dirname(__file__), '..', 'templates', 'project.html')
    with open(path, encoding='utf-8') as f:
        src = f.read()
    m = re.search(r'/(\\\[CLIP:\?\\s\*start=.*?)/gi,', src)
    assert m, 'renderChatReply card regex not found in project.html'
    return m.group(1)


class TestCanonicalRoundTrip:
    def test_matches_the_enforcement_regex_exactly(self):
        for s in CANONICAL_SAMPLES:
            assert _CLIP_MARKER_RE.findall(s) == [s]
            assert _CANONICAL_CLIP_MARKER_RE.fullmatch(s)

    def test_matches_the_naive_bracket_regex(self):
        # Values carry no ']' so even the naive [^\]]* consumers
        # (_count_clip_markers, the validator, collection _MARKER_RE)
        # parse the same span.
        for s in CANONICAL_SAMPLES:
            assert re.findall(r'\[CLIP:[^\]]*\]', s) == [s]

    def test_history_span_extraction(self):
        history = [{'role': 'assistant', 'content': CANONICAL_SAMPLES[0]}]
        assert _history_clip_spans(history) == [(130.0, 160.0, None)]

    def test_project_html_renderer_regex_parses_canonical_markers(self):
        js_pattern = _extract_project_html_regex()
        py_re = re.compile(js_pattern, re.IGNORECASE)
        # Single-project canonical form (no project=).
        m = py_re.fullmatch(CANONICAL_SAMPLES[0])
        assert m, 'project.html regex must match the canonical marker'
        start, end, title, note = m.group(1), m.group(2), m.group(3), m.group(4)
        assert (start, end) == ('00:02:10', '00:02:40')
        assert title == 'The turning point'
        assert note == "She stops (finally) telling 'her' truth."
        # Collection form (project= tolerated, groups unshifted).
        m = py_re.fullmatch(CANONICAL_SAMPLES[2])
        assert m and m.group(3) == 'Andre' and m.group(4) == 'fordi det treffer'

    def test_project_html_regex_rejects_sub5s_slivers_via_callback_guard(self):
        # The regex itself matches (it's shape-based); the callback guard
        # drops <5s — mirror that contract here so a canonical sliver can
        # never be produced upstream anyway (the floor drops it first).
        floor_out = _enforce_marker_quality_floor(
            '[CLIP: start=00:00:15 end=00:00:16 title="Tell me about that"]')
        assert floor_out == ''

    def test_collection_renderer_regex_parses_canonical_markers(self):
        # collection.js: _CLIP_MARKER_RE = /\[CLIP:?\s*([^\]]+)\]/gi with
        # per-field extraction. Python equivalence.
        body_re = re.compile(r'\[CLIP:?\s*([^\]]+)\]', re.IGNORECASE)

        def field(body, key):
            m = re.search(key + r'\s*=\s*(?:"|\')([\s\S]*?)(?:"|\')', body)
            return m.group(1) if m else None

        m = body_re.fullmatch(CANONICAL_SAMPLES[2])
        assert m
        body = m.group(1)
        assert field(body, 'project') == 'Intervju A'
        assert field(body, 'title') == 'Andre'
        assert field(body, 'note') == 'fordi det treffer'
        sm = re.search(r'start\s*=\s*([^\s\]]+)', body)
        em = re.search(r'end\s*=\s*([^\s\]]+)', body)
        assert (sm.group(1), em.group(1)) == ('00:01:00', '00:01:30')

    def test_canonical_form_survives_the_full_cleaner_byte_for_byte(self):
        for s in CANONICAL_SAMPLES:
            assert s in _clean_chat_response(f'Intro.\n{s}\nOutro.')


# ── Legitimate content must never be lost (review findings F2-F9) ──────────

class TestFloorSparesNarration:
    def test_subfloor_prose_range_is_not_wrapped_or_eaten(self):
        # F2: auto-wrap used to mint a 4s marker mid-sentence, then the
        # floor dropped marker AND sentence. Sub-floor prose ranges now
        # stay prose.
        src = 'Great line at (00:05:10 - 00:05:14), very short but punchy.'
        assert _clean_chat_response(src) == src

    def test_marker_syntax_mention_is_prose(self):
        # F2: placeholder timecode TEMPLATES (MM:SS literals) mark a
        # MENTION of the marker syntax, not a marker.
        src = 'I use [CLIP: start=MM:SS end=MM:SS title="..."] markers.'
        assert _clean_chat_response(src) == src

    def test_inline_junk_marker_loses_only_the_marker(self):
        # F2: a junk sliver sharing its line with narration loses only
        # the marker — the sentence survives.
        src = ('Also [CLIP: start=00:00:15 end=00:00:16 title="Tell me '
               'about that"] is punchy.')
        out = _enforce_marker_quality_floor(src)
        assert 'CLIP' not in out
        assert 'Also' in out and 'is punchy.' in out

    def test_marker_alone_on_line_still_drops_with_notes(self):
        src = ('[CLIP: start=00:00:15 end=00:00:16 title="Sliver"]\n'
               'Attached note line.\n\nStandalone prose.')
        out = _enforce_marker_quality_floor(src)
        assert 'Sliver' not in out and 'Attached note line.' not in out
        assert 'Standalone prose.' in out


class TestUnitSuffixedTimecodes:
    def test_tc_to_seconds_accepts_unit_suffixes(self):
        assert _tc_to_seconds('125s') == 125.0
        assert _tc_to_seconds('140 sec') == 140.0
        assert _tc_to_seconds('90 seconds') == 90.0
        assert _tc_to_seconds('MM:SS') == 0.0  # placeholder stays unparseable

    def test_unit_suffixed_marker_survives_the_cleaner(self):
        # F3: start=125s end=140s is a 15s clip the UIs play — the floor
        # must not junk-drop it (tokens stay verbatim: the collection
        # stash keys on the raw token).
        src = '[CLIP: start=125s end=140s title="Unit suffixed"]'
        assert _clean_chat_response(src) == src

    def test_history_spans_and_dedup_parse_the_suffix(self):
        history = [{'role': 'assistant',
                    'content': '[CLIP: start=125s end=140s title="A"]'}]
        assert _history_clip_spans(history) == [(125.0, 140.0, None)]


class TestAllSubFloorRescue:
    def test_keeps_the_single_longest_candidate(self):
        # F4: never claim nothing was found when something exists.
        out = ai_analysis._format_clip_cards_from_candidates([
            {'start_sec': 10.0, 'end_sec': 13.0, 'title': 'Short A', 'why': ''},
            {'start_sec': 20.0, 'end_sec': 24.5, 'title': 'Short B', 'why': ''},
        ])
        assert "couldn't find" not in out
        markers = _CLIP_MARKER_RE.findall(out)
        assert len(markers) == 1
        assert 'title="Short B"' in out  # the longest one
        # Snapped up to the floor so the UI sliver guards render it.
        (s, e), = _marker_spans(out)
        assert e - s >= ai_analysis._MIN_CLIP_MARKER_SECONDS

    def test_no_candidates_still_returns_the_honest_prose(self):
        out = ai_analysis._format_clip_cards_from_candidates([])
        assert "couldn't find" in out

    def test_zero_and_backwards_candidates_are_not_rescued(self):
        out = ai_analysis._format_clip_cards_from_candidates([
            {'start_sec': 10.0, 'end_sec': 10.0, 'title': 'Zero', 'why': ''},
            {'start_sec': 30.0, 'end_sec': 20.0, 'title': 'Backwards', 'why': ''},
        ])
        assert "couldn't find" in out


class TestSweepsSpareLegitimateProse:
    def test_lone_attr_in_prose_survives(self):
        # F5: 'Set title="My Export" in the dialog' is user-facing help,
        # not marker debris — with or without brackets elsewhere.
        src = 'Set title="My Export" in the dialog'
        assert _scrub_clip_marker_residue(src) == src
        src2 = 'Set title="My Export" in the dialog [see docs]'
        assert _scrub_clip_marker_residue(src2) == src2
        assert _clean_chat_response(src2) == src2

    def test_fcpxml_help_snippet_survives(self):
        src = ('Open the XML and find <asset-clip name="Clip 1" '
               'start="3600s" duration="60s"> in the spine.')
        assert _scrub_clip_marker_residue(src) == src

    def test_attr_pair_debris_is_still_swept(self):
        out = _scrub_clip_marker_residue('junk start=00:01:00 end=00:02:00] tail')
        assert 'start=' not in out
        assert 'junk' in out and 'tail' in out
        out = _scrub_clip_marker_residue('junk title="A" note="B"] tail')
        assert 'title=' not in out

    def test_bare_clip_reference_survives_everywhere(self):
        # F6: '[clip 3]' is a prose pointer at an earlier card.
        src = 'As shown in [clip 3] earlier.'
        assert _canonicalize_clip_markers(src) == src
        assert _scrub_clip_marker_residue(src) == src
        assert _clean_chat_response(src) == src

    def test_colon_headed_fragments_are_still_debris(self):
        assert _canonicalize_clip_markers('[CLIP: title="Just an idea"]') == ''
        out = _scrub_clip_marker_residue('Take [CLIP: start=00:01:00 end=\nNext line.')
        assert '[CLIP' not in out
        assert 'Next line.' in out


class TestTruncatedMarkerNeverFabricates:
    def test_mid_token_truncation_is_dropped(self):
        # F9: a stream cut inside the end timecode ("end=00:02:3") must
        # never fabricate a card at the wrong end time.
        out = _canonicalize_clip_markers('Intro.\n[CLIP: start=00:01:00 end=00:02:3')
        assert out == 'Intro.\n'
        cleaned = _clean_chat_response('Intro.\n[CLIP: start=00:01:00 end=00:02:3')
        assert '[CLIP' not in cleaned
        assert 'Intro.' in cleaned

    def test_bracketless_with_complete_timecodes_is_recovered(self):
        out = _canonicalize_clip_markers(
            '[CLIP: start=00:01:00 end=00:02:30 title="Cut before close')
        assert out.startswith('[CLIP: start=00:01:00 end=00:02:30 ')
        assert _CANONICAL_CLIP_MARKER_RE.search(out)


class TestSweepPerformance:
    def test_long_attr_run_scans_linearly(self):
        # ReDoS guard: requiring the closing ] INSIDE the attr-tail regex
        # made every non-terminated run fail and rescan per attr
        # (quadratic; exponential before the alternatives were made
        # disjoint). The ] is optional in the pattern and enforced in the
        # callback, so this pathological input must stay fast.
        import time
        txt = ('title="aaaa" ' * 8000) + 'and later a bracket ] here.'
        t0 = time.time()
        _scrub_clip_marker_residue(txt)
        assert time.time() - t0 < 2.0


class TestCollectionFieldExtractionMirror:
    def test_double_quote_form_wins_over_inner_apostrophes(self):
        # F8 mirror of collection.js _markerField: canonical values are
        # double-quoted and may CONTAIN apostrophes; the double-quote
        # form must be preferred so title="She said 'hello' loudly"
        # yields the full value, not 'She said'.
        body = ('start=00:01:00 end=00:01:30 '
                'title="She said \'hello\' loudly" note="the note"')

        def field(b, key):
            m = re.search(key + r'\s*=\s*"([\s\S]*?)"', b)
            if m:
                return m.group(1)
            m = re.search(key + r"\s*=\s*'([\s\S]*?)'", b)
            return m.group(1) if m else None

        assert field(body, 'title') == "She said 'hello' loudly"
        assert field(body, 'note') == 'the note'
        # single-quote fallback for old saved histories
        assert field("title='Old style'", 'title') == 'Old style'
