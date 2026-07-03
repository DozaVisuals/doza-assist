"""Tests for chat-reply CLIP-marker normalization.

Small local models (Gemma 4 e4b especially) drift on the [CLIP: ...] marker
format the chat UI parses: they use single quotes, curly Unicode quotes,
drop the colon, swap key names, or wrap the whole thing in **markdown bold**.
The frontend regex only matches the canonical shape, so without server-side
normalization those suggestions show up as raw text and the user thinks the
chat "stopped pulling clips."

These tests lock in the normalizer so a future prompt change doesn't silently
regress back to the "just text, no clip cards" behavior.
"""

import os
import re
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ai_analysis  # noqa: E402
from ai_analysis import (  # noqa: E402
    _CLIP_MARKER_RE,
    _auto_wrap_timecode_ranges,
    _clean_chat_response,
    _enforce_clip_count,
    _enforce_duration_target,
    _format_clip_title,
    _normalize_clip_markers,
    _strip_raw_json_blobs,
    _tc_to_seconds,
)


class TestNormalizeClipMarkers:
    def test_canonical_marker_passes_through_unchanged(self):
        src = 'Great moment: [CLIP: start=00:05:12 end=00:05:34 title="The turn"] — it sells the arc.'
        out = _normalize_clip_markers(src)
        assert '[CLIP: start=00:05:12 end=00:05:34 title="The turn"]' in out

    def test_single_quoted_title_is_normalized(self):
        src = "Try this: [CLIP: start=00:05:12 end=00:05:34 title='The turn']"
        out = _normalize_clip_markers(src)
        assert '[CLIP: start=00:05:12 end=00:05:34 title="The turn"]' in out

    def test_curly_unicode_quotes_are_normalized(self):
        src = 'Try this: [CLIP: start=00:05:12 end=00:05:34 title=\u201cThe turn\u201d]'
        out = _normalize_clip_markers(src)
        assert '[CLIP: start=00:05:12 end=00:05:34 title="The turn"]' in out

    def test_missing_colon_after_clip_is_fixed(self):
        # Gemma 4 e4b emitted this shape often in testing.
        src = 'Here: [CLIP start=00:05:12 end=00:05:34 title="The turn"]'
        out = _normalize_clip_markers(src)
        assert '[CLIP: start=00:05:12 end=00:05:34 title="The turn"]' in out

    def test_lowercase_clip_prefix_is_fixed(self):
        src = 'Here: [clip: start=00:05:12 end=00:05:34 title="The turn"]'
        out = _normalize_clip_markers(src)
        assert '[CLIP: start=00:05:12 end=00:05:34 title="The turn"]' in out

    def test_alternate_key_names_are_mapped(self):
        src = '[CLIP: start_time=00:05:12 end_time=00:05:34 label="The turn"]'
        out = _normalize_clip_markers(src)
        assert '[CLIP: start=00:05:12 end=00:05:34 title="The turn"]' in out

    def test_reordered_keys_are_normalized(self):
        src = '[CLIP: title="The turn" start=00:05:12 end=00:05:34]'
        out = _normalize_clip_markers(src)
        assert '[CLIP: start=00:05:12 end=00:05:34 title="The turn"]' in out

    def test_unquoted_title_is_wrapped(self):
        src = '[CLIP: start=00:05:12 end=00:05:34 title=TheTurn]'
        out = _normalize_clip_markers(src)
        assert '[CLIP: start=00:05:12 end=00:05:34 title="TheTurn"]' in out

    def test_parenthesis_variant_is_rewritten_to_brackets(self):
        src = 'Try: (CLIP: start=00:05:12 end=00:05:34 title="The turn")'
        out = _normalize_clip_markers(src)
        assert '[CLIP: start=00:05:12 end=00:05:34 title="The turn"]' in out

    def test_marker_without_timecodes_is_left_alone(self):
        # If we can't extract start+end, do not emit a half-broken card.
        src = '[CLIP: title="Just an idea"]'
        out = _normalize_clip_markers(src)
        assert out == src

    def test_multiple_markers_in_one_reply_are_all_normalized(self):
        src = (
            "Two strong beats:\n"
            "[clip: start=0:10 end=0:40 title='One']\n"
            "and later\n"
            '[CLIP start_time=1:05 end_time=1:25 label=\u201cTwo\u201d]'
        )
        out = _normalize_clip_markers(src)
        assert '[CLIP: start=0:10 end=0:40 title="One"]' in out
        assert '[CLIP: start=1:05 end=1:25 title="Two"]' in out

    def test_non_clip_brackets_are_not_touched(self):
        src = 'Tags like [note] or [BEAT: hook] must survive normalization.'
        out = _normalize_clip_markers(src)
        assert out == src


class TestCleanChatResponseIntegration:
    """_clean_chat_response is the server-side wrapper that runs BEFORE the
    reply is returned to the UI. It must normalize CLIP markers AND strip
    markdown without corrupting the normalized markers."""

    def test_markdown_bold_wrapping_a_clip_marker_is_removed(self):
        # Gemma 4 sometimes wraps the marker in **bold**, which would have
        # been mangled by the old markdown stripper.
        src = "Try this **[CLIP: start=00:05:12 end=00:05:34 title=\"The turn\"]** — it's the spine."
        out = _clean_chat_response(src)
        assert '[CLIP: start=00:05:12 end=00:05:34 title="The turn"]' in out
        assert '**' not in out

    def test_markdown_stripping_runs_outside_clip_markers(self):
        # Bold/italic in prose should still be stripped — we don't want to
        # skip markdown cleanup entirely, only protect the CLIP marker.
        src = "This is **strong**. [CLIP: start=0:10 end=0:30 title='A']"
        out = _clean_chat_response(src)
        assert 'This is strong.' in out
        assert '[CLIP: start=0:10 end=0:30 title="A"]' in out
        assert '**' not in out

    def test_variant_marker_survives_full_cleaner_pipeline(self):
        # End-to-end: a messy small-model reply should come out with a
        # canonical marker the frontend regex can parse.
        src = (
            "# Here's what I'd pull\n\n"
            "**Moment one** — [clip: start_time=00:05:12 end_time=00:05:34 "
            "title=\u201cThe turn\u201d] because it lands the decision.\n"
        )
        out = _clean_chat_response(src)
        assert '[CLIP: start=00:05:12 end=00:05:34 title="The turn"]' in out
        # Markdown header and bold are stripped.
        assert '# ' not in out
        assert '**' not in out


class TestAutoWrapTimecodeRanges:
    """When Gemma 4 writes a prose timecode range like "(00:12:34 - 00:13:00)"
    instead of a [CLIP:] marker, the chat UI renders it as static timecode
    pills — not the playable clip cards previous versions showed. This pass
    catches those ranges server-side and converts them to canonical CLIP
    markers so every referenced moment becomes a playable card."""

    def test_paren_wrapped_hms_range_is_wrapped(self):
        src = "The opening beat lands at the gala (00:00:35 - 00:00:52)."
        out = _auto_wrap_timecode_ranges(src)
        assert '[CLIP: start=00:00:35 end=00:00:52 title="Moment at 00:00:35"]' in out
        # The original paren-wrapped range is gone so the frontend doesn't
        # also render a duplicate timecode pill right next to the card.
        assert '(00:00:35 - 00:00:52)' not in out

    def test_plain_prose_range_without_parens_is_wrapped(self):
        src = "From 00:12:34 to 00:13:00 the subject shifts."
        out = _auto_wrap_timecode_ranges(src)
        assert '[CLIP: start=00:12:34 end=00:13:00 title="Moment at 00:12:34"]' in out

    def test_en_dash_and_em_dash_separators_are_handled(self):
        # Small models sometimes emit unicode dashes.
        src_en = "See (00:05:00\u201300:05:30) for the transition."
        src_em = "See (00:05:00\u201400:05:30) for the transition."
        assert '[CLIP: start=00:05:00 end=00:05:30' in _auto_wrap_timecode_ranges(src_en)
        assert '[CLIP: start=00:05:00 end=00:05:30' in _auto_wrap_timecode_ranges(src_em)

    def test_mm_ss_range_is_wrapped(self):
        # The chat UI accepts MM:SS format too — the tcToSec helper handles it.
        src = "The hook is 0:30 - 1:15."
        out = _auto_wrap_timecode_ranges(src)
        assert '[CLIP: start=0:30 end=1:15 title="Moment at 0:30"]' in out

    def test_backwards_range_is_left_alone(self):
        # If end <= start we'd emit a nonsensical clip card. Better to leave
        # the raw text so the user sees something is off.
        src = "Weirdly 00:10:00 to 00:05:00 ref."
        out = _auto_wrap_timecode_ranges(src)
        assert '[CLIP:' not in out

    def test_multiple_ranges_in_one_reply_all_wrap(self):
        src = (
            "Arc: community at the gala (00:00:35 - 00:00:52), "
            "tension rises in Rwanda (00:27:04 - 00:28:11), "
            "peaks at Nate's reflection (00:31:31 - 00:32:19)."
        )
        out = _auto_wrap_timecode_ranges(src)
        assert out.count('[CLIP:') == 3
        assert '[CLIP: start=00:00:35 end=00:00:52' in out
        assert '[CLIP: start=00:27:04 end=00:28:11' in out
        assert '[CLIP: start=00:31:31 end=00:32:19' in out

    def test_existing_clip_markers_are_not_double_wrapped(self):
        # A canonical marker already has its own timecode range — rewriting
        # it would double-encode and break the frontend regex.
        src = 'Check [CLIP: start=00:05:12 end=00:05:34 title="The turn"] now.'
        out = _auto_wrap_timecode_ranges(src)
        assert out == src

    def test_standalone_timecode_is_not_wrapped(self):
        # A single timecode in prose is still a valid pill in the chat UI —
        # don't force it into a fake range.
        src = "At 00:05:12 the speaker hesitates."
        out = _auto_wrap_timecode_ranges(src)
        assert '[CLIP:' not in out
        assert '00:05:12' in out

    def test_end_to_end_prose_range_becomes_clip_card_format(self):
        # Full pipeline: a Gemma-4-style prose reply should come out of
        # _clean_chat_response with every referenced moment wrapped in a
        # canonical CLIP marker, ready for the frontend to render as a
        # playable clip card.
        src = (
            "The emotional peak is Nate's reflection on how the experience "
            "changed his perspective (00:31:31 - 00:32:19). It lands the arc."
        )
        out = _clean_chat_response(src)
        assert '[CLIP: start=00:31:31 end=00:32:19 title="Moment at 00:31:31"]' in out


class TestFormatClipTitle:
    def test_lowercase_title_is_capitalized(self):
        assert _format_clip_title('papermaking artist process details') == 'Papermaking artist process details'

    def test_already_capitalized_title_is_unchanged(self):
        assert _format_clip_title('The turning point') == 'The turning point'

    def test_proper_nouns_preserve_internal_capitals(self):
        # Don't lowercase Chris/Crane after we capitalize the leading word.
        assert _format_clip_title('chris on the Crane Estate hilltop') == 'Chris on the Crane Estate hilltop'

    def test_extra_whitespace_is_collapsed(self):
        assert _format_clip_title('  working   with   place  ') == 'Working with place'

    def test_empty_title_passes_through(self):
        assert _format_clip_title('') == ''

    def test_normalizer_applies_sentence_case(self):
        # The full normalizer pipeline should produce a sentence-case title
        # in the rewritten marker even when the model emitted lowercase.
        src = '[CLIP: start=00:12:00 end=00:13:00 title="papermaking artist process details"]'
        out = _normalize_clip_markers(src)
        assert '[CLIP: start=00:12:00 end=00:13:00 title="Papermaking artist process details"]' in out


class TestStripRawJsonBlobs:
    def test_screenshot_shape_is_stripped(self):
        # The exact JSON dump shape from the user-reported broken chat.
        src = (
            'The overall feeling moves from technical process to deep '
            'connection with the land and community.\n'
            '{\n'
            '"type": "highlight",\n'
            '"content": "The initial part focuses on the process.",\n'
            '"context": "The early discussion about making the art."\n'
            '},\n'
            '{\n'
            '"type": "highlight",\n'
            '"content": "The middle section shifts to history and community.",\n'
            '"context": "When discussing local ecology."\n'
            '}\n'
            '[CLIP: start=00:12:00 end=00:00:35 title="Process details"]'
        )
        out = _strip_raw_json_blobs(src)
        assert '"type"' not in out
        assert '"highlight"' not in out
        assert '"content"' not in out
        assert '"context"' not in out
        # Both the prose summary and the CLIP marker must survive intact.
        assert 'The overall feeling moves from technical process' in out
        assert '[CLIP: start=00:12:00 end=00:00:35 title="Process details"]' in out

    def test_clean_chat_response_strips_json_in_full_pipeline(self):
        src = (
            'Short summary line.\n'
            '{"type": "highlight", "content": "x", "context": "y"}\n'
            '[CLIP: start=00:00:10 end=00:00:25 title="Picked moment"]'
        )
        out = _clean_chat_response(src)
        assert '"type"' not in out
        assert '"highlight"' not in out
        assert 'Short summary line.' in out
        assert '[CLIP: start=00:00:10 end=00:00:25 title="Picked moment"]' in out

    def test_legitimate_braces_in_prose_are_preserved(self):
        # We should only strip blobs that look like the model's JSON leak
        # shape. A literal `{...}` in narration with no recognized keys
        # should pass through.
        src = 'Footage from {camera A} ran long.'
        out = _strip_raw_json_blobs(src)
        assert out == src

    def test_clip_markers_with_braces_in_prose_are_preserved(self):
        # The `[CLIP:]` marker has no curly braces, so it survives. Verify
        # the stripper doesn't accidentally chew through bracket content.
        src = '[CLIP: start=00:01:00 end=00:01:30 title="Opening shot"]'
        out = _strip_raw_json_blobs(src)
        assert out == src


def _marker_spans(text):
    """[(start_sec, end_sec)] for every fully-formed marker in ``text``."""
    return [
        (_tc_to_seconds(s), _tc_to_seconds(e))
        for s, e in re.findall(r'\[CLIP:[^\]]*?start=([\d:]+)[^\]]*?end=([\d:]+)', text)
    ]


def _marker_total(text):
    return sum(e - s for s, e in _marker_spans(text))


class TestEnforceDurationTarget:
    """Code-side duration guarantee for chat replies. The pool shapes match
    what the pipeline actually passes: Layer 1 matched paragraphs
    ({'start','end','text'}) and Layer 2 aggregated candidates
    ({'start_sec','end_sec','title','why','score'})."""

    def _paragraph_pool(self, n=6, dur=40, gap=10, offset=100.0):
        return [
            {'start': offset + i * (dur + gap), 'end': offset + i * (dur + gap) + dur,
             'text': f'paragraph {i} about the topic at hand'}
            for i in range(n)
        ]

    def test_within_band_is_unchanged(self):
        text = 'Take this.\n[CLIP: start=00:00:10 end=00:01:00 title="A"]'
        assert _enforce_duration_target(text, 60.0, self._paragraph_pool()) == text

    def test_top_up_from_paragraph_pool_reaches_floor(self):
        # 30s emitted against a 120s ask (floor 96s) → deterministic top-up.
        text = 'Here.\n[CLIP: start=00:00:10 end=00:00:40 title="Opening"]'
        out = _enforce_duration_target(text, 120.0, self._paragraph_pool())
        assert _marker_total(out) >= 0.8 * 120.0
        assert _marker_total(out) <= 1.25 * 120.0
        # Original prose and marker survive.
        assert out.startswith('Here.')
        assert 'start=00:00:10 end=00:00:40' in out

    def test_top_up_accepts_layer2_candidate_shape(self):
        pool = [
            {'start_sec': 500.0 + i * 60, 'end_sec': 530.0 + i * 60,
             'title': f'Moment {i}', 'why': 'strong beat', 'score': 9 - i}
            for i in range(8)
        ]
        out = _enforce_duration_target('no markers here', 180.0, pool)
        assert _marker_total(out) >= 0.8 * 180.0
        assert 'note="strong beat"' in out

    def test_top_up_skips_overlaps_with_emitted_clips(self):
        text = '[CLIP: start=00:01:40 end=00:02:20 title="Already here"]'  # 100-140
        pool = [
            {'start': 100.0, 'end': 140.0, 'text': 'duplicate of the emitted clip'},
            {'start': 300.0, 'end': 340.0, 'text': 'fresh material elsewhere'},
        ]
        out = _enforce_duration_target(text, 100.0, pool)
        spans = _marker_spans(out)
        assert (300.0, 340.0) in spans
        assert spans.count((100.0, 140.0)) == 1

    def test_top_up_has_no_60s_cap(self):
        # _deterministic_clip_markers truncates >60s paragraphs to 45s; the
        # duration pass must keep the full span or long asks can't be met.
        pool = [{'start': 200.0, 'end': 290.0, 'text': 'one long continuous story beat'}]
        out = _enforce_duration_target('x', 90.0, pool)
        assert (200.0, 290.0) in _marker_spans(out)

    def test_trim_drops_trailing_markers_past_ceiling(self):
        # 4 × 60s = 240s against a 60s ask (ceiling 75s) → trailing markers
        # (the weakest-ranked in every emitting path) are dropped.
        text = '\n'.join(
            f'[CLIP: start=00:0{i}:00 end=00:0{i + 1}:00 title="c{i}"]' for i in range(4)
        )
        out = _enforce_duration_target(text, 60.0, [])
        assert _marker_total(out) <= 1.25 * 60.0
        assert _marker_total(out) >= 0.8 * 60.0
        assert 'start=00:00:00' in out  # first (strongest) survives

    def test_trim_never_dips_below_floor(self):
        # Two 50s markers against a 60s ask: total 100 > 75 ceiling, but
        # dropping one leaves 50 >= 48 floor → exactly one is dropped.
        text = ('[CLIP: start=00:00:00 end=00:00:50 title="a"]\n'
                '[CLIP: start=00:02:00 end=00:02:50 title="b"]')
        out = _enforce_duration_target(text, 60.0, [])
        assert len(_marker_spans(out)) == 1

    def test_empty_pool_returns_text_unchanged_when_under(self):
        text = '[CLIP: start=00:00:00 end=00:00:10 title="short"]'
        assert _enforce_duration_target(text, 300.0, []) == text
        assert _enforce_duration_target(text, 300.0, None) == text

    def test_no_target_guard(self):
        text = '[CLIP: start=00:00:00 end=00:00:10 title="short"]'
        assert _enforce_duration_target(text, 0, self._paragraph_pool()) == text
        assert _enforce_duration_target('', 60.0, self._paragraph_pool()) == ''

    def test_top_up_sanitizes_brackets_in_note(self):
        # A model-emitted 'why' containing ']' or '"' must not produce a
        # marker that truncates downstream marker parsing.
        pool = [{'start_sec': 100.0, 'end_sec': 160.0, 'title': 'T',
                 'why': 'she said ["exactly] this', 'score': 9}]
        out = _enforce_duration_target('no markers', 60.0, pool)
        markers = _CLIP_MARKER_RE.findall(out)
        assert len(markers) == 1
        assert '[' not in markers[0][len('[CLIP:'):-1]
        assert ']' not in markers[0][len('[CLIP:'):-1]


class TestTrimRemovesAttachedProse:
    """A trimmed marker's explanation paragraph must go with it — the
    rendered reply must never describe clips that no longer exist. And
    markers whose note contains ']' must trim cleanly (the naive
    [^\\]]* marker regex left literal junk like 'bracket\"]' behind)."""

    THREE_CLIPS = (
        'Here are three.\n\n'
        '[CLIP: start=00:00:00 end=00:01:00 title="A" note="n1"]\n'
        'Why A matters.\n\n'
        '[CLIP: start=00:02:00 end=00:03:00 title="B" note="has ] bracket"]\n'
        'Why B matters.\n\n'
        '[CLIP: start=00:04:00 end=00:05:00 title="C"]\n'
        'Why C matters.\n\n'
        'Overall, a strong arc.'
    )

    def test_duration_trim_drops_orphaned_prose_and_bracket_residue(self):
        out = _enforce_duration_target(self.THREE_CLIPS, 60.0, [])
        assert out.count('[CLIP:') == 1
        assert 'Why A matters.' in out          # surviving clip keeps its note
        assert 'Why B matters.' not in out      # trimmed clips lose theirs
        assert 'Why C matters.' not in out
        assert 'bracket"]' not in out           # no malformed residue
        assert 'Overall, a strong arc.' in out  # unrelated closer survives

    def test_clip_count_trim_drops_orphaned_prose_too(self):
        out = _enforce_clip_count(self.THREE_CLIPS, 1)
        assert out.count('[CLIP:') == 1
        assert 'Why A matters.' in out
        assert 'Why B matters.' not in out
        assert 'Why C matters.' not in out
        assert 'bracket"]' not in out

    def test_marker_regex_tolerates_bracket_inside_quoted_note(self):
        marker = '[CLIP: start=00:00:00 end=00:00:30 title="A" note="he said [wow] there"]'
        found = _CLIP_MARKER_RE.findall(f'intro\n{marker}\nafter')
        assert found == [marker]

    def test_consecutive_lines_without_blank_separators(self):
        text = (
            'Opening.\n'
            '[CLIP: start=00:00:00 end=00:01:00 title="A"]\n'
            'Why A matters.\n'
            '[CLIP: start=00:02:00 end=00:03:00 title="B"]\n'
            'Why B matters.'
        )
        out = _enforce_duration_target(text, 60.0, [])
        assert out.count('[CLIP:') == 1
        assert 'Why A matters.' in out
        assert 'Why B matters.' not in out


class TestDurationAskPipelineRegression:
    """End-to-end guard for the live bug: 'give me 1 minute of selects' was
    misparsed as clip-count 1 and the reply hard-trimmed to ONE clip."""

    def _transcript(self):
        segments = []
        for i in range(20):
            start = i * 30.0
            segments.append({
                'start': start, 'end': start + 30.0,
                'start_formatted': f'{int(start) // 3600:02d}:{(int(start) % 3600) // 60:02d}:{int(start) % 60:02d}',
                'text': f'great selects material segment {i}',
                'speaker': 'A',
            })
        return {'segments': segments}

    def test_one_minute_ask_is_not_trimmed_to_one_clip(self):
        # Model emits three 20s markers (60s total — exactly the ask).
        reply = (
            'Three picks.\n'
            '[CLIP: start=00:00:00 end=00:00:20 title="Moment one"]\n'
            '[CLIP: start=00:01:00 end=00:01:20 title="Moment two"]\n'
            '[CLIP: start=00:02:00 end=00:02:20 title="Moment three"]'
        )
        with patch.object(ai_analysis, '_call_ai_chat', return_value=reply):
            out = ai_analysis.chat_about_transcript(
                self._transcript(), 'give me 1 minute of selects',
            )
        assert len(_marker_spans(out)) == 3, (
            f'duration ask was trimmed to {len(_marker_spans(out))} clip(s): {out!r}'
        )

    def test_one_minute_ask_tops_up_a_short_reply(self):
        # Model emits a single 20s marker against the 60s ask — the
        # deterministic pass extends from the retrieval pool.
        reply = 'One pick.\n[CLIP: start=00:00:00 end=00:00:20 title="Moment one"]'
        with patch.object(ai_analysis, '_call_ai_chat', return_value=reply):
            out = ai_analysis.chat_about_transcript(
                self._transcript(), 'give me 1 minute of selects',
            )
        assert _marker_total(out) >= 0.8 * 60.0

    def test_explicit_count_ask_still_trims(self):
        # The count path is untouched when no duration parses.
        reply = (
            '[CLIP: start=00:00:00 end=00:00:20 title="Moment one"]\n'
            '[CLIP: start=00:01:00 end=00:01:20 title="Moment two"]\n'
            '[CLIP: start=00:02:00 end=00:02:20 title="Moment three"]'
        )
        with patch.object(ai_analysis, '_call_ai_chat', return_value=reply):
            out = ai_analysis.chat_about_transcript(
                self._transcript(), 'find 2 clips about selects',
            )
        assert len(_marker_spans(out)) == 2

    def test_count_wins_over_narrative_fact_duration(self):
        # 'give me 3 clips from her 5 minute speech': the un-anchored
        # 5-minute mention used to co-parse as a duration target and
        # silently disable count enforcement (~9 clips came back). The
        # explicit count=3 must be enforced; no duration machinery runs.
        reply = '\n'.join(
            f'[CLIP: start=00:0{i}:00 end=00:0{i}:20 title="M{i}"]'
            for i in range(5)
        )
        with patch.object(ai_analysis, '_call_ai_chat', return_value=reply):
            out = ai_analysis.chat_about_transcript(
                self._transcript(), 'give me 3 clips from her 5 minute speech',
            )
        assert len(_marker_spans(out)) == 3, (
            f'expected count=3 enforcement, got {len(_marker_spans(out))}: {out!r}'
        )
