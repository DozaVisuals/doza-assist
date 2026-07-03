"""Tests for query-intent classification and quantity hint extraction.

Lock in the user-visible behavior: synthesis queries skip strict keyword
anchoring (so the chat reasons editorially), and quantity hints from the
query control how many clips come back. Twice-reported regression — see
feedback memory.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_analysis import (  # noqa: E402
    _detect_explicit_clip_count,
    _extract_clip_count_from_message,
    _extract_query_keywords,
    _extract_quantity_hint,
    _is_conversational_query,
    _is_synthesis_query,
    _parse_user_clip_count,
    parse_target_duration_seconds,
)


class TestSynthesisDetection:
    def test_best_n_is_synthesis(self):
        assert _is_synthesis_query("whats the best 1 social media clip in this whole interview?")
        assert _is_synthesis_query("give me the best moment")
        assert _is_synthesis_query("top 3 highlights")

    def test_format_words_trigger_synthesis(self):
        assert _is_synthesis_query("pull a tiktok clip")
        assert _is_synthesis_query("make a reel")
        assert _is_synthesis_query("what would work for instagram?")

    def test_lookup_queries_not_synthesis(self):
        assert not _is_synthesis_query("what did she say about Moose Hill?")
        assert not _is_synthesis_query("when does he mention his father")
        assert not _is_synthesis_query("did the trustees come up?")

    def test_empty_message_not_synthesis(self):
        assert not _is_synthesis_query("")
        assert not _is_synthesis_query(None)


class TestQuantityHint:
    def test_digit_with_clip_noun(self):
        assert _extract_quantity_hint("best 1 social media clip") == 1
        assert _extract_quantity_hint("top 3 highlights please") == 3
        assert _extract_quantity_hint("give me 5 quotes") == 5

    def test_word_form_with_clip_noun(self):
        assert _extract_quantity_hint("a single highlight") == 1
        assert _extract_quantity_hint("two strongest moments") == 2
        assert _extract_quantity_hint("a few soundbites") == 3

    def test_number_without_clip_noun_returns_none(self):
        # Timecodes, ages, dates shouldn't trigger.
        assert _extract_quantity_hint("what did she say at 1:35?") is None
        assert _extract_quantity_hint("he was 12 years old") is None
        assert _extract_quantity_hint("in 1980 he started") is None

    def test_no_quantity_returns_none(self):
        assert _extract_quantity_hint("what's the best moment?") is None
        assert _extract_quantity_hint("show me anything good") is None

    def test_out_of_range_clamped_to_none(self):
        assert _extract_quantity_hint("give me 50 clips") is None  # absurd
        assert _extract_quantity_hint("0 clips") is None

    def test_clip_noun_can_have_filler_words_between(self):
        assert _extract_quantity_hint("best 1 social media clip") == 1
        assert _extract_quantity_hint("3 really powerful moments") == 3


class TestKeywordExtractionAfterStopwordExpansion:
    """The format/quality stopword expansion is what stops "best 1 social
    media clip" from anchoring on every paragraph containing 'social' or
    'media' or 'interview'."""

    def test_format_words_dropped(self):
        phrases, words = _extract_query_keywords("best 1 social media clip")
        # 'social', 'media', 'clip', 'best' should all be stopwords now.
        for w in ('social', 'media', 'clip', 'best'):
            assert w not in words, f"{w!r} should be stopword"

    def test_generic_content_nouns_dropped(self):
        phrases, words = _extract_query_keywords("the best moment in the whole interview")
        for w in ('whole', 'interview', 'moment', 'best'):
            assert w not in words

    def test_proper_noun_topics_kept(self):
        phrases, words = _extract_query_keywords("what did she say about Moose Hill")
        assert 'moose' in words
        assert 'hill' in words

    def test_synthesis_query_extracts_no_searchable_terms(self):
        # The compound failure mode the user hit: every word in this query
        # is either a stopword (under the expansion) or filler. Result:
        # zero search terms → no anchoring → the model reasons editorially
        # instead of literal-searching.
        phrases, words = _extract_query_keywords(
            "whats the best 1 social media clip in this whole interview?"
        )
        assert words == []
        assert phrases == []


class TestParseTargetDurationSeconds:
    """Shared duration-target parser — code-side, because the bundled Gemma
    models can't do arithmetic. Any regression here reopens the tester bug
    where a 14-minute ask returned 5.5 minutes."""

    @pytest.mark.parametrize("message,expected", [
        # The tester's exact Story Builder ask.
        ("build me a 14 minute paranormal investigation video with intro, "
         "investigation, arc, and conclusion", 840.0),
        # The chat misparse repro.
        ("give me 1 minute of selects", 60.0),
        # Positional reference — NOT a target.
        ("the moment at 14 minutes in", None),
        # Hyphenated adjective form.
        ("give me a 14-minute cut", 840.0),
        # Seconds / abbreviations.
        ("pull 90 sec for instagram", 90.0),
        ("build a 90 second teaser", 90.0),
        # Fractional hours.
        ("a 1.5 hour edit", 5400.0),
        # Word-number forms.
        ("give me one minute of selects", 60.0),
        ("give me two minutes of the best moments", 120.0),
        ("half an hour of highlights", 1800.0),
        ("a minute and a half of selects", 90.0),
        # mm:ss literal, anchored to a request.
        ("make a 2:30 highlight reel", 150.0),
        # Bare timecode in prose is a position, never a target.
        ("what did she say at 1:35?", None),
        # Source-footage duration is not the ask; the deliverable is.
        ("turn this 40 minute interview into a 5 minute cut", 300.0),
        # Positional lead-ins.
        ("the first 2 minutes", None),
        ("the 2 minute mark", None),
        # Per-clip length, not a total.
        ("give me 3 clips of 30 seconds each", None),
        # No duration at all.
        ("what's the emotional arc?", None),
        ("find 2 clips", None),
        ("", None),
        (None, None),
    ])
    def test_cases(self, message, expected):
        assert parse_target_duration_seconds(message) == expected

    @pytest.mark.parametrize("message", [
        # Content durations the speaker merely MENTIONS — a lone
        # un-anchored ('plain') duration must never become a target.
        # Honoring it activated enforcement and flooded locate answers
        # with clip cards (verified: 10800s / 600s / 120s misparses).
        "find the part where he says it took 3 hours to set up",
        "find the moment 10 minutes before the ending",
        "find the section where she spends 2 minutes describing the house",
        # Explicit count + narrative-fact duration: the duration must NOT
        # parse (so count=3 enforcement runs, see below).
        "give me 3 clips from her 5 minute speech",
    ])
    def test_unanchored_content_durations_return_none(self, message):
        assert parse_target_duration_seconds(message) is None

    @pytest.mark.parametrize("message", [
        # Relative/comparative adjustments — never a total-output ask.
        "cut 30 seconds off",
        "make it 30 seconds shorter",
        "make it 2 minutes shorter",
        # Per-clip length phrased with 'each' BEFORE the number.
        "make each clip 30 seconds long",
        # Chat filler: bare article + unit without deliverable context.
        "give me a second, checking something",
        "give me a sec",
        "give me a minute",
        # Plausibility floor: sub-15s "targets" are noise.
        "cut me a 5 second teaser",
        "give me 10 seconds",
    ])
    def test_adjustment_and_filler_phrases_return_none(self, message):
        assert parse_target_duration_seconds(message) is None

    def test_article_duration_with_deliverable_context_still_parses(self):
        # The article guard must not eat real asks.
        assert parse_target_duration_seconds("give me a minute of selects") == 60.0
        assert parse_target_duration_seconds("a minute and a half of selects") == 90.0


class TestExplicitCountWinsOverUnanchoredDuration:
    """'give me 3 clips from her 5 minute speech': the 5-minute narrative
    fact used to co-parse as a duration target, silently discarding the
    explicit count=3 (count enforcement is skipped whenever a duration
    parses). With plain durations dropped, the count wins."""

    MESSAGE = "give me 3 clips from her 5 minute speech"

    def test_duration_does_not_parse(self):
        assert parse_target_duration_seconds(self.MESSAGE) is None

    def test_count_parses(self):
        assert _detect_explicit_clip_count(self.MESSAGE) == 3


class TestDurationNotMisparsedAsClipCount:
    """Regression: 'give me 1 minute of selects' was parsed as clip-count 1
    by the bare-digit-after-verb pattern, then _enforce_clip_count trimmed
    the reply to ONE clip. Every count parser must refuse numbers attached
    to time units."""

    def test_give_me_1_minute_is_not_clip_count_1(self):
        assert _detect_explicit_clip_count("give me 1 minute of selects") is None

    def test_pull_90_seconds_is_not_clip_count_90(self):
        assert _detect_explicit_clip_count("pull 90 seconds for instagram") is None

    def test_give_me_2_minutes_is_not_clip_count_2(self):
        assert _detect_explicit_clip_count("give me 2 minutes of the best moments") is None

    def test_word_form_duration_is_not_clip_count(self):
        assert _detect_explicit_clip_count("give me one minute of selects") is None
        assert _detect_explicit_clip_count("another minute of the good stuff") is None
        assert _detect_explicit_clip_count("two more minutes of selects") is None

    def test_plain_counts_still_parse(self):
        assert _detect_explicit_clip_count("give me 3") == 3
        assert _detect_explicit_clip_count("find 2 clips") == 2
        assert _detect_explicit_clip_count("one more clip") == 1
        assert _detect_explicit_clip_count("just one") == 1
        assert _detect_explicit_clip_count("a few more") == 3

    def test_salvage_count_parser_ignores_durations(self):
        # "give me one" is a fixed count-1 pattern; the time unit after it
        # must push the parse back to the default.
        assert _extract_clip_count_from_message("give me one minute of selects", default=3) == 3
        assert _extract_clip_count_from_message("find 2 clips") == 2

    def test_layer2_count_parser_ignores_durations(self):
        assert _parse_user_clip_count("give me 2 minutes of highlights") is None
        assert _parse_user_clip_count("find me 2 clips about the bakery") == 2


class TestExtractiveVerbRouting:
    """'build/make/cut/create/assemble me an X-minute...' asks must reach
    the clip pipeline — they used to classify as conversational, so the
    duration machinery (and clip salvage) never ran."""

    @pytest.mark.parametrize("message", [
        "build me a 2-minute cut",
        "make a 90 second teaser",
        "cut a highlight reel from this",
        "create a montage of the best moments",
        "assemble a 3 minute sequence",
    ])
    def test_assembly_verbs_are_extractive(self, message):
        assert not _is_conversational_query(message)

    def test_discussion_stays_conversational(self):
        # First token is 'what', not an extractive verb — mid-sentence
        # 'make' must not flip the classification.
        assert _is_conversational_query("what do you make of her arc?")
        assert _is_conversational_query("no clips, just tell me the story")

    @pytest.mark.parametrize("message", [
        # Bare assembly verbs are ambiguous English. Without an anchored
        # duration or a deliverable noun, these are DISCUSSION — on
        # >60-min projects the extractive route returns clip-cards-only
        # and never answers the question in prose.
        "make sense of what she's saying about her mother",
        "cut to the chase — what is this interview really about?",
        "create a description of the subject's arc",
        "make it shorter",
        "cut down on the jargon when you answer me",
    ])
    def test_bare_assembly_verbs_without_anchor_stay_conversational(self, message):
        assert _is_conversational_query(message)

    @pytest.mark.parametrize("message", [
        # 'me'-suffixed forms stay unconditionally extractive.
        "build me a story from this",
        "make me something punchy",
        "cut me a teaser from the interview",
        # Bare verbs flip when a deliverable noun corroborates...
        "cut a highlight reel from this",
        "create a montage of the best moments",
        # ...or when an anchored duration target does.
        "make a 90 second teaser",
        "assemble a 3 minute sequence",
    ])
    def test_anchored_assembly_asks_are_extractive(self, message):
        assert not _is_conversational_query(message)
