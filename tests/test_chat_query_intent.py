"""Tests for query-intent classification and quantity hint extraction.

Lock in the user-visible behavior: synthesis queries skip strict keyword
anchoring (so the chat reasons editorially), and quantity hints from the
query control how many clips come back. Twice-reported regression — see
feedback memory.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_analysis import (  # noqa: E402
    _extract_query_keywords,
    _extract_quantity_hint,
    _is_synthesis_query,
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
