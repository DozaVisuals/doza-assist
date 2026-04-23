"""Tests for Layer 1 keyword pre-retrieval in the chat prompt.

The Trustees regression: on a 102-minute, 1,167-segment FCPXML project the
chat returned irrelevant clips from the opening of the transcript when the
user asked "find moments about moose hill" — even though "Moose Hill" appeared
13 times in the middle. Content fit in Gemma 4 e2b's 32k window; the cause
was attention/recency bias against middle-of-context tokens.

Layer 1 fix: when the user's question contains concrete keywords, extract
the matching paragraphs from the structured transcript and inject them in a
RELEVANT EXCERPTS block placed AFTER the full transcript (but before the
FINAL REMINDER). Recency bias then works FOR the answer.

These tests lock in:
  * keyword extraction drops stopwords, keeps topical terms, preserves
    multi-word phrases
  * phrase matches take priority over individual words
  * matched paragraphs include ±1 context
  * the RELEVANT EXCERPTS block lands AFTER the transcript, BEFORE the
    FINAL REMINDER, so recency bias lines up with the answer
  * Layer 1 never triggers on short transcripts — the current working path
    must not change
  * no keyword matches means no block at all (not an empty block with a
    misleading header) — this is the case that'll route to Layer 2 later
"""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ai_analysis  # noqa: E402


def _capture_system_prompt(transcript, message, **kwargs):
    captured = {}

    def fake_call(prompt, system_prompt=""):
        captured['system'] = system_prompt
        captured['user'] = prompt
        return "stub reply"

    with patch.object(ai_analysis, '_call_ai_chat', side_effect=fake_call):
        ai_analysis.chat_about_transcript(transcript, message, **kwargs)
    return captured


def _make_short_transcript_with_topic(topic_phrase, topic_positions, total_segments=80):
    """Build a <60-min transcript where `topic_phrase` appears only at the
    given segment indices. Segments run 30s each so total duration stays
    under the _LONG_CHAT_SECONDS threshold — Layer 1 only runs on short
    interviews now; long interviews route to Layer 2 which doesn't use the
    RELEVANT EXCERPTS block in a single system prompt.
    """
    segs = []
    t = 0.0
    seg_seconds = 30.0
    # 80 segments × 30s = 2400s = 40 min → under the 60-min threshold.
    for s_idx in range(total_segments):
        base_text = (
            f"{topic_phrase} is the focus here."
            if s_idx in topic_positions
            else "Generic filler sentence without the target."
        )
        segs.append({
            'start': t,
            'end': t + seg_seconds,
            'start_formatted':
                f"{int(t)//3600:02d}:{(int(t)%3600)//60:02d}:{int(t)%60:02d}.000",
            'text': base_text,
            'speaker': 'Chris',
        })
        t += seg_seconds
    return {'segments': segs}


class TestKeywordExtraction:
    def test_drops_stopwords(self):
        phrases, words = ai_analysis._extract_query_keywords(
            "what did they say about the budget?"
        )
        assert 'budget' in words
        assert 'the' not in words
        assert 'what' not in words
        assert 'did' not in words

    def test_keeps_multiword_phrase(self):
        phrases, _ = ai_analysis._extract_query_keywords(
            "find moments about moose hill"
        )
        assert 'moose hill' in phrases

    def test_phrase_broken_by_stopword(self):
        # "climate of moose hill" → "climate" is a single word, "moose hill"
        # is a preserved phrase. The preposition "of" breaks the run.
        phrases, words = ai_analysis._extract_query_keywords(
            "talk about the climate of moose hill"
        )
        assert 'moose hill' in phrases
        assert 'climate' in words
        assert 'climate moose hill' not in phrases

    def test_drops_short_tokens(self):
        _, words = ai_analysis._extract_query_keywords("it is ok")
        # 'ok' is only 2 chars, should be dropped along with stopwords.
        assert 'ok' not in words

    def test_empty_message_returns_empty(self):
        phrases, words = ai_analysis._extract_query_keywords("")
        assert phrases == []
        assert words == []

    def test_all_stopwords_returns_empty_words(self):
        phrases, words = ai_analysis._extract_query_keywords("what do you think")
        assert words == []
        assert phrases == []


class TestFindRelevantParagraphs:
    def _p(self, start, end, text, speaker='Chris'):
        return {'start': start, 'end': end, 'text': text, 'speaker': speaker}

    def test_substring_match_is_case_insensitive(self):
        paragraphs = [
            self._p(0, 60, "Opening monologue."),
            self._p(60, 120, "We went up to Moose Hill that morning."),
            self._p(120, 180, "Closing thought."),
        ]
        matched = ai_analysis._find_relevant_paragraphs(
            paragraphs, phrases=['moose hill'], words=['moose', 'hill']
        )
        texts = [p['text'] for p in matched]
        assert any('Moose Hill' in t for t in texts)

    def test_context_paragraphs_included(self):
        paragraphs = [
            self._p(0, 60, "Opening."),
            self._p(60, 120, "Lead-in."),
            self._p(120, 180, "Moose Hill moment."),
            self._p(180, 240, "Lead-out."),
            self._p(240, 300, "Closing."),
        ]
        matched = ai_analysis._find_relevant_paragraphs(
            paragraphs, phrases=['moose hill'], words=[]
        )
        texts = [p['text'] for p in matched]
        assert "Lead-in." in texts
        assert "Moose Hill moment." in texts
        assert "Lead-out." in texts
        # ±1 only — shouldn't sweep in Opening/Closing.
        assert "Opening." not in texts
        assert "Closing." not in texts

    def test_phrase_match_prevents_word_fallback(self):
        # Paragraph A has the full phrase; paragraph B only has "hill".
        # If phrase matched, word-level scan must NOT run (else we'd pull in
        # every "hill" on the timeline — the Trustees bug was about "Moose
        # Hill" specifically).
        paragraphs = [
            self._p(0, 60, "Opening."),
            self._p(60, 120, "We hiked Moose Hill that day."),
            self._p(120, 180, "Nothing relevant here."),
            self._p(180, 240, "Rolling hill country but unrelated topic."),
            self._p(240, 300, "Closing."),
        ]
        matched = ai_analysis._find_relevant_paragraphs(
            paragraphs, phrases=['moose hill'], words=['moose', 'hill']
        )
        texts = [p['text'] for p in matched]
        assert "We hiked Moose Hill that day." in texts
        # The unrelated "hill country" paragraph must not show up.
        assert not any('hill country' in t for t in texts)

    def test_word_fallback_triggers_only_when_no_phrase_hit(self):
        paragraphs = [
            self._p(0, 60, "Opening."),
            self._p(60, 120, "He mentioned budget constraints."),
            self._p(120, 180, "Unrelated topic."),
        ]
        # Phrase "moose hill" yields zero hits; word-level scan on "budget"
        # should kick in as fallback.
        matched = ai_analysis._find_relevant_paragraphs(
            paragraphs, phrases=['moose hill'], words=['budget']
        )
        texts = [p['text'] for p in matched]
        assert any('budget' in t for t in texts)

    def test_no_matches_returns_empty(self):
        paragraphs = [self._p(0, 60, "Just some text.")]
        matched = ai_analysis._find_relevant_paragraphs(
            paragraphs, phrases=['nothing matches'], words=['nothing']
        )
        assert matched == []

    def test_overlapping_context_is_deduped(self):
        # Two hits two paragraphs apart → their context windows overlap.
        # Result must be a flat unique list, not a repeated paragraph.
        paragraphs = [
            self._p(0, 60, "A"),
            self._p(60, 120, "target 1"),
            self._p(120, 180, "B"),
            self._p(180, 240, "target 2"),
            self._p(240, 300, "C"),
        ]
        matched = ai_analysis._find_relevant_paragraphs(
            paragraphs, phrases=[], words=['target']
        )
        starts = [p['start'] for p in matched]
        # Unique and in order.
        assert starts == sorted(set(starts))
        # Should cover indices 0..4 (all of them) because the two ±1 windows overlap.
        assert len(matched) == 5


class TestLayer1IntegrationIntoPrompt:
    """Layer 1 runs on short (<60 min) interviews with keyword matches. Long
    interviews bypass the single-prompt path entirely and go to Layer 2.
    """

    def test_excerpts_block_present_on_short_transcript_with_match(self):
        t = _make_short_transcript_with_topic(
            "Moose Hill", topic_positions={40, 41, 42}, total_segments=80
        )
        sys_prompt = _capture_system_prompt(t, "what did they say about moose hill?")['system']
        assert 'RELEVANT EXCERPTS' in sys_prompt

    def test_excerpts_block_lands_after_transcript_before_reminder(self):
        # Whole point of Layer 1: recency bias. Block MUST sit between the
        # transcript and the final reminder, not above the transcript.
        t = _make_short_transcript_with_topic(
            "Moose Hill", topic_positions={40, 41, 42}, total_segments=80
        )
        sys_prompt = _capture_system_prompt(t, "what did they say about moose hill?")['system']
        transcript_idx = sys_prompt.index('TRANSCRIPT:')
        excerpts_idx = sys_prompt.index('RELEVANT EXCERPTS')
        reminder_idx = sys_prompt.index('FINAL REMINDER')
        assert transcript_idx < excerpts_idx < reminder_idx

    def test_excerpts_block_omitted_when_no_keyword_match_short(self):
        # Vague questions (no concrete terms) on short interviews yield no
        # Layer 1 block — the main chat prompt runs unchanged.
        t = _make_short_transcript_with_topic(
            "Moose Hill", topic_positions={40, 41, 42}, total_segments=80
        )
        sys_prompt = _capture_system_prompt(t, "what do you think?")['system']
        assert 'RELEVANT EXCERPTS' not in sys_prompt

    def test_excerpts_block_contains_matched_timecodes(self):
        # Segments 40, 41, 42 at 30s/segment → 20:00, 20:30, 21:00.
        # Context ±2 adds 38, 39, 43, 44.
        t = _make_short_transcript_with_topic(
            "Moose Hill", topic_positions={40, 41, 42}, total_segments=80
        )
        sys_prompt = _capture_system_prompt(t, "moose hill moments please")['system']
        block = sys_prompt.split('RELEVANT EXCERPTS')[1].split('FINAL REMINDER')[0]
        # Matched hits themselves.
        assert '[00:20:00-00:20:30]' in block
        assert '[00:20:30-00:21:00]' in block
        assert '[00:21:00-00:21:30]' in block

    def test_full_transcript_still_present_alongside_excerpts(self):
        # Layer 1 AUGMENTS the prompt — it doesn't replace the full transcript.
        # The model still needs global context for follow-up questions.
        t = _make_short_transcript_with_topic(
            "Moose Hill", topic_positions={40}, total_segments=80
        )
        sys_prompt = _capture_system_prompt(t, "moose hill?")['system']
        transcript_block = sys_prompt.split(
            'TRANSCRIPT:', 1
        )[1].split('RELEVANT EXCERPTS', 1)[0]
        # 80 segments on a short interview → 80 per-segment lines.
        segment_lines = [
            ln for ln in transcript_block.splitlines()
            if ln.startswith('[') and ']' in ln
        ]
        assert len(segment_lines) >= 70, (
            "Layer 1 must augment, not replace, the full transcript. "
            f"Only found {len(segment_lines)} segment lines."
        )

    def test_long_interview_bypasses_layer1_single_prompt_path(self):
        # Long interviews route to Layer 2. The single-prompt stub we install
        # on _call_ai_chat therefore never fires — instead _call_ai_json gets
        # the calls. This test locks in the routing so a future change that
        # accidentally sent long interviews back through _call_ai_chat would
        # be caught.
        from unittest.mock import patch

        segs = []
        t = 0.0
        for _ in range(2000):  # 2000 × 5s = 10000s ≈ 166 min, long
            segs.append({
                'start': t, 'end': t + 5,
                'start_formatted': f"{int(t)//3600:02d}:{(int(t)%3600)//60:02d}:{int(t)%60:02d}.000",
                'text': 'Moose Hill was mentioned once.',
                'speaker': 'Chris',
            })
            t += 5

        chat_calls = []
        json_calls = []

        def fake_chat(*args, **kwargs):
            chat_calls.append(1)
            return "stub"

        def fake_json(*args, **kwargs):
            json_calls.append(1)
            return '{"candidates": []}'

        with patch.object(ai_analysis, '_call_ai_chat', side_effect=fake_chat), \
             patch.object(ai_analysis, '_call_ai_json', side_effect=fake_json):
            ai_analysis.chat_about_transcript({'segments': segs}, "moose hill?")

        # Long interview → Layer 2 only. Zero _call_ai_chat invocations.
        assert chat_calls == [], (
            "Long interviews must NOT touch the single-prompt Layer 1 path — "
            f"got {len(chat_calls)} _call_ai_chat invocations."
        )
        assert len(json_calls) >= 1, (
            "Long interviews must run Layer 2 per-chunk JSON calls — "
            f"got {len(json_calls)} _call_ai_json invocations."
        )
