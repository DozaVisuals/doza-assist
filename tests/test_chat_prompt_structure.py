"""Tests for the chat system-prompt structure.

Long transcripts (FCPXML imports of 1hr+ timelines run 1000+ segments) pushed
the [CLIP:] output contract tens of thousands of tokens above the generation
point in the system prompt. Small models (Gemma 4 e4b especially) have strong
recency bias and defaulted to prose summaries, so the chat stopped rendering
clip cards on long projects even though short MP4s still worked.

The fix is to restate the contract AFTER the transcript so it's fresh in
context right before generation. These tests lock that structure in so a
future prompt refactor can't silently re-surface the long-transcript bug.
"""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ai_analysis  # noqa: E402


def _capture_system_prompt(transcript, message, **kwargs):
    """Run chat_about_transcript with _call_ai_chat stubbed, capture the system prompt."""
    captured = {}

    def fake_call(system_message, messages=None, **kwargs):
        # Matches the current _call_ai_chat(system_message, messages,
        # num_ctx=...) contract. The chat prompt was refactored from a
        # single combined string into a system message (which holds the
        # CHAT_SYSTEM_PROMPT + FINAL REMINDER contract) plus a messages
        # array (which holds the transcript / PRE-ANALYZED MOMENTS /
        # RELEVANT EXCERPTS in a user-role turn). These tests assert on the
        # whole prompt the model sees, so capture the concatenation —
        # system first, then each message's content in order.
        parts = [system_message or '']
        for m in (messages or []):
            if isinstance(m, dict) and m.get('content'):
                parts.append(m['content'])
        captured['system'] = '\n'.join(parts)
        captured['system_only'] = system_message or ''
        captured['messages'] = messages
        return "stub reply"

    with patch.object(ai_analysis, '_call_ai_chat', side_effect=fake_call):
        ai_analysis.chat_about_transcript(transcript, message, **kwargs)
    return captured


def _make_transcript(n_segments=3):
    segments = []
    for i in range(n_segments):
        start = i * 30
        end = start + 25
        segments.append({
            'start': start,
            'end': end,
            'start_formatted': f"{start//3600:02d}:{(start%3600)//60:02d}:{start%60:02d}.000",
            'text': f"Sample segment {i} text.",
            'speaker': 'Speaker',
        })
    return {'segments': segments}


class TestClipContractInSystemPrompt:
    """The clip-output contract must reach the model on every chat call.

    History: the original fix restated the contract in a trailing "FINAL
    REMINDER" section placed AFTER the transcript, to beat Gemma 4's recency
    bias on long FCPXML projects. The chat prompt was since refactored — the
    transcript moved into a user-role message and the full contract now lives
    in the dedicated ``CHAT_SYSTEM_PROMPT`` (``prompts/chat-system-prompt.md``)
    rather than a duplicated reminder block. These tests track the same
    anti-regression intent against the current structure: the system prompt
    must still (1) name the exact [CLIP:] marker syntax, (2) steer away from
    defaulting to prose, and (3) tell the model how to handle question types
    so synthesis/abstract questions don't lose their clip markers.
    """

    def _system(self, **kw):
        return _capture_system_prompt(_make_transcript(), "what did they say?", **kw)['system_only']

    def test_clip_marker_format_is_named(self):
        # The exact marker syntax must be in the model's working memory or it
        # can't emit parseable clips.
        sys_prompt = self._system()
        assert '[CLIP:' in sys_prompt
        assert 'start=' in sys_prompt
        assert 'end=' in sys_prompt
        assert 'title=' in sys_prompt

    def test_prompt_steers_away_from_prose_default(self):
        # The Trustees FCPXML symptom: model paraphrased instead of citing.
        # The prompt must distinguish prose answers from clip output.
        sys_prompt = self._system().lower()
        assert 'prose' in sys_prompt or 'paragraph' in sys_prompt

    def test_prompt_covers_question_type_handling(self):
        # The bug surfaced on synthesis/abstract questions ("what's the most
        # revealing thing"). The current prompt handles this via explicit
        # conversational / extractive / hybrid orientation rather than a
        # "synthesis" keyword — verify that orientation is present.
        sys_prompt = self._system().lower()
        assert 'conversational' in sys_prompt
        assert 'hybrid' in sys_prompt or 'extractive' in sys_prompt


class TestTranscriptStaysInPrompt:
    def test_transcript_segments_are_rendered_with_timecodes(self):
        sys_prompt = _capture_system_prompt(_make_transcript(3), "x")['system']
        # At least one formatted segment line should make it in.
        assert '[00:00:00-00:00:25]' in sys_prompt
        assert 'Sample segment 0 text.' in sys_prompt

    def test_analysis_block_is_appended_when_provided(self):
        analysis = {
            'story_beats': [
                {'start': '00:01:00', 'end': '00:01:30', 'label': 'Anchor Beat'}
            ]
        }
        sys_prompt = _capture_system_prompt(
            _make_transcript(), "x", analysis=analysis
        )['system']
        assert 'PRE-ANALYZED MOMENTS' in sys_prompt
        assert 'Anchor Beat' in sys_prompt

    def test_analysis_block_is_omitted_when_absent(self):
        # When analysis is None the chat still works — the prompt just
        # skips the PRE-ANALYZED MOMENTS *list* cleanly. (The phrase itself
        # appears in the grounding rule, which is fine — we only care that
        # no stale/empty anchor list gets injected.)
        sys_prompt = _capture_system_prompt(
            _make_transcript(), "x", analysis=None
        )['system']
        assert 'PRE-ANALYZED MOMENTS (real timecodes' not in sys_prompt
        # The transcript must still reach the model even with no analysis.
        assert 'TRANSCRIPT:' in sys_prompt


def _make_monologue_transcript(duration_seconds, segment_seconds=5.0, speaker='Chris'):
    """Build a single-speaker transcript covering [0, duration_seconds]."""
    segs = []
    t = 0.0
    i = 0
    while t < duration_seconds:
        end = min(t + segment_seconds, duration_seconds)
        segs.append({
            'start': t,
            'end': end,
            'start_formatted': f"{int(t)//3600:02d}:{(int(t)%3600)//60:02d}:{int(t)%60:02d}.000",
            'end_formatted': f"{int(end)//3600:02d}:{(int(end)%3600)//60:02d}:{int(end)%60:02d}.000",
            'text': f"Monologue line {i}.",
            'speaker': speaker,
        })
        t += segment_seconds
        i += 1
    return {'segments': segs}


class TestParagraphGroupingOnLongTranscripts:
    """Regression coverage for the original Trustees bug: a 102-minute,
    1,167-segment FCPXML project returned bare "[133]" because the per-segment
    prompt blew past Gemma 4's attention window. Paragraph grouping (merging
    adjacent same-speaker segments into ≤60s paragraphs) collapses ~1,170
    lines to ~110.

    Long transcripts now route through Layer 2 (chunked search) instead of the
    single-prompt path, so the grouping invariant is verified at the
    ``_build_paragraphs`` / ``_chunk_paragraphs`` level rather than against a
    captured system prompt.
    """

    def test_short_transcript_still_uses_per_segment_format(self):
        # ST Gala (47 min) is below threshold and currently produces rich
        # clip cards. The per-segment format must be preserved there so we
        # don't change behavior on working projects.
        t = _make_monologue_transcript(30 * 60, segment_seconds=5.0)
        sys_prompt = _capture_system_prompt(t, "x")['system']
        # Every segment should get its own line — look for several adjacent
        # timecode starts that the paragraph-grouping pass would have merged.
        assert '[00:00:00-00:00:05]' in sys_prompt
        assert '[00:00:05-00:00:10]' in sys_prompt
        assert '[00:00:10-00:00:15]' in sys_prompt

    def test_long_transcript_collapses_adjacent_same_speaker_segments(self):
        # 90-minute monologue → one paragraph per 60-second window, not one
        # line per 5-second segment. That's the ~10x line reduction that
        # keeps Gemma 4 on the [CLIP:] contract inside each Layer 2 chunk.
        t = _make_monologue_transcript(90 * 60, segment_seconds=5.0)
        paragraphs = ai_analysis._build_paragraphs(t)

        starts = [p['start'] for p in paragraphs]
        ends = [p['end'] for p in paragraphs]

        # First paragraph spans 0 → 60s, not 0 → 5s.
        assert starts[0] == 0
        assert ends[0] == 60, f"Expected first paragraph to span 60s; got {ends[0]}s"

        # Paragraph text concatenates adjacent segments' text with spaces.
        assert 'Monologue line 0.' in paragraphs[0]['text']
        assert 'Monologue line 5.' in paragraphs[0]['text']

    def test_long_transcript_paragraph_count_is_dramatically_smaller(self):
        # Lock in the size-reduction invariant. 90 min at 5s/segment = 1,080
        # segments; paragraph grouping at 60s cap yields ~90 paragraphs.
        t = _make_monologue_transcript(90 * 60, segment_seconds=5.0)
        paragraphs = ai_analysis._build_paragraphs(t)
        assert len(paragraphs) < 200, (
            f"Expected paragraph grouping to reduce count well below the "
            f"1,080-segment raw count; got {len(paragraphs)} paragraphs."
        )

    def test_paragraph_grouping_preserves_speaker_turns(self):
        # When speaker changes, paragraphs must break — a single paragraph
        # must never contain lines from two different speakers, or Layer 2
        # clip cards would attribute speech to the wrong person.
        segs = []
        t = 0.0
        for speaker in ('Chris', 'Amanda'):
            for _ in range(int(40 * 60 / 5)):
                segs.append({
                    'start': t, 'end': t + 5,
                    'start_formatted': f"{int(t)//3600:02d}:{(int(t)%3600)//60:02d}:{int(t)%60:02d}.000",
                    'text': f"line from {speaker}",
                    'speaker': speaker,
                })
                t += 5
        paragraphs = ai_analysis._build_paragraphs({'segments': segs})
        for p in paragraphs:
            # Each paragraph has exactly one speaker (property of _build_paragraphs).
            assert p['speaker'] in ('Chris', 'Amanda')
            # Belt-and-suspenders: the merged text doesn't name both speakers.
            assert not (
                'line from Chris' in p['text'] and 'line from Amanda' in p['text']
            ), f"Paragraph bridged speaker switch: {p!r}"
        # Both speakers are represented in the output.
        speakers_present = {p['speaker'] for p in paragraphs}
        assert speakers_present == {'Chris', 'Amanda'}
