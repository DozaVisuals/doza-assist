"""Content-preservation contracts for the chat cleaning pipeline.

The 1.0.32 chat rework's core invariant: no cleaning pass may delete
non-empty natural-language prose. The pre-rework pipeline deleted numbered
"1. Header: text" lines, "> " blockquoted quotes, follow-up offers,
"Label: content" lines, and whole prose lines that shared a line with an
invalid marker — which is how specific, grounded answers reached the user
as dangling headers ("Thematically, we are looking at:" followed by
nothing). These tests replay those exact failure shapes and pin the
preserving behavior, plus the grounding plumbing added around them
(payload-aware num_ctx, history compaction, content-lookup detection).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ai_analysis  # noqa: E402
from ai_analysis import (  # noqa: E402
    _build_chat_messages,
    _clean_chat_response,
    _compact_history_turn,
    _estimate_chat_num_ctx,
    _is_content_lookup_query,
    _strip_essay_scaffolding,
    _strip_meta_preamble,
    _strip_no_answer_placeholders,
    _strip_trailing_repetition,
    _validate_clip_markers_in_text,
)


def _segments(n=6, seg_len=25, gap=5):
    segs = []
    t = 0
    for i in range(n):
        segs.append({
            'start': t,
            'end': t + seg_len,
            'text': f'Segment {i} about the chestnut trees and the river.',
            'speaker': 'Mae',
        })
        t += seg_len + gap
    return segs


class TestEssayScaffoldingPreservesContent:
    def test_numbered_theme_list_survives(self):
        # The screenshot bug: a theme list under a colon header was deleted
        # wholesale, leaving "Thematically, we are looking at:" dangling.
        src = (
            "Thematically, we are looking at:\n\n"
            "1. Memory and Loss: how the land holds what people forget.\n"
            "2. Survival: the fight to keep the oral history alive.\n"
        )
        out = _strip_essay_scaffolding(src)
        assert 'Memory and Loss: how the land holds what people forget.' in out
        assert 'Survival: the fight to keep the oral history alive.' in out

    def test_blockquote_text_survives_as_quoted_prose(self):
        src = 'She names it directly:\n> I never told anyone this before.\nThat is the turn.'
        out = _strip_essay_scaffolding(src)
        assert 'I never told anyone this before.' in out
        assert not any(line.lstrip().startswith('>') for line in out.split('\n'))

    def test_already_quoted_blockquote_not_double_wrapped(self):
        src = '> "I never told anyone this before."'
        out = _strip_essay_scaffolding(src)
        assert out.count('“') == 0
        assert '"I never told anyone this before."' in out

    def test_fabrication_confessions_still_stripped(self):
        src = 'A strong opener (hypothetical selection) for the piece.'
        out = _strip_essay_scaffolding(src)
        assert 'hypothetical' not in out


class TestMetaPreamblePreservesContent:
    def test_follow_up_offers_survive(self):
        src = 'The strongest beat is the whale story.\nWould you like me to pull it as a clip?'
        out = _strip_meta_preamble(src)
        assert 'Would you like me to pull it as a clip?' in out

    def test_label_colon_content_survives_above_marker(self):
        src = (
            'Speakers: Mae Babcock and her daughter.\n'
            'Setup: the flood that took the archive.\n'
            '[CLIP: start=00:01:00 end=00:01:30 title="The flood"]'
        )
        out = _strip_meta_preamble(src)
        assert 'Speakers: Mae Babcock and her daughter.' in out
        assert 'Setup: the flood that took the archive.' in out

    def test_leading_bracketed_reasoning_still_stripped(self):
        src = ('[The user is asking for the main theme. I will summarize '
               'the transcript and select a moment.]\nThe theme is memory.')
        out = _strip_meta_preamble(src)
        assert 'user is asking' not in out
        assert 'The theme is memory.' in out

    def test_suggested_response_label_still_stripped(self):
        src = 'Suggested Response:\nThe theme is memory.'
        out = _strip_meta_preamble(src)
        assert 'Suggested Response' not in out
        assert 'The theme is memory.' in out


class TestNoAnswerPlaceholderNarrowing:
    def test_bracketed_placeholder_still_stripped(self):
        assert _strip_no_answer_placeholders('[No matches found]').strip() == ''

    def test_substantive_negative_answer_survives(self):
        src = ('No direct quotes about the fire exist, but she circles it '
               'twice — once at 12:40 and again near the end.')
        out = _strip_no_answer_placeholders(src)
        assert 'she circles it twice' in out


class TestCleanChatResponseScreenshotShapes:
    def test_theme_list_with_bold_labels_survives_end_to_end(self):
        # The full pipeline replay of the vague-answer screenshot: bold
        # numbered theme items must reach the frontend intact (the old
        # pipeline unwrapped the bold FIRST, then deleted the resulting
        # "1. Header: text" lines).
        src = (
            "Thematically, we are looking at:\n\n"
            "1. **Memory and Loss:** how the land holds what people forget.\n"
            "2. **Survival:** the fight to keep the oral history alive.\n"
        )
        out = _clean_chat_response(src)
        assert 'Memory and Loss' in out
        assert 'how the land holds what people forget' in out
        assert 'the fight to keep the oral history alive' in out

    def test_single_bracketed_timecode_unwraps_not_deletes(self):
        out = _clean_chat_response('She names the cost at [00:14:22] and it lands.')
        assert '00:14:22' in out
        assert '[00:14:22]' not in out

    def test_verbatim_quote_survives_the_pipeline(self):
        src = 'Her strongest line:\n> We were never on the map to begin with.\n'
        out = _clean_chat_response(src)
        assert 'We were never on the map to begin with.' in out


class TestValidatorKeepsSharedProse:
    def test_invalid_marker_in_prose_sentence_keeps_sentence(self):
        segs = _segments()
        src = ('The turn lands when she names the cost '
               '[CLIP: start=02:00:00 end=02:00:30 title="Naming the cost"] '
               'and the room goes quiet.')
        out = _validate_clip_markers_in_text(src, segs)
        assert 'The turn lands when she names the cost' in out
        assert '[CLIP:' not in out

    def test_marker_only_line_still_dropped_whole(self):
        segs = _segments()
        src = 'Here it is:\n[CLIP: start=02:00:00 end=02:00:30 title="Ghost"]'
        out = _validate_clip_markers_in_text(src, segs)
        assert '[CLIP:' not in out
        assert 'Ghost' not in out

    def test_auto_wrapped_moment_at_title_is_exempt_from_anchor_check(self):
        segs = _segments()
        # 'moment' appears nowhere in the segment texts near this window but
        # the mechanical title must never trigger the cross-reference drop.
        src = 'Look here [CLIP: start=00:00:05 end=00:00:20 title="Moment at 00:00:05"] first.'
        out = _validate_clip_markers_in_text(src, segs)
        assert 'title="Moment at 00:00:05"' in out


class TestPayloadAwareNumCtx:
    def test_bigger_payload_gets_bigger_window(self):
        small = _estimate_chat_num_ctx('sys', [{'role': 'user', 'content': 'x' * 1000}])
        big = _estimate_chat_num_ctx('s' * 40000, [
            {'role': 'user', 'content': 'x' * 60000},
            {'role': 'assistant', 'content': 'y' * 5000},
        ])
        assert small == 8192
        assert big > small

    def test_system_prompt_alone_counts(self):
        # The old estimator gave the system prompt a fixed 4096-token slack
        # the real prompt outgrew — the new one measures it.
        no_sys = _estimate_chat_num_ctx('', [{'role': 'user', 'content': 'x' * 30000}])
        with_sys = _estimate_chat_num_ctx('s' * 30000, [{'role': 'user', 'content': 'x' * 30000}])
        assert with_sys >= no_sys
        assert with_sys > 8192

    def test_ceiling_is_32k(self):
        assert _estimate_chat_num_ctx('s' * 400000, [
            {'role': 'user', 'content': 'x' * 400000}
        ]) == 32768


class TestHistoryHygiene:
    def _messages(self, message, history):
        segs = _segments()
        _, msgs = _build_chat_messages(
            message, history, 'Interview', segs,
            formatted='[00:00:00-00:00:25] Mae: hello.',
            analysis_block='', relevant_excerpts_block='', profile_id=None,
        )
        return msgs

    def test_trailing_duplicate_of_current_message_is_dropped(self):
        history = [
            {'role': 'user', 'content': 'earlier question'},
            {'role': 'assistant', 'content': 'earlier answer'},
            {'role': 'user', 'content': 'whats this all about'},
        ]
        msgs = self._messages('whats this all about', history)
        user_turns = [m['content'] for m in msgs if m['role'] == 'user']
        dupes = [c for c in user_turns if c.strip() == 'whats this all about']
        assert len(dupes) == 0  # the final turn carries the reminder tail
        finals = [c for c in user_turns if c.startswith('whats this all about')]
        assert len(finals) == 1

    def test_assistant_history_markers_compact_to_references(self):
        long_answer = (
            'Here are your picks.\n'
            '[CLIP: start=00:00:10 end=00:00:40 title="The whale story" note="Opens the piece"]\n'
            '[CLIP: start=00:01:00 end=00:01:30 title="Naming the cost"]\n'
        )
        history = [
            {'role': 'user', 'content': 'find moments'},
            {'role': 'assistant', 'content': long_answer},
        ]
        msgs = self._messages('and the theme?', history)
        replayed = [m['content'] for m in msgs if m['role'] == 'assistant'
                    and 'whale story' in m['content']]
        assert replayed, 'assistant turn should be replayed'
        assert '[CLIP:' not in replayed[0]
        assert '‣ The whale story (00:00:10–00:00:40)' in replayed[0]

    def test_long_assistant_history_is_capped(self):
        history = [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'word ' * 2000},
        ]
        msgs = self._messages('next?', history)
        replayed = [m['content'] for m in msgs if m['role'] == 'assistant'
                    and m['content'].startswith('word')]
        assert replayed and len(replayed[0]) <= 1210

    def test_compact_history_turn_direct(self):
        text = 'Intro.\n[CLIP: start=0:10 end=0:40 title="A"]\nOutro.'
        out = _compact_history_turn(text)
        assert '‣ A (0:10–0:40)' in out
        assert 'Intro.' in out and 'Outro.' in out


class TestContentLookupDetection:
    def test_positives(self):
        for msg in (
            'what does she say about chestnut trees?',
            'what did they say about the river?',
            'where does he talk about the flood?',
            'how does she describe the archive?',
            'did she ever mention her mother?',
            'does he talk about the funding?',
            'what was said about the merger?',       # passive
            'tell me what she says about the dam',    # imperative
            'what year did that happen?',             # factual-wh
        ):
            assert _is_content_lookup_query(msg), msg

    def test_negatives(self):
        for msg in (
            'whats this all about',
            'whats the actual story',
            'what do you think about the pacing?',
            'how should I open the piece?',
            'find me the strongest moments',
            'what would you say makes it work?',      # assistant-directed
            'what does that say about the piece?',    # demonstrative subject
            'what do these clips tell us?',           # demonstrative subject
            'what does it say about memory?',         # expletive subject
        ):
            assert not _is_content_lookup_query(msg), msg

    def test_no_clips_signal_suppresses_lookup(self):
        # An explicit no-clips instruction is a hard signal the grounding
        # tail and salvage backstop must not override with cards.
        assert not _is_content_lookup_query(
            'no clips, just tell me what she says about the fire')


class TestRepetitionGuardsRespectStructure:
    def test_punctuation_only_tail_is_not_trimmed(self):
        src = 'Solid list follows.\n\n' + ('-' * 40)
        assert _strip_trailing_repetition(src) == src

    def test_letter_pattern_repetition_is_trimmed_to_boundary(self):
        src = ('She lands the point cleanly. ' + 'and so on ' * 12)
        out = _strip_trailing_repetition(src)
        assert 'and so on and so on' not in out
        assert out.endswith('.')


class TestChatStopTokens:
    def test_no_prose_like_stop_sequences(self):
        # "[No specific answer" as a STOP hard-cut generation mid-reply;
        # the placeholder shape is handled post-generation instead.
        from ai_providers.ollama_provider import DEFAULT_STOP_TOKENS
        for stop in DEFAULT_STOP_TOKENS:
            assert 'No ' not in stop
