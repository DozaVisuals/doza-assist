"""Tests for _build_chat_analysis_index.

The chat model kept hallucinating timecodes and fabricating quotes because it
was chatting against a bare transcript with no anchor list of vetted moments.
_build_chat_analysis_index surfaces the Story Builder's pre-computed beats,
soundbites, and social clips as a compact timecoded menu the chat model can
cite from. These tests lock in the shape of that menu so a prompt change
doesn't quietly regress back to the "poetic hallucinations" behavior.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_analysis import _build_chat_analysis_index  # noqa: E402


class TestBuildChatAnalysisIndex:
    def test_none_returns_empty_string(self):
        # Callers concat the result unconditionally, so None must be "".
        assert _build_chat_analysis_index(None) == ""

    def test_empty_dict_returns_empty_string(self):
        assert _build_chat_analysis_index({}) == ""

    def test_non_dict_returns_empty_string(self):
        assert _build_chat_analysis_index("not a dict") == ""
        assert _build_chat_analysis_index([1, 2, 3]) == ""

    def test_dict_with_only_empty_lists_returns_empty_string(self):
        analysis = {'story_beats': [], 'strongest_soundbites': [], 'social_clips': []}
        assert _build_chat_analysis_index(analysis) == ""

    def test_story_beats_are_formatted_with_timecodes(self):
        analysis = {
            'story_beats': [
                {
                    'start': '00:05:12',
                    'end': '00:05:28',
                    'label': 'Opening Hook',
                    'why': 'Sets the stakes of the whole arc',
                },
            ]
        }
        out = _build_chat_analysis_index(analysis)
        assert 'STORY BEATS:' in out
        assert '[00:05:12-00:05:28]' in out
        assert 'Opening Hook' in out
        assert 'Sets the stakes of the whole arc' in out

    def test_strongest_soundbites_are_quoted(self):
        analysis = {
            'strongest_soundbites': [
                {
                    'start': '00:12:34',
                    'end': '00:12:50',
                    'text': 'I never thought it would end this way.',
                    'why': 'Most vulnerable moment',
                }
            ]
        }
        out = _build_chat_analysis_index(analysis)
        assert 'STRONGEST SOUNDBITES:' in out
        assert '[00:12:34-00:12:50]' in out
        assert '"I never thought it would end this way."' in out
        assert 'Most vulnerable moment' in out

    def test_social_clips_are_listed(self):
        analysis = {
            'social_clips': [
                {
                    'start': '00:27:04',
                    'end': '00:28:11',
                    'title': 'Rwanda turning point',
                    'why': 'Tight arc, strong hook',
                }
            ]
        }
        out = _build_chat_analysis_index(analysis)
        assert 'SOCIAL CLIPS:' in out
        assert '[00:27:04-00:28:11]' in out
        assert 'Rwanda turning point' in out

    def test_all_three_sections_appear_when_populated(self):
        analysis = {
            'story_beats': [{'start': '00:01:00', 'end': '00:01:30', 'label': 'A'}],
            'strongest_soundbites': [{'start': '00:02:00', 'end': '00:02:20', 'text': 'B'}],
            'social_clips': [{'start': '00:03:00', 'end': '00:03:30', 'title': 'C'}],
        }
        out = _build_chat_analysis_index(analysis)
        assert 'STORY BEATS:' in out
        assert 'STRONGEST SOUNDBITES:' in out
        assert 'SOCIAL CLIPS:' in out
        # Order matters for the model's attention — beats first (story spine),
        # then soundbites (citable quotes), then social (standalone moments).
        assert out.index('STORY BEATS:') < out.index('STRONGEST SOUNDBITES:')
        assert out.index('STRONGEST SOUNDBITES:') < out.index('SOCIAL CLIPS:')

    def test_items_missing_timecodes_are_skipped(self):
        # A beat with no start/end can't be cited as a [CLIP:] marker. Better
        # to drop it than emit a half-broken anchor.
        analysis = {
            'story_beats': [
                {'label': 'No timecode'},
                {'start': '00:01:00', 'end': '00:01:30', 'label': 'Has timecode'},
            ]
        }
        out = _build_chat_analysis_index(analysis)
        assert 'Has timecode' in out
        assert 'No timecode' not in out

    def test_numeric_seconds_are_converted_to_hhmmss(self):
        # The Story Builder sometimes returns raw seconds; the chat prompt
        # needs canonical HH:MM:SS so the model can copy them verbatim into
        # [CLIP:] markers.
        analysis = {
            'story_beats': [{'start': 312, 'end': 328, 'label': 'Beat'}]
        }
        out = _build_chat_analysis_index(analysis)
        assert '[00:05:12-00:05:28]' in out

    def test_long_descriptions_are_truncated(self):
        # The index is meant to be compact — a runaway description would
        # balloon the system prompt and crowd out the transcript itself.
        long_why = 'x' * 500
        analysis = {
            'story_beats': [{
                'start': '00:01:00', 'end': '00:01:30',
                'label': 'Beat', 'why': long_why,
            }]
        }
        out = _build_chat_analysis_index(analysis)
        # Truncation adds an ellipsis — the full 500 chars must not survive.
        assert 'x' * 500 not in out
        assert '…' in out

    def test_prefix_describes_the_list_as_canonical(self):
        # The model treats the section header as an instruction. The prefix
        # must signal "these are the real timecodes — prefer citing from
        # here" or the grounding rule loses its teeth.
        analysis = {
            'story_beats': [{'start': '00:01:00', 'end': '00:01:30', 'label': 'A'}]
        }
        out = _build_chat_analysis_index(analysis)
        assert 'PRE-ANALYZED MOMENTS' in out
        assert 'real timecodes' in out.lower()

    def test_malformed_items_are_silently_skipped(self):
        # A string or None inside the list (small-model drift) must not
        # crash the index builder — just skip it and emit what's valid.
        analysis = {
            'story_beats': [
                "oops a string",
                None,
                {'start': '00:01:00', 'end': '00:01:30', 'label': 'Good'},
            ]
        }
        out = _build_chat_analysis_index(analysis)
        assert 'Good' in out
        assert '00:01:00' in out

    def test_non_list_sections_are_ignored(self):
        # normalize_analysis guarantees lists, but defensive coding protects
        # against raw analysis dicts that skipped normalization.
        analysis = {
            'story_beats': "not a list",
            'strongest_soundbites': {'also': 'not a list'},
            'social_clips': [{'start': '00:01:00', 'end': '00:01:30', 'title': 'ok'}],
        }
        out = _build_chat_analysis_index(analysis)
        assert 'ok' in out
        # No section headers for the malformed ones.
        assert 'STORY BEATS:' not in out
        assert 'STRONGEST SOUNDBITES:' not in out
