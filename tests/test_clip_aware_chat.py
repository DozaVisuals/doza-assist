"""Tests for clip-aware chat — the conditional swap that replaces the
pre-analyzed moments block with an enriched <editor_selections> block when
the editor has clips in labeled_sections.

Covers:
  - Speaker lookup from transcript segments
  - Chronological sort + sequential indexing
  - Text truncation when over the 4,074-char ceiling
  - Conditional swap (clips present → swap; absent → legacy behavior)
  - Feature flag (DOZA_DISABLE_CLIP_AWARE_CHAT=1 reverts to legacy)
  - Framing paragraph variants (My Style ON vs OFF)
"""
import os
import sys
import importlib
import unittest

# Make the core dir importable when tests are run from any cwd
HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.dirname(HERE)
if CORE not in sys.path:
    sys.path.insert(0, CORE)

import ai_analysis  # noqa: E402


def _segments():
    return [
        {'start': 0.0,    'end': 10.0,   'speaker': 'Host',  'text': 'Welcome.'},
        {'start': 10.0,   'end': 20.0,   'speaker': 'Guest', 'text': 'Thanks.'},
        {'start': 600.0,  'end': 610.0,  'speaker': 'Guest', 'text': 'Pivotal.'},
        {'start': 1200.0, 'end': 1210.0, 'speaker': 'Host',  'text': 'Tell me more.'},
    ]


def _clips_unsorted():
    return [
        {'id': 5, 'start': 600.0,  'end': 612.0,  'color': 'blue',  'text': 'Pivot'},
        {'id': 1, 'start': 10.0,   'end': 22.0,   'color': 'green', 'text': 'Open'},
        {'id': 3, 'start': 1200.0, 'end': 1215.0, 'color': 'red',   'text': 'Probe'},
    ]


class TestEditorSelectionsBlock(unittest.TestCase):
    def test_empty_input_returns_empty_string(self):
        self.assertEqual(ai_analysis._build_editor_selections_block([], _segments()), '')
        self.assertEqual(ai_analysis._build_editor_selections_block(None, _segments()), '')

    def test_chronological_sort_and_indexing(self):
        block = ai_analysis._build_editor_selections_block(_clips_unsorted(), _segments())
        # The clip starting at 10s should be SELECTED 1, not the one with id=1
        self.assertIn('[SELECTED 1] [00:00:10', block)
        self.assertIn('[SELECTED 2] [00:10:00', block)
        self.assertIn('[SELECTED 3] [00:20:00', block)

    def test_speaker_lookup(self):
        block = ai_analysis._build_editor_selections_block(_clips_unsorted(), _segments())
        self.assertIn('Speaker: Guest', block)
        self.assertIn('Speaker: Host', block)

    def test_truncation_when_over_budget(self):
        # 30 clips × 500-char texts will be over the 4,074-char cap
        fat_clips = [
            {'id': i, 'start': i * 60.0, 'end': i * 60.0 + 10, 'text': 'X' * 500}
            for i in range(1, 30)
        ]
        fat_segs = [
            {'start': i * 60.0, 'end': i * 60.0 + 10, 'speaker': 'S', 'text': 'X'}
            for i in range(1, 30)
        ]
        block = ai_analysis._build_editor_selections_block(fat_clips, fat_segs)
        # Allow some slack for XML scaffolding overhead beyond the cap
        self.assertLessEqual(len(block), ai_analysis._CLIP_AWARE_MAX_CHARS + 200)
        # Truncated lines should end with an ellipsis after 50 X's
        self.assertIn('X' * 50 + '…', block)

    def test_block_wraps_in_xml_tags(self):
        block = ai_analysis._build_editor_selections_block(_clips_unsorted(), _segments())
        self.assertTrue(block.startswith('<editor_selections>'))
        self.assertIn('</editor_selections>', block)
        self.assertIn('Total selections: 3', block)


class TestConditionalSwap(unittest.TestCase):
    def setUp(self):
        # Ensure feature flag is on for these tests
        os.environ.pop('DOZA_DISABLE_CLIP_AWARE_CHAT', None)
        importlib.reload(ai_analysis)
        self.analysis_block = ai_analysis._build_chat_analysis_index({
            'story_beats': [{'start': '00:01:00', 'end': '00:02:00', 'label': 'Hook'}],
            'strongest_soundbites': [{'start': '00:05:00', 'end': '00:05:30', 'text': 'line'}],
        })
        self.assertIn('PRE-ANALYZED MOMENTS', self.analysis_block)

    def test_with_clips_swaps_to_editor_selections(self):
        sys_msg, msgs = ai_analysis._build_chat_messages(
            'q', [], 'Test', _segments(), 'TRANSCRIPT',
            self.analysis_block, '',
            profile_id=None, labeled_sections=_clips_unsorted(),
        )
        trans_msg = next(m['content'] for m in msgs if 'TRANSCRIPT' in m['content'])
        self.assertIn('<editor_selections>', trans_msg)
        self.assertNotIn('PRE-ANALYZED MOMENTS', trans_msg)
        # Framing paragraph added to system prompt
        self.assertTrue('two layers' in sys_msg or 'three layers' in sys_msg)

    def test_without_clips_keeps_legacy_behavior(self):
        sys_msg, msgs = ai_analysis._build_chat_messages(
            'q', [], 'Test', _segments(), 'TRANSCRIPT',
            self.analysis_block, '',
            profile_id=None, labeled_sections=None,
        )
        trans_msg = next(m['content'] for m in msgs if 'TRANSCRIPT' in m['content'])
        self.assertNotIn('<editor_selections>', trans_msg)
        self.assertIn('PRE-ANALYZED MOMENTS', trans_msg)
        self.assertNotIn('two layers', sys_msg)
        self.assertNotIn('three layers', sys_msg)

    def test_empty_clips_list_keeps_legacy_behavior(self):
        sys_msg, msgs = ai_analysis._build_chat_messages(
            'q', [], 'Test', _segments(), 'TRANSCRIPT',
            self.analysis_block, '',
            profile_id=None, labeled_sections=[],
        )
        trans_msg = next(m['content'] for m in msgs if 'TRANSCRIPT' in m['content'])
        self.assertNotIn('<editor_selections>', trans_msg)
        self.assertIn('PRE-ANALYZED MOMENTS', trans_msg)


class TestFeatureFlag(unittest.TestCase):
    def tearDown(self):
        os.environ.pop('DOZA_DISABLE_CLIP_AWARE_CHAT', None)
        importlib.reload(ai_analysis)

    def test_flag_default_is_on(self):
        os.environ.pop('DOZA_DISABLE_CLIP_AWARE_CHAT', None)
        importlib.reload(ai_analysis)
        self.assertTrue(ai_analysis._clip_aware_chat_enabled())

    def test_flag_disabled_reverts_to_legacy(self):
        os.environ['DOZA_DISABLE_CLIP_AWARE_CHAT'] = '1'
        importlib.reload(ai_analysis)
        self.assertFalse(ai_analysis._clip_aware_chat_enabled())

        analysis_block = ai_analysis._build_chat_analysis_index({
            'story_beats': [{'start': '00:01:00', 'end': '00:02:00', 'label': 'Hook'}],
        })
        sys_msg, msgs = ai_analysis._build_chat_messages(
            'q', [], 'Test', _segments(), 'TRANSCRIPT',
            analysis_block, '',
            profile_id=None, labeled_sections=_clips_unsorted(),
        )
        trans_msg = next(m['content'] for m in msgs if 'TRANSCRIPT' in m['content'])
        # Flag off → behave as if no clips present
        self.assertNotIn('<editor_selections>', trans_msg)
        self.assertIn('PRE-ANALYZED MOMENTS', trans_msg)


class TestFramingParagraph(unittest.TestCase):
    def test_my_style_on_variant(self):
        framing = ai_analysis._build_clip_aware_framing(True)
        self.assertIn('three layers', framing)
        # The framing must name the label the style block actually ships
        # under — 'STYLE CONTEXT' — not a tag that never appears in the
        # chat context (the old '<storytelling_foundation>' reference was
        # dangling: chat never injects that block).
        self.assertIn('STYLE CONTEXT', framing)
        self.assertIn('editorial patterns', framing)

    def test_my_style_off_variant(self):
        framing = ai_analysis._build_clip_aware_framing(False)
        self.assertIn('two layers', framing)
        self.assertNotIn('storytelling_foundation', framing)
        self.assertNotIn('three layers', framing)


if __name__ == '__main__':
    unittest.main()
