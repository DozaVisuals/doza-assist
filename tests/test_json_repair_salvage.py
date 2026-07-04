"""Truncated-JSON repair — salvage the complete leading elements.

The live bug: a 15-20 minute story ask made the model emit a 40-clip JSON
that overran its output-token budget and cut off mid-array. The old
_repair_truncated_json only closed open braces/brackets around the partial
tail, which is still invalid JSON — so ALL the complete leading clips were
thrown away, _parse_json_response fell through to its error dict, and the
story endpoint reported "The AI returned 0 clips". Repair must instead cut
back to the last complete element boundary inside the deepest open array,
drop the partial tail, and close what remains.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ai_analysis  # noqa: E402


def _story_json(n_clips=40):
    """A realistic verbose story response — long editorial notes (the 32B
    variant writes long ones) and embedded escaped quotes."""
    clips = [{
        'order': i + 1,
        'seg_id': f'SEG{i:03d}',
        'title': f'clip {i} with a "quoted" phrase',
        'start_time': f'00:{i:02d}:00',
        'end_time': f'00:{i:02d}:40',
        'editorial_note': (
            'ROLE: context — '
            + 'a long editorial note about why this moment matters, ' * 4
        ),
    } for i in range(n_clips)]
    body = {
        'story_title': 'Paranormal Investigation Cut',
        'target_duration': '17 minutes',
        'reasoning': 'a fairly long reasoning paragraph ' * 15,
        'clips': clips,
    }
    return json.dumps(body, indent=2), clips


class TestRepairSalvagesLeadingClips:
    def test_sweep_every_500_chars_yields_clean_clip_prefix(self):
        # The exact repro shape: cut the response at every 500-char step and
        # demand a parseable dict whose clips are a clean prefix of the
        # originals. Zero clips is acceptable only while the cut lands
        # before the first clip object has closed.
        full, clips = _story_json()
        first_clip_end = full.index('}', full.index('"clips"'))
        for cut in range(1000, len(full), 500):
            parsed = ai_analysis._parse_json_response(full[:cut])
            assert isinstance(parsed, dict), f'cut={cut}: no dict'
            assert 'error' not in parsed, f'cut={cut}: fell to the error dict'
            got = parsed.get('clips') or []
            assert got == clips[:len(got)], (
                f'cut={cut}: clips are not a clean prefix ({len(got)} clips)'
            )
            if cut > first_clip_end + 1:
                assert got, f'cut={cut}: complete clips existed but none salvaged'

    def test_mid_string_cut_drops_partial_clip_whole(self):
        # Cut inside clip 5's editorial_note string: clips 0-4 survive, the
        # half-written clip 5 is dropped entirely (a clip missing its
        # timecodes helps nobody downstream).
        full, clips = _story_json(8)
        cut = full.index('SEG005')  # inside clip 5's opening fields
        parsed = ai_analysis._parse_json_response(full[:cut])
        assert parsed.get('clips') == clips[:5]

    def test_cut_between_elements_keeps_all_complete_clips(self):
        full, clips = _story_json(6)
        # Just after clip 3's closing brace (before the comma/next element).
        end_of_clip_3 = -1
        for _ in range(4):
            end_of_clip_3 = full.index('},', end_of_clip_3 + 1)
        parsed = ai_analysis._parse_json_response(full[:end_of_clip_3 + 1])
        assert parsed.get('clips') == clips[:4]

    def test_untruncated_json_is_untouched(self):
        full, clips = _story_json(5)
        parsed = ai_analysis._parse_json_response(full)
        assert parsed['clips'] == clips
        assert parsed['story_title'] == 'Paranormal Investigation Cut'


class TestRepairEdgeShapes:
    def test_cut_before_any_complete_clip_yields_empty_array(self):
        repaired = ai_analysis._repair_truncated_json('{"clips": [{"order')
        assert json.loads(repaired) == {'clips': []}

    def test_primitive_array_keeps_comma_sealed_elements(self):
        repaired = ai_analysis._repair_truncated_json('{"a": [1, 2, 3')
        # The trailing 3 might itself be truncated (e.g. from 30) — only
        # comma-sealed primitives are trusted.
        assert json.loads(repaired) == {'a': [1, 2]}

    def test_objects_only_cut_keeps_complete_pairs(self):
        repaired = ai_analysis._repair_truncated_json(
            '{"story_title": "X", "reasoning": "cut mid-sente'
        )
        assert json.loads(repaired) == {'story_title': 'X'}

    def test_dangling_key_is_not_kept(self):
        # A bare key with no value must not survive ("key" alone is invalid).
        repaired = ai_analysis._repair_truncated_json('{"story_title": "X", "clips"')
        assert json.loads(repaired) == {'story_title': 'X'}

    def test_nothing_salvageable_returns_empty_shell(self):
        repaired = ai_analysis._repair_truncated_json('{"story_title": "Par')
        assert json.loads(repaired) == {}

    def test_escaped_quotes_do_not_break_string_tracking(self):
        text = '{"clips": [{"t": "say \\"boo\\""}, {"t": "half \\"quo'
        repaired = ai_analysis._repair_truncated_json(text)
        assert json.loads(repaired) == {'clips': [{'t': 'say "boo"'}]}


class TestBalancedJsonWithTrailingProse:
    """Complete JSON + trailing prose with an ODD quote count left the scan
    with an empty stack but in_str=True — the early return was skipped,
    both cut_frame searches found nothing in the empty stack, and
    stack[0] raised IndexError. That escaped _parse_json_response (only
    json.loads was wrapped) and crashed all 12 call sites, bypassing the
    story fallback. This shape is routine on the force_json=False retries,
    where JSON wrapped in prose is the EXPECTED output."""

    def test_trailing_prose_with_odd_quote_count_parses_the_json(self):
        parsed = ai_analysis._parse_json_response(
            '{"summary": "done"} Note: I omitted the "resolution beat'
        )
        assert parsed == {'summary': 'done'}

    def test_prose_wrapped_json_with_truncated_commentary(self):
        parsed = ai_analysis._parse_json_response(
            'Here is the story JSON:\n'
            '{"story_beats": [{"beat": "hook"}], "summary": "ok"}\n'
            'I picked "the reveal because it'
        )
        assert parsed == {'story_beats': [{'beat': 'hook'}], 'summary': 'ok'}

    def test_trailing_prose_with_even_quotes_still_degrades_gracefully(self):
        # Balanced trailing quotes take the structurally-complete early
        # return — pinned so both trailing-junk shapes stay crash-free.
        parsed = ai_analysis._parse_json_response(
            '{"summary": "done"} Note: I omitted the "resolution" beat'
        )
        assert isinstance(parsed, dict)

    def test_repair_with_no_balanced_prefix_returns_text_unchanged(self):
        # Nothing ever closed at the top level and nothing is open —
        # unreachable via _parse_json_response (which slices from '{'),
        # but the repair itself must not crash on it.
        assert ai_analysis._repair_truncated_json('no json "here') == 'no json "here'

    def test_repair_bug_cannot_crash_the_parse_contract(self, monkeypatch):
        # Belt-and-braces: even if the repair itself raises, the caller
        # degrades to the tolerant error dict — a repair bug must never
        # take down a caller again.
        def _boom(text):
            raise IndexError('list index out of range')

        monkeypatch.setattr(ai_analysis, '_repair_truncated_json', _boom)
        parsed = ai_analysis._parse_json_response('{"clips": [{"or')
        assert parsed.get('error') == 'Failed to parse AI response'
