"""Regression tests for _repair_truncated_json (the AI-analysis empty-results bug).

Ollama (/api/generate, format='json') stops mid-object when its reply runs past
num_predict (done_reason='length'). Token-verbose languages like German hit this
constantly. The old repair only closed open braces at the raw cut point, leaving a
dangling key/value -> structurally invalid JSON -> the whole reply (including its
complete leading items) was discarded -> "Analysis produced no results".

The rewrite rewinds to the last fully-formed element of the deepest still-open
container and closes the rest, so every COMPLETE item survives and only the
half-written trailing one is dropped. These tests pin that contract.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_analysis import _repair_truncated_json as repair  # noqa: E402
from ai_analysis import _parse_json_response  # noqa: E402


# Each case: a truncated buffer, and the count of complete soundbite objects that
# MUST survive (verbatim, with their data intact).
_SOUNDBITE_TRUNCATIONS = [
    # cut on a dangling key (no colon yet)
    ('{"strongest_soundbites": [{"text": "a", "start": "1"}, '
     '{"text": "b", "start": "2"}, {"text"', 2),
    # cut mid-string value
    ('{"strongest_soundbites": [{"text": "a", "start": "1"}, {"text": "hello wor', 1),
    # cut right after a key + colon
    ('{"strongest_soundbites": [{"text": "a", "start": "1"}, {"text":', 1),
    # cut mid-key
    ('{"strongest_soundbites": [{"text": "a", "start": "1"}, {"te', 1),
    # cut mid-scalar value
    ('{"items": [{"rank": 1}, {"rank": 2', 1),
]


@pytest.mark.parametrize('raw,survivors', _SOUNDBITE_TRUNCATIONS)
def test_truncations_parse_and_keep_complete_items(raw, survivors):
    repaired = repair(raw)
    obj = json.loads(repaired)  # must be valid JSON now
    # the list lives under whichever single key the object has
    (arr,) = list(obj.values())
    complete = [el for el in arr if isinstance(el, dict) and el]  # drop salvaged {}
    assert len(complete) == survivors


def test_nested_arrays_preserved():
    """A clip carrying a nested tags array survives whole; the dangling tail dies."""
    raw = '{"clips": [{"tags": ["x", "y"], "start": "1"}], "extra'
    obj = json.loads(repair(raw))
    assert obj['clips'] == [{'tags': ['x', 'y'], 'start': '1'}]


def test_truncated_inside_nested_array():
    raw = '{"clips": [{"tags": ["x", "y'
    obj = json.loads(repair(raw))
    # the half-written clip collapses to {} (dropped downstream); valid JSON either way
    assert isinstance(obj['clips'], list)


@pytest.mark.parametrize('valid', [
    '{"a": [1, 2, {"b": 3}]}',
    '{"clips": [{"tags": ["x", "y"], "start": "1"}]}',
    '{"strongest_soundbites": []}',
    '{}',
])
def test_already_valid_json_is_untouched(valid):
    """Idempotent: complete JSON must be returned byte-identical (it never reaches
    repair in practice, but the safety net must not corrupt it)."""
    assert repair(valid) == valid
    json.loads(repair(valid))


def test_just_opening_brace():
    assert json.loads(repair('{')) == {}


def test_parse_json_response_recovers_truncated_soundbites():
    """End-to-end through the real parser: a truncated reply that the OLD repair
    threw away now yields the recovered list key (not the error fallback dict)."""
    truncated = ('{"strongest_soundbites": [{"text": "real quote one", '
                 '"start": "00:01:00", "end": "00:01:10", "why": "thesis"}, '
                 '{"text": "second quote but cut off here mid')
    parsed = _parse_json_response(truncated)
    assert 'strongest_soundbites' in parsed
    complete = [s for s in parsed['strongest_soundbites'] if s]
    assert len(complete) == 1
    assert complete[0]['text'] == 'real quote one'
    assert 'error' not in parsed
