"""Clip edges keep sub-second precision end to end.

The transcript lines the model copies from, and every marker the app builds
from float seconds, used to floor both edges to whole seconds, so a clip
could open on the tail of the previous sentence and clip its own last word."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ai_analysis  # noqa: E402
from ai_analysis import _seconds_to_tc, _seconds_to_tc_frac, _tc_to_seconds  # noqa: E402


@pytest.mark.parametrize("value,expected", [
    (0, "00:00:00"), (83, "00:01:23"), (83.0, "00:01:23"),
    (83.456, "00:01:23.456"), (83.4, "00:01:23.4"), (83.9996, "00:01:24"),
    (3725.25, "01:02:05.25"), (-1, "00:00:00"), (None, "00:00:00"), ("x", "00:00:00"),
])
def test_frac_serializer(value, expected):
    assert _seconds_to_tc_frac(value) == expected


def test_whole_second_serializer_unchanged():
    assert _seconds_to_tc(83.9) == "00:01:23"


@pytest.mark.parametrize("value", [12.4, 15.9, 3725.25, 0.5])
def test_round_trip_through_parser(value):
    assert _tc_to_seconds(_seconds_to_tc_frac(value)) == pytest.approx(value)


def test_transcript_lines_carry_fractional_edges():
    transcript = {'segments': [
        {'start': 12.4, 'end': 15.9, 'start_formatted': '00:00:12.400',
         'speaker': 'Ana', 'text': 'We never looked back.'},
        {'start': 16.0, 'end': 20.0, 'start_formatted': '00:00:16.000',
         'speaker': 'Ana', 'text': 'Not once.'},
    ]}
    out = ai_analysis._format_transcript_for_ai(transcript)
    assert '[00:00:12.4-00:00:15.9] Ana: We never looked back.' in out
    assert '[00:00:16-00:00:20] Ana: Not once.' in out


def test_paragraph_lines_carry_fractional_edges():
    lines = ai_analysis._format_paragraphs_as_lines([
        {'start': 61.25, 'end': 70.75, 'speaker': 'Ana', 'text': 'hello'},
    ])
    assert '[00:01:01.25-00:01:10.75]' in lines
