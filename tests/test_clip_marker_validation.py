"""Tests for server-side timecode validation of [CLIP:] markers in chat
replies. Small models occasionally hallucinate timecodes that fall outside
the transcript — _validate_clip_markers_in_text catches those before the
frontend renders a "Play" button that scrubs to silence.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_analysis import _validate_clip_markers_in_text  # noqa: E402


SEGMENTS = [
    {'start': 0.0, 'end': 30.0, 'text': 'opening'},
    {'start': 30.0, 'end': 60.0, 'text': 'middle'},
    {'start': 60.0, 'end': 90.0, 'text': 'closing'},
]


def test_marker_inside_transcript_is_kept():
    text = '[CLIP: start=00:00:10 end=00:00:20 title="x"]'
    out = _validate_clip_markers_in_text(text, SEGMENTS)
    assert text in out


def test_marker_well_past_end_is_dropped():
    text = (
        'Here is the moment.\n'
        '[CLIP: start=99:99:99 end=99:99:99 title="hallucinated"]\n'
    )
    out = _validate_clip_markers_in_text(text, SEGMENTS)
    assert '[CLIP:' not in out


def test_grace_window_keeps_near_edge_markers():
    # Transcript ends at 90s. A marker starting at 91s is within the 5s grace —
    # still kept.
    text = '[CLIP: start=00:01:31 end=00:01:35 title="trailing"]'
    out = _validate_clip_markers_in_text(text, SEGMENTS)
    assert text in out


def test_marker_far_past_grace_is_dropped():
    # Transcript ends at 90s. 120s start is way past the 5s grace.
    text = '[CLIP: start=00:02:00 end=00:02:30 title="off the end"]'
    out = _validate_clip_markers_in_text(text, SEGMENTS)
    assert '[CLIP:' not in out


def test_mixed_good_and_bad_markers():
    text = (
        '[CLIP: start=00:00:05 end=00:00:25 title="good"]\n'
        '[CLIP: start=99:00:00 end=99:00:30 title="bad"]\n'
        '[CLIP: start=00:01:10 end=00:01:25 title="also good"]\n'
    )
    out = _validate_clip_markers_in_text(text, SEGMENTS)
    assert 'title="good"' in out
    assert 'title="also good"' in out
    assert 'title="bad"' not in out


def test_empty_segments_pass_through():
    # Without a transcript to validate against we can't drop anything.
    text = '[CLIP: start=00:00:10 end=00:00:20 title="x"]'
    assert _validate_clip_markers_in_text(text, []) == text


def test_no_markers_returns_text_unchanged():
    text = "Plain prose with no clip markers at all."
    assert _validate_clip_markers_in_text(text, SEGMENTS) == text


def test_unparseable_start_kept_for_frontend_to_handle():
    # If we can't parse the timecode at all, leave the marker in place
    # rather than silently dropping a possibly-valid clip.
    text = '[CLIP: start=not-a-time end=still-not title="x"]'
    out = _validate_clip_markers_in_text(text, SEGMENTS)
    assert text in out
