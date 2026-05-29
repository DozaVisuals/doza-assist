"""BUG-01: structured-analysis timecode validation against transcript segments.

Pure-function tests for ``ai_analysis._validate_clip_timecodes`` — no model
calls. The validator is the structured-output counterpart to the chat path's
``_validate_clip_markers_in_text``: it checks each emitted HH:MM:SS start/end
against the real transcript segments, repairs soundbites/social clips via
verbatim-text match, and drops anything it can't confidently anchor.
"""
import ai_analysis as A

# A 30-second, 4-segment transcript.
SEGMENTS = [
    {"start": 0.0,  "end": 6.0,  "text": "Welcome to the show today."},
    {"start": 6.0,  "end": 14.0, "text": "I grew up in a small fishing town on the coast."},
    {"start": 14.0, "end": 22.0, "text": "My father taught me everything about the sea."},
    {"start": 22.0, "end": 30.0, "text": "We lost the boat in the storm of ninety three."},
]


def test_anchored_clip_kept_unchanged():
    clips = [{"text": "My father taught me everything about the sea.",
              "start": "00:00:14", "end": "00:00:22"}]
    out = A._validate_clip_timecodes(clips, SEGMENTS, text_key="text", kind="soundbite")
    assert len(out) == 1
    assert out[0]["start"] == "00:00:14"
    assert out[0]["end"] == "00:00:22"


def test_hallucinated_timecode_with_matching_quote_is_repaired():
    # start is hours outside the 30s transcript, but the verbatim quote matches
    # segment #3 (14-22s); repair must snap to that segment's real boundaries.
    clips = [{"text": "My father taught me everything about the sea.",
              "start": "02:31:14", "end": "02:31:30"}]
    out = A._validate_clip_timecodes(clips, SEGMENTS, text_key="text", kind="soundbite")
    assert len(out) == 1
    assert out[0]["start"] == "00:00:14"
    assert out[0]["end"] == "00:00:22"


def test_multi_segment_quote_spans_the_matching_run():
    # Quote covers segments #2 and #3 verbatim -> snap start of #2, end of #3.
    quote = ("I grew up in a small fishing town on the coast. "
             "My father taught me everything about the sea.")
    clips = [{"text": quote, "start": "09:09:09", "end": "09:09:30"}]
    out = A._validate_clip_timecodes(clips, SEGMENTS, text_key="text", kind="soundbite")
    assert len(out) == 1
    assert out[0]["start"] == "00:00:06"
    assert out[0]["end"] == "00:00:22"


def test_hallucinated_timecode_no_text_match_is_dropped():
    clips = [{"text": "Something never uttered anywhere in this interview at all.",
              "start": "02:31:14", "end": "02:31:30"}]
    out = A._validate_clip_timecodes(clips, SEGMENTS, text_key="text", kind="soundbite")
    assert out == []


def test_beat_anchored_is_kept():
    clips = [{"label": "The loss", "description": "emotional low point",
              "start": "00:00:22", "end": "00:00:30"}]
    out = A._validate_clip_timecodes(clips, SEGMENTS, text_key=None, kind="story beat")
    assert len(out) == 1
    assert out[0]["start"] == "00:00:22"


def test_beat_unanchored_is_dropped_not_relocated():
    # No verbatim text => never relocate; an unanchored beat is dropped.
    clips = [{"label": "The loss", "description": "emotional low point",
              "start": "02:31:14", "end": "02:31:30"}]
    out = A._validate_clip_timecodes(clips, SEGMENTS, text_key=None, kind="story beat")
    assert out == []


def test_end_overrun_is_clamped_to_transcript_end():
    clips = [{"text": "We lost the boat in the storm of ninety three.",
              "start": "00:00:22", "end": "00:05:00"}]  # end far past the 30s end
    out = A._validate_clip_timecodes(clips, SEGMENTS, text_key="text", kind="soundbite")
    assert len(out) == 1
    assert out[0]["start"] == "00:00:22"
    assert out[0]["end"] == "00:00:30"


def test_no_segments_returns_clips_unchanged():
    clips = [{"text": "x", "start": "99:99:99"}]
    assert A._validate_clip_timecodes(clips, [], text_key="text") == clips


def test_missing_start_left_untouched():
    # A clip with no timecode is a separate concern from a hallucinated one;
    # it must not be dropped just for lacking a start.
    clips = [{"label": "beat", "description": "d", "start": ""}]
    out = A._validate_clip_timecodes(clips, SEGMENTS, text_key=None, kind="story beat")
    assert len(out) == 1


def test_every_surviving_timecode_is_anchored():
    # Mixed batch: 1 valid, 1 repairable, 1 droppable. No survivor may carry a
    # start outside the real transcript window.
    clips = [
        {"text": "My father taught me everything about the sea.",
         "start": "00:00:14", "end": "00:00:22"},                 # valid
        {"text": "We lost the boat in the storm of ninety three.",
         "start": "01:00:00", "end": "01:00:08"},                 # repairable
        {"text": "Entirely fabricated sentence with no anchor.",
         "start": "03:03:03", "end": "03:03:20"},                 # droppable
    ]
    out = A._validate_clip_timecodes(clips, SEGMENTS, text_key="text", kind="soundbite")
    assert len(out) == 2  # the fabricated one is gone
    for c in out:
        sec = A._tc_to_seconds(c["start"])
        assert 0.0 <= sec <= 30.0
