"""Tests for ai_analysis.normalize_analysis.

Small/fast local models (notably Gemma 4 e4b) drift on the requested JSON
schema — they'll emit ``beat_description`` instead of ``description``,
``start_time`` instead of ``start``, or put the list under a differently-named
top-level key. The normalizer accepts those variants so the templates, which
read canonical keys, still render correctly.

These tests lock in the mapping so a future "improvement" to the model prompt
doesn't silently regress existing broken data on disk.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_analysis import normalize_analysis  # noqa: E402


class TestStoryBeats:
    def test_passthrough_canonical_shape(self):
        a = {
            "story_beats": [
                {"label": "Opening Hook", "description": "Why this works",
                 "start": "00:00:45", "end": "00:01:02", "order": 1},
            ]
        }
        out = normalize_analysis(a)
        beat = out["story_beats"][0]
        assert beat["label"] == "Opening Hook"
        assert beat["description"] == "Why this works"
        assert beat["start"] == "00:00:45"
        assert beat["end"] == "00:01:02"

    def test_aliases_beat_description_and_time_fields(self):
        # The exact shape Gemma 4 e4b returned for the Trustees project.
        a = {
            "story_beats": [
                {"beat_description": "Introduction to cyclical nature.",
                 "start_time": "0:00", "end_time": "0:02:10"},
            ]
        }
        out = normalize_analysis(a)
        beat = out["story_beats"][0]
        assert beat["description"] == "Introduction to cyclical nature."
        assert beat["start"] == "0:00"
        assert beat["end"] == "0:02:10"
        # Label synthesized from description (first sentence)
        assert beat["label"] == "Introduction to cyclical nature"

    def test_synthesizes_label_from_long_description(self):
        # First sentence has 18 words — over the 12-word threshold, so the
        # synthesizer falls back to first 8 words + ellipsis.
        a = {
            "story_beats": [
                {"description": "The artist reflects on years of sustained work in printmaking and describes the collaborations that shaped the practice"},
            ]
        }
        out = normalize_analysis(a)
        label = out["story_beats"][0]["label"]
        assert label == "The artist reflects on years of sustained work…"

    def test_label_fallback_when_no_description(self):
        a = {"story_beats": [{"start": "0:00", "end": "0:10"}]}
        out = normalize_analysis(a)
        assert out["story_beats"][0]["label"] == "Story Beat"

    def test_idempotent(self):
        raw = {
            "story_beats": [
                {"beat_description": "Hello world", "start_time": "0:00", "end_time": "0:05"}
            ]
        }
        once = normalize_analysis(raw)
        twice = normalize_analysis(once)
        assert once == twice


class TestSocialClips:
    def test_aliases(self):
        a = {
            "social_clips": [
                {"name": "Big moment", "start_time": "1:00", "end_time": "1:30",
                 "quote": "the actual line", "rationale": "it lands"},
            ]
        }
        out = normalize_analysis(a)
        c = out["social_clips"][0]
        assert c["title"] == "Big moment"
        assert c["start"] == "1:00"
        assert c["end"] == "1:30"
        assert c["text"] == "the actual line"
        assert c["why"] == "it lands"


class TestSoundbites:
    def test_aliases(self):
        a = {"strongest_soundbites": [{"quote": "printmaking is breath", "start_tc": "2:14"}]}
        out = normalize_analysis(a)
        sb = out["strongest_soundbites"][0]
        assert sb["text"] == "printmaking is breath"
        assert sb["start"] == "2:14"


class TestTopLevelKeyDrift:
    """analyze_transcript uses _first_present_list to accept alternate top-level
    keys (e.g. `beats` instead of `story_beats`). This is tested via
    analyze_transcript directly by stubbing the story/social calls."""

    def test_beats_alias_surfaces_story_beats(self, monkeypatch):
        import ai_analysis
        monkeypatch.setattr(ai_analysis, "_analyze_story", lambda t, n: {
            "summary": "short",
            "beats": [{"description": "intro", "start_time": "0:00", "end_time": "0:10"}],
        })
        monkeypatch.setattr(ai_analysis, "_analyze_social", lambda t, n: {"clips": []})
        monkeypatch.setattr(ai_analysis, "_format_transcript_for_ai", lambda t: "x")
        out = ai_analysis.analyze_transcript({"segments": [{"start": 0, "end": 1, "text": "x"}]})
        assert len(out["story_beats"]) == 1
        assert out["story_beats"][0]["description"] == "intro"

    def test_clips_alias_surfaces_social_clips(self, monkeypatch):
        import ai_analysis
        monkeypatch.setattr(ai_analysis, "_analyze_story", lambda t, n: {})
        monkeypatch.setattr(ai_analysis, "_analyze_social", lambda t, n: {
            "clips": [{"name": "x", "start_time": "1:00", "end_time": "1:15", "quote": "q"}],
        })
        monkeypatch.setattr(ai_analysis, "_format_transcript_for_ai", lambda t: "x")
        out = ai_analysis.analyze_transcript({"segments": [{"start": 0, "end": 1, "text": "x"}]})
        assert len(out["social_clips"]) == 1
        assert out["social_clips"][0]["title"] == "x"
        assert out["social_clips"][0]["text"] == "q"


class TestEdgeCases:
    def test_none_analysis(self):
        assert normalize_analysis(None) == {}

    def test_empty_analysis(self):
        assert normalize_analysis({}) == {}

    def test_non_dict_beat_filtered(self):
        a = {"story_beats": [{"description": "ok"}, "garbage", None, 42, {"beat_description": "fine"}]}
        out = normalize_analysis(a)
        # Non-dict items dropped, dicts normalized.
        assert len(out["story_beats"]) == 2

    def test_preserves_unknown_fields_on_beat(self):
        # Fields we don't know about shouldn't be dropped — future-proof.
        a = {"story_beats": [{"description": "x", "start": "0:00", "end": "0:05", "custom": "keep me"}]}
        out = normalize_analysis(a)
        assert out["story_beats"][0]["custom"] == "keep me"
