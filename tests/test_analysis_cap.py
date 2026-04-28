"""Tests for the post-merge cap+rerank pass that enforces ≤7 items per
analysis category. Locks in three behaviors:

1. Hard cap: never more than ANALYSIS_PER_CATEGORY_CAP per list, even when
   the chunked path floods the accumulator with 20+ raw beats.
2. Time-overlap dedupe: the same moment showing up in two chunks (or a
   chunk-boundary overlap zone) collapses to one survivor.
3. Vector-aware ranking: when ``segment_vectors`` are passed in, items
   anchored to "high"-narrative-score windows beat items anchored to "low"
   ones; without vectors we fall back to length so longer well-formed
   clips win.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_analysis import (  # noqa: E402
    ANALYSIS_PER_CATEGORY_CAP,
    _cap_and_rank_analysis,
)


def _beat(start_sec, end_sec, label='Beat', beat_type='context', description='x'):
    return {
        'label': label,
        'description': description,
        'beat_type': beat_type,
        'start': f"{start_sec//3600:02d}:{(start_sec%3600)//60:02d}:{start_sec%60:02d}",
        'end': f"{end_sec//3600:02d}:{(end_sec%3600)//60:02d}:{end_sec%60:02d}",
    }


def _vec(start_sec, end_sec, score='medium'):
    return {
        'timecode_in': f"{start_sec//3600:02d}:{(start_sec%3600)//60:02d}:{start_sec%60:02d}",
        'timecode_out': f"{end_sec//3600:02d}:{(end_sec%3600)//60:02d}:{end_sec%60:02d}",
        'narrative_score': score,
    }


class TestHardCap:
    def test_caps_story_beats_to_seven(self):
        # 12 distinct beats spread across the timeline → trim to 7.
        beats = [_beat(i * 60, i * 60 + 30, label=f'b{i}') for i in range(12)]
        out = _cap_and_rank_analysis({'story_beats': beats})
        assert len(out['story_beats']) == ANALYSIS_PER_CATEGORY_CAP

    def test_caps_soundbites_to_seven(self):
        sbs = [
            {'text': f'q{i}', 'start': f'00:0{i}:00', 'end': f'00:0{i}:30', 'why': 'z'}
            for i in range(9)
        ]
        out = _cap_and_rank_analysis({'strongest_soundbites': sbs})
        assert len(out['strongest_soundbites']) == ANALYSIS_PER_CATEGORY_CAP

    def test_caps_social_clips_to_seven(self):
        clips = [
            {'title': f'c{i}', 'start': f'00:0{i}:00', 'end': f'00:0{i}:30', 'text': 't'}
            for i in range(10)
        ]
        out = _cap_and_rank_analysis({'social_clips': clips})
        assert len(out['social_clips']) == ANALYSIS_PER_CATEGORY_CAP

    def test_caps_themes_and_broll(self):
        themes = [f'theme-{i}' for i in range(15)]
        broll = [f'broll-{i}' for i in range(15)]
        out = _cap_and_rank_analysis({'themes': themes, 'broll_suggestions': broll})
        assert len(out['themes']) == ANALYSIS_PER_CATEGORY_CAP
        assert len(out['broll_suggestions']) == ANALYSIS_PER_CATEGORY_CAP

    def test_under_cap_passes_through(self):
        beats = [_beat(i * 60, i * 60 + 30) for i in range(3)]
        out = _cap_and_rank_analysis({'story_beats': beats})
        assert len(out['story_beats']) == 3


class TestDedupe:
    def test_time_overlap_dedupes(self):
        # Two beats at nearly the same window — only one should survive.
        beats = [
            _beat(60, 120, label='from chunk A'),
            _beat(63, 117, label='from chunk B'),
        ]
        out = _cap_and_rank_analysis({'story_beats': beats})
        assert len(out['story_beats']) == 1

    def test_distinct_windows_kept(self):
        # 3 disjoint windows — all kept.
        beats = [
            _beat(0, 30),
            _beat(120, 150),
            _beat(300, 330),
        ]
        out = _cap_and_rank_analysis({'story_beats': beats})
        assert len(out['story_beats']) == 3

    def test_themes_dedupe_case_insensitively(self):
        themes = ['Resilience', 'resilience', 'Art', 'ART', 'Loss']
        out = _cap_and_rank_analysis({'themes': themes})
        assert len(out['themes']) == 3
        # First-seen casing wins.
        assert out['themes'][0] == 'Resilience'


class TestVectorRanking:
    def test_high_score_vectors_outrank_low(self):
        # 9 beats — only 7 survive the cap. The 7 anchored to "high" vector
        # windows should win over the 2 anchored to "low".
        high_beats = [_beat(i * 60, i * 60 + 30, label=f'high-{i}') for i in range(7)]
        low_beats = [_beat(1000 + i * 60, 1000 + i * 60 + 30, label=f'low-{i}') for i in range(2)]
        beats = high_beats + low_beats
        vectors = [
            _vec(i * 60, i * 60 + 30, 'high') for i in range(7)
        ] + [
            _vec(1000 + i * 60, 1000 + i * 60 + 30, 'low') for i in range(2)
        ]
        out = _cap_and_rank_analysis({'story_beats': beats}, segment_vectors=vectors)
        kept_labels = {b['label'] for b in out['story_beats']}
        assert all(label.startswith('high-') for label in kept_labels)


class TestBeatTypeDiversity:
    def test_diversity_keeps_one_per_type_when_possible(self):
        # 4 hooks vs 1 each of context/turn/resolution. With diversity, the cap
        # should keep at least one of each non-hook type rather than stacking
        # all 4 hooks.
        beats = [
            _beat(i * 60, i * 60 + 30, label=f'hook-{i}', beat_type='hook')
            for i in range(4)
        ] + [
            _beat(500, 530, label='context-1', beat_type='context'),
            _beat(700, 730, label='turn-1', beat_type='turn'),
            _beat(900, 930, label='resolution-1', beat_type='resolution'),
        ]
        out = _cap_and_rank_analysis({'story_beats': beats})
        kept_types = {b['beat_type'] for b in out['story_beats']}
        assert {'context', 'turn', 'resolution'}.issubset(kept_types)


class TestIdempotence:
    def test_calling_twice_yields_same_output(self):
        beats = [_beat(i * 60, i * 60 + 30) for i in range(20)]
        once = _cap_and_rank_analysis({'story_beats': beats})
        twice = _cap_and_rank_analysis(once)
        assert twice == once
