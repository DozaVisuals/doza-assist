"""Tests for the long-interview chunking path in analyze_transcript.

Small models lose attention on long transcripts — they'll analyze the first
5-10 minutes of a 1h 42m interview and return nothing for the rest. Chunking
breaks the transcript into ~15-minute slices, runs analysis per-chunk, and
merges. These tests lock in the chunk boundaries, the merging logic, and the
short-vs-long routing threshold.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ai_analysis  # noqa: E402
from ai_analysis import (  # noqa: E402
    ANALYSIS_PER_CATEGORY_CAP,
    CHUNK_MINUTES,
    _iter_transcript_chunks,
    _seconds_to_tc,
    analyze_transcript,
)


def _mk_segments(total_seconds, step=5.0):
    """Build a transcript's worth of segments covering [0, total_seconds]."""
    segs = []
    t = 0.0
    i = 0
    while t < total_seconds:
        end = min(t + step, total_seconds)
        segs.append({
            'start': t,
            'end': end,
            'start_formatted': _seconds_to_tc(t),
            'text': f"line {i}",
            'speaker': 'A',
        })
        t += step
        i += 1
    return segs


class TestChunkIteration:
    def test_short_transcript_yields_single_chunk(self):
        segs = _mk_segments(5 * 60)  # 5 minutes
        chunks = list(_iter_transcript_chunks(segs))
        assert len(chunks) == 1
        assert chunks[0]['segments'] == segs

    def test_long_transcript_chunks_at_target(self):
        # 1h 42m ≈ 6120s. With 15-min chunks we expect ~7 chunks.
        segs = _mk_segments(6120)
        chunks = list(_iter_transcript_chunks(segs))
        assert len(chunks) == pytest.approx(7, abs=1)
        # Every chunk should span at least 14 minutes (some slack for rounding) —
        # except possibly the last.
        for c in chunks[:-1]:
            assert c['end_seconds'] - c['start_seconds'] >= 14 * 60

    def test_chunks_cover_all_segments_exactly_once(self):
        segs = _mk_segments(45 * 60)  # 45 min
        chunks = list(_iter_transcript_chunks(segs))
        seen = []
        for c in chunks:
            seen.extend(c['segments'])
        assert seen == segs

    def test_chunks_are_monotonic(self):
        segs = _mk_segments(50 * 60)
        chunks = list(_iter_transcript_chunks(segs))
        for a, b in zip(chunks, chunks[1:]):
            assert a['end_seconds'] <= b['start_seconds']

    def test_empty_segments_yields_nothing(self):
        assert list(_iter_transcript_chunks([])) == []


class TestAnalyzeTranscriptRouting:
    """Short interviews go through the single-call path; long ones chunk."""

    def test_short_transcript_uses_single_call(self, monkeypatch):
        segs = _mk_segments(5 * 60)  # under 15-min threshold
        calls = {'story': 0, 'social': 0}

        def _stub_story(text, name, **kwargs):
            calls['story'] += 1
            return {
                'summary': 'short', 'suggested_title': 'T',
                'story_beats': [{'label': 'hook', 'start': '0:00', 'end': '0:30',
                                 'description': 'x'}],
                'strongest_soundbites': [],
            }

        def _stub_social(text, name, **kwargs):
            calls['social'] += 1
            return {'social_clips': [{'title': 'c', 'start': '0:10', 'end': '0:20',
                                      'text': 'q'}]}

        monkeypatch.setattr(ai_analysis, '_analyze_story', _stub_story)
        monkeypatch.setattr(ai_analysis, '_analyze_social', _stub_social)

        out = analyze_transcript({'segments': segs}, project_name='Short', analysis_type='all')
        assert calls == {'story': 1, 'social': 1}
        assert out['summary'] == 'short'
        assert len(out['story_beats']) == 1
        assert len(out['social_clips']) == 1

    def test_long_transcript_triggers_chunked_path(self, monkeypatch):
        # 1h 5m → should split into roughly 5 chunks.
        segs = _mk_segments(65 * 60)
        calls = {'story': [], 'social': []}

        def _stub_story(text, name, **kwargs):
            calls['story'].append(name)
            n = len(calls['story'])
            # Distinct timecodes per chunk so the post-merge dedup pass doesn't
            # collapse identical ranges across chunks. Each chunk returns 2
            # beats and 1 soundbite anchored to its own ~15-min window so the
            # merged + capped output reflects actual chunk coverage.
            base = (n - 1) * 15 * 60
            return {
                'summary': f'sum {n}',
                'suggested_title': f'T{n}',
                'story_beats': [
                    {'label': f'beat-{n}-1', 'description': 'x',
                     'start': _seconds_to_tc(base + 30),
                     'end': _seconds_to_tc(base + 60)},
                    {'label': f'beat-{n}-2', 'description': 'y',
                     'start': _seconds_to_tc(base + 90),
                     'end': _seconds_to_tc(base + 120)},
                ],
                'strongest_soundbites': [{'text': f'sb-{n}',
                                          'start': _seconds_to_tc(base + 45),
                                          'end': _seconds_to_tc(base + 60),
                                          'why': 'z'}],
                'themes': ['art', 'resilience'],
            }

        def _stub_social(text, name, **kwargs):
            calls['social'].append(name)
            n = len(calls['social'])
            base = (n - 1) * 15 * 60
            return {'social_clips': [
                {'title': f'clip-{n}',
                 'start': _seconds_to_tc(base + 30),
                 'end': _seconds_to_tc(base + 50),
                 'text': 'hi'},
            ]}

        # Stub the overall-summary synthesis call so it doesn't try to hit
        # Ollama/Claude from a unit test. This is the pass that folds all
        # per-chunk summaries into one sidebar overview.
        synth_calls = {'count': 0, 'seen_summaries': None, 'seen_titles': None}

        def _stub_synthesize(summaries, titles, project_name):
            synth_calls['count'] += 1
            synth_calls['seen_summaries'] = list(summaries)
            synth_calls['seen_titles'] = list(titles)
            return {
                'summary': 'overall across all chunks',
                'suggested_title': 'Overall Title',
            }

        monkeypatch.setattr(ai_analysis, '_analyze_story', _stub_story)
        monkeypatch.setattr(ai_analysis, '_analyze_social', _stub_social)
        monkeypatch.setattr(ai_analysis, '_synthesize_overall_summary', _stub_synthesize)

        out = analyze_transcript({'segments': segs}, project_name='Long', analysis_type='all')

        # Each chunk triggered exactly one story + one social call.
        assert len(calls['story']) == len(calls['social'])
        assert len(calls['story']) >= 4  # 65 min / 15 = ~4-5
        # Chunk labels include the "part X/N" suffix so the model sees context.
        assert all('part' in name for name in calls['story'])
        # Coverage: chunks merged into the accumulator. The post-merge cap
        # enforces ≤ ANALYSIS_PER_CATEGORY_CAP per category, but chunks stay
        # represented (every list is non-empty when a chunk produced items).
        assert len(out['story_beats']) <= ANALYSIS_PER_CATEGORY_CAP
        assert len(out['story_beats']) == min(2 * len(calls['story']), ANALYSIS_PER_CATEGORY_CAP)
        assert len(out['strongest_soundbites']) <= ANALYSIS_PER_CATEGORY_CAP
        assert len(out['strongest_soundbites']) == min(len(calls['story']), ANALYSIS_PER_CATEGORY_CAP)
        assert len(out['social_clips']) <= ANALYSIS_PER_CATEGORY_CAP
        assert len(out['social_clips']) == min(len(calls['social']), ANALYSIS_PER_CATEGORY_CAP)
        # The synthesis pass ran once with every chunk's summary/title, so the
        # sidebar describes the whole timeline — not just the first 15 minutes.
        assert synth_calls['count'] == 1
        assert synth_calls['seen_summaries'] == [f'sum {i}' for i in range(1, len(calls['story']) + 1)]
        assert synth_calls['seen_titles'] == [f'T{i}' for i in range(1, len(calls['story']) + 1)]
        assert out['summary'] == 'overall across all chunks'
        assert out['suggested_title'] == 'Overall Title'
        # Themes deduped across chunks.
        assert set(out['themes']) == {'art', 'resilience'}

    def test_chunked_path_tolerates_one_chunk_failing(self, monkeypatch):
        segs = _mk_segments(50 * 60)
        call_count = {'story': 0}

        def _flaky_story(text, name, **kwargs):
            call_count['story'] += 1
            if call_count['story'] == 2:
                raise RuntimeError("model timed out on chunk 2")
            n = call_count['story']
            base = (n - 1) * 15 * 60
            return {'story_beats': [{'label': f'x{n}', 'description': 'y',
                                     'start': _seconds_to_tc(base + 30),
                                     'end': _seconds_to_tc(base + 60)}]}

        monkeypatch.setattr(ai_analysis, '_analyze_story', _flaky_story)
        monkeypatch.setattr(ai_analysis, '_analyze_social', lambda t, n, **k: {'social_clips': []})
        # Synthesis also runs on the surviving chunk summaries — stub it out
        # so this test doesn't try to hit an AI backend.
        monkeypatch.setattr(
            ai_analysis, '_synthesize_overall_summary',
            lambda summaries, titles, name: {'summary': '', 'suggested_title': ''},
        )

        out = analyze_transcript({'segments': segs}, project_name='Flaky', analysis_type='story')
        # One chunk failed, the rest survived. Each surviving chunk emitted a
        # distinct-timecode beat so the cap (>= number of survivors here) does
        # not trim them.
        assert len(out['story_beats']) == call_count['story'] - 1
        assert call_count['story'] >= 2


class TestSynthesizeOverallSummary:
    """The summary in the project sidebar must cover the whole transcript, not
    just the first 15-minute chunk. _synthesize_overall_summary is what folds
    per-chunk summaries into one overall summary — regression coverage for the
    multi-interview FCPXML case where the old code only kept chunk 1."""

    def test_single_chunk_passes_through_without_ai_call(self, monkeypatch):
        # A single summary should never trigger a synthesis AI call — the one
        # summary IS the overall summary.
        called = {'count': 0}
        monkeypatch.setattr(ai_analysis, '_call_ai',
                            lambda *a, **k: (called.__setitem__('count', called['count'] + 1) or '{}'))
        out = ai_analysis._synthesize_overall_summary(
            ['only one chunk'], ['Only Title'], 'P',
        )
        assert called['count'] == 0
        assert out == {'summary': 'only one chunk', 'suggested_title': 'Only Title'}

    def test_empty_input_returns_empty(self, monkeypatch):
        called = {'count': 0}
        monkeypatch.setattr(ai_analysis, '_call_ai',
                            lambda *a, **k: (called.__setitem__('count', called['count'] + 1) or '{}'))
        out = ai_analysis._synthesize_overall_summary([], [], 'P')
        assert called['count'] == 0
        assert out == {'summary': '', 'suggested_title': ''}

    def test_multiple_chunks_synthesizes_via_ai(self, monkeypatch):
        # Multi-chunk case: one meta-call combines the per-chunk summaries into
        # a single 3-4 sentence overview covering every section.
        captured = {}

        def _fake_call_ai(prompt, system_prompt=''):
            captured['prompt'] = prompt
            return (
                '{"summary": "Covers interviews A, B, and C across the whole '
                'timeline.", "suggested_title": "Three Voices"}'
            )

        monkeypatch.setattr(ai_analysis, '_call_ai', _fake_call_ai)
        out = ai_analysis._synthesize_overall_summary(
            ['interview A opener', 'interview B mid', 'interview C close'],
            ['A', 'B', 'C'],
            'Triple',
        )
        # The synthesis prompt includes all chunk summaries so the model has
        # the full timeline to work from.
        assert 'interview A opener' in captured['prompt']
        assert 'interview B mid' in captured['prompt']
        assert 'interview C close' in captured['prompt']
        assert out['summary'].startswith('Covers interviews A, B, and C')
        assert out['suggested_title'] == 'Three Voices'

    def test_falls_back_to_joined_summaries_on_ai_failure(self, monkeypatch):
        # If the synthesis call dies we must not silently return the old
        # "first chunk only" sidebar — join everything so the user at least
        # sees coverage of the whole transcript.
        def _boom(*a, **k):
            raise RuntimeError("ollama down")

        monkeypatch.setattr(ai_analysis, '_call_ai', _boom)
        out = ai_analysis._synthesize_overall_summary(
            ['first part', 'second part', 'third part'],
            ['T1', 'T2', 'T3'],
            'Triple',
        )
        assert 'first part' in out['summary']
        assert 'second part' in out['summary']
        assert 'third part' in out['summary']
        # First title is a reasonable fallback when synthesis didn't produce one.
        assert out['suggested_title'] == 'T1'

    def test_falls_back_when_ai_returns_unparseable_json(self, monkeypatch):
        # Small models sometimes return prose despite the "JSON only"
        # instruction. Treat unparseable output the same as a failure rather
        # than letting an empty summary reach the UI.
        monkeypatch.setattr(
            ai_analysis, '_call_ai',
            lambda *a, **k: "Here is the summary: it's about three people.",
        )
        out = ai_analysis._synthesize_overall_summary(
            ['chunk 1 text', 'chunk 2 text'], ['T1', 'T2'], 'Double',
        )
        assert 'chunk 1 text' in out['summary']
        assert 'chunk 2 text' in out['summary']


class TestSocialClipTitleSynthesis:
    """When the model returns only {start, end, text} for a social clip, we
    synthesize a title from the text so the UI isn't blank."""

    def test_title_synthesized_from_text_when_missing(self):
        from ai_analysis import normalize_analysis
        a = {'social_clips': [
            {'start': '0:30', 'end': '0:45', 'text': "I'm going to make this a really good piece of work one day."},
        ]}
        out = normalize_analysis(a)
        title = out['social_clips'][0]['title']
        assert title  # not empty
        # Title is derived from the text — first sentence or first-N words.
        assert title.startswith("I'm going to make this")

    def test_existing_title_preserved(self):
        from ai_analysis import normalize_analysis
        a = {'social_clips': [
            {'title': 'Big moment', 'start': '0:30', 'end': '0:45', 'text': 'q'},
        ]}
        out = normalize_analysis(a)
        assert out['social_clips'][0]['title'] == 'Big moment'
