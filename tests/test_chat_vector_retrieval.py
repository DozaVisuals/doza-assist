"""Tests for the segment_vectors.json plumbing in the chat retrieval path.

When vectors are present, three behaviors should kick in:
1. theme_tag overlap with the user's query becomes additional search vocab.
2. high-narrative-score windows get folded into the relevant excerpts pool.
3. Layer 2 narrows chunk fan-out on synthesis queries to high-score chunks.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_analysis import (  # noqa: E402
    _augment_with_high_score_vectors,
    _collect_theme_phrases_from_vectors,
    _find_relevant_paragraphs,
    _find_vector_anchored_chunk_indices,
    _rerank_candidates_globally,
)


def _vec(start_sec, end_sec, score='medium', tags=None):
    return {
        'timecode_in': f"{start_sec//3600:02d}:{(start_sec%3600)//60:02d}:{start_sec%60:02d}",
        'timecode_out': f"{end_sec//3600:02d}:{(end_sec%3600)//60:02d}:{end_sec%60:02d}",
        'narrative_score': score,
        'theme_tags': tags or [],
    }


def _seg(start, end, text, speaker='A'):
    return {'start': start, 'end': end, 'text': text, 'speaker': speaker}


class TestThemeVocab:
    def test_query_word_matches_theme_tag(self):
        vectors = [_vec(0, 30, tags=['resilience', 'work'])]
        out = _collect_theme_phrases_from_vectors(vectors, "show me the resilience moments")
        assert 'resilience' in out

    def test_no_query_no_match(self):
        vectors = [_vec(0, 30, tags=['resilience'])]
        assert _collect_theme_phrases_from_vectors(vectors, '') == []

    def test_no_overlap_returns_empty(self):
        vectors = [_vec(0, 30, tags=['resilience'])]
        out = _collect_theme_phrases_from_vectors(vectors, "what about the dog?")
        assert out == []

    def test_multiword_tag_partial_match(self):
        vectors = [_vec(0, 30, tags=['family pressure'])]
        out = _collect_theme_phrases_from_vectors(vectors, "tell me about family")
        assert 'family pressure' in out


class TestThemePhrasesInRetrieval:
    def test_theme_phrases_expand_match_pool(self):
        # "career change" appears as a theme tag in the query universe but the
        # paragraph text never contains "career". Theme matching rescues it.
        paragraphs = [
            _seg(0, 30, "I worked at the firm for ten years."),
            _seg(60, 90, "I quit and started my own thing."),
        ]
        # No literal hits on "career change", but the theme phrase matches
        # "started my own thing" via substring on the theme tag itself.
        matched = _find_relevant_paragraphs(
            paragraphs, phrases=[], words=[], theme_phrases=['started']
        )
        assert any('started' in p['text'].lower() for p in matched)


class TestAugmentWithVectors:
    def test_high_score_vectors_pulled_in(self):
        segments = [
            _seg(0, 30, "opening words"),
            _seg(60, 90, "the high moment"),
            _seg(180, 210, "closing"),
        ]
        # Empty match; high vector at 60-90 should still surface.
        vectors = [
            _vec(60, 90, score='high', tags=[]),
            _vec(0, 30, score='low', tags=[]),
        ]
        out = _augment_with_high_score_vectors(
            matched_paragraphs=[],
            segments=segments,
            segment_vectors=vectors,
            theme_phrases=[],
        )
        assert any(p['text'] == 'the high moment' for p in out)
        assert not any(p['text'] == 'opening words' for p in out)

    def test_no_vectors_returns_input_unchanged(self):
        matched = [_seg(0, 30, "a")]
        out = _augment_with_high_score_vectors(matched, [_seg(0, 30, "a")], None, [])
        assert out == matched


class TestVectorAnchoredChunks:
    def _chunk(start, end, text='x'):  # pragma: no cover - helper only
        return [_seg(start, end, text)]

    def test_high_score_filter_picks_overlapping_chunk(self):
        # Two chunks: 0-300, 300-600. High-score vector at 400-450 → only
        # chunk 1 (the second) qualifies.
        chunks = [
            [_seg(0, 300, 'first')],
            [_seg(300, 600, 'second')],
        ]
        vectors = [_vec(400, 450, score='high')]
        out = _find_vector_anchored_chunk_indices(chunks, vectors, score_filter='high')
        assert out == [1]

    def test_match_filter_uses_theme_phrases(self):
        chunks = [
            [_seg(0, 300, 'first')],
            [_seg(300, 600, 'second')],
        ]
        vectors = [_vec(50, 80, score='medium', tags=['family'])]
        out = _find_vector_anchored_chunk_indices(
            chunks, vectors, score_filter='match', theme_phrases=['family'],
        )
        assert out == [0]

    def test_limit_caps_chunk_count(self):
        chunks = [[_seg(i * 100, i * 100 + 90, f'c{i}')] for i in range(8)]
        vectors = [_vec(i * 100 + 10, i * 100 + 80, score='high') for i in range(8)]
        out = _find_vector_anchored_chunk_indices(chunks, vectors, score_filter='high', limit=3)
        assert len(out) == 3
        # Indices come back chronologically even after the limit-by-overlap step.
        assert out == sorted(out)


class TestGlobalRerank:
    def test_skips_when_few_candidates(self):
        cands = [{'title': 'a', 'start_sec': 0, 'end_sec': 30, 'why': 'x', 'score': 5}]
        out = _rerank_candidates_globally(cands, 'pick the best', top_k=5)
        assert out == cands

    def test_falls_back_to_local_aggregator_on_ai_failure(self, monkeypatch):
        import ai_analysis
        monkeypatch.setattr(ai_analysis, '_call_ai_json', lambda *a, **k: '')
        cands = [
            {'title': f'c{i}', 'start_sec': i * 100, 'end_sec': i * 100 + 30,
             'why': '', 'score': 10 - i}
            for i in range(8)
        ]
        out = ai_analysis._rerank_candidates_globally(cands, 'pick best', top_k=5)
        # Falls back to local aggregator → top_k clips by score, chrono order.
        assert len(out) == 5
        # Highest-scored 5 (i=0..4) should be the survivors.
        kept_titles = {c['title'] for c in out}
        assert kept_titles == {f'c{i}' for i in range(5)}

    def test_ai_picks_drive_the_selection(self, monkeypatch):
        import ai_analysis
        monkeypatch.setattr(
            ai_analysis, '_call_ai_json',
            lambda *a, **k: '{"picks": [3, 7, 1]}',
        )
        cands = [
            {'title': f'c{i}', 'start_sec': i * 100, 'end_sec': i * 100 + 30,
             'why': '', 'score': 5}
            for i in range(8)
        ]
        out = ai_analysis._rerank_candidates_globally(cands, 'pick best', top_k=5)
        kept_titles = [c['title'] for c in out]
        # Indices 1, 3, 7 → titles c1, c3, c7. Output is chrono-sorted.
        assert kept_titles == ['c1', 'c3', 'c7']
