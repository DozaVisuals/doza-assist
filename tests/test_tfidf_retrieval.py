"""Tests for the TF-IDF paragraph retrieval index.

Locks in: build → query → save/load round-trip → ranking quality on a
small synthetic corpus. The point isn't perfect IR (TF-IDF is a baseline)
but that the index returns *topically* relevant paragraphs ahead of
generic ones for typical chat queries.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from doza_assist.retrieval import (  # noqa: E402
    TfidfIndex, build_paragraph_index, load_index, save_index,
)


PARAGRAPHS = [
    {'start': 0, 'end': 30, 'speaker': 'A',
     'text': 'I lost my father when I was twelve. It changed how I saw everything.'},
    {'start': 30, 'end': 60, 'speaker': 'A',
     'text': 'The trustees of the estate met every Tuesday to review the budget.'},
    {'start': 60, 'end': 90, 'speaker': 'A',
     'text': 'Moose Hill rises above the village; from the top you can see three counties.'},
    {'start': 90, 'end': 120, 'speaker': 'A',
     'text': 'My father taught me how to read a map by lantern light.'},
    {'start': 120, 'end': 150, 'speaker': 'A',
     'text': 'The trustees voted unanimously against the proposal.'},
    {'start': 150, 'end': 180, 'speaker': 'A',
     'text': 'Just generic filler that nobody would search for.'},
]


class TestRanking:
    def test_specific_query_picks_specific_paragraph(self):
        idx = TfidfIndex.build(PARAGRAPHS)
        out = idx.query_paragraphs('Moose Hill view', k=2)
        assert out
        assert 'Moose Hill' in out[0]['text']

    def test_father_query_outranks_filler(self):
        idx = TfidfIndex.build(PARAGRAPHS)
        out = idx.query_paragraphs('what did he say about his father', k=2)
        assert out
        # Both father paragraphs (0 and 3) should rank ahead of filler.
        top_texts = [p['text'] for p in out]
        assert any('father' in t for t in top_texts)
        assert not any('generic filler' in t for t in top_texts)

    def test_trustees_query_brings_back_trustees_paragraphs(self):
        idx = TfidfIndex.build(PARAGRAPHS)
        out = idx.query_paragraphs('what did the trustees decide', k=3)
        assert out
        # Both trustees paragraphs should appear ahead of Moose Hill.
        kept_starts = [p['start'] for p in out]
        assert 30 in kept_starts
        assert 120 in kept_starts

    def test_empty_query_returns_nothing(self):
        idx = TfidfIndex.build(PARAGRAPHS)
        assert idx.query_paragraphs('', k=5) == []
        assert idx.query_paragraphs('the the the', k=5) == []

    def test_no_match_returns_empty(self):
        idx = TfidfIndex.build(PARAGRAPHS)
        # Query has no shared content terms with any paragraph.
        out = idx.query_paragraphs('blockchain quantum cryptocurrency', k=5)
        assert out == []


class TestPersistence:
    def test_save_and_load_round_trip(self):
        idx = TfidfIndex.build(PARAGRAPHS)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'idx.json')
            assert save_index(idx, path)
            loaded = load_index(path)
        assert loaded is not None
        # Same query returns same top hit.
        assert idx.query_paragraphs('Moose Hill', k=1)[0]['text'] == \
               loaded.query_paragraphs('Moose Hill', k=1)[0]['text']

    def test_load_missing_returns_none(self):
        assert load_index('/nonexistent/path/idx.json') is None

    def test_load_invalid_json_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'bad.json')
            with open(path, 'w') as f:
                f.write('not json')
            assert load_index(path) is None

    def test_load_wrong_version_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'v0.json')
            with open(path, 'w') as f:
                json.dump({'version': 0, 'paragraphs': [], 'idf': {}, 'vectors': []}, f)
            assert load_index(path) is None


class TestBuildFromTranscript:
    def test_build_paragraph_index_from_transcript(self):
        # Two same-speaker segments under 60s should merge into one paragraph.
        transcript = {'segments': [
            {'start': 0, 'end': 5, 'text': 'I went to Moose Hill yesterday.', 'speaker': 'A'},
            {'start': 5, 'end': 10, 'text': 'The view was incredible.', 'speaker': 'A'},
            {'start': 10, 'end': 20, 'text': 'Then we discussed the trustees.', 'speaker': 'B'},
        ]}
        idx = build_paragraph_index(transcript)
        # Three segments, two paragraphs (split on speaker change).
        assert len(idx.paragraphs) == 2
        out = idx.query_paragraphs('Moose Hill', k=1)
        assert 'Moose' in out[0]['text']
