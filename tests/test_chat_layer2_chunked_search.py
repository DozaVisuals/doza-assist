"""Tests for Layer 2 chunked search on long (>60 min) interviews.

Design contract:
- Long interviews bypass the single-prompt path. They split the
  paragraph-grouped transcript into overlapping ~7k-token chunks and run a
  per-chunk JSON-output Ollama call. Candidates are aggregated across chunks,
  deduplicated by timecode overlap, sorted by score, and the top-K are
  emitted as [CLIP:] markers server-side.

These tests exercise each stage in isolation (chunking, prompt building,
JSON parsing with regex fallback, aggregation, card formatting) plus the
top-level integration path.

None of these tests hit Ollama — they patch `_call_ai_json` with canned
responses so the full Layer 2 flow runs deterministically in CI.
"""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ai_analysis  # noqa: E402


def _make_paragraphs(n, text_len=400, paragraph_seconds=60, speaker='Chris'):
    """Build N structured paragraphs covering [0, N*paragraph_seconds]."""
    out = []
    for i in range(n):
        start = i * paragraph_seconds
        out.append({
            'speaker': speaker,
            'start': start,
            'end': start + paragraph_seconds,
            'text': 'X' * text_len,
        })
    return out


def _make_long_transcript(minutes=90, segment_seconds=5.0, speaker='Chris', text='sample text'):
    segs = []
    t = 0.0
    total = int(minutes * 60 / segment_seconds)
    for _ in range(total):
        segs.append({
            'start': t, 'end': t + segment_seconds,
            'start_formatted':
                f"{int(t)//3600:02d}:{(int(t)%3600)//60:02d}:{int(t)%60:02d}.000",
            'text': text,
            'speaker': speaker,
        })
        t += segment_seconds
    return {'segments': segs}


class TestChunkParagraphs:
    def test_single_chunk_when_under_budget(self):
        paragraphs = _make_paragraphs(5, text_len=200)  # way under 7k tokens
        chunks = ai_analysis._chunk_paragraphs(paragraphs)
        assert len(chunks) == 1
        assert len(chunks[0]) == 5

    def test_multiple_chunks_when_over_budget(self):
        # 7000 tokens × 3.8 chars/tok ≈ 26,600 chars per chunk.
        # 100 paragraphs × 1000 chars = 100k chars → at least 3 chunks.
        paragraphs = _make_paragraphs(100, text_len=1000)
        chunks = ai_analysis._chunk_paragraphs(paragraphs)
        assert len(chunks) >= 3

    def test_chunks_overlap_by_configured_paragraphs(self):
        paragraphs = _make_paragraphs(100, text_len=1000)
        chunks = ai_analysis._chunk_paragraphs(paragraphs, overlap_paragraphs=3)
        # Last 3 paragraphs of chunk[0] should equal first 3 of chunk[1].
        assert chunks[0][-3:] == chunks[1][:3], (
            "Overlap zone must carry the same paragraph dicts — without it, "
            "moments that span a chunk boundary lose their lead-in/lead-out."
        )

    def test_empty_input_returns_empty_list(self):
        assert ai_analysis._chunk_paragraphs([]) == []

    def test_pathological_single_huge_paragraph_is_preserved(self):
        # A single paragraph that exceeds the token budget should still emit,
        # not loop forever.
        giant = [{
            'speaker': 'Chris', 'start': 0, 'end': 3600,
            'text': 'X' * 1_000_000,
        }]
        chunks = ai_analysis._chunk_paragraphs(giant)
        assert len(chunks) == 1
        assert chunks[0] == giant

    def test_forward_progress_guard(self):
        # If overlap >= chunk size, next_i would regress. The guard forces
        # i += 1 instead. Verify by using overlap larger than chunks produce.
        paragraphs = _make_paragraphs(10, text_len=5000)  # each ~5k chars
        chunks = ai_analysis._chunk_paragraphs(
            paragraphs, tokens_per_chunk=1500, overlap_paragraphs=50
        )
        # If the guard is missing, this would hang. Success = returns quickly
        # and covers all paragraphs.
        covered = set()
        for ch in chunks:
            for p in ch:
                covered.add(p['start'])
        assert len(covered) == 10


class TestBuildChunkSearchPrompt:
    def test_prompt_contains_chunk_content(self):
        chunk = _make_paragraphs(3, text_len=50)
        sys_p, user_p = ai_analysis._build_chunk_search_prompt(
            chunk, "find key moments", phrases=[], words=[],
            chunk_idx=0, total_chunks=3, project_name="Test"
        )
        assert 'EXCERPT:' in sys_p
        assert '[00:00:00-00:01:00]' in sys_p  # first paragraph timecode

    def test_prompt_includes_keyword_hint_when_phrases_present(self):
        chunk = _make_paragraphs(2, text_len=50)
        sys_p, _ = ai_analysis._build_chunk_search_prompt(
            chunk, "moose hill stuff", phrases=['moose hill'], words=['moose', 'hill'],
            chunk_idx=0, total_chunks=2, project_name="T"
        )
        assert 'moose hill' in sys_p
        # Crucially, the hint must say "prioritize NOT filter" so the model
        # still picks up neighboring context paragraphs without the keyword.
        assert 'prioritize' in sys_p.lower()
        assert 'not as a filter' in sys_p.lower() or 'not a filter' in sys_p.lower()

    def test_prompt_omits_keyword_hint_when_no_keywords(self):
        chunk = _make_paragraphs(2, text_len=50)
        sys_p, _ = ai_analysis._build_chunk_search_prompt(
            chunk, "what's here?", phrases=[], words=[],
            chunk_idx=0, total_chunks=2, project_name="T"
        )
        # No keyword hint section when there are no keywords.
        assert 'suggests these search terms' not in sys_p

    def test_prompt_requests_json_output(self):
        chunk = _make_paragraphs(1, text_len=50)
        sys_p, _ = ai_analysis._build_chunk_search_prompt(
            chunk, "x", phrases=[], words=[],
            chunk_idx=0, total_chunks=1, project_name="T"
        )
        # The JSON contract must be explicit — the parser expects it.
        assert 'JSON' in sys_p
        assert '"candidates"' in sys_p
        assert '"score"' in sys_p


class TestParseChunkResponse:
    def setup_method(self):
        self.chunk = _make_paragraphs(10, text_len=100, paragraph_seconds=60)
        # Chunk spans 0 to 600 seconds.

    def test_parses_well_formed_json(self):
        response = (
            '{"candidates": ['
            '{"title": "Opening", "start": "00:01:00", "end": "00:02:00", '
            '"score": 8, "why": "strong opening"}'
            ']}'
        )
        out = ai_analysis._parse_chunk_response(response, self.chunk)
        assert len(out) == 1
        assert out[0]['title'] == 'Opening'
        assert out[0]['start_sec'] == 60
        assert out[0]['end_sec'] == 120
        assert out[0]['score'] == 8
        assert out[0]['source'] == 'json'

    def test_parses_json_wrapped_in_code_fence(self):
        # Gemma sometimes emits ```json ... ``` despite instructions.
        response = (
            '```json\n'
            '{"candidates": [{"title": "X", "start": "00:03:00", "end": "00:04:00", '
            '"score": 7, "why": "y"}]}\n'
            '```'
        )
        out = ai_analysis._parse_chunk_response(response, self.chunk)
        assert len(out) == 1
        assert out[0]['start_sec'] == 180

    def test_parses_json_with_prose_prefix(self):
        # Models sometimes write "Here is the JSON: {...}" despite being told not to.
        response = (
            'Sure — here is the JSON:\n'
            '{"candidates": [{"title": "X", "start": "00:02:00", "end": "00:02:30", '
            '"score": 6, "why": "z"}]}'
        )
        out = ai_analysis._parse_chunk_response(response, self.chunk)
        assert len(out) == 1
        assert out[0]['start_sec'] == 120

    def test_regex_fallback_when_json_unparseable(self):
        # Model wrote prose instead of JSON. Parser salvages timecodes.
        response = (
            "The best moment is from 00:02:00 to 00:03:00 — a strong revelation. "
            "Another good one is 00:04:30 - 00:05:15."
        )
        out = ai_analysis._parse_chunk_response(response, self.chunk)
        assert len(out) >= 1
        assert all(c['source'] == 'regex_fallback' for c in out)
        # Low-confidence score (no ranking info from the model).
        assert all(c['score'] == 3 for c in out)
        # First range extracted.
        assert out[0]['start_sec'] == 120
        assert out[0]['end_sec'] == 180

    def test_empty_response_returns_empty(self):
        assert ai_analysis._parse_chunk_response('', self.chunk) == []
        assert ai_analysis._parse_chunk_response('   ', self.chunk) == []

    def test_empty_candidates_list_is_valid(self):
        # "Nothing in this chunk" is a legitimate answer.
        out = ai_analysis._parse_chunk_response('{"candidates": []}', self.chunk)
        assert out == []

    def test_candidates_clamped_to_chunk_timespan(self):
        # Chunk spans 0–600s. A hallucinated candidate at 01:00:00 must be
        # rejected (it's far outside this chunk's window).
        response = (
            '{"candidates": ['
            '{"title": "Hallucinated", "start": "01:00:00", "end": "01:01:00", '
            '"score": 10, "why": "fake"}'
            ']}'
        )
        out = ai_analysis._parse_chunk_response(response, self.chunk)
        assert out == [], "Out-of-chunk timecodes must be rejected"

    def test_invalid_end_before_start_is_rejected(self):
        response = (
            '{"candidates": ['
            '{"title": "Bad", "start": "00:05:00", "end": "00:04:00", '
            '"score": 5, "why": "inverted"}'
            ']}'
        )
        out = ai_analysis._parse_chunk_response(response, self.chunk)
        assert out == []

    def test_score_clamped_to_1_10(self):
        response = (
            '{"candidates": ['
            '{"title": "OverScore", "start": "00:01:00", "end": "00:02:00", '
            '"score": 99, "why": "y"},'
            '{"title": "UnderScore", "start": "00:03:00", "end": "00:04:00", '
            '"score": -5, "why": "y"}'
            ']}'
        )
        out = ai_analysis._parse_chunk_response(response, self.chunk)
        assert len(out) == 2
        scores = {c['title']: c['score'] for c in out}
        assert scores['OverScore'] == 10
        assert scores['UnderScore'] == 1


class TestAggregateCandidates:
    def test_dedupes_overlapping_candidates(self):
        # Two candidates at essentially the same moment — keep the stronger one.
        cands = [
            {'title': 'A', 'start_sec': 100, 'end_sec': 160, 'score': 5, 'why': '', 'source': 'json'},
            {'title': 'A-dup', 'start_sec': 110, 'end_sec': 165, 'score': 9, 'why': '', 'source': 'json'},
        ]
        out = ai_analysis._aggregate_chunk_candidates(cands, top_k=5)
        assert len(out) == 1
        assert out[0]['title'] == 'A-dup'  # higher score won

    def test_keeps_non_overlapping(self):
        cands = [
            {'title': 'A', 'start_sec': 0, 'end_sec': 30, 'score': 8, 'why': '', 'source': 'json'},
            {'title': 'B', 'start_sec': 100, 'end_sec': 130, 'score': 7, 'why': '', 'source': 'json'},
            {'title': 'C', 'start_sec': 200, 'end_sec': 230, 'score': 6, 'why': '', 'source': 'json'},
        ]
        out = ai_analysis._aggregate_chunk_candidates(cands, top_k=5)
        assert len(out) == 3

    def test_top_k_limits_output(self):
        cands = [
            {'title': f'M{i}', 'start_sec': i * 100, 'end_sec': i * 100 + 30,
             'score': 10 - i, 'why': '', 'source': 'json'}
            for i in range(10)
        ]
        out = ai_analysis._aggregate_chunk_candidates(cands, top_k=5)
        assert len(out) == 5

    def test_final_order_is_chronological(self):
        # Aggregation picks by score then re-sorts by start time so the user
        # sees the moments in timeline order.
        cands = [
            {'title': 'Late-high', 'start_sec': 900, 'end_sec': 960, 'score': 10, 'why': '', 'source': 'json'},
            {'title': 'Early-low', 'start_sec': 60, 'end_sec': 120, 'score': 6, 'why': '', 'source': 'json'},
            {'title': 'Mid-mid', 'start_sec': 450, 'end_sec': 510, 'score': 8, 'why': '', 'source': 'json'},
        ]
        out = ai_analysis._aggregate_chunk_candidates(cands, top_k=5)
        starts = [c['start_sec'] for c in out]
        assert starts == sorted(starts)

    def test_empty_input_returns_empty(self):
        assert ai_analysis._aggregate_chunk_candidates([]) == []


class TestFormatClipCards:
    def test_emits_clip_markers(self):
        cands = [
            {'title': 'Opening', 'start_sec': 60, 'end_sec': 120,
             'score': 8, 'why': "strong lead-in", 'source': 'json'},
            {'title': 'Climax', 'start_sec': 1800, 'end_sec': 1860,
             'score': 9, 'why': "peak emotional beat", 'source': 'json'},
        ]
        out = ai_analysis._format_clip_cards_from_candidates(cands)
        # The editorial "why" ships inside the marker as note="..." so the
        # frontend can render it inside the card, not floating below.
        assert '[CLIP: start=00:01:00 end=00:02:00 title="Opening" note="strong lead-in"]' in out
        assert '[CLIP: start=00:30:00 end=00:31:00 title="Climax" note="peak emotional beat"]' in out

    def test_embeds_note_attribute_for_editorial_why(self):
        cands = [{
            'title': 'Moose hill reveal', 'start_sec': 300, 'end_sec': 360,
            'score': 9, 'why': "Chris explains why the hilltop view reframed the project.",
            'source': 'json',
        }]
        out = ai_analysis._format_clip_cards_from_candidates(cands)
        assert 'note="Chris explains why the hilltop view reframed the project."' in out
        # Why sentence should NOT also appear as free-floating prose outside the marker.
        assert out.count('Chris explains why') == 1

    def test_escapes_double_quotes_in_why_and_title(self):
        cands = [{
            'title': 'He said "hello"', 'start_sec': 10, 'end_sec': 20,
            'score': 7, 'why': 'She replied "goodbye"', 'source': 'json',
        }]
        out = ai_analysis._format_clip_cards_from_candidates(cands)
        assert 'title="He said \'hello\'"' in out
        assert 'note="She replied \'goodbye\'"' in out

    def test_empty_candidates_returns_apology(self):
        out = ai_analysis._format_clip_cards_from_candidates([])
        assert '[CLIP:' not in out
        assert 'could' in out.lower() or "couldn't" in out.lower() or 'rephras' in out.lower()

    def test_regex_fallback_candidates_emit_bare_marker(self):
        # No "why" text on fallback candidates — just the marker.
        cands = [{
            'title': 'Highlighted moment', 'start_sec': 100, 'end_sec': 160,
            'score': 3, 'why': '', 'source': 'regex_fallback',
        }]
        out = ai_analysis._format_clip_cards_from_candidates(cands)
        assert '[CLIP: start=00:01:40 end=00:02:40 title="Highlighted moment"]' in out


class TestLayer2Integration:
    """End-to-end tests that run chat_about_transcript on a long interview
    with _call_ai_json patched to return canned JSON. Verify the whole
    pipeline stitches together correctly.
    """

    def test_long_interview_routes_to_layer2(self):
        t = _make_long_transcript(minutes=90)
        json_calls = []

        def fake_json(sys_p, user_p, **kwargs):
            json_calls.append((sys_p, user_p))
            return '{"candidates": [{"title": "X", "start": "00:05:00", "end": "00:06:00", "score": 7, "why": "y"}]}'

        with patch.object(ai_analysis, '_call_ai_json', side_effect=fake_json):
            reply = ai_analysis.chat_about_transcript(t, "what's the best moment?")

        # At least one per-chunk call happened.
        assert len(json_calls) >= 1
        # Every chunk prompt advertises itself as an excerpt.
        for sys_p, _ in json_calls:
            assert 'EXCERPT' in sys_p
        # Final reply has CLIP markers.
        assert '[CLIP:' in reply

    def test_long_interview_aggregates_candidates_across_chunks(self):
        t = _make_long_transcript(minutes=90)

        # Simulate each chunk returning a distinct candidate with varying scores.
        # The aggregator should keep the top-K; with scores 5, 6, 7, 8, 9 across
        # non-overlapping timestamps all should survive.
        call_count = [0]

        def fake_json(sys_p, user_p, **kwargs):
            call_count[0] += 1
            idx = call_count[0]
            # Pick a timestamp inside the first chunk so it's always valid.
            # Score varies so we can see aggregation working.
            mm = idx * 3  # 00:03, 00:06, 00:09, …
            return (
                '{"candidates": [{"title": "Clip ' + str(idx) + '", '
                '"start": "00:0' + str(mm // 10) + ':' + f'{mm % 10}0' + '", '
                '"end": "00:0' + str((mm + 1) // 10) + ':' + f'{(mm + 1) % 10}0' + '", '
                '"score": ' + str(idx) + ', "why": "reason"}]}'
            )

        with patch.object(ai_analysis, '_call_ai_json', side_effect=fake_json):
            reply = ai_analysis.chat_about_transcript(t, "find good clips")

        # Should contain at least one [CLIP: start=...] marker.
        import re
        markers = re.findall(r'\[CLIP:\s*start=(\d{2}:\d{2}:\d{2})', reply)
        assert len(markers) >= 1

    def test_long_interview_keyword_hint_passed_into_chunks(self):
        # When the keyword doesn't appear literally anywhere in the transcript
        # we fall back to full scan with the non-strict hint; every chunk
        # prompt still mentions the keyword for prioritization.
        t = _make_long_transcript(minutes=90)
        captured_prompts = []

        def fake_json(sys_p, user_p, **kwargs):
            captured_prompts.append(sys_p)
            return '{"candidates": []}'

        with patch.object(ai_analysis, '_call_ai_json', side_effect=fake_json):
            ai_analysis.chat_about_transcript(t, "what about moose hill?")

        assert all('moose hill' in p.lower() for p in captured_prompts)
        # Non-strict mode when no chunk anchors — model is told keywords are a
        # prioritization hint, not a filter.
        assert any('prioritize, NOT as a filter' in p for p in captured_prompts)

    def test_long_interview_runs_concurrently(self):
        # Verify the pool is used — without concurrency, N chunks take N× wall
        # time. With concurrency=4, 4 chunks run in parallel.
        import time
        # Pad segment text so chunks actually split (short text fits in 1 chunk
        # even on a 2-hour interview).
        t = _make_long_transcript(
            minutes=120, text='sample text ' + 'x' * 500
        )

        chunks_seen = []

        def slow_json(sys_p, user_p, **kwargs):
            chunks_seen.append(time.time())
            time.sleep(0.1)
            return '{"candidates": []}'

        start = time.time()
        with patch.object(ai_analysis, '_call_ai_json', side_effect=slow_json):
            ai_analysis.chat_about_transcript(t, "x")
        elapsed = time.time() - start

        # With concurrency=4 and N chunks, wall time ≈ ceil(N/4) × 0.1 + overhead.
        # Without concurrency it'd be N × 0.1. Use a loose bound.
        assert len(chunks_seen) >= 2  # sanity: multiple chunks
        serial_lower_bound = 0.1 * len(chunks_seen)
        assert elapsed < serial_lower_bound * 0.8, (
            f"Expected concurrent execution but wall time {elapsed:.2f}s approaches "
            f"serial bound {serial_lower_bound:.2f}s for {len(chunks_seen)} chunks."
        )

    def test_long_interview_empty_results_returns_apology(self):
        t = _make_long_transcript(minutes=90)

        def fake_json(*args, **kwargs):
            return '{"candidates": []}'

        with patch.object(ai_analysis, '_call_ai_json', side_effect=fake_json):
            reply = ai_analysis.chat_about_transcript(t, "find something obscure")

        assert '[CLIP:' not in reply
        # Some kind of "nothing found" message.
        assert len(reply) > 0

    def test_short_interview_never_calls_json_path(self):
        # Short interviews go through the single-prompt Layer 1 path, not Layer 2.
        t = _make_long_transcript(minutes=30)  # under 60 min
        json_calls = []

        def fake_json(*args, **kwargs):
            json_calls.append(1)
            return '{"candidates": []}'

        def fake_chat(prompt, system_prompt=""):
            return 'stub'

        with patch.object(ai_analysis, '_call_ai_json', side_effect=fake_json), \
             patch.object(ai_analysis, '_call_ai_chat', side_effect=fake_chat):
            ai_analysis.chat_about_transcript(t, "x")

        assert json_calls == [], (
            "Short interviews must never hit Layer 2 — "
            f"got {len(json_calls)} _call_ai_json invocations."
        )


class TestFindKeywordChunkIndices:
    """Identifies chunks where the user's search terms literally appear.

    Layer 2 uses this to narrow search scope on specific-topic queries — the
    reason small models were returning Crane Estate clips for "moose hill"
    queries is that every chunk ran and non-keyword chunks got to score.
    """

    def _chunk(self, text):
        return [{'text': text, 'start': 0, 'end': 60, 'speaker': 'Chris'}]

    def test_empty_keywords_returns_empty_list(self):
        chunks = [self._chunk('blah')]
        assert ai_analysis._find_keyword_chunk_indices(chunks, [], []) == []

    def test_matches_phrase_substring_case_insensitively(self):
        chunks = [self._chunk('He walked to Moose Hill Farm at dawn.')]
        assert ai_analysis._find_keyword_chunk_indices(
            chunks, ['moose hill'], []
        ) == [0]

    def test_matches_word_as_whole_token(self):
        chunks = [
            self._chunk('The chestnut trees towered over us.'),
            self._chunk('We ate pechestnutpie at noon.'),  # "chestnut" inside another word
        ]
        hits = ai_analysis._find_keyword_chunk_indices(chunks, [], ['chestnut'])
        assert hits == [0], (
            "word matching must be whole-token — 'pechestnutpie' is not a chestnut hit"
        )

    def test_returns_only_matching_chunk_indices(self):
        chunks = [
            self._chunk('Opening remarks about the weather.'),
            self._chunk('Moose Hill is a real place.'),
            self._chunk('Back to the weather.'),
            self._chunk('More about Moose Hill Farm.'),
        ]
        assert ai_analysis._find_keyword_chunk_indices(
            chunks, ['moose hill'], []
        ) == [1, 3]

    def test_phrase_takes_precedence_over_word(self):
        chunks = [self._chunk('Moose Hill Farm was beautiful.')]
        # Both phrase and words listed; phrase hits, we short-circuit.
        hits = ai_analysis._find_keyword_chunk_indices(
            chunks, ['moose hill'], ['moose', 'hill']
        )
        assert hits == [0]

    def test_phrase_match_excludes_word_only_chunks(self):
        # The bug this guards: a "moose hill" query anchored onto chunks
        # that only contained "hill" (unrelated Crane Estate context)
        # because word matching ran alongside phrase matching. Phrases
        # must suppress word-only hits in OTHER chunks.
        chunks = [
            self._chunk('Opening weather remarks.'),
            self._chunk('Moose Hill Farm was the turning point.'),
            self._chunk('The Crane Estate sits on a hill overlooking the marsh.'),
            self._chunk('Back to Moose Hill the next summer.'),
        ]
        hits = ai_analysis._find_keyword_chunk_indices(
            chunks, ['moose hill'], ['moose', 'hill']
        )
        assert hits == [1, 3], (
            "chunk 2 ('hill' without 'moose') must not anchor when a "
            "phrase match exists elsewhere"
        )

    def test_falls_back_to_word_matches_when_no_phrase_hits(self):
        # If no chunk contains the full phrase, word matches are the only
        # signal we have — fall back to them rather than returning empty.
        chunks = [
            self._chunk('We saw a moose in the field.'),
            self._chunk('The chestnut was in bloom.'),
        ]
        hits = ai_analysis._find_keyword_chunk_indices(
            chunks, ['moose hill'], ['moose', 'hill']
        )
        assert hits == [0]


class TestKeywordAnchoringInLayer2:
    """End-to-end: when the transcript literally contains the user's keyword,
    Layer 2 must restrict search to keyword-anchored chunks and switch the
    prompt into strict mode."""

    def _transcript_with_keyword(self, minutes, keyword_at_second, keyword_text='moose hill'):
        # Build a long transcript with the keyword embedded at one location.
        segs = []
        t = 0.0
        total = int(minutes * 60 / 5.0)
        for _ in range(total):
            # Inject keyword once near the target time. Elsewhere: filler.
            is_hit = abs(t - keyword_at_second) < 5
            text = (f'We walked to {keyword_text} farm this morning. '
                    if is_hit else 'sample text ' + 'x' * 400)
            segs.append({
                'start': t, 'end': t + 5.0,
                'start_formatted':
                    f"{int(t)//3600:02d}:{(int(t)%3600)//60:02d}:{int(t)%60:02d}.000",
                'text': text, 'speaker': 'Chris',
            })
            t += 5.0
        return {'segments': segs}

    def test_keyword_match_restricts_search_to_anchored_chunks(self):
        t = self._transcript_with_keyword(minutes=90, keyword_at_second=600)
        captured_prompts = []

        def fake_json(sys_p, user_p, **kwargs):
            captured_prompts.append(sys_p)
            return '{"candidates": []}'

        with patch.object(ai_analysis, '_call_ai_json', side_effect=fake_json):
            ai_analysis.chat_about_transcript(t, 'what about moose hill?')

        # Every prompt must be for a chunk that actually contains the phrase.
        assert len(captured_prompts) >= 1
        for p in captured_prompts:
            excerpt = p.split('EXCERPT:', 1)[-1].lower()
            assert 'moose hill' in excerpt, (
                'Anchored search must not run chunks that lack the keyword.'
            )

    def test_strict_mode_prompt_when_anchored(self):
        t = self._transcript_with_keyword(minutes=90, keyword_at_second=600)
        captured_prompts = []

        def fake_json(sys_p, user_p, **kwargs):
            captured_prompts.append(sys_p)
            return '{"candidates": []}'

        with patch.object(ai_analysis, '_call_ai_json', side_effect=fake_json):
            ai_analysis.chat_about_transcript(t, 'what about moose hill?')

        # Strict prompt tells the model the excerpt contains the terms and
        # it should return at least one candidate — not default to empty.
        strict_phrase = 'you should return at least one candidate'
        assert any(strict_phrase in p for p in captured_prompts), (
            'Anchored chunks must use strict keyword prompt'
        )

    def test_no_keyword_match_falls_back_to_full_scan(self):
        # Transcript has no 'zebra'; all chunks should be searched.
        t = _make_long_transcript(
            minutes=90, text='sample text ' + 'x' * 400
        )
        chunk_count = [0]

        def fake_json(sys_p, user_p, **kwargs):
            chunk_count[0] += 1
            return '{"candidates": []}'

        with patch.object(ai_analysis, '_call_ai_json', side_effect=fake_json):
            ai_analysis.chat_about_transcript(t, 'tell me about zebras')

        # No anchor found → full scan over multiple chunks.
        assert chunk_count[0] >= 2

    def test_abstract_query_falls_back_to_full_scan(self):
        # No keywords extracted → every chunk searched.
        t = _make_long_transcript(
            minutes=90, text='sample text ' + 'x' * 400
        )
        chunk_count = [0]

        def fake_json(sys_p, user_p, **kwargs):
            chunk_count[0] += 1
            return '{"candidates": []}'

        with patch.object(ai_analysis, '_call_ai_json', side_effect=fake_json):
            ai_analysis.chat_about_transcript(t, 'find the best moment')

        assert chunk_count[0] >= 2
