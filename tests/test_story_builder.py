"""Story Builder tests — vector chunking + /story/build recovery path.

The failure mode we're guarding against: a 100-minute transcript caused the
pre-chunked ``generate_segment_vectors`` to silently return zero segments,
``/analyze`` then skipped saving ``segment_vectors.json``, and ``/story/build``
fell through to the raw-transcript path which also couldn't cope with the
long input — the editor saw an "Untitled" story with zero clips and no error.

These tests lock in:
  - ``generate_segment_vectors`` chunks long transcripts and merges segments
    with globally-unique seg_ids.
  - ``/story/build`` auto-generates missing vectors on demand and persists
    them for reuse.
  - The endpoint surfaces an explicit error (not an empty "success") when the
    model returns zero clips.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ai_analysis  # noqa: E402
import app as app_module  # noqa: E402


def _mk_segments(total_seconds, step=5.0):
    segs = []
    t = 0.0
    i = 0
    while t < total_seconds:
        end = min(t + step, total_seconds)
        segs.append({
            'start': t, 'end': end,
            'start_formatted': f"{int(t)//3600:02d}:{(int(t)%3600)//60:02d}:{int(t)%60:02d}",
            'text': f"line {i}", 'speaker': 'A',
        })
        t += step
        i += 1
    return segs


# ---------- generate_segment_vectors chunking -------------------------------

class TestGenerateSegmentVectorsChunking:
    def test_short_transcript_single_call(self, monkeypatch):
        calls = []

        def _stub(text, name):
            calls.append(name)
            return [
                {'seg_id': 'SEG001', 'timecode_in': '00:00:00', 'timecode_out': '00:00:30',
                 'thread_title': 'intro', 'memory_type': 'episodic',
                 'narrative_score': 'high', 'beat_type': 'hook', 'theme_tags': ['a']},
            ]
        monkeypatch.setattr(ai_analysis, '_generate_vectors_single_chunk', _stub)

        out = ai_analysis.generate_segment_vectors(
            {'segments': _mk_segments(300)}, project_name='Short',
        )
        assert len(calls) == 1
        assert len(out) == 1

    def test_long_transcript_chunks_and_merges(self, monkeypatch):
        calls = []

        def _stub(text, name):
            calls.append(name)
            n = len(calls)
            # Each chunk restarts numbering at SEG001 — the writer must
            # renumber to globally-unique IDs before _normalize sees them.
            return [
                {'seg_id': 'SEG001', 'timecode_in': '00:00:00', 'timecode_out': '00:00:30',
                 'thread_title': f'thread {n}-a', 'memory_type': 'episodic',
                 'narrative_score': 'high', 'beat_type': 'hook', 'theme_tags': ['t']},
                {'seg_id': 'SEG002', 'timecode_in': '00:01:00', 'timecode_out': '00:02:00',
                 'thread_title': f'thread {n}-b', 'memory_type': 'semantic',
                 'narrative_score': 'medium', 'beat_type': 'context', 'theme_tags': ['t']},
            ]
        monkeypatch.setattr(ai_analysis, '_generate_vectors_single_chunk', _stub)

        out = ai_analysis.generate_segment_vectors(
            {'segments': _mk_segments(65 * 60)}, project_name='Long',
        )
        # ~4-5 chunks × 2 segments each.
        assert len(calls) >= 4
        assert len(out) == 2 * len(calls)
        # Every seg_id is globally unique.
        ids = [s['seg_id'] for s in out]
        assert len(ids) == len(set(ids))
        # Chunk labels include the "part X/N" suffix.
        assert all('part' in name for name in calls)

    def test_chunk_failure_is_tolerated(self, monkeypatch):
        call_count = {'n': 0}

        def _flaky(text, name):
            call_count['n'] += 1
            if call_count['n'] == 2:
                raise RuntimeError("boom")
            return [
                {'seg_id': 'SEG001', 'timecode_in': '00:00:00', 'timecode_out': '00:00:30',
                 'thread_title': 'ok', 'memory_type': 'episodic',
                 'narrative_score': 'high', 'beat_type': 'hook', 'theme_tags': ['t']},
            ]
        monkeypatch.setattr(ai_analysis, '_generate_vectors_single_chunk', _flaky)

        out = ai_analysis.generate_segment_vectors(
            {'segments': _mk_segments(50 * 60)}, project_name='Flaky',
        )
        # The failing chunk contributes zero; the rest still produce output.
        assert len(out) == call_count['n'] - 1
        assert call_count['n'] >= 2


# ---------- /story/build endpoint behavior ----------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setitem(app_module.app.config, "PROJECTS_DIR", str(tmp_path / "projects"))
    os.makedirs(app_module.app.config["PROJECTS_DIR"], exist_ok=True)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


@pytest.fixture
def transcribed_project(client, tmp_path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF____WAVE")
    resp = client.post("/create", json={
        "source_path": str(audio),
        "project_name": "Story Test",
    })
    pid = resp.get_json()["project_id"]
    meta_path = Path(app_module.app.config["PROJECTS_DIR"]) / pid / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["transcript"] = {"segments": _mk_segments(600), "language": "en"}
    meta_path.write_text(json.dumps(meta))
    return pid


class TestStoryBuildMenuPruning:
    """The clip-selection pass builds a menu from segment_vectors and sends
    it to the model. Low-scored segments are explicitly filler per the
    vector-generation prompt and the model won't pick them — excluding them
    from the menu shrinks the prompt ~35% on long interviews, which is what
    keeps the 8B-Gemma clip-selection call inside its timeout."""

    def _vectors(self):
        return [
            {'seg_id': f'SEG{i:03d}', 'timecode_in': f'00:{i:02d}:00',
             'timecode_out': f'00:{i:02d}:30',
             'thread_title': f't{i}', 'memory_type': 'episodic',
             'narrative_score': score, 'beat_type': 'hook',
             'theme_tags': ['x'],
             'transcript_excerpt': f'excerpt {i}', 'frozen': True}
            for i, score in enumerate(['high', 'medium', 'low', 'high', 'low'], start=1)
        ]

    def test_menu_drops_low_score_segments(self, monkeypatch):
        captured = {}

        def _stub_call_ai(prompt, system_prompt=""):
            captured['prompt'] = prompt
            return '{"clips": []}'

        monkeypatch.setattr(ai_analysis, '_call_ai', _stub_call_ai)
        monkeypatch.setattr(ai_analysis, 'inject_my_style', lambda p, profile_id=None: p)

        ai_analysis._build_story_from_vectors(
            self._vectors(), message="build", project_name="T",
        )
        prompt = captured['prompt']
        # "high" and "medium" segments appear in the menu.
        assert 'SEG001' in prompt  # high
        assert 'SEG002' in prompt  # medium
        assert 'SEG004' in prompt  # high
        # "low" segments are pruned out.
        assert 'SEG003' not in prompt
        assert 'SEG005' not in prompt

    def test_fallback_when_all_segments_are_low(self, monkeypatch):
        # Pathological case: every segment was scored "low". Restore the full
        # menu so the user still gets some attempt.
        all_low = [
            dict(s, narrative_score='low') for s in self._vectors()
        ]
        captured = {}
        monkeypatch.setattr(ai_analysis, '_call_ai', lambda p, s="": (captured.update(prompt=p), '{"clips": []}')[1])
        monkeypatch.setattr(ai_analysis, 'inject_my_style', lambda p, profile_id=None: p)

        ai_analysis._build_story_from_vectors(all_low, message="build", project_name="T")
        for i in range(1, 6):
            assert f'SEG{i:03d}' in captured['prompt']


class TestStoryBuildEndpoint:
    def test_auto_generates_vectors_when_missing(self, client, transcribed_project, monkeypatch):
        # No segment_vectors.json on disk. The endpoint should call
        # generate_segment_vectors on demand and persist the result.
        generated = [
            {'seg_id': 'SEG001', 'timecode_in': '00:00:00', 'timecode_out': '00:00:30',
             'thread_title': 't', 'memory_type': 'episodic',
             'narrative_score': 'high', 'beat_type': 'hook', 'theme_tags': ['x'],
             'transcript_excerpt': 'hi', 'frozen': True},
        ]
        vec_calls = []

        def _fake_generate(transcript, project_name="Interview"):
            vec_calls.append(project_name)
            return generated

        def _fake_build(transcript, message, project_name="Interview",
                        segment_vectors=None, profile_id=None):
            # The endpoint must pass the auto-generated vectors through.
            assert segment_vectors == generated
            return {
                'story_title': 'A Good Story',
                'target_duration': '1 min',
                'reasoning': 'because',
                'clips': [{'order': 1, 'seg_id': 'SEG001', 'title': 'x',
                           'start_time': '00:00:00', 'end_time': '00:00:30',
                           'transcript': 'hi', 'editorial_note': ''}],
            }

        monkeypatch.setattr("ai_analysis.generate_segment_vectors", _fake_generate)
        monkeypatch.setattr("ai_analysis.build_story", _fake_build)

        resp = client.post(
            f"/project/{transcribed_project}/story/build",
            json={"message": "build me a story"},
        )
        assert resp.status_code == 200, resp.data
        body = resp.get_json()
        assert body["status"] == "built"
        assert body["build"]["story_title"] == "A Good Story"
        assert len(body["build"]["clips"]) == 1
        assert len(vec_calls) == 1

        vectors_path = Path(app_module.app.config["PROJECTS_DIR"]) / transcribed_project / "segment_vectors.json"
        assert vectors_path.exists()
        assert json.loads(vectors_path.read_text()) == generated

    def test_reuses_existing_vectors_without_regenerating(self, client, transcribed_project, monkeypatch):
        # Pre-populate segment_vectors.json.
        existing = [
            {'seg_id': 'SEG042', 'timecode_in': '00:05:00', 'timecode_out': '00:05:30',
             'thread_title': 'cached', 'memory_type': 'episodic',
             'narrative_score': 'high', 'beat_type': 'hook', 'theme_tags': ['x'],
             'transcript_excerpt': 'cached excerpt', 'frozen': True},
        ]
        vec_path = Path(app_module.app.config["PROJECTS_DIR"]) / transcribed_project / "segment_vectors.json"
        vec_path.parent.mkdir(parents=True, exist_ok=True)
        vec_path.write_text(json.dumps(existing))

        def _no_regen(*a, **kw):
            raise AssertionError("generate_segment_vectors should NOT run when cache exists")
        monkeypatch.setattr("ai_analysis.generate_segment_vectors", _no_regen)
        monkeypatch.setattr("ai_analysis.build_story", lambda *a, segment_vectors=None, **kw: {
            'story_title': 'Y', 'clips': [{'seg_id': 'SEG042', 'title': 'cached', 'start_time': '00:05:00', 'end_time': '00:05:30', 'order': 1}],
        })

        resp = client.post(
            f"/project/{transcribed_project}/story/build",
            json={"message": "build"},
        )
        assert resp.status_code == 200, resp.data

    def test_empty_clips_returns_error_not_success(self, client, transcribed_project, monkeypatch):
        """The bug: a zero-clip AI response used to save a broken 'success' build."""
        monkeypatch.setattr("ai_analysis.generate_segment_vectors", lambda *a, **kw: [])
        monkeypatch.setattr("ai_analysis.build_story", lambda *a, **kw: {
            'story_title': 'Untitled', 'target_duration': '', 'reasoning': '', 'clips': [],
        })

        resp = client.post(
            f"/project/{transcribed_project}/story/build",
            json={"message": "build me something"},
        )
        assert resp.status_code == 500
        body = resp.get_json()
        assert 'error' in body
        assert '0 clips' in body['error'].lower() or 'zero clips' in body['error'].lower()

        # Nothing persisted — an empty build on disk would confuse the UI
        # list later.
        builds_path = Path(app_module.app.config["PROJECTS_DIR"]) / transcribed_project / "story_builds.json"
        assert not builds_path.exists()
