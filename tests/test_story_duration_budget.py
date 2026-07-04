"""Story Builder duration-budget tests.

The live bug: a tester asked for a 14-minute build from a 40:11 timeline and
got 5.5 minutes, reproducibly. The requested duration was never parsed or
enforced in code — the only budget was a prose rule the small Gemma models
can't execute (they'd have to sum 40+ dur=Ns menu values), and the stored
target_duration was the model's own unvalidated echo.

These tests lock in the deterministic layer:
  - parse_target_duration_seconds runs at the top of build_story;
  - the vector-menu prompt carries a numeric TARGET TOTAL RUNTIME line with
    a 0.9-1.1x band and a clip count derived from the ACTUAL menu average
    (replacing the misleading "3-4 clips per minute" heuristic);
  - _enforce_duration_budget measures the hydrated total, tops up from
    unused segment vectors, trims overshoot, and reports honest numbers
    (actual_duration_seconds / duration_enforced / duration_shortfall_note);
  - /story/build persists those fields and surfaces the shortfall note.
"""

import json
import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ai_analysis  # noqa: E402
import app as app_module  # noqa: E402


def _tc(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _mk_vectors(n=60, dur=40, scores=('high', 'medium', 'medium', 'low')):
    """n segments of ``dur`` seconds each, scores cycling through ``scores``.

    Defaults model the tester's timeline: 60 x 40s = a 40-minute menu.
    """
    out = []
    for i in range(n):
        start = i * dur
        out.append({
            'seg_id': f'SEG{i:03d}',
            'timecode_in': _tc(start),
            'timecode_out': _tc(start + dur),
            'thread_title': f'thread {i}',
            'memory_type': 'episodic' if i % 2 == 0 else 'semantic',
            'narrative_score': scores[i % len(scores)],
            'beat_type': ('hook', 'context', 'pressure', 'turn', 'resolution')[i % 5],
            'theme_tags': ['t'],
            'transcript_excerpt': f'excerpt {i}',
            'frozen': True,
        })
    return out


def _clip_total(clips):
    return sum(
        ai_analysis._tc_to_seconds(c['end_time']) - ai_analysis._tc_to_seconds(c['start_time'])
        for c in clips
    )


def _model_response(seg_ids):
    return json.dumps({
        'story_title': 'Stubbed',
        'target_duration': '5 minutes',  # deliberately wrong model echo
        'reasoning': 'r',
        'clips': [
            {'order': i + 1, 'seg_id': sid, 'title': f'c{i}',
             'editorial_note': 'ROLE: context'}
            for i, sid in enumerate(seg_ids)
        ],
    })


class TestStoryDurationPromptInjection:
    def _capture(self, monkeypatch, message, vectors=None):
        captured = {}

        def _stub(prompt, system_prompt="", **kwargs):
            captured['prompt'] = prompt
            captured['system'] = system_prompt
            return '{"clips": []}'

        monkeypatch.setattr(ai_analysis, '_call_ai', _stub)
        monkeypatch.setattr(ai_analysis, 'inject_my_style', lambda p, profile_id=None: p)
        ai_analysis.build_story(
            {'segments': []}, message=message, project_name='T',
            segment_vectors=vectors if vectors is not None else _mk_vectors(),
        )
        return captured

    def test_target_line_carries_menu_derived_count(self, monkeypatch):
        cap = self._capture(monkeypatch, 'build me a 14 minute cut')
        prompt = cap['prompt']
        assert 'TARGET TOTAL RUNTIME: 840s (14:00)' in prompt
        # Menu is 60 x 40s -> ceil(840/40) = 21 segments, capped to the
        # 18-segment prompt ask (an uncapped 40-segment demand made the
        # model truncate its JSON mid-array; the code-side top-up owns the
        # remainder). The capped ask states the total the count CAN
        # deliver (18 x 40s), not the unreachable 756-924s band.
        assert math.ceil(840 / 40) == 21  # above the cap by construction
        assert 'at least 18 segments' in prompt
        assert 'must sum to between' not in prompt
        assert 'should total at least 720s' in prompt

    def test_uncapped_target_line_carries_band(self, monkeypatch):
        # ceil(240/40) = 6 <= 18: the band wording survives verbatim.
        cap = self._capture(monkeypatch, 'build me a 4 minute cut')
        prompt = cap['prompt']
        assert 'TARGET TOTAL RUNTIME: 240s (4:00)' in prompt
        assert 'must sum to between 216s and 264s' in prompt
        assert 'at least 6 segments' in prompt

    def test_no_target_keeps_prompt_free_of_budget_line(self, monkeypatch):
        cap = self._capture(monkeypatch, 'build me a story about the harbor')
        assert 'TARGET TOTAL RUNTIME' not in cap['prompt']

    @pytest.mark.parametrize("message", [
        # Story briefs mentioning CONTENT durations — narrative facts, not
        # deliverable asks. These used to parse (10800s / 21600s / 300s!)
        # and bury the model's curated arc in auto-filler segments, each
        # noted 'Added to reach the requested runtime.'
        'tell the story of the 3 hour rescue',
        'build the arc of the night the fire took 6 hours to contain',
        'a story about the 5 minute standing ovation she got',
    ])
    def test_narrative_fact_durations_do_not_trigger_budget(self, monkeypatch, message):
        cap = self._capture(monkeypatch, message)
        assert 'TARGET TOTAL RUNTIME' not in cap['prompt'], message

    def test_misleading_clips_per_minute_heuristic_is_gone(self, monkeypatch):
        # The old prose rule assumed 15-20s clips; vector segments run
        # 25-50s, which is how 14-minute asks came back at 5.5 minutes.
        cap = self._capture(monkeypatch, 'build me a 14 minute cut')
        assert '3-4 clips per minute' not in cap['system']
        assert 'TARGET TOTAL RUNTIME' in cap['system']

    def test_raw_transcript_path_also_gets_budget_line(self, monkeypatch):
        captured = {}

        def _stub(prompt, system_prompt="", **kwargs):
            captured['prompt'] = prompt
            captured['system'] = system_prompt
            return '{"clips": []}'

        monkeypatch.setattr(ai_analysis, '_call_ai', _stub)
        monkeypatch.setattr(ai_analysis, 'inject_my_style', lambda p, profile_id=None: p)
        ai_analysis.build_story(
            {'segments': [{'start': 0.0, 'end': 30.0, 'text': 'hello',
                           'start_formatted': '00:00:00', 'speaker': 'A'}]},
            message='build me a 4 minute cut', project_name='T',
            segment_vectors=None,
        )
        assert 'TARGET TOTAL RUNTIME: 240s (4:00)' in captured['prompt']
        assert '3-4 clips per minute' not in captured['system']


class TestEnforceDurationBudget:
    def test_top_up_reaches_band_and_prefers_non_low(self):
        vectors = _mk_vectors()
        # Model picked 12 x 40s = 480s against an 840s ask.
        clips = [{
            'order': i + 1, 'seg_id': f'SEG{i:03d}', 'title': f'c{i}',
            'start_time': _tc(i * 40), 'end_time': _tc(i * 40 + 40),
            'transcript': 'x', 'editorial_note': '',
            'narrative_score': vectors[i]['narrative_score'],
            'beat_type': 'hook' if i == 0 else ('resolution' if i == 11 else 'context'),
        } for i in range(12)]
        out, meta = ai_analysis._enforce_duration_budget(clips, 840.0, vectors)
        total = _clip_total(out)
        assert 0.9 * 840 <= total <= 1.15 * 840
        assert meta['duration_enforced'] is True
        assert meta['actual_duration_seconds'] == pytest.approx(total)
        assert 'duration_shortfall_note' not in meta
        # Enough non-low material exists -> no 'low' segment drafted.
        added = [c for c in out if c.get('editorial_note') == 'Added to reach the requested runtime.']
        assert added and all(c['narrative_score'] != 'low' for c in added)
        # Closer stays last; orders renumbered sequentially.
        assert out[-1]['beat_type'] == 'resolution'
        assert [c['order'] for c in out] == list(range(1, len(out) + 1))

    def test_top_up_dips_into_low_only_when_hard_floor_unreachable(self):
        # 6 non-low (240s) + 6 low (240s); ask 480s -> hard floor 384s is
        # unreachable on non-low alone, so low segments are drafted.
        vectors = _mk_vectors(n=12, dur=40, scores=('medium', 'low'))
        clips = [{
            'order': 1, 'seg_id': 'SEG000', 'title': 'c',
            'start_time': _tc(0), 'end_time': _tc(40),
            'narrative_score': 'medium', 'beat_type': 'hook',
        }]
        out, meta = ai_analysis._enforce_duration_budget(clips, 480.0, vectors)
        assert any(c.get('narrative_score') == 'low' for c in out)
        assert _clip_total(out) >= 0.8 * 480

    def test_shortfall_note_when_all_material_cannot_reach_target(self):
        vectors = _mk_vectors(n=8, dur=40, scores=('high', 'medium'))  # 320s total
        clips = [{
            'order': i + 1, 'seg_id': f'SEG{i:03d}', 'title': f'c{i}',
            'start_time': _tc(i * 40), 'end_time': _tc(i * 40 + 40),
            'narrative_score': 'high', 'beat_type': 'context',
        } for i in range(3)]
        out, meta = ai_analysis._enforce_duration_budget(clips, 840.0, vectors)
        assert meta['duration_enforced'] is True
        note = meta.get('duration_shortfall_note')
        assert note and '14:00' in note
        # Everything usable was included.
        assert _clip_total(out) == pytest.approx(320.0)

    def test_trim_removes_weakest_never_hook_or_closer(self):
        # 10 x 60s = 600s against a 240s ask (ceiling 276s).
        clips = [{
            'order': n + 1, 'seg_id': f'S{n}', 'title': f'c{n}',
            'start_time': _tc(n * 70), 'end_time': _tc(n * 70 + 60),
            'beat_type': 'hook' if n == 0 else ('resolution' if n == 9 else 'context'),
            'narrative_score': 'low' if n in (3, 5, 7) else 'medium',
        } for n in range(10)]
        out, meta = ai_analysis._enforce_duration_budget(clips, 240.0, None)
        total = _clip_total(out)
        assert total <= 1.15 * 240
        assert total >= 0.9 * 240
        assert out[0]['beat_type'] == 'hook'
        assert out[-1]['beat_type'] == 'resolution'
        # The low-scored middles went first.
        surviving_ids = {c['seg_id'] for c in out}
        assert {'S3', 'S5', 'S7'} & surviving_ids == set()
        assert meta['duration_enforced'] is True

    def test_within_band_reports_without_modifying(self):
        clips = [{
            'order': 1, 'seg_id': 'SEG000', 'title': 'c',
            'start_time': _tc(0), 'end_time': _tc(120),
            'narrative_score': 'high', 'beat_type': 'hook',
        }]
        out, meta = ai_analysis._enforce_duration_budget(clips, 120.0, _mk_vectors())
        assert out == clips
        assert meta['duration_enforced'] is False
        assert meta['actual_duration_seconds'] == pytest.approx(120.0)

    def test_two_clip_overshoot_is_shortened_into_band(self):
        # Verified defect: two 40s clips vs a 30s target shipped at 80s
        # (2.67x) untouched — the removal branch required len(clips) > 2
        # and only ever dropped interior clips. The longest clips' ends
        # are now tightened toward the target instead.
        clips = [{
            'order': 1, 'seg_id': 'S0', 'title': 'a',
            'start_time': _tc(0), 'end_time': _tc(40), 'beat_type': 'hook',
        }, {
            'order': 2, 'seg_id': 'S1', 'title': 'b',
            'start_time': _tc(100), 'end_time': _tc(140), 'beat_type': 'resolution',
        }]
        out, meta = ai_analysis._enforce_duration_budget(clips, 30.0, None)
        total = _clip_total(out)
        assert total <= 1.15 * 30
        assert meta['duration_enforced'] is True
        assert meta['actual_duration_seconds'] == pytest.approx(total)
        # Both clips survive; only end_time moved, and only downward
        # (start never moves, so segment boundaries hold).
        assert len(out) == 2
        assert out[0]['start_time'] == _tc(0)
        assert out[1]['start_time'] == _tc(100)
        assert ai_analysis._tc_to_seconds(out[0]['end_time']) <= 40
        assert ai_analysis._tc_to_seconds(out[1]['end_time']) <= 140

    def test_single_clip_overshoot_is_shortened(self):
        clips = [{
            'order': 1, 'seg_id': 'S0', 'title': 'a',
            'start_time': _tc(0), 'end_time': _tc(80), 'beat_type': 'hook',
        }]
        out, meta = ai_analysis._enforce_duration_budget(clips, 30.0, None)
        total = _clip_total(out)
        assert 0.9 * 30 <= total <= 1.15 * 30
        assert meta['duration_enforced'] is True

    def test_shortening_never_goes_below_15s(self):
        # 10s + 100s clips vs a 20s target: the 100s clip clamps at 15s,
        # the 10s clip is never touched, and the loop stops cleanly even
        # though the ceiling stays out of reach.
        clips = [{
            'order': 1, 'seg_id': 'S0', 'title': 'a',
            'start_time': _tc(0), 'end_time': _tc(10), 'beat_type': 'hook',
        }, {
            'order': 2, 'seg_id': 'S1', 'title': 'b',
            'start_time': _tc(100), 'end_time': _tc(200), 'beat_type': 'resolution',
        }]
        out, meta = ai_analysis._enforce_duration_budget(clips, 20.0, None)
        spans = [_clip_total([c]) for c in out]
        assert spans[0] == pytest.approx(10.0)   # untouched (already < 15s)
        assert spans[1] == pytest.approx(15.0)   # clamped at the floor
        assert meta['duration_enforced'] is True

    def test_hook_and_closer_survive_removal_then_get_shortened(self):
        # 10 x 40s vs a 30s target: whole-clip removal strands hook+closer
        # at 80s (still 2.67x) — the shortening pass lands them in band.
        clips = [{
            'order': i + 1, 'seg_id': f'S{i}', 'title': f'c{i}',
            'start_time': _tc(i * 50), 'end_time': _tc(i * 50 + 40),
            'beat_type': 'hook' if i == 0 else ('resolution' if i == 9 else 'context'),
            'narrative_score': 'medium',
        } for i in range(10)]
        out, meta = ai_analysis._enforce_duration_budget(clips, 30.0, None)
        assert out[0]['beat_type'] == 'hook'
        assert out[-1]['beat_type'] == 'resolution'
        assert _clip_total(out) <= 1.15 * 30
        assert meta['duration_enforced'] is True


class TestBuildStoryDurationIntegration:
    def test_14_minute_ask_lands_in_band(self, monkeypatch):
        # The tester's regime: 40-minute menu, model returns 12 clips
        # (~480s), ask is 840s. The budget layer must close the gap.
        vectors = _mk_vectors()
        monkeypatch.setattr(
            ai_analysis, '_call_ai',
            lambda p, s="", **kw: _model_response([f'SEG{i:03d}' for i in range(12)]),
        )
        monkeypatch.setattr(ai_analysis, 'inject_my_style', lambda p, profile_id=None: p)

        result = ai_analysis.build_story(
            {'segments': []},
            message=('build me a 14 minute paranormal investigation video '
                     'with intro, investigation, arc, and conclusion'),
            project_name='T', segment_vectors=vectors,
        )
        total = _clip_total(result['clips'])
        assert 0.9 * 840 <= total <= 1.15 * 840, f'total {total}s outside band'
        # target_duration is the PARSED ask, not the model's '5 minutes' echo.
        assert result['target_duration'] == '14:00'
        assert result['duration_enforced'] is True
        assert result['actual_duration_seconds'] == pytest.approx(total)
        assert 'duration_shortfall_note' not in result

    def test_shortfall_flows_through_build_story(self, monkeypatch):
        vectors = _mk_vectors(n=10, dur=40)  # 400s of material, 840s ask
        monkeypatch.setattr(
            ai_analysis, '_call_ai',
            lambda p, s="", **kw: _model_response(['SEG000', 'SEG001', 'SEG002']),
        )
        monkeypatch.setattr(ai_analysis, 'inject_my_style', lambda p, profile_id=None: p)
        result = ai_analysis.build_story(
            {'segments': []}, message='build me a 14 minute cut',
            project_name='T', segment_vectors=vectors,
        )
        assert result.get('duration_shortfall_note')

    def test_no_duration_ask_leaves_result_shape_unchanged(self, monkeypatch):
        vectors = _mk_vectors(n=6)
        monkeypatch.setattr(
            ai_analysis, '_call_ai',
            lambda p, s="", **kw: _model_response(['SEG000', 'SEG001']),
        )
        monkeypatch.setattr(ai_analysis, 'inject_my_style', lambda p, profile_id=None: p)
        result = ai_analysis.build_story(
            {'segments': []}, message='build me a story about the harbor',
            project_name='T', segment_vectors=vectors,
        )
        # Model echo preserved, no budget fields — today's behavior.
        assert result['target_duration'] == '5 minutes'
        assert 'duration_enforced' not in result
        assert 'actual_duration_seconds' not in result
        assert len(result['clips']) == 2


# ---------- /story/build endpoint persistence --------------------------------

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
        "project_name": "Duration Test",
    })
    pid = resp.get_json()["project_id"]
    meta_path = Path(app_module.app.config["PROJECTS_DIR"]) / pid / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["transcript"] = {
        "segments": [{"start": 0.0, "end": 30.0, "text": "hi",
                      "start_formatted": "00:00:00", "speaker": "A"}],
        "language": "en",
    }
    meta_path.write_text(json.dumps(meta))
    return pid


class TestStoryBuildEndpointDurationFields:
    def _stub_result(self, **extra):
        base = {
            'story_title': 'Budgeted',
            'target_duration': '14:00',
            'reasoning': 'r',
            'clips': [{'order': 1, 'seg_id': 'SEG001', 'title': 'x',
                       'start_time': '00:00:00', 'end_time': '00:00:40',
                       'transcript': 'hi', 'editorial_note': ''}],
        }
        base.update(extra)
        return base

    def test_budget_fields_are_persisted(self, client, transcribed_project, monkeypatch):
        monkeypatch.setattr("ai_analysis.generate_segment_vectors", lambda *a, **kw: [])
        monkeypatch.setattr("ai_analysis.build_story", lambda *a, **kw: self._stub_result(
            actual_duration_seconds=812.0, duration_enforced=True,
        ))
        resp = client.post(
            f"/project/{transcribed_project}/story/build",
            json={"message": "build me a 14 minute cut"},
        )
        assert resp.status_code == 200, resp.data
        build = resp.get_json()["build"]
        assert build["actual_duration_seconds"] == 812.0
        assert build["duration_enforced"] is True
        assert build["target_duration"] == "14:00"

        builds_path = Path(app_module.app.config["PROJECTS_DIR"]) / transcribed_project / "story_builds.json"
        persisted = json.loads(builds_path.read_text())[0]
        assert persisted["actual_duration_seconds"] == 812.0
        assert persisted["duration_enforced"] is True

    def test_shortfall_note_surfaces_top_level(self, client, transcribed_project, monkeypatch):
        note = "The build reaches 5:20 of the 14:00 requested."
        monkeypatch.setattr("ai_analysis.generate_segment_vectors", lambda *a, **kw: [])
        monkeypatch.setattr("ai_analysis.build_story", lambda *a, **kw: self._stub_result(
            actual_duration_seconds=320.0, duration_enforced=True,
            duration_shortfall_note=note,
        ))
        resp = client.post(
            f"/project/{transcribed_project}/story/build",
            json={"message": "build me a 14 minute cut"},
        )
        assert resp.status_code == 200, resp.data
        body = resp.get_json()
        assert body["duration_shortfall_note"] == note
        assert body["build"]["duration_shortfall_note"] == note

    def test_no_budget_fields_when_absent(self, client, transcribed_project, monkeypatch):
        monkeypatch.setattr("ai_analysis.generate_segment_vectors", lambda *a, **kw: [])
        monkeypatch.setattr("ai_analysis.build_story", lambda *a, **kw: self._stub_result())
        resp = client.post(
            f"/project/{transcribed_project}/story/build",
            json={"message": "build me a story"},
        )
        assert resp.status_code == 200, resp.data
        body = resp.get_json()
        assert "duration_shortfall_note" not in body
        assert "duration_enforced" not in body["build"]


class TestShortfallNoteReachesTheUI:
    """The backend copies duration_shortfall_note top-level 'so the UI can
    toast it' — this pins the other half of that contract: the story-build
    response handler in project.html actually reads the field and toasts
    it. Without a consumer the honest-numbers note was a dead-end payload
    field and a silently short build showed 'Story built!'."""

    def test_build_story_handler_toasts_shortfall_note(self):
        template = Path(__file__).resolve().parents[1] / "templates" / "project.html"
        html = template.read_text()
        assert "data.duration_shortfall_note" in html, (
            "buildStory() no longer consumes duration_shortfall_note — "
            "the shortfall warning would never reach the user"
        )
        # The consumer lives in the success branch, warning-styled toast.
        idx = html.index("data.duration_shortfall_note")
        window = html[idx:idx + 400]
        assert "showToast(data.duration_shortfall_note" in window


# ---------- prompt-ask cap ----------------------------------------------------

TESTER_RANGE_PROMPT = (
    'build a 15 to 20 minute paranormal investigation video that has a '
    'defined intro, investigation, and ending'
)


class TestStoryPromptClipAskCap:
    """The live 0-clips bug, part 1: a 1200s target over a ~30s menu
    demanded "at least 40 segments — do not stop early", pushing the model
    to write past its output budget and truncate the JSON mid-array. The
    demanded count is now capped; the code-side top-up owns the rest."""

    def test_count_above_cap_is_capped_and_pressure_dropped(self):
        block = ai_analysis._story_duration_prompt_block(1200, 30)
        assert 'at least 18 segments' in block
        assert 'Do not stop early' not in block
        assert 'completed automatically' in block

    def test_count_below_cap_keeps_full_demand(self):
        block = ai_analysis._story_duration_prompt_block(300, 30)
        assert 'at least 10 segments' in block
        assert 'Do not stop early' in block

    def test_capped_block_replaces_band_with_count_consistent_total(self):
        # 18 x 30s can only promise ~540s — demanding the 1080-1320s band
        # in the same block was contradictory (the model either chased the
        # band into the output blowout the cap exists to stop, or obeyed
        # the cap and "failed" the band).
        block = ai_analysis._story_duration_prompt_block(1200, 30)
        assert 'TARGET TOTAL RUNTIME: 1200s (20:00)' in block
        assert 'must sum to between' not in block
        assert 'should total at least 540s' in block
        assert 'completed automatically' in block

    def test_uncapped_block_keeps_band_wording(self):
        block = ai_analysis._story_duration_prompt_block(300, 30)
        assert 'must sum to between 270s and 330s' in block
        assert 'completed automatically' not in block

    def test_capped_build_swaps_the_system_band_bullet(self, monkeypatch):
        # The user-prompt block and the system bullet must agree: for a
        # capped ask the system prompt's "must sum inside the stated band /
        # undershooting is the most common failure" line is replaced with
        # the count-consistent instruction.
        captured = {}

        def _stub(prompt, system_prompt="", **kwargs):
            captured['system'] = system_prompt
            return '{"clips": []}'

        monkeypatch.setattr(ai_analysis, '_call_ai', _stub)
        monkeypatch.setattr(ai_analysis, 'inject_my_style',
                            lambda p, profile_id=None: p)
        ai_analysis.build_story(
            {'segments': []}, message='build me a 14 minute cut',
            project_name='T', segment_vectors=_mk_vectors(),
        )  # ceil(840/40) = 21 > 18 -> capped
        assert 'must sum inside the stated band' not in captured['system']
        assert 'undershooting by stopping early' not in captured['system']
        assert 'completed automatically' in captured['system']
        assert 'TARGET TOTAL RUNTIME' in captured['system']

        ai_analysis.build_story(
            {'segments': []}, message='build me a 4 minute cut',
            project_name='T', segment_vectors=_mk_vectors(),
        )  # ceil(240/40) = 6 <= 18 -> band bullet kept verbatim
        assert 'must sum inside the stated band' in captured['system']
        assert ('undershooting by stopping early is the most common failure'
                in captured['system'])

    def test_raw_path_capped_build_swaps_the_system_band_bullet(self, monkeypatch):
        captured = {}

        def _stub(prompt, system_prompt="", **kwargs):
            captured['system'] = system_prompt
            return '{"clips": []}'

        monkeypatch.setattr(ai_analysis, '_call_ai', _stub)
        monkeypatch.setattr(ai_analysis, 'inject_my_style',
                            lambda p, profile_id=None: p)
        transcript = {'segments': [{'start': 0.0, 'end': 30.0, 'text': 'hi',
                                    'start_formatted': '00:00:00',
                                    'speaker': 'A'}]}
        ai_analysis.build_story(
            transcript, message='build me a 15 minute cut',
            project_name='T', segment_vectors=None,
        )  # ceil(900/20) = 45 > 18 -> capped
        assert 'must sum inside the stated band' not in captured['system']
        assert 'completed automatically' in captured['system']

        ai_analysis.build_story(
            transcript, message='build me a 4 minute cut',
            project_name='T', segment_vectors=None,
        )  # ceil(240/20) = 12 <= 18 -> band bullet kept verbatim
        assert ("(end_time - start_time) values must sum inside the stated "
                "band") in captured['system']


# ---------- num_predict threading ---------------------------------------------

class _SpyProvider:
    """Records every generate() kwargs dict; returns an unusable body so the
    fallback path gets exercised too."""

    def __init__(self, response='{"clips": []}'):
        self.calls = []
        self._response = response

    def generate(self, system_prompt, user_or_messages, task_type="general",
                 **kwargs):
        self.calls.append(kwargs)
        return self._response


class TestStoryNumPredictThreading:
    """The live 0-clips bug, part 2: story builds inherited the provider's
    4096-token output default — a verbose 40-clip JSON does not fit. Both
    story calls must raise the budget; nothing else may change."""

    def _spy(self, monkeypatch):
        spy = _SpyProvider()
        import ai_providers
        monkeypatch.setattr(
            ai_providers, 'get_active_provider', lambda model_resolver=None: spy,
        )
        monkeypatch.setattr(ai_analysis, 'inject_my_style',
                            lambda p, profile_id=None: p)
        return spy

    def test_vector_story_call_raises_output_budget(self, monkeypatch):
        spy = self._spy(monkeypatch)
        ai_analysis.build_story(
            {'segments': []}, message='build a story', project_name='T',
            segment_vectors=_mk_vectors(n=6),
        )
        assert spy.calls[-1].get('num_predict') == 8192

    def test_raw_story_call_raises_output_budget(self, monkeypatch):
        spy = self._spy(monkeypatch)
        ai_analysis.build_story(
            {'segments': [{'start': 0.0, 'end': 30.0, 'text': 'hello',
                           'start_formatted': '00:00:00', 'speaker': 'A'}]},
            message='build a story', project_name='T', segment_vectors=None,
        )
        assert spy.calls[-1].get('num_predict') == 8192

    def test_other_callers_keep_the_provider_default(self, monkeypatch):
        spy = self._spy(monkeypatch)
        ai_analysis._call_ai('plain analysis prompt')
        # No num_predict key at all — the provider's own default applies.
        assert 'num_predict' not in spy.calls[-1]


# ---------- timeout sized to the output budget ---------------------------------

class TestStoryTimeoutSizing:
    """The raised 8192-token budget decodes for 600-850s on the default 8GB
    profile, but _call_ai sized the HTTP timeout with a no-arg
    recommended_analysis_timeout() call (504s, off the 2000-token
    representative) — ReadTimeout killed healthy story generations
    mid-decode and bypassed both the JSON repair and the fallback. The
    budget must reach the timeout function."""

    def _spy_timeout(self, monkeypatch, seen):
        import model_config

        def _spy(hw_info=None, num_predict=None):
            seen.append(num_predict)
            return 600

        monkeypatch.setattr(model_config, 'recommended_analysis_timeout', _spy)
        spy = _SpyProvider()
        import ai_providers
        monkeypatch.setattr(ai_providers, 'get_active_provider',
                            lambda model_resolver=None: spy)
        return spy

    def test_call_ai_threads_num_predict_into_timeout(self, monkeypatch):
        seen = []
        self._spy_timeout(monkeypatch, seen)
        ai_analysis._call_ai('p', num_predict=8192)
        assert seen == [8192]

    def test_call_ai_without_budget_passes_none(self, monkeypatch):
        # Default-path callers stay byte-identical: None falls back to the
        # representative output size inside recommended_analysis_timeout.
        seen = []
        self._spy_timeout(monkeypatch, seen)
        ai_analysis._call_ai('p')
        assert seen == [None]

    def test_story_build_timeout_covers_the_honest_decode(self, monkeypatch):
        # End-to-end on the default profile (8GB Apple Silicon, medium):
        # the story call's timeout must cover the 600-850s an 8192-token
        # generation honestly needs (the old no-arg call returned 504s).
        import model_config
        monkeypatch.setattr(model_config, 'detect_hardware_tier',
                            lambda: {'arch': 'arm64', 'ram_gb': 8.0})
        monkeypatch.setattr(model_config, 'load_model_config',
                            lambda: {'tier': 'medium'})
        spy = _SpyProvider()
        import ai_providers
        monkeypatch.setattr(ai_providers, 'get_active_provider',
                            lambda model_resolver=None: spy)
        monkeypatch.setattr(ai_analysis, 'inject_my_style',
                            lambda p, profile_id=None: p)
        ai_analysis.build_story(
            {'segments': []}, message='build me a 14 minute cut',
            project_name='T', segment_vectors=_mk_vectors(),
        )
        assert spy.calls[-1].get('num_predict') == 8192
        assert spy.calls[-1].get('timeout') >= 850


class TestProviderTimeoutFallsBackToDeterministicBuild:
    """A mid-generation provider death (typically the read timeout on slow
    hardware) used to escape _build_story_from_vectors before the parse —
    bypassing the JSON repair AND the fallback, so the endpoint 500'd even
    though a fully classified menu existed."""

    def test_provider_error_routes_into_fallback(self, monkeypatch):
        from ai_providers import ProviderError

        def _boom(prompt, system_prompt="", **kwargs):
            raise ProviderError('Ollama timed out mid-request')

        monkeypatch.setattr(ai_analysis, '_call_ai', _boom)
        monkeypatch.setattr(ai_analysis, 'inject_my_style',
                            lambda p, profile_id=None: p)
        result = ai_analysis.build_story(
            {'segments': []}, message=TESTER_RANGE_PROMPT, project_name='T',
            segment_vectors=_mk_vectors(),
        )
        assert result['fallback_build'] is True
        total = _clip_total(result['clips'])
        assert 0.9 * 1050 <= total <= 1.15 * 1050

    def test_provider_error_reraised_when_no_fallback_possible(self, monkeypatch):
        # Degenerate vectors (no usable timecodes) leave nothing to build
        # from — the typed provider message must survive to the endpoint
        # instead of a bare 0-clip result.
        from ai_providers import ProviderError

        def _boom(prompt, system_prompt="", **kwargs):
            raise ProviderError('Ollama timed out mid-request')

        monkeypatch.setattr(ai_analysis, '_call_ai', _boom)
        monkeypatch.setattr(ai_analysis, 'inject_my_style',
                            lambda p, profile_id=None: p)
        degenerate = [{'seg_id': 'S1', 'timecode_in': '00:00:10',
                       'timecode_out': '00:00:10', 'narrative_score': 'high',
                       'transcript_excerpt': ''}]
        with pytest.raises(ProviderError):
            ai_analysis.build_story(
                {'segments': []}, message='build me a 14 minute cut',
                project_name='T', segment_vectors=degenerate,
            )


# ---------- deterministic fallback build --------------------------------------

class TestDeterministicFallbackBuild:
    """The live 0-clips bug, part 3: when the model's response is unusable
    end-to-end (garbage, unsalvageable truncation, or nothing surviving
    hydration), the build must fall back to a deterministic assembly from
    the segment menu instead of surfacing "The AI returned 0 clips"."""

    def _build(self, monkeypatch, response, message=TESTER_RANGE_PROMPT,
               vectors=None):
        monkeypatch.setattr(ai_analysis, '_call_ai',
                            lambda p, s="", **kw: response)
        monkeypatch.setattr(ai_analysis, 'inject_my_style',
                            lambda p, profile_id=None: p)
        return ai_analysis.build_story(
            {'segments': []}, message=message, project_name='T',
            segment_vectors=vectors if vectors is not None else _mk_vectors(),
        )

    def test_garbage_response_builds_fallback_in_band(self, monkeypatch):
        # The tester's exact prompt (parses to 1050s) + a model that answers
        # in prose. The fallback must land inside the enforcement band.
        result = self._build(monkeypatch, 'sorry, I could not pick clips')
        assert result['fallback_build'] is True
        assert result['clips']
        total = _clip_total(result['clips'])
        assert 0.9 * 1050 <= total <= 1.15 * 1050, f'total {total}s off band'
        assert result['target_duration'] == '17:30'
        assert 'deterministically' in result['reasoning']

    def test_truncated_beyond_salvage_builds_fallback(self, monkeypatch):
        # Cut before the first clip object completes — repair yields an
        # empty clips array, so the fallback takes over.
        result = self._build(monkeypatch, '{"story_title": "X", "clips": [{"or')
        assert result['fallback_build'] is True
        assert result['clips']

    def test_hydration_wipeout_builds_fallback(self, monkeypatch):
        # Valid JSON, but every seg_id is unknown and no timecodes came
        # back — hydration drops everything.
        response = json.dumps({'story_title': 'X', 'clips': [
            {'order': 1, 'seg_id': 'NOPE1'}, {'order': 2, 'seg_id': 'NOPE2'},
        ]})
        result = self._build(monkeypatch, response)
        assert result['fallback_build'] is True
        assert result['clips']

    def test_fallback_prefers_strong_tiers_chronologically(self, monkeypatch):
        result = self._build(monkeypatch, 'garbage')
        clips = result['clips']
        # Enough high/medium material exists on the 40-minute menu — no
        # 'low' filler drafted for a 1050s ask.
        assert all(c.get('narrative_score') != 'low' for c in clips)
        starts = [ai_analysis._tc_to_seconds(c['start_time']) for c in clips]
        assert starts == sorted(starts), 'fallback must stay chronological'
        assert clips[0]['beat_type'] == 'hook'
        assert clips[-1]['beat_type'] == 'resolution'

    def test_fallback_without_target_sizes_modestly(self, monkeypatch):
        result = self._build(monkeypatch, 'garbage',
                             message='build me a story about the harbor')
        assert result['fallback_build'] is True
        assert 8 <= len(result['clips']) <= 12
        # No duration parsed -> no budget fields.
        assert 'duration_enforced' not in result

    def test_successful_model_build_carries_no_fallback_flag(self, monkeypatch):
        result = self._build(monkeypatch,
                             _model_response([f'SEG{i:03d}' for i in range(12)]))
        assert 'fallback_build' not in result

    def test_no_vectors_and_no_parse_still_returns_empty(self, monkeypatch):
        # Raw-transcript path with an unusable response: no menu to fall
        # back on, so the endpoint's 0-clips error remains reachable.
        monkeypatch.setattr(ai_analysis, '_call_ai',
                            lambda p, s="", **kw: 'garbage')
        monkeypatch.setattr(ai_analysis, 'inject_my_style',
                            lambda p, profile_id=None: p)
        result = ai_analysis.build_story(
            {'segments': [{'start': 0.0, 'end': 30.0, 'text': 'hello',
                           'start_formatted': '00:00:00', 'speaker': 'A'}]},
            message=TESTER_RANGE_PROMPT, segment_vectors=None,
        )
        assert not result.get('clips')


class TestFallbackTimelineCoverage:
    """98 high-scored 40s segments across ~100 minutes with a 1050s ask:
    the greedy version consumed the strong pool from the front and returned
    only the first 27 minutes — everything after minute 27 (including any
    actual ending) was absent, and the arbitrary early cut point was tagged
    'resolution' (a slot the budget pass protects). Oversupplied pools are
    now stride-sampled across the whole timeline, with the closer reserved
    for the final third."""

    def _interview_vectors(self, n=98, spacing=60, dur=40, scores=('high',)):
        out = []
        for i in range(n):
            start = 60 + i * spacing
            out.append({
                'seg_id': f's{i:03d}',
                'timecode_in': _tc(start),
                'timecode_out': _tc(start + dur),
                'thread_title': f'thread {i}',
                'memory_type': 'episodic',
                'narrative_score': scores[i % len(scores)],
                'beat_type': 'context',
                'theme_tags': ['t'],
                'transcript_excerpt': f'excerpt {i}',
            })
        return out

    def test_picks_span_the_whole_timeline(self):
        vectors = self._interview_vectors()
        clips = ai_analysis._fallback_story_clips(vectors, target_seconds=1050)
        starts = [ai_analysis._tc_to_seconds(c['start_time']) for c in clips]
        assert starts == sorted(starts), 'draft must stay chronological'
        # Coverage reaches past the 60-minute mark, not just the opening.
        assert max(starts) > 3600
        # The closer comes from the final third of the interview.
        span_lo, span_hi = 60, 60 + 97 * 60
        assert clips[-1]['beat_type'] == 'resolution'
        assert starts[-1] >= span_lo + (span_hi - span_lo) * 2 / 3
        # Sized near the ask so the budget pass lands it in band.
        total = _clip_total(clips)
        assert 0.9 * 1050 <= total <= 1.15 * 1050

    def test_stride_prefers_high_over_medium_within_windows(self):
        # Alternating high/medium: every stride window contains a 'high',
        # so the sampled draft should be all-high while still spanning the
        # timeline.
        vectors = self._interview_vectors(scores=('high', 'medium'))
        clips = ai_analysis._fallback_story_clips(vectors, target_seconds=1050)
        assert all(c['narrative_score'] == 'high' for c in clips)
        starts = [ai_analysis._tc_to_seconds(c['start_time']) for c in clips]
        assert starts == sorted(starts)
        assert max(starts) > 3600

    def test_undersupplied_pool_keeps_greedy_take_everything(self):
        # Strong pool smaller than the ask: all of it is used (top-up and
        # shortfall handling belong to the budget pass), same as before.
        vectors = self._interview_vectors(n=10)
        clips = ai_analysis._fallback_story_clips(vectors, target_seconds=1050)
        assert len(clips) == 10

    def test_no_target_draft_also_samples_the_timeline(self):
        vectors = self._interview_vectors()
        clips = ai_analysis._fallback_story_clips(vectors)
        assert 8 <= len(clips) <= 12
        starts = [ai_analysis._tc_to_seconds(c['start_time']) for c in clips]
        assert starts == sorted(starts)
        assert max(starts) > 3600

    def test_reasoning_copy_matches_the_sampling(self, monkeypatch):
        # The user-facing copy claimed 'strongest moments' while the old
        # algorithm took the EARLIEST — the copy must describe the draft.
        monkeypatch.setattr(ai_analysis, '_call_ai',
                            lambda p, s="", **kw: 'garbage')
        monkeypatch.setattr(ai_analysis, 'inject_my_style',
                            lambda p, profile_id=None: p)
        result = ai_analysis.build_story(
            {'segments': []}, message=TESTER_RANGE_PROMPT, project_name='T',
            segment_vectors=self._interview_vectors(),
        )
        assert result['fallback_build'] is True
        assert 'strongest moments' not in result['reasoning']
        assert 'sampled across the full timeline' in result['reasoning']


class TestStoryBuildEndpointFallbackFlag:
    """/story/build must pass fallback_build through to the persisted build
    and the response so the UI can surface it later."""

    def test_fallback_flag_is_persisted_and_returned(self, client,
                                                     transcribed_project,
                                                     monkeypatch):
        monkeypatch.setattr("ai_analysis.generate_segment_vectors",
                            lambda *a, **kw: [])
        monkeypatch.setattr("ai_analysis.build_story", lambda *a, **kw: {
            'story_title': 'Draft', 'target_duration': '17:30',
            'reasoning': 'deterministic', 'fallback_build': True,
            'clips': [{'order': 1, 'seg_id': 'SEG001', 'title': 'x',
                       'start_time': '00:00:00', 'end_time': '00:00:40',
                       'transcript': 'hi', 'editorial_note': ''}],
        })
        resp = client.post(
            f"/project/{transcribed_project}/story/build",
            json={"message": TESTER_RANGE_PROMPT},
        )
        assert resp.status_code == 200, resp.data
        build = resp.get_json()["build"]
        assert build["fallback_build"] is True

        builds_path = (Path(app_module.app.config["PROJECTS_DIR"])
                       / transcribed_project / "story_builds.json")
        persisted = json.loads(builds_path.read_text())[0]
        assert persisted["fallback_build"] is True
