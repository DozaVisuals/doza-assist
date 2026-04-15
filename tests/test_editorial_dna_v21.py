"""
Smoke tests for Editorial DNA v2.1.

Covers:
  - Profile CRUD (create, rename, activate, delete)
  - Legacy v1 → v2.1 migration
  - Snapshot creation + delta computation
  - Injector picks up the active profile and session override
  - Analysis pass tolerates missing LLM (falls back cleanly)

These tests monkeypatch editorial_dna.profiles.EDNA_ROOT to a tmp dir so they
don't touch the user's real profile data. The LLM client (_call_ai) is
patched to return canned JSON so the tests don't require Ollama or Claude.

Run:
    python -m pytest tests/test_editorial_dna_v21.py -v
"""

import os
import json
import sys
import pytest

# Make the project importable when tests run from the repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Redirect every editorial_dna path to a fresh tmp dir for each test."""
    from editorial_dna import profiles as edp
    edna_root = tmp_path / 'edna'
    profiles_dir = edna_root / 'profiles'
    os.makedirs(profiles_dir, exist_ok=True)
    monkeypatch.setattr(edp, 'EDNA_ROOT', str(edna_root))
    monkeypatch.setattr(edp, 'LEGACY_PROFILE_PATH', str(edna_root / 'profile.json'))
    monkeypatch.setattr(edp, 'PROFILES_DIR', str(profiles_dir))
    monkeypatch.setattr(edp, 'INDEX_PATH', str(edna_root / 'profiles_index.json'))
    yield


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace _call_ai so analysis.py returns deterministic JSON."""
    canned = {
        'opening_style': 'emotional hook',
        'opening_style_confidence': 0.72,
        'emotional_arc_shape': 'peak-and-resolve',
        'resolution_style': 'reflective',
        'common_themes': ['resilience', 'family', 'loss'],
        'subject_focus': 'personal stories',
        'vulnerability_pattern': 'gradual',
        'favors_chronological': False,
        'uses_intercutting': True,
        'tone': 'warm and observational',
        'formality_level': 'conversational',
        'uses_narration': False,
    }

    def fake(prompt):
        return json.dumps(canned)

    import ai_analysis
    monkeypatch.setattr(ai_analysis, '_call_ai', fake)
    # analysis.py imports _call_ai at module load, so patch its local reference too
    import editorial_dna.analysis as ea
    monkeypatch.setattr(ea, '_call_ai', fake)
    import editorial_dna.summarizer as es
    monkeypatch.setattr(es, '_call_ai', lambda p: 'A calm, observational editor who lets subjects breathe.')
    return canned


# ---------------------------------------------------------------------------
# Profile CRUD
# ---------------------------------------------------------------------------

def test_create_and_list_profiles():
    from editorial_dna import profiles as edp
    pid = edp.create_profile('Doc Style', description='For long-form docs')
    assert pid
    listed = edp.list_profiles()
    assert len(listed) == 1
    assert listed[0]['name'] == 'Doc Style'
    assert listed[0]['is_active'] is True  # first profile auto-activates


def test_create_multiple_and_switch_active():
    from editorial_dna import profiles as edp
    pid1 = edp.create_profile('Doc Style')
    pid2 = edp.create_profile('Social Cuts')

    assert edp.get_active_profile_id() == pid1  # first one stays active
    edp.set_active(pid2)
    assert edp.get_active_profile_id() == pid2

    listed = edp.list_profiles()
    active_entry = next(p for p in listed if p['is_active'])
    assert active_entry['id'] == pid2


def test_rename_profile():
    from editorial_dna import profiles as edp
    pid = edp.create_profile('Old Name')
    edp.rename_profile(pid, 'New Name')
    profile = edp.get_profile(pid)
    assert profile['name'] == 'New Name'
    assert profile['summary']['profile_name'] == 'New Name'
    listed = edp.list_profiles()
    assert listed[0]['name'] == 'New Name'


def test_delete_profile_clears_active_when_empty():
    from editorial_dna import profiles as edp
    pid = edp.create_profile('Solo')
    edp.delete_profile(pid)
    assert edp.get_active_profile_id() is None
    assert edp.list_profiles() == []


def test_delete_profile_promotes_next_when_others_exist():
    from editorial_dna import profiles as edp
    pid1 = edp.create_profile('First')
    pid2 = edp.create_profile('Second')
    edp.set_active(pid1)
    edp.delete_profile(pid1)
    assert edp.get_active_profile_id() == pid2


# ---------------------------------------------------------------------------
# Migration from v1
# ---------------------------------------------------------------------------

def test_migration_from_v1_profile(fake_llm):
    from editorial_dna import profiles as edp
    legacy = {
        'profile_version': '1.0',
        'feature_name': 'My Style',
        'created_at': '2025-10-01T12:00:00',
        'active': True,
        'source_files': [
            {'filename': 'a.mp4', 'imported_at': '2025-10-01T12:00:00', 'duration_seconds': 180},
            {'filename': 'b.mp4', 'imported_at': '2025-10-02T12:00:00', 'duration_seconds': 240},
        ],
        'speech_pacing': {'words_per_minute': 140, 'avg_beat_length': 14.2, 'rhythm_descriptor': 'measured and deliberate'},
        'structural_rhythm': {'energy_arc': 'builds to the end', 'longest_beat_position': 'last_third'},
        'soundbite_craft': {'avg_soundbite_length': 12.5, 'clean_cut_ratio': 0.78},
        'story_shape': {'opening_style': 'cold_quote', 'closing_style': 'button', 'uses_callbacks': True},
        'content_patterns': {'topic_count': 3},
        'natural_language_summary': 'A calm editor who lets subjects breathe.',
    }
    with open(edp.LEGACY_PROFILE_PATH, 'w') as f:
        json.dump(legacy, f)

    # First public call triggers migration
    listed = edp.list_profiles()
    assert len(listed) == 1
    pid = listed[0]['id']
    assert listed[0]['name'] == 'My Style'

    profile = edp.get_profile(pid)
    assert profile['natural_language_summary'] == 'A calm editor who lets subjects breathe.'
    assert profile['speech_pacing']['words_per_minute'] == 140
    # Structured summary was populated from v1 metrics where possible
    assert profile['summary']['projects_analyzed'] == 2
    assert profile['summary']['total_runtime_seconds'] == 420
    assert profile['summary']['narrative_patterns']['pacing_signature'] == 'measured and deliberate'
    assert profile['summary']['narrative_patterns']['emotional_arc_shape'] == 'builds to the end'
    assert profile['summary']['structural_preferences']['tends_to_open_with'] == 'cold_quote'

    # An initial snapshot was taken
    from editorial_dna import snapshots
    snaps = snapshots.list_snapshots(pid)
    assert len(snaps) == 1
    assert snaps[0]['delta_from_previous']['is_first_snapshot'] is True

    # Legacy file was renamed, not deleted
    assert not os.path.isfile(edp.LEGACY_PROFILE_PATH)
    assert os.path.isfile(edp.LEGACY_PROFILE_PATH + '.migrated')


# ---------------------------------------------------------------------------
# Snapshots + delta
# ---------------------------------------------------------------------------

def test_snapshot_delta_detects_changes(fake_llm):
    from editorial_dna import profiles as edp
    from editorial_dna import snapshots
    from editorial_dna.models import empty_style_profile_summary

    pid = edp.create_profile('Test')

    # Seed summary #1
    s1 = empty_style_profile_summary(pid, 'Test')
    s1['projects_analyzed'] = 2
    s1['narrative_patterns']['opening_style'] = 'narration'
    s1['narrative_patterns']['average_clip_length_seconds'] = 14.0
    s1['thematic_patterns']['common_themes'] = ['family', 'work']
    s1['voice_characteristics']['tone'] = 'calm'
    edp.save_summary(pid, s1)
    snap1 = snapshots.create_snapshot(pid, note='Import 1')
    assert snap1['delta_from_previous']['is_first_snapshot'] is True

    # Seed summary #2 with changes
    s2 = empty_style_profile_summary(pid, 'Test')
    s2['projects_analyzed'] = 4
    s2['narrative_patterns']['opening_style'] = 'emotional hook'   # changed
    s2['narrative_patterns']['average_clip_length_seconds'] = 11.0  # ~21% shorter
    s2['thematic_patterns']['common_themes'] = ['family', 'legacy']  # removed "work", added "legacy"
    s2['voice_characteristics']['tone'] = 'urgent'  # changed
    edp.save_summary(pid, s2)
    snap2 = snapshots.create_snapshot(pid, note='Import 2')

    d = snap2['delta_from_previous']
    assert d['is_first_snapshot'] is False
    assert d['opening_style_changed'] is True
    assert 'legacy' in d['new_themes_detected']
    assert 'work' in d['themes_no_longer_present']
    assert d['tone_shift'] is not None
    # A shorter clip length should be reported as "shorter"
    assert any('shorter' in c for c in d['changes'])
    assert any('legacy' in c for c in d['changes'])
    assert any('2 new' in c or 'added 2' in c for c in d['changes'])


# ---------------------------------------------------------------------------
# Injector — active profile + session override
# ---------------------------------------------------------------------------

def test_injector_uses_active_profile(fake_llm):
    from editorial_dna import profiles as edp
    from editorial_dna.injector import inject_my_style

    pid = edp.create_profile('Doc Style')
    edp.save_system_prompt(pid, 'DOC-STYLE-PROMPT-MARKER')

    result = inject_my_style('BASE SYSTEM PROMPT')
    assert 'DOC-STYLE-PROMPT-MARKER' in result
    assert 'BASE SYSTEM PROMPT' in result
    assert 'Doc Style' in result


def test_injector_respects_session_override(fake_llm):
    from editorial_dna import profiles as edp
    from editorial_dna.injector import inject_my_style

    pid1 = edp.create_profile('Doc Style')
    edp.save_system_prompt(pid1, 'DOC-MARKER')
    pid2 = edp.create_profile('Social Cuts')
    edp.save_system_prompt(pid2, 'SOCIAL-MARKER')
    edp.set_active(pid1)

    # No override → active profile
    default_result = inject_my_style('BASE')
    assert 'DOC-MARKER' in default_result
    assert 'SOCIAL-MARKER' not in default_result

    # Session override → overridden profile
    overridden = inject_my_style('BASE', profile_id=pid2)
    assert 'SOCIAL-MARKER' in overridden
    assert 'DOC-MARKER' not in overridden


def test_injector_skipped_when_profile_toggled_off(fake_llm):
    from editorial_dna import profiles as edp
    from editorial_dna.injector import inject_my_style

    pid = edp.create_profile('Doc Style')
    edp.save_system_prompt(pid, 'DOC-MARKER')
    edp.set_profile_active_toggle(pid, False)
    result = inject_my_style('BASE SYSTEM PROMPT')
    assert 'DOC-MARKER' not in result
    assert result == 'BASE SYSTEM PROMPT'


# ---------------------------------------------------------------------------
# Analysis pass tolerance
# ---------------------------------------------------------------------------

def test_analysis_runs_with_transcripts(fake_llm):
    from editorial_dna import profiles as edp
    from editorial_dna import analysis

    pid = edp.create_profile('Test')
    metrics = {
        'speech_pacing': {'rhythm_descriptor': 'conversational', 'avg_beat_length': 10.0},
        'structural_rhythm': {'energy_arc': 'steady'},
        'soundbite_craft': {'avg_soundbite_length': 15.0, 'clean_cut_ratio': 0.8},
        'story_shape': {'opening_style': 'cold_quote', 'closing_style': 'button'},
        'content_patterns': {'topic_count': 2},
    }
    source_files = [{'filename': 'a.mp4', 'duration_seconds': 300}]
    transcripts = "Once upon a time, there was a subject who told a story about resilience and family."

    summary = analysis.generate_structured_summary(
        pid, 'Test', metrics, source_files, transcripts_text=transcripts,
    )
    # Fields populated from metrics pass through verbatim
    assert summary['narrative_patterns']['pacing_signature'] == 'conversational'
    # Fields from the fake LLM
    assert 'resilience' in summary['thematic_patterns']['common_themes']
    assert summary['voice_characteristics']['tone'] == 'warm and observational'
    assert summary['projects_analyzed'] == 1


def test_analysis_without_transcripts_uses_placeholders(fake_llm):
    from editorial_dna import analysis
    from editorial_dna.models import PLACEHOLDER_NARRATIVE

    summary = analysis.generate_structured_summary(
        'x', 'Test', {}, [{'filename': 'a.mp4', 'duration_seconds': 100}],
        transcripts_text='',
    )
    # No LLM was called (no transcripts), so narrative/thematic/voice stay at placeholder
    assert summary['thematic_patterns']['subject_focus'] == PLACEHOLDER_NARRATIVE
    assert summary['voice_characteristics']['tone'] == PLACEHOLDER_NARRATIVE


def test_analysis_survives_bad_llm_json(monkeypatch):
    """If the LLM returns garbage, analysis should still return a valid summary."""
    import ai_analysis
    import editorial_dna.analysis as ea
    import editorial_dna.summarizer as es
    monkeypatch.setattr(ai_analysis, '_call_ai', lambda p: 'not json at all!!')
    monkeypatch.setattr(ea, '_call_ai', lambda p: 'not json at all!!')
    monkeypatch.setattr(es, '_call_ai', lambda p: 'A calm editor.')

    summary = ea.generate_structured_summary(
        'x', 'Test', {'speech_pacing': {'rhythm_descriptor': 'calm'}},
        [{'filename': 'a.mp4', 'duration_seconds': 100}],
        transcripts_text='some transcript text',
    )
    # Summary still has all keys, even though the LLM call failed to parse
    assert 'narrative_patterns' in summary
    assert summary['narrative_patterns']['pacing_signature'] == 'calm'


# ---------------------------------------------------------------------------
# Export / import round-trip
# ---------------------------------------------------------------------------

def test_export_import_roundtrip(fake_llm):
    from editorial_dna import profiles as edp
    pid1 = edp.create_profile('Doc Style')
    pid2 = edp.create_profile('Social Cuts')
    edp.rename_profile(pid1, 'My Doc Style')

    bundle = edp.export_all()
    assert bundle['schema_version']
    assert len(bundle['profiles']) == 2

    # Wipe and re-import
    edp.delete_profile(pid1)
    edp.delete_profile(pid2)
    assert edp.list_profiles() == []

    imported = edp.import_bundle(bundle)
    assert len(imported) == 2
    names = [p['name'] for p in edp.list_profiles()]
    assert 'My Doc Style' in names
    assert 'Social Cuts' in names
