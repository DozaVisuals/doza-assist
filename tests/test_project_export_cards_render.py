"""Export-tab card rendering + Documents-card category wiring.

Field bug (2026-08-23, Direct 1.0.44): on an FCPXML round-trip project the
Export tab renders ONLY the round-trip category card (rtCat* checkbox ids),
but the Documents card's Selects export read categories exclusively from the
unified card's exportCat* ids — the collector always returned [] and the
"Check at least one category above" toast fired with clips visibly checked.

Pins:
  - The two category cards are mutually exclusive per project shape
    (plain/transcribed → unified exportCat*; round-trip → rtCat*).
  - The Documents card ships the variant-aware collector
    (_collectDocumentCategories) that reads whichever card is present and
    normalizes to the documents route's vocabulary (labels/social/story/
    soundbites — NOT the round-trip writer's 'client_selects').
  - The round-trip export button is platform-labeled (rt-export key) so
    "Export to Final Cut Pro" can no longer sit above a DELIVER TO row
    promising DaVinci Resolve (the companion 1.0.44 field bug).
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module


@pytest.fixture
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def _seed(pid, tmp_path, fcpxml_source=None):
    pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    pdir.mkdir(parents=True)
    meta = {
        'id': pid, 'name': pid, 'status': 'transcribed',
        # Nonexistent on purpose: skips the start-TC / framerate ffprobe paths.
        'source_path': f'/nonexistent/{pid}-source.mov',
        'transcript': {'language': 'en', 'segments': [
            {'start': 0.0, 'end': 2.0, 'text': 'hello', 'speaker': 'SPEAKER_00',
             'start_formatted': '00:00:00.000', 'end_formatted': '00:00:02.000',
             'words': []}]},
    }
    if fcpxml_source is not None:
        meta['fcpxml_source'] = fcpxml_source
    json.dump(meta, open(pdir / 'meta.json', 'w'))


def _rt_source(tmp_path):
    stored = tmp_path / 'stored.fcpxml'
    stored.write_text('<fcpxml version="1.14"/>')
    return {
        'container_type': 'sync-clip',
        'timeline_audio_rendered': True,
        'is_multi_source': True,
        'sequence_framerate': 25.0,
        'stored_fcpxml_path': str(stored),
    }


def _page(client, pid):
    resp = client.get(f'/project/{pid}')
    assert resp.status_code == 200
    return resp.get_data(as_text=True)


def test_plain_project_renders_unified_card_only(client, tmp_path):
    _seed('plain1', tmp_path)
    html = _page(client, 'plain1')
    assert 'id="exportCategories"' in html
    for cid in ('exportCatLabels', 'exportCatSocial',
                'exportCatStory', 'exportCatTranscript'):
        assert f'id="{cid}"' in html
    assert 'id="rtExportCategories"' not in html
    assert 'id="rtCatLabels"' not in html


def test_roundtrip_project_renders_rt_card_only(client, tmp_path):
    # Proves the bug's precondition: NONE of the exportCat* ids the old
    # collector read exist on a round-trip page.
    _seed('rt1', tmp_path, fcpxml_source=_rt_source(tmp_path))
    html = _page(client, 'rt1')
    assert 'id="rtExportCategories"' in html
    for cid in ('rtCatLabels', 'rtCatSocial',
                'rtCatStory', 'rtCatSoundbites'):
        assert f'id="{cid}"' in html
    assert 'id="exportCategories"' not in html
    assert 'id="exportCatLabels"' not in html


@pytest.mark.parametrize('shape', ['plain', 'roundtrip'])
def test_documents_card_ships_variant_aware_collector(client, tmp_path, shape):
    pid = f'doc-{shape}'
    _seed(pid, tmp_path,
          fcpxml_source=_rt_source(tmp_path) if shape == 'roundtrip' else None)
    html = _page(client, pid)
    # The Documents card renders on both page shapes…
    assert 'id="docWhat"' in html
    assert 'function downloadDocument' in html
    # …and its Selects collector is the variant-aware one, mapping the
    # round-trip card's checkboxes to the documents route's vocabulary.
    assert 'function _collectDocumentCategories' in html
    assert "rtCatLabels: 'labels'" in html
    assert "rtCatSoundbites: 'soundbites'" in html
    assert '_collectDocumentCategories()' in html
    # The round-trip WRITER key must never leak into the documents payload —
    # the route silently drops it (spurious "No selects" 400).
    assert "rtCatLabels: 'client_selects'" in html  # rt writer keeps its own map
    assert 'documents route silently drops' in html  # the guard comment ships


def test_rt_export_button_is_platform_labeled(client, tmp_path):
    _seed('rt2', tmp_path, fcpxml_source=_rt_source(tmp_path))
    html = _page(client, 'rt2')
    assert 'data-export-label-key="rt-export"' in html
    assert "'rt-export'" in html
    assert 'Export FCPXML for DaVinci Resolve' in html
