"""Inline transcript correction (transcript_edit.py).

Ground rules pinned here:
- media timing never changes: no op alters an existing word's start/end
- every mutation flips _transcript_hash and marks derived_stale
- the paragraph partial carries data-seg / data-w so the page can address a
  word without inventing ids
"""

import copy
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module  # noqa: E402
from transcribe import format_timestamp  # noqa: E402


@pytest.fixture
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    app_module.app.config['UPLOAD_FOLDER'] = str(tmp_path / 'uploads')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    Path(app_module.app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def _words(start, tokens, step=0.4):
    out = []
    t = start
    for tok in tokens:
        out.append({'start': round(t, 3), 'end': round(t + step, 3), 'word': ' ' + tok})
        t += step
    return out


def _seg(start, tokens, speaker, step=0.4):
    words = _words(start, tokens, step)
    return {
        'start': words[0]['start'],
        'end': words[-1]['end'],
        'text': ' '.join(tokens),
        'speaker': speaker,
        'start_formatted': format_timestamp(words[0]['start']),
        'end_formatted': format_timestamp(words[-1]['end']),
        'words': words,
    }


def _meta(pid):
    segs = [
        _seg(0.0, ['Tale', 'of', 'two', 'halves'], 'SPEAKER_00'),
        _seg(2.0, ['Second', 'half', 'was', 'better'], 'SPEAKER_00'),
        _seg(6.0, ['What', 'did', 'you', 'learn'], 'SPEAKER_01'),
        {'start': 9.0, 'end': 11.0, 'text': 'segment level only', 'speaker': 'SPEAKER_01',
         'start_formatted': '00:00:09.000', 'end_formatted': '00:00:11.000'},
    ]
    return {
        'id': pid,
        'name': 'Edit Test',
        'status': 'transcribed',
        'transcript': {'segments': segs, 'language': 'en', 'duration': 11.0, 'engine': 'parakeet-mlx'},
        'speaker_names': {'SPEAKER_00': 'Reporter'},
        'labeled_sections': [],
        'color_labels': {},
    }


@pytest.fixture
def project(client):
    pid = 'edit-test'
    pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    pdir.mkdir(parents=True)
    (pdir / 'meta.json').write_text(json.dumps(_meta(pid)), encoding='utf-8')
    return pid


def _load(pid):
    return json.loads((Path(app_module.app.config['PROJECTS_DIR']) / pid / 'meta.json').read_text(encoding='utf-8'))


# ── commit 1: addressing + paragraph refresh ─────────────────────────────

def test_paragraph_partial_carries_seg_and_word_indices(client, project):
    resp = client.get(f'/project/{project}')
    html = resp.get_data(as_text=True)
    assert 'data-seg="0" data-w="0"' in html
    assert 'data-seg="1" data-w="3"' in html
    assert 'data-seg="2" data-w="0"' in html
    # Segment-level span (no words[]) is addressable at the segment with w=-1.
    assert 'data-seg="3" data-w="-1"' in html


def test_group_into_paragraphs_first_index_is_contiguous():
    segs = _load_segments_for_grouping()
    paras = app_module.group_into_paragraphs(segs)
    assert paras[0]['first_index'] == 0
    seen = 0
    for para in paras:
        assert para['first_index'] == seen
        seen += len(para['segments'])
    assert seen == len(segs)


def _load_segments_for_grouping():
    return _meta('x')['transcript']['segments']


def test_paragraph_html_returns_the_paragraph_containing_time(client, project):
    resp = client.get(f'/project/{project}/transcript/paragraph-html?start=6.5')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['start'] == 6.0
    assert data['first_index'] == 2
    assert 'class="para-block"' in data['html']
    assert 'data-seg="2" data-w="0"' in data['html']
    assert 'data-seg="0"' not in data['html']


def test_paragraph_html_requires_start(client, project):
    assert client.get(f'/project/{project}/transcript/paragraph-html').status_code == 400
    assert client.get('/project/nope/transcript/paragraph-html?start=0').status_code == 404


def test_paragraph_html_multi_badge(client, project):
    resp = client.get(f'/project/{project}/transcript/paragraph-html?start=0&multi=1&color=blue')
    html = resp.get_json()['html']
    assert 'para-project-badge' in html and 'var(--blue)' in html
