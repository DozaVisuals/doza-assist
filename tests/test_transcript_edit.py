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


# ── commit 2: edit-word / delete-word ─────────────────────────────────────

def _post(client, pid, op, body):
    return client.post(f'/project/{pid}/transcript/{op}', json=body)


def _timing(segments):
    """Every word's (start, end) plus every segment's (start, end)."""
    return [
        (s['start'], s['end'], [(w['start'], w['end']) for w in s.get('words', [])])
        for s in segments
    ]


def test_edit_word_changes_only_that_word(client, project):
    before = _load(project)
    resp = _post(client, project, 'edit-word', {'seg': 1, 'w': 1, 'expected': 'half', 'text': 'halves'})
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    after = _load(project)
    seg = after['transcript']['segments'][1]
    assert seg['words'][1]['word'] == ' halves'        # leading-space convention kept
    assert seg['text'] == 'Second halves was better'
    assert _timing(before['transcript']['segments']) == _timing(after['transcript']['segments'])
    # Every other word is byte-identical.
    for i, s in enumerate(before['transcript']['segments']):
        for j, w in enumerate(s.get('words', [])):
            if (i, j) != (1, 1):
                assert after['transcript']['segments'][i]['words'][j] == w
    assert after['derived_stale'] is True
    assert after['transcript_edited_at']
    assert data['paragraph_start'] == 0.0 and data['seg'] == 1 and data['w'] == 1
    assert data['has_analysis'] is False


def test_edit_word_expected_mismatch_returns_409_with_current(client, project):
    resp = _post(client, project, 'edit-word', {'seg': 0, 'w': 0, 'expected': 'Tail', 'text': 'Tale'})
    assert resp.status_code == 409
    assert resp.get_json()['current'] == ' Tale'
    assert _load(project)['transcript']['segments'][0]['words'][0]['word'] == ' Tale'
    assert 'derived_stale' not in _load(project)


def test_edit_word_expected_ignores_leading_space(client, project):
    resp = _post(client, project, 'edit-word', {'seg': 0, 'w': 0, 'expected': ' Tale ', 'text': 'Tail'})
    assert resp.status_code == 200
    assert _load(project)['transcript']['segments'][0]['words'][0]['word'] == ' Tail'


def test_edit_segment_level_text(client, project):
    resp = _post(client, project, 'edit-word', {'seg': 3, 'expected': 'segment level only',
                                                'text': 'segment level text'})
    assert resp.status_code == 200
    seg = _load(project)['transcript']['segments'][3]
    assert seg['text'] == 'segment level text'
    assert seg['start'] == 9.0 and seg['end'] == 11.0
    # A worded segment refuses the segment-level form.
    resp = _post(client, project, 'edit-word', {'seg': 0, 'expected': 'Tale of two halves', 'text': 'x'})
    assert resp.status_code == 400


def test_edit_word_rejects_empty_and_bad_indices(client, project):
    assert _post(client, project, 'edit-word', {'seg': 0, 'w': 0, 'expected': 'Tale', 'text': '  '}).status_code == 400
    assert _post(client, project, 'edit-word', {'seg': 9, 'w': 0, 'expected': 'x', 'text': 'y'}).status_code == 404
    assert _post(client, project, 'edit-word', {'seg': 0, 'w': 9, 'expected': 'x', 'text': 'y'}).status_code == 404
    assert _post(client, 'nope', 'edit-word', {'seg': 0, 'w': 0, 'expected': 'x', 'text': 'y'}).status_code == 404


def test_edit_word_flips_hash_and_analysis_freshness(client, project):
    meta = _load(project)
    h0 = app_module._transcript_hash(meta['transcript'])
    meta['analysis'] = {'summary': 's', 'story_beats': []}
    meta['analysis_cache'] = {h0: {'all': {'analysis': meta['analysis'], 'cached_at': 'x'}}}
    (Path(app_module.app.config['PROJECTS_DIR']) / project / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    assert app_module._analysis_is_fresh(_load(project)) is True

    resp = _post(client, project, 'edit-word', {'seg': 0, 'w': 2, 'expected': 'two', 'text': 'three'})
    assert resp.status_code == 200
    assert resp.get_json()['has_analysis'] is True
    after = _load(project)
    assert app_module._transcript_hash(after['transcript']) != h0
    assert app_module._analysis_is_fresh(after) is False


def test_edit_rebuilds_paragraph_index(client, project):
    idx = Path(app_module.app.config['PROJECTS_DIR']) / project / 'paragraph_index.json'
    assert not idx.exists()
    _post(client, project, 'edit-word', {'seg': 0, 'w': 0, 'expected': 'Tale', 'text': 'Story'})
    assert idx.exists()
    data = json.loads(idx.read_text(encoding='utf-8'))
    assert any('Story' in p.get('text', '') for p in data.get('paragraphs', []))


def test_delete_word_is_the_inverse_of_a_word(client, project):
    before = _load(project)
    resp = _post(client, project, 'delete-word', {'seg': 1, 'w': 3, 'expected': 'better'})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['removed']['word'] == ' better'
    after = _load(project)
    seg = after['transcript']['segments'][1]
    assert [w['word'] for w in seg['words']] == [' Second', ' half', ' was']
    assert seg['text'] == 'Second half was'
    # Segment bounds and every remaining word's timing are untouched.
    assert (seg['start'], seg['end']) == (before['transcript']['segments'][1]['start'],
                                          before['transcript']['segments'][1]['end'])
    assert seg['words'] == before['transcript']['segments'][1]['words'][:3]


def test_delete_word_refuses_to_empty_a_segment(client, project):
    seg = _load(project)['transcript']['segments'][0]
    for w in list(seg['words'][1:]):
        assert _post(client, project, 'delete-word', {'seg': 0, 'w': 1, 'expected': w['word']}).status_code == 200
    resp = _post(client, project, 'delete-word', {'seg': 0, 'w': 0, 'expected': 'Tale'})
    assert resp.status_code == 400
    assert _post(client, project, 'delete-word', {'seg': 0, 'w': 0, 'expected': 'Nope'}).status_code == 400


def test_edit_does_not_clobber_concurrent_label_save(client, project):
    """The edit runs under update_project's lock and writes only what it
    touched; a labels save made between page load and the edit survives."""
    client.post(f'/project/{project}/labels', json={
        'color_labels': {}, 'labeled_sections': [{'start': 2.0, 'end': 3.6, 'color': 'blue', 'text': 'Second half was'}],
    })
    assert _post(client, project, 'edit-word', {'seg': 2, 'w': 0, 'expected': 'What', 'text': 'So what'}).status_code == 200
    after = _load(project)
    assert after['labeled_sections'][0]['color'] == 'blue'
    assert after['transcript']['segments'][2]['words'][0]['word'] == ' So what'
