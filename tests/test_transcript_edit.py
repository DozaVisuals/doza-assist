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


# ── commit 3: split-segment / merge-segment ──────────────────────────────

def _all_word_timings(segments):
    return [(w['start'], w['end']) for s in segments for w in s.get('words', [])]


def test_split_segment_recomputes_bounds_and_keeps_word_times(client, project):
    before = _load(project)['transcript']['segments']
    resp = _post(client, project, 'split-segment', {'seg': 1, 'at_w': 2})
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    segs = _load(project)['transcript']['segments']
    assert len(segs) == len(before) + 1
    left, right = segs[1], segs[2]
    assert [w['word'] for w in left['words']] == [' Second', ' half']
    assert [w['word'] for w in right['words']] == [' was', ' better']
    assert left['text'] == 'Second half' and right['text'] == 'was better'
    assert (left['start'], left['end']) == (2.0, 2.8)
    assert (right['start'], right['end']) == (2.8, 3.6)
    assert left['end_formatted'] == format_timestamp(2.8)
    assert right['start_formatted'] == format_timestamp(2.8)
    assert right['speaker'] == 'SPEAKER_00' and 'speaker_manual' not in right
    # Word timings across the whole transcript are unchanged; later segments untouched.
    assert _all_word_timings(segs) == _all_word_timings(before)
    assert segs[3:] == before[2:]
    assert data['new_seg'] == 2 and data['speaker_changed'] is False
    assert data['paragraph_start'] == 0.0 and data['paragraph_end'] == 3.6


def test_split_rejects_first_word_and_bad_indices(client, project):
    assert _post(client, project, 'split-segment', {'seg': 1, 'at_w': 0}).status_code == 400
    assert _post(client, project, 'split-segment', {'seg': 1, 'at_w': 4}).status_code == 400
    assert _post(client, project, 'split-segment', {'seg': 3, 'at_w': 1}).status_code == 400   # no words
    assert _post(client, project, 'split-segment', {'seg': 8, 'at_w': 1}).status_code == 404
    assert len(_load(project)['transcript']['segments']) == 4


def test_split_with_new_speaker_label_marks_manual(client, project):
    resp = _post(client, project, 'split-segment', {'seg': 1, 'at_w': 2, 'new_speaker': 'SPEAKER_01'})
    assert resp.status_code == 200
    data = resp.get_json()
    segs = _load(project)['transcript']['segments']
    assert segs[2]['speaker'] == 'SPEAKER_01' and segs[2]['speaker_manual'] is True
    assert segs[1]['speaker'] == 'SPEAKER_00' and 'speaker_manual' not in segs[1]
    assert data['speaker_changed'] is True and data['speaker'] == 'SPEAKER_01'
    # The back half now opens its own paragraph, so the refresh range spans both.
    assert data['paragraph_start'] == 0.0 and data['paragraph_end'] == 3.6
    assert {s['raw'] for s in data['speakers']} == {'SPEAKER_00', 'SPEAKER_01'}


def test_split_with___new___mints_unused_label(client, project):
    meta = _load(project)
    meta['diarization'] = {'speakers': ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02']}
    (Path(app_module.app.config['PROJECTS_DIR']) / project / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    resp = _post(client, project, 'split-segment', {'seg': 0, 'at_w': 2, 'new_speaker': '__new__'})
    assert resp.status_code == 200
    after = _load(project)
    assert after['transcript']['segments'][1]['speaker'] == 'SPEAKER_03'
    assert after['transcript']['segments'][1]['speaker_manual'] is True
    assert after['diarization']['speakers'][-1] == 'SPEAKER_03'
    assert 'SPEAKER_03' not in after.get('speaker_names', {})
    assert resp.get_json()['speaker'] == 'SPEAKER_03'


def test_split_then_merge_round_trips(client, project):
    before = _load(project)['transcript']['segments']
    assert _post(client, project, 'split-segment', {'seg': 1, 'at_w': 1}).status_code == 200
    resp = _post(client, project, 'merge-segment', {'seg': 1})
    assert resp.status_code == 200, resp.get_json()
    after = _load(project)['transcript']['segments']
    assert after == before
    assert resp.get_json()['paragraph_start'] == 0.0


def test_merge_refuses_different_speakers(client, project):
    resp = _post(client, project, 'merge-segment', {'seg': 1})   # SPEAKER_00 + SPEAKER_01
    assert resp.status_code == 409
    assert resp.get_json()['current'] == ['SPEAKER_00', 'SPEAKER_01']
    assert _post(client, project, 'merge-segment', {'seg': 3}).status_code == 400   # nothing after
    assert len(_load(project)['transcript']['segments']) == 4


def test_merge_keeps_speaker_manual_if_either_half_had_it(client, project):
    assert _post(client, project, 'split-segment', {'seg': 1, 'at_w': 2, 'new_speaker': 'SPEAKER_00'}).status_code == 200
    segs = _load(project)['transcript']['segments']
    assert segs[2]['speaker_manual'] is True and 'speaker_manual' not in segs[1]
    assert _post(client, project, 'merge-segment', {'seg': 1}).status_code == 200
    merged = _load(project)['transcript']['segments'][1]
    assert merged['speaker_manual'] is True
    assert merged['text'] == 'Second half was better' and (merged['start'], merged['end']) == (2.0, 3.6)


def test_merge_segment_level_pair(client, project):
    meta = _load(project)
    meta['transcript']['segments'].append({'start': 11.0, 'end': 12.0, 'text': 'and more', 'speaker': 'SPEAKER_01',
                                           'start_formatted': '00:00:11.000', 'end_formatted': '00:00:12.000'})
    (Path(app_module.app.config['PROJECTS_DIR']) / project / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    assert _post(client, project, 'merge-segment', {'seg': 3}).status_code == 200
    seg = _load(project)['transcript']['segments'][3]
    assert seg['text'] == 'segment level only and more' and seg['end'] == 12.0
    # Mixed worded / segment-level pair is refused.
    assert _post(client, project, 'merge-segment', {'seg': 2}).status_code == 400


def test_speakers_endpoint_resolves_display_names(client, project):
    resp = client.get(f'/project/{project}/transcript/speakers')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['speakers'] == [{'raw': 'SPEAKER_00', 'display': 'Reporter'},
                                {'raw': 'SPEAKER_01', 'display': 'SPEAKER_01'}]
    assert data['new_speaker'] == '__new__'
