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


# ── commit 4: reassign-speaker ────────────────────────────────────────────

def _diarize(project):
    meta = _load(project)
    meta['diarization'] = {'model': 'x', 'status': 'done', 'completed_at': '2026-09-09T00:00:00',
                           'speakers': ['SPEAKER_00', 'SPEAKER_01']}
    (Path(app_module.app.config['PROJECTS_DIR']) / project / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    (Path(app_module.app.config['PROJECTS_DIR']) / project / 'diarization_status.json').write_text(
        json.dumps({'status': 'done'}), encoding='utf-8')


def test_reassign_on_diarized_project_bypasses_gate_and_marks_manual(client, project):
    _diarize(project)
    # The existing per-segment route is gated on diarized projects...
    gated = client.post(f'/project/{project}/update-speaker-range',
                        json={'start': 6.0, 'end': 8.0, 'speaker': 'SPEAKER_00'})
    assert gated.status_code == 409
    # ...the new one is not.
    before = _load(project)['transcript']['segments']
    resp = _post(client, project, 'reassign-speaker', {'seg': 2, 'speaker': 'SPEAKER_00'})
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    after = _load(project)['transcript']['segments']
    assert after[2]['speaker'] == 'SPEAKER_00' and after[2]['speaker_manual'] is True
    assert data['previous_speaker'] == 'SPEAKER_01' and data['previous_manual'] is False
    assert data['speaker_changed'] is True
    # Nothing else moved: same text, same bounds, same words, other segments identical.
    assert {k: v for k, v in after[2].items() if k not in ('speaker', 'speaker_manual')} == \
        {k: v for k, v in before[2].items() if k != 'speaker'}
    assert [s for i, s in enumerate(after) if i != 2] == [s for i, s in enumerate(before) if i != 2]
    # Segment 2 sits after a 2.4 s gap so it heads its own paragraph either
    # way; before the change it shared a paragraph with segment 3, so the
    # refresh range is the union: 6.0 to the end of segment 3.
    assert data['paragraph_start'] == 6.0 and data['paragraph_end'] == 11.0


def test_reassign___new___mints_unused_label_and_extends_diarization_list(client, project):
    _diarize(project)
    resp = _post(client, project, 'reassign-speaker', {'seg': 0, 'speaker': '__new__'})
    assert resp.status_code == 200
    after = _load(project)
    assert after['transcript']['segments'][0]['speaker'] == 'SPEAKER_02'
    assert after['diarization']['speakers'] == ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02']
    assert 'SPEAKER_02' not in after['speaker_names']
    # Minting again skips the label now in use.
    resp = _post(client, project, 'reassign-speaker', {'seg': 3, 'speaker': '__new__'})
    assert _load(project)['transcript']['segments'][3]['speaker'] == 'SPEAKER_03'


def test_reassign_without_diarization_list_still_mints(client, project):
    resp = _post(client, project, 'reassign-speaker', {'seg': 1, 'speaker': '__new__'})
    assert resp.status_code == 200
    after = _load(project)
    assert after['transcript']['segments'][1]['speaker'] == 'SPEAKER_02'
    assert 'diarization' not in after


def test_reassign_undo_restores_label_and_clears_manual(client, project):
    assert _post(client, project, 'reassign-speaker', {'seg': 2, 'speaker': 'SPEAKER_00'}).status_code == 200
    resp = _post(client, project, 'reassign-speaker', {'seg': 2, 'speaker': 'SPEAKER_01', 'speaker_manual': False})
    assert resp.status_code == 200
    seg = _load(project)['transcript']['segments'][2]
    assert seg['speaker'] == 'SPEAKER_01' and 'speaker_manual' not in seg
    assert _load(project)['transcript']['segments'] == _meta(project)['transcript']['segments']


def test_reassign_rejects_empty_speaker(client, project):
    assert _post(client, project, 'reassign-speaker', {'seg': 2, 'speaker': ''}).status_code == 400
    assert _post(client, project, 'reassign-speaker', {'seg': 2}).status_code == 400
    assert _post(client, project, 'reassign-speaker', {'seg': 9, 'speaker': 'SPEAKER_00'}).status_code == 404


def test_reassign_flips_hash(client, project):
    h0 = app_module._transcript_hash(_load(project)['transcript'])
    _post(client, project, 'reassign-speaker', {'seg': 2, 'speaker': 'SPEAKER_00'})
    assert app_module._transcript_hash(_load(project)['transcript']) != h0
    assert _load(project)['derived_stale'] is True


# ── commit 5: snapshot refresh ────────────────────────────────────────────

def _with_snapshots(project):
    meta = _load(project)
    meta['labeled_sections'] = [
        {'start': 2.0, 'end': 3.6, 'color': 'blue', 'text': 'Second half was better', 'title': 'keep me'},
        {'start': 6.0, 'end': 7.6, 'color': 'green', 'text': 'What did you learn'},
    ]
    meta['analysis'] = {
        'summary': 's',
        'strongest_soundbites': [
            {'text': 'Second half was better', 'start': '00:00:02', 'end': '00:00:03.6', 'why': 'w'},
            {'text': 'untouched bite', 'start': '00:00:09', 'end': '00:00:11', 'why': 'w'},
        ],
        'social_clips': [
            {'rank': 1, 'title': 't', 'start': '00:00:00', 'end': '00:00:04', 'text': 'Tale of two halves Second half was better'},
        ],
        'story_beats': [{'order': 1, 'label': 'b', 'description': 'd', 'start': '00:00:02', 'end': '00:00:03.6'}],
    }
    meta['quote_sheet_draft'] = {'generated_at': 'x', 'mode': 'auto', 'speakers': [
        {'name': 'Reporter', 'quotes': [
            {'raw_text': 'Second half was better', 'cleaned_text': 'Second half was better.',
             'topic_header': 'On halves', 'timecode_start': 2.0, 'timecode_end': 3.6, 'include_in_export': True},
            {'raw_text': 'segment level only', 'cleaned_text': 'Segment level only.',
             'topic_header': 'On x', 'timecode_start': 9.0, 'timecode_end': 11.0, 'include_in_export': True},
        ]},
    ]}
    (Path(app_module.app.config['PROJECTS_DIR']) / project / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    return meta


def test_edit_refreshes_only_overlapping_snapshots(client, project):
    before = _with_snapshots(project)
    resp = _post(client, project, 'edit-word', {'seg': 1, 'w': 3, 'expected': 'better', 'text': 'stronger'})
    assert resp.status_code == 200
    assert resp.get_json()['refreshed'] == {'labeled_sections': 1, 'strongest_soundbites': 1,
                                            'social_clips': 1, 'quotes': 1}
    after = _load(project)
    secs = after['labeled_sections']
    assert secs[0]['text'] == 'Second half was stronger'
    assert (secs[0]['start'], secs[0]['end'], secs[0]['color'], secs[0]['title']) == (2.0, 3.6, 'blue', 'keep me')
    assert secs[1] == before['labeled_sections'][1]
    bites = after['analysis']['strongest_soundbites']
    assert bites[0]['text'] == 'Second half was stronger'
    assert (bites[0]['start'], bites[0]['end'], bites[0]['why']) == ('00:00:02', '00:00:03.6', 'w')
    assert bites[1] == before['analysis']['strongest_soundbites'][1]
    assert after['analysis']['social_clips'][0]['text'] == 'Tale of two halves Second half was stronger'
    assert after['analysis']['story_beats'] == before['analysis']['story_beats']
    quotes = after['quote_sheet_draft']['speakers'][0]['quotes']
    assert quotes[0]['raw_text'] == 'Second half was stronger'
    assert quotes[0]['cleaned_text'] == 'Second half was better.'      # the user's copy, untouched
    assert (quotes[0]['timecode_start'], quotes[0]['timecode_end']) == (2.0, 3.6)
    assert quotes[1] == before['quote_sheet_draft']['speakers'][0]['quotes'][1]


def test_label_text_follows_the_200_char_rule(client, project):
    meta = _load(project)
    long_words = _words(2.0, ['w%02d' % i for i in range(120)], step=0.01)
    meta['transcript']['segments'][1] = {
        'start': long_words[0]['start'], 'end': long_words[-1]['end'], 'speaker': 'SPEAKER_00',
        'text': ' '.join(w['word'].strip() for w in long_words), 'words': long_words,
        'start_formatted': format_timestamp(2.0), 'end_formatted': format_timestamp(long_words[-1]['end']),
    }
    meta['labeled_sections'] = [{'start': 2.0, 'end': 3.3, 'color': 'blue', 'text': 'old'}]
    (Path(app_module.app.config['PROJECTS_DIR']) / project / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    assert _post(client, project, 'edit-word', {'seg': 1, 'w': 0, 'expected': 'w00', 'text': 'first'}).status_code == 200
    text = _load(project)['labeled_sections'][0]['text']
    assert text.startswith('first w01 w02') and len(text) == 200


def test_reassign_leaves_snapshots_alone(client, project):
    before = _with_snapshots(project)
    resp = _post(client, project, 'reassign-speaker', {'seg': 1, 'speaker': 'SPEAKER_01'})
    assert resp.status_code == 200 and 'refreshed' not in resp.get_json()
    after = _load(project)
    assert after['labeled_sections'] == before['labeled_sections']
    assert after['analysis'] == before['analysis']
    assert after['quote_sheet_draft'] == before['quote_sheet_draft']


def test_split_and_merge_refresh_the_original_range(client, project):
    _with_snapshots(project)
    assert _post(client, project, 'split-segment', {'seg': 1, 'at_w': 2}).status_code == 200
    # Text is unchanged by a split, so the snapshots re-derive to the same words.
    assert _load(project)['labeled_sections'][0]['text'] == 'Second half was better'
    assert _load(project)['quote_sheet_draft']['speakers'][0]['quotes'][0]['raw_text'] == 'Second half was better'
    assert _post(client, project, 'delete-word', {'seg': 2, 'w': 1, 'expected': 'better'}).status_code == 200
    assert _load(project)['labeled_sections'][0]['text'] == 'Second half was'
    assert _post(client, project, 'merge-segment', {'seg': 1}).status_code == 200
    assert _load(project)['analysis']['strongest_soundbites'][0]['text'] == 'Second half was'


def test_refresh_survives_missing_or_malformed_snapshots(client, project):
    meta = _load(project)
    meta['labeled_sections'] = [{'start': 'x'}, 'junk', {'start': 2.0, 'end': 3.6, 'color': 'blue', 'text': 'old'}]
    meta['analysis'] = {'strongest_soundbites': ['junk', {'text': 'no times'}]}
    meta['quote_sheet_draft'] = {'speakers': ['junk', {'quotes': [{'timecode_start': 'x'}]}]}
    (Path(app_module.app.config['PROJECTS_DIR']) / project / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    resp = _post(client, project, 'edit-word', {'seg': 1, 'w': 0, 'expected': 'Second', 'text': 'Latter'})
    assert resp.status_code == 200
    assert _load(project)['labeled_sections'][2]['text'] == 'Latter half was better'
