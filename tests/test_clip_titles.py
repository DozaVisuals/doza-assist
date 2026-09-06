"""Generated clip titles (1.0.47): clip_titles helpers, the
POST /project/<id>/clips/titles route with a stubbed model, and the page
helpers in static/clip_text.js under Node."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402
import clip_titles  # noqa: E402

CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_TEXT_JS = os.path.join(CORE_DIR, 'static', 'clip_text.js')
NODE = shutil.which('node')
needs_node = pytest.mark.skipif(NODE is None, reason='node is not installed')

SENTENCE = ('And how do we help as Daybright? Yeah, great question. '
            'We started by listening to the people who call in every day.')


def _segments():
    words = SENTENCE.split()
    return [{
        'start': 0.0, 'end': 12.0, 'speaker': 'SPEAKER_00',
        'words': [{'word': w, 'start': i * 0.5, 'end': i * 0.5 + 0.4} for i, w in enumerate(words)],
    }, {
        'start': 12.0, 'end': 20.0, 'speaker': 'SPEAKER_01',
        'text': 'That is what changed everything for us.',
    }]


# ── helpers ────────────────────────────────────────────────────────────────

def test_transcript_for_range_uses_word_timings_then_segment_text():
    segs = _segments()
    assert clip_titles.transcript_for_range(segs, 0, 1.1) == 'And how do'
    assert clip_titles.transcript_for_range(segs, 12, 20) == 'That is what changed everything for us.'
    assert clip_titles.transcript_for_range(segs, 5, 5) == ''


def test_fragment_detection():
    tr = clip_titles.transcript_for_range(_segments(), 0, 12)
    # The brush stores the first 200 characters of the painted words.
    assert clip_titles.is_transcript_fragment('And how do we help as Daybright? Yeah, great ques...', tr)
    # AI Analysis stores the first 40 characters of a soundbite.
    assert clip_titles.is_transcript_fragment(SENTENCE[:40], tr)
    # A Story Brief quote from the middle of the range.
    assert clip_titles.is_transcript_fragment('we started by listening to the people who call', tr)
    # A Chat headline is not the transcript talking.
    assert not clip_titles.is_transcript_fragment('It Is That Simple as a Challenge', tr)
    # Empty text always needs a title; with no transcript a text is kept.
    assert clip_titles.is_transcript_fragment('', tr)
    assert not clip_titles.is_transcript_fragment('Some title', '')
    # An AI select stores its overlapping segments, so its text can begin a
    # sentence before the clip's start: the wider window recognizes it.
    ctx = clip_titles.transcript_for_range(_segments(), 0, 20)
    tail = clip_titles.transcript_for_range(_segments(), 3.5, 20)
    assert tail.startswith('Yeah, great question')
    assert clip_titles.is_transcript_fragment(
        'help as Daybright? Yeah, great question. We started by listening to the people who call', tail, ctx)
    # One transcription difference in a long fragment still counts.
    assert clip_titles.is_transcript_fragment(
        'help as Daybright? Yeah, great question. We started by listening to the folks who call in', tail, ctx)
    # A headline stays a headline even with the wider window.
    assert not clip_titles.is_transcript_fragment('Listening Is The Whole Job', tail, ctx)


def test_first_line_takes_whole_sentences_up_to_the_cap():
    # Short enough to fit whole: unchanged.
    assert clip_titles.first_line(SENTENCE) == SENTENCE
    long_tail = SENTENCE + ' ' + 'More words follow here and keep going for quite a while longer than the cap.'
    assert clip_titles.first_line(long_tail) == 'And how do we help as Daybright? Yeah, great question.'
    # A clip that starts on the tail of a sentence keeps reading.
    assert clip_titles.first_line('daybreak? Yeah, that is a great question. ' + 'x ' * 100) == 'daybreak? Yeah, that is a great question.'
    long = 'word ' * 60
    line = clip_titles.first_line(long)
    assert line.endswith('…') and len(line) <= 141
    assert clip_titles.first_line('') == ''


def test_parse_titles_accepts_dict_list_and_junk():
    assert clip_titles.parse_titles({'titles': {'1': '"Listening First."', '2': '2. A Question of Help'}}, 2) == {
        1: 'Listening First', 2: 'A Question of Help'}
    assert clip_titles.parse_titles({'titles': ['One', 'Two', 'Three']}, 2) == {1: 'One', 2: 'Two'}
    assert clip_titles.parse_titles({'error': 'parse'}, 2) == {}
    assert clip_titles.parse_titles(None, 2) == {}
    assert clip_titles.parse_titles({'titles': {'1': ''}}, 1) == {}


def test_clean_title_caps_run_on_titles_at_a_word_boundary():
    t = clip_titles.clean_title('This title just keeps going ' * 6)
    assert len(t) <= clip_titles.TITLE_MAX_CHARS and not t.endswith(' ')


def test_build_prompt_numbers_excerpts_and_appends_language_directive():
    p = clip_titles.build_prompt([
        {'transcript': 'first words here', 'speaker': 'Maya Chen', 'duration': 75},
        {'transcript': 'second words', 'speaker': '', 'duration': 8},
    ], '\n\nOUTPUT LANGUAGE: Norwegian')
    assert 'Excerpt 1 (Maya Chen, 1:15 long):' in p
    assert 'Excerpt 2 (8s long):' in p
    assert p.index('"first words here"') < p.index('"second words"')
    assert p.endswith('OUTPUT LANGUAGE: Norwegian')
    assert '—' not in p


def test_build_prompt_trims_long_excerpts_head_and_tail():
    long = ' '.join(f'w{i}' for i in range(600))
    p = clip_titles.build_prompt([{'transcript': long, 'speaker': '', 'duration': 30}])
    assert ' ... ' in p and 'w0 w1' in p and 'w599' in p
    assert len(p) < len(long)


# ── title_clips orchestration (no Flask) ───────────────────────────────────

def _project(sections):
    return {'id': 'p1', 'name': 'P', 'labeled_sections': sections,
            'transcript': {'segments': _segments()},
            'speaker_names': {'SPEAKER_00': 'Maya Chen'}}


def test_title_clips_generates_for_fragments_and_carries_headlines():
    calls = []

    def model(prompt, system):
        calls.append(prompt)
        return {'titles': {'1': 'Listening To The Callers'}}

    project = _project([
        {'start': 0, 'end': 12, 'color': 'blue', 'text': SENTENCE[:60]},   # brush fragment
        {'start': 12, 'end': 20, 'color': 'green', 'text': 'It Changed Everything'},  # chat headline
    ])
    res = clip_titles.title_clips(project, None, model)
    assert len(calls) == 1 and 'Maya Chen' in calls[0]
    assert res['generated'] == 1 and res['carried'] == 1 and res['failed'] == 0
    assert res['changed'] is True
    secs = res['sections']
    assert secs[0]['title'] == 'Listening To The Callers' and secs[0]['title_auto'] is True
    assert secs[0]['text'] == SENTENCE[:60]  # the fragment stays as text
    assert secs[1]['title'] == 'It Changed Everything' and 'title_auto' not in secs[1]
    statuses = [i['status'] for i in res['items']]
    assert statuses == ['generated', 'carried']
    assert res['items'][0]['lead'] == SENTENCE


def test_title_clips_requested_clip_not_yet_saved_is_titled_but_not_stored():
    def model(prompt, system):
        return {'titles': {'1': 'Fresh Title'}}

    project = _project([])
    res = clip_titles.title_clips(project, [{'start': 0, 'end': 12, 'text': SENTENCE[:50]}], model,
                                  include_transcript=True)
    assert res['items'][0]['title'] == 'Fresh Title'
    assert res['items'][0]['transcript'].startswith('And how do we help')
    assert res['changed'] is False and res['sections'] == []


def test_title_clips_skips_titled_clips_and_survives_model_errors():
    def model(prompt, system):
        raise RuntimeError('Ollama is not running')

    project = _project([
        {'start': 0, 'end': 12, 'color': 'blue', 'text': SENTENCE[:60], 'title': 'Already Named'},
        {'start': 12, 'end': 20, 'color': 'blue', 'text': 'That is what changed everything for us.'},
    ])
    res = clip_titles.title_clips(project, None, model)
    assert res['kept'] == 0  # untitled-only scan never lists the titled clip
    assert res['failed'] == 1 and res['generated'] == 0
    assert res['error'] == 'Ollama is not running'
    assert res['changed'] is False
    # explicit request for the titled clip reports it as kept
    res2 = clip_titles.title_clips(project, [{'start': 0, 'end': 12}], model)
    assert res2['items'][0]['status'] == 'kept' and res2['items'][0]['title'] == 'Already Named'


def test_title_clips_batches_by_batch_size():
    calls = []

    def model(prompt, system):
        calls.append(prompt)
        n = prompt.count('Excerpt ')
        return {'titles': {str(i): f'T{i}' for i in range(1, n + 1)}}

    sections = [{'start': i * 0.5, 'end': i * 0.5 + 2, 'color': 'blue', 'text': ''} for i in range(5)]
    res = clip_titles.title_clips(_project(sections), None, model, batch_size=2)
    assert len(calls) == 3 and res['generated'] == 5


# ── the route ──────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    app_module.app.config['EXPORTS_DIR'] = str(tmp_path / 'exports')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    Path(app_module.app.config['EXPORTS_DIR']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def _make_project(pid, **extra):
    pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    pdir.mkdir(parents=True)
    meta = {'id': pid, 'name': 'P', 'status': 'complete',
            'transcript': {'segments': _segments()}}
    meta.update(extra)
    (pdir / 'meta.json').write_text(json.dumps(meta))
    return pdir


def _read_meta(pid):
    return json.loads((Path(app_module.app.config['PROJECTS_DIR']) / pid / 'meta.json').read_text())


def test_route_titles_stored_clips_and_persists(client, monkeypatch):
    seen = {}

    def fake_model(prompt, system):
        seen['prompt'] = prompt
        seen['system'] = system
        return {'titles': {'1': 'Listening To The Callers'}}

    monkeypatch.setattr(app_module, '_clip_title_model_call', fake_model)
    _make_project('p1', labeled_sections=[
        {'start': 0, 'end': 12, 'color': 'blue', 'text': SENTENCE[:60]},
        {'start': 12, 'end': 20, 'color': 'green', 'text': 'It Changed Everything', 'speaker': 'Sam'},
    ])
    res = client.post('/project/p1/clips/titles', json={})
    assert res.status_code == 200
    data = res.get_json()
    assert data['generated'] == 1 and data['carried'] == 1 and data['failed'] == 0
    assert data['error'] is None
    assert [t['title'] for t in data['titles']] == ['Listening To The Callers', 'It Changed Everything']
    meta = _read_meta('p1')
    assert meta['labeled_sections'][0]['title'] == 'Listening To The Callers'
    assert meta['labeled_sections'][0]['title_auto'] is True
    assert meta['labeled_sections'][1] == {'start': 12, 'end': 20, 'color': 'green',
                                           'text': 'It Changed Everything', 'speaker': 'Sam',
                                           'title': 'It Changed Everything'}
    assert seen['system'] == clip_titles.SYSTEM_PROMPT
    assert 'Excerpt 1' in seen['prompt'] and 'Excerpt 2' not in seen['prompt']


def test_route_explicit_sections_before_the_labels_save_lands(client, monkeypatch):
    monkeypatch.setattr(app_module, '_clip_title_model_call',
                        lambda p, s: {'titles': {'1': 'Fresh Title'}})
    _make_project('p2', labeled_sections=[])
    res = client.post('/project/p2/clips/titles', json={
        'sections': [{'start': 0, 'end': 12, 'text': SENTENCE[:50]}], 'include_transcript': True})
    data = res.get_json()
    assert data['titles'][0]['title'] == 'Fresh Title'
    assert data['titles'][0]['transcript'].startswith('And how do we help')
    assert _read_meta('p2')['labeled_sections'] == []


def test_route_model_down_returns_200_with_error_and_no_title(client, monkeypatch):
    def boom(prompt, system):
        raise RuntimeError('Ollama is not running')
    monkeypatch.setattr(app_module, '_clip_title_model_call', boom)
    _make_project('p3', labeled_sections=[{'start': 0, 'end': 12, 'color': 'blue', 'text': SENTENCE[:60]}])
    res = client.post('/project/p3/clips/titles', json={})
    assert res.status_code == 200
    data = res.get_json()
    assert data['failed'] == 1 and data['error'] == 'Ollama is not running'
    assert data['titles'][0]['title'] == ''
    assert 'title' not in _read_meta('p3')['labeled_sections'][0]


def test_route_no_model_call_when_every_clip_is_titled(client, monkeypatch):
    calls = []
    monkeypatch.setattr(app_module, '_clip_title_model_call', lambda p, s: calls.append(p))
    _make_project('p4', labeled_sections=[{'start': 0, 'end': 12, 'color': 'blue', 'text': 'x', 'title': 'Named'}])
    data = client.post('/project/p4/clips/titles', json={}).get_json()
    assert calls == [] and data['generated'] == 0 and data['titles'] == []


def test_route_rejects_cross_origin_and_missing_project(client, monkeypatch):
    monkeypatch.setattr(app_module, '_clip_title_model_call', lambda p, s: {'titles': {}})
    _make_project('p5', labeled_sections=[])
    assert client.post('/project/p5/clips/titles', json={},
                       headers={'Origin': 'https://share.doza.ai'}).status_code == 403
    assert client.post('/project/nope/clips/titles', json={}).status_code == 404
    assert client.post('/project/p5/clips/titles', json={'sections': 'x'}).status_code == 400


def test_labels_round_trip_keeps_title_fields(client):
    _make_project('p6')
    body = {'color_labels': {}, 'labeled_sections': [
        {'start': 1, 'end': 3, 'color': 'blue', 'text': 'raw words', 'title': 'Real Title', 'title_auto': True}]}
    assert client.post('/project/p6/labels', json=body).status_code == 200
    assert _read_meta('p6')['labeled_sections'] == body['labeled_sections']


def test_exports_prefer_the_title_in_the_note():
    from exporters.documents import selects_rows
    project = _project([{'start': 0, 'end': 12, 'color': 'blue', 'text': SENTENCE[:60], 'title': 'Listening First'},
                        {'start': 12, 'end': 20, 'color': 'blue', 'text': 'plain'}])
    project['color_labels'] = {'blue': 'Hero'}
    rows = selects_rows(project, ['labels'])
    assert [r['note'] for r in rows] == ['Listening First', 'plain']


# ── the page helpers under Node ─────────────────────────────────────────────

def _node(script):
    out = subprocess.run([NODE, '-e', script], capture_output=True, text=True, check=True, cwd=CORE_DIR)
    return json.loads(out.stdout.strip())


@needs_node
def test_clip_text_helpers_under_node():
    res = _node(f"""
        const t = require({json.dumps(CLIP_TEXT_JS)});
        const secs = [{{ start: 1, end: 3, text: 'raw words' }}, {{ start: 5, end: 9, text: 'x', title: 'Named' }}];
        const needs = [t.needsTitle(secs[0]), t.needsTitle(secs[1]), t.needsTitle({{ text: 'x', _titleFailed: true }}), t.needsTitle({{ text: 'x', _titling: true }})];
        const applied = t.applyTitle(secs, {{ start: 1.2, end: 3.1, title: 'Good Title', title_auto: true }});
        console.log(JSON.stringify({{
            lead: t.firstLine({json.dumps(SENTENCE)} + ' More words follow here and keep going for quite a while longer than the cap.'),
            tail: t.firstLine('daybreak? Yeah, that is a great question. ' + 'x '.repeat(100)),
            leadLong: t.firstLine('word '.repeat(60)).length,
            pending: t.displayTitle({{ text: 'raw', _titling: true }}),
            titled: t.displayTitle({{ text: 'raw', title: 'It Is That Simple' }}),
            plain: t.displayTitle({{ text: 'raw' }}),
            needs, applied, secs,
            again: t.applyTitle(secs, {{ start: 1, end: 3, title: 'Good Title' }}),
        }}));
    """)
    assert res['lead'] == 'And how do we help as Daybright? Yeah, great question.'
    assert res['tail'] == 'daybreak? Yeah, that is a great question.'
    assert res['leadLong'] <= 141
    assert res['pending'] == '' and res['titled'] == 'It Is That Simple' and res['plain'] == 'raw'
    assert res['needs'] == [True, False, False, False]
    assert res['applied'] is True
    assert res['secs'][0] == {'start': 1, 'end': 3, 'text': 'raw words', 'title': 'Good Title', 'title_auto': True}
    assert res['again'] is False


def test_no_em_dashes_in_added_copy():
    for rel in ('clip_titles.py', 'static/clip_text.js'):
        assert '—' not in Path(CORE_DIR, rel).read_text(encoding='utf-8'), rel
