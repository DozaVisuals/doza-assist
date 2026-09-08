"""Story So Far (2026-09-07): what the editor is making + moments they
passed on. Persisted per project, injected into the final chat turn,
honoured by retrieval, count top-ups and the reply."""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import ai_analysis as A  # noqa: E402
import app as app_module  # noqa: E402


STORY = {'making': 'a 4-minute recruiting film', 'passed': [
    {'start': 100.0, 'end': 130.0, 'title': 'The gumbo analogy'},
]}


def _segs(minutes=90):
    segs, t, i = [], 0.0, 0
    while t < minutes * 60:
        segs.append({'speaker': 'SPEAKER_00', 'start': t, 'end': t + 18.0,
                     'text': f'the fire in the barn, moment {i}.'})
        t += 20.0
        i += 1
    return segs


# ── prompt tail ─────────────────────────────────────────────────────────────

def test_tail_lists_making_and_passed():
    tail = A._story_so_far_tail(STORY)
    assert tail.startswith('STORY SO FAR')
    assert "I'm making: a 4-minute recruiting film" in tail
    assert 'The gumbo analogy' in tail and '00:01:40–00:02:10' in tail


def test_tail_empty_when_nothing_known():
    assert A._story_so_far_tail(None) == ''
    assert A._story_so_far_tail({}) == ''
    assert A._story_so_far_tail({'making': '', 'passed': []}) == ''


def test_tail_rides_final_turn_only_both_modes():
    segs = _segs(1)
    for reminder in (True, False):
        _sys, msgs = A._build_chat_messages(
            'what is this about?', [], 'P', segs, 'TRANSCRIPT', '', '', None,
            include_final_reminder=reminder, story_so_far=STORY, followups_hint=True,
        )
        last = msgs[-1]['content']
        assert 'STORY SO FAR' in last
        assert last.index('STORY SO FAR') < last.index('FOLLOW-UPS:')
        assert all('STORY SO FAR' not in m['content'] for m in msgs[:-1])


# ── passed spans ────────────────────────────────────────────────────────────

def test_passed_spans_parse_and_tolerate_junk():
    assert A._story_passed_spans(STORY) == [(100.0, 130.0)]
    assert A._story_passed_spans({'passed': [{'start': 'x'}, 'nope', {'start': 5, 'end': 2}]}) == []


def test_drop_passed_markers_removes_reissued_moment():
    text = ('Two moments:\n'
            '[CLIP: start=00:01:42 end=00:02:05 title="Gumbo again"]\n'
            '[CLIP: start=00:10:00 end=00:10:20 title="Keep me"]')
    out = A._drop_passed_markers(text, STORY)
    assert 'Gumbo again' not in out and 'Keep me' in out
    assert A._drop_passed_markers(text, None) == text


def test_long_retrieve_excludes_passed_spans():
    tr = {'segments': _segs(90), 'language': 'en'}
    phrases, words = A._extract_query_keywords('the fire in the barn')
    out = A._chat_long_retrieve(tr, 'the fire in the barn', phrases, words, [], [], [], None,
                                passed_spans=[(100.0, 130.0)])
    assert out
    assert not any(min(p['end'], 130.0) - max(p['start'], 100.0) > 0 for p in out)


def test_layer1_reply_drops_passed_moment(monkeypatch):
    tr = {'segments': _segs(5), 'language': 'en'}
    reply = ('Here you go.\n'
             '[CLIP: start=00:01:40 end=00:02:10 title="Gumbo"]\n'
             '[CLIP: start=00:03:00 end=00:03:18 title="Fire"]\n'
             'FOLLOW-UPS: one question here | two question here | three question here')
    monkeypatch.setattr(A, '_call_ai_chat', lambda *a, **k: reply)
    monkeypatch.setattr(A, '_salvage_clips_if_missing', lambda cleaned, *a, **k: cleaned)
    monkeypatch.setattr(A, '_sticky_chat_num_ctx', lambda *a, **k: 8192)
    out = A.chat_about_transcript(tr, 'find me 2 clips about the fire', history=[],
                                  project_name='P', story_so_far=STORY)
    assert 'Fire' in out and 'Gumbo' not in out and 'FOLLOW' not in out


# ── auto "making" ───────────────────────────────────────────────────────────

@pytest.mark.parametrize('message,expected', [
    ('build me a 90 second teaser', True),
    ("I'm making a recruiting film for the benefits team", True),
    ('this is for a 4 minute documentary', True),
    ('what is the emotional arc?', False),
    ('find me 3 clips about the fire', False),
    ('make it shorter', False),
])
def test_detect_story_making(message, expected):
    assert bool(A._detect_story_making(message)) is expected


# ── routes ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    pid = 'story-test'
    d = Path(app_module.app.config['PROJECTS_DIR']) / pid
    d.mkdir()
    (d / 'meta.json').write_text(json.dumps({
        'id': pid, 'name': 'Story test', 'filename': 'x.wav', 'filepath': str(d / 'x.wav'),
        'transcript': {'segments': _segs(2), 'language': 'en'},
    }))
    c = app_module.app.test_client()
    c.pid = pid
    return c


def test_story_routes_round_trip(client):
    pid = client.pid
    assert client.get(f'/project/{pid}/story-so-far').get_json() == {}
    r = client.post(f'/project/{pid}/story-so-far', json={'making': '  a 60 second cut  '})
    assert r.status_code == 200 and r.get_json()['making'] == 'a 60 second cut'
    r = client.post(f'/project/{pid}/story-so-far/pass', json={'start': 10, 'end': 25, 'title': 'Nope'})
    assert r.get_json()['passed'] == [{'start': 10.0, 'end': 25.0, 'title': 'Nope'}]
    # duplicate span is deduped, junk rejected
    client.post(f'/project/{pid}/story-so-far/pass', json={'start': 10, 'end': 25, 'title': 'Nope again'})
    r = client.post(f'/project/{pid}/story-so-far', json={'passed': [{'start': 10, 'end': 25}, {'start': 9, 'end': 3}, 'junk']})
    assert len(r.get_json()['passed']) == 1
    r = client.post(f'/project/{pid}/story-so-far/pass', json={'start': 10, 'end': 25, 'undo': True})
    assert r.get_json()['passed'] == []
    assert client.get(f'/project/{pid}/story-so-far').get_json()['making'] == 'a 60 second cut'


def test_story_routes_404_and_400(client):
    assert client.get('/project/nope/story-so-far').status_code == 404
    assert client.post(f'/project/{client.pid}/story-so-far/pass', json={'title': 'x'}).status_code == 400


def test_auto_making_never_overwrites_a_hand_edit():
    stored = {'story_so_far': {'making': 'my own line'}}
    app_module._story_auto_making(stored, 'build me a 90 second teaser')
    assert stored['story_so_far']['making'] == 'my own line'
    stored = {}
    app_module._story_auto_making(stored, 'build me a 90 second teaser')
    assert stored['story_so_far']['making'] == 'build me a 90 second teaser'
    stored = {}
    app_module._story_auto_making(stored, 'what is the theme?')
    assert 'story_so_far' not in stored
