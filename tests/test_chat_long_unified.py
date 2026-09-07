"""Long-interview unified chat path (2026-09-06).

Covers the audit findings that motivated it:
  * the speaker-name anchor no longer bypasses the clip-noun reference
    guards ("why did you pick those Posey clips?" is discussion);
  * retrieval helpers carry the project's speaker display names;
  * the long path answers in prose with markers through ONE model call —
    never the card-only chunked search, never the "couldn't find" apology;
  * follow-up chips are split off the reply and never reach the text;
  * the interview map summarises the timeline from segment vectors;
  * cloud providers with a big window get the full transcript regardless
    of length; local Ollama keeps the duration threshold.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import ai_analysis as A  # noqa: E402


# ── fixtures ────────────────────────────────────────────────────────────────

def _long_transcript(minutes=90, named=False):
    """A synthetic interview: two speakers alternating 20-second segments."""
    segs = []
    t = 0.0
    labels = ('Mae Babcock', 'Tom Posey') if named else ('SPEAKER_00', 'SPEAKER_01')
    topics = [
        'the river flooded the lower field and we lost the whole crop',
        'my father built this house with his own hands after the war',
        'the fire started in the barn and spread before anyone woke',
        'we never talked about money at the table it was not done',
        'the town council voted against us three years running',
    ]
    i = 0
    while t < minutes * 60:
        segs.append({
            'speaker': labels[i % 2],
            'start': t,
            'end': t + 18.0,
            'text': f"{topics[i % len(topics)]} and that was moment number {i}.",
        })
        t += 20.0
        i += 1
    return {'segments': segs, 'language': 'en'}


NAMES = {'SPEAKER_00': 'Mae Babcock', 'SPEAKER_01': 'Tom Posey'}

VECTORS = [
    {'seg_id': 'SEG001', 'timecode_in': '00:00:40', 'timecode_out': '00:01:00',
     'thread_title': 'The fire in the barn', 'narrative_score': 'high',
     'theme_tags': ['fire', 'loss'], 'transcript_excerpt': 'the fire started in the barn'},
    {'seg_id': 'SEG002', 'timecode_in': '00:20:00', 'timecode_out': '00:20:20',
     'thread_title': 'Money at the table', 'narrative_score': 'medium',
     'theme_tags': ['money', 'family'], 'transcript_excerpt': 'we never talked about money'},
    {'seg_id': 'SEG003', 'timecode_in': '01:10:00', 'timecode_out': '01:10:20',
     'thread_title': 'Council votes', 'narrative_score': 'low',
     'theme_tags': ['politics'], 'transcript_excerpt': 'the town council voted'},
]


# ── classifier: reference guard runs before the speaker anchor ─────────────

@pytest.mark.parametrize('labels', [
    [{'speaker': 'Mae Babcock'}, {'speaker': 'Tom Posey'}],
    [{'speaker': 'SPEAKER_00'}, {'speaker': 'SPEAKER_01'}],
])
@pytest.mark.parametrize('message,expected', [
    ('why did you pick those clips?', True),
    ('why did you pick those Posey clips?', True),
    ("those Mae clips were perfect, what's the theme?", True),
    ('no clips, what is Mae\'s arc?', True),
    ('find me 3 clips about the fire', False),
    ('pull the best Posey moment', False),
    ('show me clips of Mae talking about the river', False),
    ('what are the strongest emotional moments?', False),
    ('build me a 2 minute cut', False),
    ("what's the theme of this interview?", True),
])
def test_classifier_reference_guard_precedes_speaker_anchor(labels, message, expected):
    assert A._is_conversational_query(message, segments=labels) is expected


def test_named_speaker_thematic_question_still_reaches_retrieval():
    # "what's the story with Mae?" names a speaker with no clip noun: it
    # stays extractive (retrieval + marker pressure) on named segments and
    # conversational on raw labels — both now end in a prose answer.
    named = [{'speaker': 'Mae Babcock'}]
    assert A._is_conversational_query("what's the story with Mae?", segments=named) is False


# ── speaker display names in retrieval ─────────────────────────────────────

def test_apply_speaker_names_maps_raw_labels():
    items = [{'speaker': 'SPEAKER_00', 'text': 'a'}, {'speaker': 'Guest', 'text': 'b'}]
    out = A._apply_speaker_names(items, NAMES)
    assert out[0]['speaker'] == 'Mae Babcock'
    assert out[1]['speaker'] == 'Guest'
    assert items[0]['speaker'] == 'SPEAKER_00'  # input untouched


def test_build_paragraphs_applies_names_after_merge():
    tr = _long_transcript(minutes=2)
    paras = A._build_paragraphs(tr, speaker_names=NAMES)
    assert paras and all(p['speaker'] in ('Mae Babcock', 'Tom Posey') for p in paras)
    raw = A._build_paragraphs(tr)
    assert len(raw) == len(paras)


def test_chunk_lines_carry_display_names():
    tr = _long_transcript(minutes=2)
    lines = A._format_paragraphs_as_lines(A._build_paragraphs(tr, speaker_names=NAMES))
    assert 'SPEAKER_00' not in lines and 'Mae Babcock' in lines


# ── follow-up chips ─────────────────────────────────────────────────────────

def test_split_followups_strips_tail_and_returns_three():
    text = ('Dana frames the rule as a constraint.\n\n'
            '[CLIP: start=01:30:57 end=01:31:14 title="The rule"]\n\n'
            '**FOLLOW-UPS:** Show the moment before this | Find the counterpoint | Add both to the story')
    body, chips = A._split_followups(text)
    assert 'FOLLOW' not in body
    assert body.endswith('title="The rule"]')
    assert chips == ['Show the moment before this', 'Find the counterpoint', 'Add both to the story']


def test_split_followups_only_last_line_and_caps_at_three():
    text = 'She said "follow-ups: none" on camera.\nFollow-ups: a question here | b question here | c question here | d question here'
    body, chips = A._split_followups(text)
    assert body == 'She said "follow-ups: none" on camera.'
    assert len(chips) == 3


def test_split_followups_no_tail():
    assert A._split_followups('plain answer') == ('plain answer', [])
    assert A._split_followups('') == ('', [])


def test_build_chat_messages_followups_hint_rides_final_turn_only():
    segs = _long_transcript(minutes=1)['segments']
    _sys, msgs = A._build_chat_messages(
        'what is this about?', [], 'P', segs, 'TRANSCRIPT', '', '', None,
        followups_hint=True,
    )
    assert 'FOLLOW-UPS:' in msgs[-1]['content']
    assert all('FOLLOW-UPS:' not in m['content'] for m in msgs[:-1])
    _sys2, msgs2 = A._build_chat_messages(
        'what is this about?', [], 'P', segs, 'TRANSCRIPT', '', '', None,
    )
    assert 'FOLLOW-UPS:' not in msgs2[-1]['content']


# ── interview map ───────────────────────────────────────────────────────────

def test_interview_map_groups_by_window_and_marks_high():
    block = A._build_interview_map_block(VECTORS, 90 * 60)
    assert block.startswith('INTERVIEW MAP')
    lines = block.splitlines()[1:]
    assert lines[0].startswith('[00:00:00–00:15:00]') and '★ The fire in the barn' in lines[0]
    assert any(l.startswith('[00:15:00–00:30:00]') and 'Money at the table' in l for l in lines)
    assert any(l.startswith('[01:00:00–01:15:00]') and 'Council votes' in l for l in lines)


def test_interview_map_empty_without_vectors():
    assert A._build_interview_map_block(None, 100) == ''
    assert A._build_interview_map_block([], 100) == ''


# ── retrieval ───────────────────────────────────────────────────────────────

def test_long_retrieve_keyword_hits_carry_names_and_stay_in_budget():
    tr = _long_transcript(minutes=90)
    phrases, words = A._extract_query_keywords('what does Mae say about the fire?')
    out = A._chat_long_retrieve(tr, 'what does Mae say about the fire?', phrases, words,
                                [], [], VECTORS, NAMES)
    assert out, 'retrieval must find the fire paragraphs'
    assert all(p['speaker'] in ('Mae Babcock', 'Tom Posey') for p in out)
    assert any('fire' in (p.get('text') or '') for p in out)
    total = sum(len(p.get('text') or '') + 40 for p in out)
    assert total <= A._long_chat_excerpt_budget_chars()
    starts = [p['start'] for p in out]
    assert starts == sorted(starts)


def test_long_retrieve_speaker_anchor_without_keywords():
    tr = _long_transcript(minutes=90)
    out = A._chat_long_retrieve(tr, "what's the story with Mae?", [], [], [], [], VECTORS, NAMES)
    assert out and all(p['speaker'] == 'Mae Babcock' for p in out)
    assert len(out) <= A._LONG_CHAT_SPEAKER_CAP


def test_long_retrieve_falls_back_to_high_score_vectors():
    tr = _long_transcript(minutes=90)
    out = A._chat_long_retrieve(tr, 'how would you open this film?', [], [], [], [], VECTORS, NAMES)
    assert out, 'abstract asks still get the editorial-signal excerpts'


def test_long_retrieve_empty_transcript():
    assert A._chat_long_retrieve({'segments': []}, 'anything', [], [], [], [], VECTORS, NAMES) == []


# ── the unified path: one call, prose + cards, no chunk scans ──────────────

def _fake_stream(reply):
    def _stream(system_message, messages, num_ctx, **kw):
        for piece in reply.split(' '):
            yield ('token', piece + ' ')
        yield ('raw', reply)
    return _stream


def test_unified_stream_answers_with_prose_and_cards(monkeypatch):
    tr = _long_transcript(minutes=90)
    reply = ('The fire is the hinge of the whole interview.\n\n'
             '[CLIP: start=00:00:40 end=00:00:58 title="The fire in the barn" note="Where it turns"]\n\n'
             'FOLLOW-UPS: What happened after the fire? | Show me the council vote | Build a 60 second cut')
    monkeypatch.setattr(A, '_stream_chat_events', _fake_stream(reply))
    monkeypatch.setattr(A, '_call_ai_json', lambda *a, **k: pytest.fail('chunked search must not run'))
    monkeypatch.setattr(A, '_salvage_clips_if_missing', lambda cleaned, *a, **k: cleaned)
    monkeypatch.setattr(A, '_long_chat_threshold', lambda: 60 * 60)
    monkeypatch.setattr(A, '_ollama_is_active', lambda: True)
    monkeypatch.setattr(A, '_sticky_chat_num_ctx', lambda *a, **k: 16384)

    events = list(A.chat_about_transcript_stream(
        tr, 'why did you pick those Mae clips?', history=[], project_name='P',
        analysis={'summary': 'A farm family and a fire.', 'themes': ['loss']},
        segment_vectors=VECTORS, speaker_names=NAMES,
    ))
    kinds = [e[0] for e in events]
    assert 'progress' in kinds and 'token' in kinds and 'done' in kinds
    assert 'followups' in kinds
    done = [e for e in events if e[0] == 'done'][0][1]
    assert 'The fire is the hinge' in done
    assert '[CLIP:' in done
    assert 'FOLLOW' not in done
    chips = [e for e in events if e[0] == 'followups'][0][1]
    assert len(chips) == 3
    prog = [e[1] for e in events if e[0] == 'progress']
    assert any(p.startswith('Reading ') for p in prog)
    assert "couldn't find moments" not in done


def test_unified_blocking_mirrors_stream(monkeypatch):
    tr = _long_transcript(minutes=90)
    reply = 'Mae carries the arc.\n\nFOLLOW-UPS: a longer question | another question | third one'
    monkeypatch.setattr(A, '_call_ai_chat', lambda *a, **k: reply)
    monkeypatch.setattr(A, '_call_ai_json', lambda *a, **k: pytest.fail('chunked search must not run'))
    monkeypatch.setattr(A, '_salvage_clips_if_missing', lambda cleaned, *a, **k: cleaned)
    monkeypatch.setattr(A, '_long_chat_threshold', lambda: 60 * 60)
    monkeypatch.setattr(A, '_sticky_chat_num_ctx', lambda *a, **k: 16384)
    out = A.chat_about_transcript(tr, "what's the theme?", history=[], project_name='P',
                                  analysis=None, segment_vectors=VECTORS, speaker_names=NAMES)
    assert out.strip() == 'Mae carries the arc.'


def test_legacy_flag_restores_chunked_routing(monkeypatch):
    tr = _long_transcript(minutes=90)
    monkeypatch.setenv('DOZA_CHAT_LEGACY_CHUNKED', '1')
    monkeypatch.setattr(A, '_long_chat_threshold', lambda: 60 * 60)
    called = {}

    def _fake_chunked(*a, **k):
        called['yes'] = True
        return '[CLIP: start=00:00:40 end=00:00:58 title="x"]'
    monkeypatch.setattr(A, '_chat_layer2_chunked_search', _fake_chunked)
    out = A.chat_about_transcript(tr, 'find me 3 clips about the fire', history=[],
                                  project_name='P', segment_vectors=VECTORS, speaker_names=NAMES)
    assert called.get('yes') and '[CLIP:' in out


def test_long_context_has_map_and_names_but_not_selections():
    tr = _long_transcript(minutes=90)
    ctx = A._build_long_chat_context('P', tr['segments'], {'summary': 'S'}, [{'start': 0, 'end': 10}],
                                     speaker_names=NAMES, segment_vectors=VECTORS)
    assert 'INTERVIEW MAP' in ctx
    assert 'Mae Babcock' in ctx and 'SPEAKER_00' not in ctx
    assert '<editor_selections>' not in ctx


# ── provider-aware routing ──────────────────────────────────────────────────

def test_cloud_provider_gets_full_transcript(monkeypatch):
    import ai_providers
    tr = _long_transcript(minutes=90)
    monkeypatch.setattr(ai_providers, 'current_provider_name', lambda: 'anthropic')
    assert A._cloud_full_transcript_ok(tr) is True
    monkeypatch.setattr(ai_providers, 'current_provider_name', lambda: 'ollama')
    assert A._cloud_full_transcript_ok(tr) is False


def test_cloud_provider_still_chunks_when_transcript_exceeds_window(monkeypatch):
    import ai_providers
    tr = _long_transcript(minutes=9 * 60)
    # Real interviews run ~1,400 chars per minute; the synthetic fixture is
    # sparse, so pad each segment to a realistic density.
    for seg in tr['segments']:
        seg['text'] = seg['text'] * 6
    monkeypatch.setattr(ai_providers, 'current_provider_name', lambda: 'anthropic')
    assert A._cloud_full_transcript_ok(tr) is False


# ── prewarm covers long projects now ───────────────────────────────────────

def test_prewarm_builds_long_prefix(monkeypatch):
    tr = _long_transcript(minutes=90)
    seen = {}

    def _fake_chat(system_message, messages, num_ctx=32768, **kw):
        seen['messages'] = messages
        seen['num_ctx'] = num_ctx
        return ''
    monkeypatch.setattr(A, '_call_ai_chat', _fake_chat)
    monkeypatch.setattr(A, '_ollama_is_active', lambda: True)
    monkeypatch.setattr(A, '_long_chat_threshold', lambda: 60 * 60)
    A._PREWARM_STATE.clear()
    ok = A.prewarm_chat_context(tr, project_name='PrewarmLong', analysis={'summary': 'S'},
                                speaker_names=NAMES, segment_vectors=VECTORS)
    assert ok is True
    joined = '\n'.join(m['content'] for m in seen['messages'])
    assert 'INTERVIEW MAP' in joined
    assert 'FOLLOW-UPS' not in joined
