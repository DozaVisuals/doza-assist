"""Analysis speed without touching what the model is asked: the four
per-chunk calls share one prompt prefix (so the model server reads the
transcript once per chunk), and analysis and chat share one context size
(so the runner is not reloaded on every switch). Both hold for whichever
Gemma model the editor picks."""
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.abspath(os.path.join(HERE, '..'))
if CORE not in sys.path:
    sys.path.insert(0, CORE)

import ai_analysis as aa  # noqa: E402

TRANSCRIPT = "[00:00:01] SPEAKER_00: We started in a garage.\n[00:00:09] SPEAKER_01: And nearly went broke."
DIRECTIVE = "\n\nWrite every description in Norwegian."


def _capture(monkeypatch):
    calls = []

    def fake_call(prompt, system_prompt='', **kw):
        calls.append((system_prompt, prompt))
        return '{}'
    monkeypatch.setattr(aa, '_call_ai', fake_call)
    return calls


def test_the_four_chunk_calls_share_one_prefix(monkeypatch):
    calls = _capture(monkeypatch)
    aa._analyze_story_soundbites(TRANSCRIPT, 'Founder interview', 7, language_directive_text=DIRECTIVE)
    aa._analyze_story_beats(TRANSCRIPT, 'Founder interview', 7, language_directive_text=DIRECTIVE)
    aa._analyze_story_overview(TRANSCRIPT, 'Founder interview', language_directive_text=DIRECTIVE)
    aa._analyze_social(TRANSCRIPT, 'Founder interview', clips_target=7, language_directive_text=DIRECTIVE)
    # The fake reply is empty, so each builder also fires its rare retry;
    # judge the first (normal) call of each of the four.
    calls = [c for c in calls if 'Re-analyze' not in c[1] and 'NO MARKDOWN' not in c[1]]
    assert len(calls) == 4
    systems = {c[0] for c in calls}
    assert len(systems) == 1, 'one system prompt for every per-chunk call'
    head = aa._analysis_prompt_head('Founder interview', TRANSCRIPT)
    assert all(c[1].startswith(head) for c in calls), 'every call opens with the same project + transcript'
    # Each call's own instructions and the language directive come AFTER the transcript.
    for system, prompt in calls:
        assert 'Norwegian' not in system
        assert prompt.rstrip().endswith('Write every description in Norwegian.')
        assert prompt.index('TRANSCRIPT:') < prompt.index('Return')
    # The four asks are still distinct.
    tails = {p[len(head):][:60] for _, p in calls}
    assert len(tails) == 4


def test_shared_num_ctx_grows_once_and_respects_the_ceiling(monkeypatch):
    monkeypatch.setattr(aa, '_get_ollama_model', lambda: 'gemma4:e4b')
    aa._SHARED_NUM_CTX_HWM.clear()
    small = [{'content': 'x' * 2000}]
    big = [{'content': 'x' * int(2.8 * 27000)}]
    assert aa._shared_num_ctx('sys', small, 32768) == 8192
    assert aa._shared_num_ctx('sys', big, 32768) == 32768
    # Grow-only: a small payload after a big one keeps the rung (no reload).
    assert aa._shared_num_ctx('sys', small, 32768) == 32768
    # The machine ceiling always wins.
    assert aa._shared_num_ctx('sys', big, 16384) == 16384
    # A different model starts its own mark.
    monkeypatch.setattr(aa, '_get_ollama_model', lambda: 'gemma4:26b')
    assert aa._shared_num_ctx('sys', small, 32768) == 8192


def test_analysis_and_chat_converge_on_one_rung(monkeypatch):
    monkeypatch.setattr(aa, '_get_ollama_model', lambda: 'gemma4:e4b')
    aa._SHARED_NUM_CTX_HWM.clear(); aa._NUM_CTX_HWM.clear()
    sent = {}

    class _Prov:
        def generate(self, system_prompt, prompt, **kw):
            sent.update(kw); return '{}'
    fake_providers = types.ModuleType('ai_providers')
    fake_providers.get_active_provider = lambda model_resolver=None: _Prov()
    monkeypatch.setitem(sys.modules, 'ai_providers', fake_providers)
    aa._call_ai('x' * int(2.8 * 15000), 'sys')          # a 15-minute chunk
    analysis_ctx = sent['num_ctx']
    assert analysis_ctx in (24576, 32768)
    chat_ctx = aa._sticky_chat_num_ctx('Founder interview', 'sys', [{'content': 'short question'}])
    assert chat_ctx == analysis_ctx, 'chat after analysis reuses the analysis rung'
