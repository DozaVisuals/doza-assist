"""Fixes from the 2026-09-08 observed test session: capped analysis output
is detected and retried, an oversized chat payload answers from retrieval
instead of losing the transcript, /health exists, and analyze chunk prompts
share a stable head."""
import json
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.abspath(os.path.join(HERE, '..'))
if CORE not in sys.path:
    sys.path.insert(0, CORE)


def test_health_route_is_live():
    os.environ.setdefault('OLLAMA_HOST', 'http://127.0.0.1:1')
    import app as core_app
    client = core_app.app.test_client()
    r = client.get('/health')
    assert r.status_code == 200
    assert r.get_json()['ok'] is True and r.get_json()['ready'] is True


def test_chat_payload_overflow_detection(monkeypatch):
    import ai_analysis as aa
    small = [{'role': 'user', 'content': 'hello'}]
    assert aa._chat_payload_overflows('system', small) is False
    huge = [{'role': 'user', 'content': 'x' * (2.8 * 40000).__int__()}]
    assert aa._chat_payload_overflows('system', huge) is True
    # The machine's chat ceiling lowers the bar (8 GB machines run at 16K).
    mb = types.ModuleType('memory_budget')
    mb.chat_num_ctx_ceiling = lambda: 16384
    monkeypatch.setitem(sys.modules, 'memory_budget', mb)
    mid = [{'role': 'user', 'content': 'x' * int(2.8 * 14000)}]
    assert aa._chat_payload_overflows('system', mid) is True


def test_capped_analysis_output_is_retried_with_a_larger_cap(monkeypatch):
    from ai_providers import ollama_provider as op
    calls = []

    class _Resp:
        def __init__(self, body):
            self.status_code = 200
            self._body = body
        def json(self):
            return self._body

    def fake_post(url, json=None, timeout=None):
        calls.append(json['options']['num_predict'])
        if len(calls) == 1:
            return _Resp({'response': '{"beats": [', 'done_reason': 'length',
                          'eval_count': 8192, 'eval_duration': 1e9,
                          'prompt_eval_count': 100, 'prompt_eval_duration': 1e8})
        return _Resp({'response': '{"beats": []}', 'done_reason': 'stop',
                      'eval_count': 12, 'eval_duration': 1e8,
                      'prompt_eval_count': 100, 'prompt_eval_duration': 1e8})
    monkeypatch.setattr(op, '_post_with_reconnect', fake_post)
    prov = op.OllamaProvider.__new__(op.OllamaProvider)
    prov.base_url = 'http://127.0.0.1:1'
    prov.model = 'gemma4:e4b'
    monkeypatch.setattr(prov, '_resolve_model', lambda override=None: 'gemma4:e4b', raising=False)
    out = prov.generate('sys', 'prompt', task_type='analysis')
    assert out == '{"beats": []}'
    assert calls == [8192, 16384]


def test_capped_output_is_logged(capsys):
    from ai_providers import ollama_provider as op
    op._log_timing('analysis', 'gemma4:e4b', 32768,
                   {'prompt_eval_count': 10, 'prompt_eval_duration': 1e8,
                    'eval_count': 4096, 'eval_duration': 1e9, 'done_reason': 'length'})
    out = capsys.readouterr().out
    assert 'done_reason=length' in out and '[ai-warn]' in out


def test_analyze_chunks_keep_a_stable_prompt_head():
    src = open(os.path.join(CORE, 'ai_analysis.py')).read()
    assert 'chunk_label = project_name' in src
    assert '[Part {i+1} of {len(chunks)} of this interview' in src
    assert "· part {i+1}/{len(chunks)}" not in src
