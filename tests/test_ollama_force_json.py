"""Tests for the Ollama analysis free-form fallback (force_json).

Larger local models (gemma4:26b/31b) can degenerate under the strict
format='json' decoding grammar and return an empty / "{}" body, which made
the whole analysis come back blank ("Analysis incomplete: the AI model
returned an unexpected response"). The fix: the per-pass retry drops the
grammar (force_json=False) so the model answers in free form and the
tolerant parser salvages the JSON.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_providers.ollama_provider import OllamaProvider  # noqa: E402


class _FakeResp:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload


def _capture_post(captured, response_text=""):
    def fake_post(url, json=None, timeout=None, **kwargs):
        captured.append(json)
        return _FakeResp({"response": response_text})
    return fake_post


def test_analysis_default_uses_json_grammar(monkeypatch):
    import ai_providers.ollama_provider as op
    captured = []
    monkeypatch.setattr(op.requests, "post", _capture_post(captured, '{"ok":1}'))
    provider = OllamaProvider(model_resolver=lambda: "gemma4:e4b")

    provider.generate("sys", "prompt", task_type="analysis")

    body = captured[-1]
    assert body["format"] == "json", "default analysis call must use format='json'"
    assert "stop" not in body["options"]


def test_analysis_force_json_false_drops_grammar_and_adds_stops(monkeypatch):
    import ai_providers.ollama_provider as op
    captured = []
    monkeypatch.setattr(op.requests, "post", _capture_post(captured, 'here you go: {"ok":1}'))
    provider = OllamaProvider(model_resolver=lambda: "gemma4:26b")

    out = provider.generate("sys", "prompt", task_type="analysis", force_json=False)

    body = captured[-1]
    assert "format" not in body, "free-form fallback must NOT send format='json'"
    assert body["options"]["stop"] == op.DEFAULT_STOP_TOKENS
    # The raw free-form text is returned verbatim — the caller's parser
    # is responsible for extracting the JSON from any prose wrapper.
    assert out == 'here you go: {"ok":1}'


def test_story_soundbites_recovers_via_freeform_retry(monkeypatch):
    """End-to-end at the analysis layer: the first (grammar) call blanks
    out like gemma4:26b does; the free-form retry returns usable JSON and
    the pass recovers real soundbites instead of warning."""
    import ai_analysis

    calls = []

    def fake_call_ai(prompt, system_prompt="", task_type="analysis", force_json=True):
        calls.append(force_json)
        if force_json:
            return "{}"  # degenerate empty body, as the large model emits
        # Free-form retry: model answers with JSON wrapped in prose.
        return (
            'Sure, here are the soundbites:\n'
            '{"strongest_soundbites": [{"text": "a real quote", '
            '"start": "00:00:05", "end": "00:00:09", "why": "thesis"}]}'
        )

    monkeypatch.setattr(ai_analysis, "_call_ai", fake_call_ai)

    result = ai_analysis._analyze_story_soundbites("transcript", "Proj", 7)

    assert calls == [True, False], "should try grammar first, then free-form"
    sb = result.get("strongest_soundbites")
    assert sb and sb[0]["text"] == "a real quote"
