"""Tests for POST /api/ai-model/pull — the in-UI Gemma download endpoint.

The route streams Ollama's /api/pull event stream to the client so the AI
Model modal can render a progress bar. These tests monkeypatch the outgoing
``requests.post`` to Ollama so they run offline.
"""

import json
import os
import sys
from io import BytesIO

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module  # noqa: E402


class _FakeStreamingResponse:
    """Minimal stand-in for a streaming ``requests`` response.

    Emits bytes by default — Ollama's /api/pull response has no charset
    header, so ``requests.iter_lines(decode_unicode=True)`` returns bytes in
    practice. This mirrors that so regressions aren't masked by a too-forgiving
    stub.
    """

    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code

    def iter_lines(self, decode_unicode=False):
        for line in self._lines:
            # Mimic real upstream: hand back bytes regardless of the flag.
            yield line.encode('utf-8') if isinstance(line, str) else line


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setitem(app_module.app.config, "PROJECTS_DIR", str(tmp_path / "projects"))
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


class TestAIModelPull:
    def test_rejects_invalid_tier(self, client):
        resp = client.post("/api/ai-model/pull", json={"tier": "bogus"})
        assert resp.status_code == 400
        assert "invalid tier" in resp.get_json()["error"]

    def test_streams_ollama_pull_events(self, client, monkeypatch):
        pull_events = [
            '{"status":"pulling manifest"}',
            '{"status":"downloading","digest":"abc123","total":1000,"completed":250}',
            '{"status":"downloading","digest":"abc123","total":1000,"completed":750}',
            '{"status":"verifying sha256"}',
            '{"status":"success"}',
        ]
        captured = {}

        def _fake_post(url, json=None, stream=None, timeout=None):
            captured["url"] = url
            captured["body"] = json
            return _FakeStreamingResponse(pull_events)

        monkeypatch.setattr("requests.post", _fake_post)

        resp = client.post("/api/ai-model/pull", json={"tier": "small"})
        assert resp.status_code == 200
        assert resp.mimetype == "application/x-ndjson"

        body = resp.get_data(as_text=True)
        lines = [line for line in body.strip().split("\n") if line]
        assert len(lines) == len(pull_events)
        events = [__import__("json").loads(line) for line in lines]
        assert events[0]["status"] == "pulling manifest"
        assert events[1]["completed"] == 250
        assert events[-1]["status"] == "success"

        # Correct Ollama endpoint + variant from the tier mapping.
        assert captured["url"] == "http://localhost:11434/api/pull"
        assert captured["body"]["name"] == "gemma4:e2b"  # small tier
        assert captured["body"]["stream"] is True

    def test_connection_error_produces_friendly_terminus(self, client, monkeypatch):
        import requests

        def _fake_post(url, json=None, stream=None, timeout=None):
            raise requests.exceptions.ConnectionError("refused")

        monkeypatch.setattr("requests.post", _fake_post)

        resp = client.post("/api/ai-model/pull", json={"tier": "medium"})
        assert resp.status_code == 200  # streaming — 200 even on error
        body = resp.get_data(as_text=True)
        events = [__import__("json").loads(line) for line in body.strip().split("\n") if line]
        assert len(events) == 1
        assert events[0]["status"] == "error"
        assert "Ollama" in events[0]["message"]

    def test_upstream_non_200_reports_error(self, client, monkeypatch):
        def _fake_post(url, json=None, stream=None, timeout=None):
            return _FakeStreamingResponse([], status_code=500)

        monkeypatch.setattr("requests.post", _fake_post)

        resp = client.post("/api/ai-model/pull", json={"tier": "large"})
        body = resp.get_data(as_text=True)
        events = [__import__("json").loads(line) for line in body.strip().split("\n") if line]
        assert events[-1]["status"] == "error"
        assert "500" in events[-1]["message"]
