"""Tests for the Ollama connection-refused recovery helper.

The bundled Ollama can die mid-batch (typically an OOM on a heavy model load
on a lower-RAM machine). The Electron supervisor relaunches it on the same
port, but that leaves a few-second window where the socket is refused. The
``_post_with_reconnect`` helper retries connection errors (and ONLY connection
errors) across that window so an in-flight analyze call doesn't fail the whole
interview with "[Errno 61] Connection refused".
"""
import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_providers import ollama_provider  # noqa: E402


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    # Don't actually wait out the backoff in tests.
    monkeypatch.setattr(ollama_provider.time, "sleep", lambda *_: None)


class _Resp:
    status_code = 200


def test_retries_then_succeeds(monkeypatch):
    # Ollama is refused twice (still restarting), then comes back.
    calls = {"n": 0}

    def fake_post(url, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise requests.exceptions.ConnectionError("refused")
        return _Resp()

    monkeypatch.setattr(ollama_provider.requests, "post", fake_post)
    resp = ollama_provider._post_with_reconnect("http://x/api/generate", json={}, timeout=180)
    assert resp.status_code == 200
    assert calls["n"] == 3, "should retry past the transient refusals"


def test_gives_up_after_budget_and_reraises(monkeypatch):
    # Ollama never comes back — re-raise the connection error (don't hang).
    calls = {"n": 0}

    def always_refused(url, **kwargs):
        calls["n"] += 1
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(ollama_provider.requests, "post", always_refused)
    with pytest.raises(requests.exceptions.ConnectionError):
        ollama_provider._post_with_reconnect("http://x/api/generate", json={}, timeout=180)
    # 1 initial attempt + len(_CONNECT_BACKOFF) retries.
    assert calls["n"] == len(ollama_provider._CONNECT_BACKOFF) + 1


def test_does_not_retry_read_timeout(monkeypatch):
    # A read timeout is the model being slow, not Ollama being down — that's
    # the caller's model-aware timeout's job. Retrying would multiply a long
    # wait. It must propagate on the first attempt.
    calls = {"n": 0}

    def slow(url, **kwargs):
        calls["n"] += 1
        raise requests.exceptions.ReadTimeout("read timed out")

    monkeypatch.setattr(ollama_provider.requests, "post", slow)
    with pytest.raises(requests.exceptions.ReadTimeout):
        ollama_provider._post_with_reconnect("http://x/api/generate", json={}, timeout=180)
    assert calls["n"] == 1, "read timeouts must not be retried"


def test_succeeds_first_try_no_retry(monkeypatch):
    calls = {"n": 0}

    def ok(url, **kwargs):
        calls["n"] += 1
        return _Resp()

    monkeypatch.setattr(ollama_provider.requests, "post", ok)
    resp = ollama_provider._post_with_reconnect("http://x/api/chat", json={}, timeout=300)
    assert resp.status_code == 200
    assert calls["n"] == 1
