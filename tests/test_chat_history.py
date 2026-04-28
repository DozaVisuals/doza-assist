"""Persistence tests for the AI Chat conversation history.

Verifies that POSTing to /project/<id>/chat writes the user message + model
reply into meta.json, and that DELETE /project/<id>/chat wipes it.

The chat route calls ``chat_about_transcript`` which hits Ollama → Claude.
We monkeypatch it to return a fixed reply so these tests are offline and
deterministic.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setitem(app_module.app.config, "PROJECTS_DIR", str(tmp_path / "projects"))
    os.makedirs(app_module.app.config["PROJECTS_DIR"], exist_ok=True)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


@pytest.fixture
def transcribed_project(client, tmp_path, monkeypatch):
    """Spin up a project with a minimal transcript so the chat route accepts it."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFF____WAVE")
    resp = client.post("/create", json={
        "source_path": str(audio),
        "project_name": "Chat Test",
    })
    assert resp.status_code == 200, resp.data
    pid = resp.get_json()["project_id"]

    # Inject a tiny transcript directly so the chat route's transcript check passes.
    meta_path = Path(app_module.app.config["PROJECTS_DIR"]) / pid / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["transcript"] = {
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "Hello world.", "speaker": "A"},
        ],
        "language": "en",
    }
    meta_path.write_text(json.dumps(meta))

    # Stub the AI call — this test is about persistence, not model output.
    def _fake_chat(transcript, message, history=None, project_name="", analysis=None,
                   profile_id=None, segment_vectors=None, paragraph_index=None):
        return f"You said: {message}"

    monkeypatch.setattr("ai_analysis.chat_about_transcript", _fake_chat)
    return pid


def _read_meta(pid):
    meta_path = Path(app_module.app.config["PROJECTS_DIR"]) / pid / "meta.json"
    return json.loads(meta_path.read_text())


class TestChatHistoryPersists:
    def test_fresh_project_has_no_chat_history(self, client, transcribed_project):
        meta = _read_meta(transcribed_project)
        assert meta.get("chat_history", []) == []

    def test_single_turn_saves_both_messages(self, client, transcribed_project):
        resp = client.post(
            f"/project/{transcribed_project}/chat",
            json={"message": "What's the emotional arc?", "history": []},
        )
        assert resp.status_code == 200, resp.data
        assert resp.get_json()["reply"] == "You said: What's the emotional arc?"

        meta = _read_meta(transcribed_project)
        history = meta.get("chat_history", [])
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "What's the emotional arc?"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "You said: What's the emotional arc?"
        # Timestamps are added but optional to the caller.
        assert "ts" in history[0]
        assert "ts" in history[1]

    def test_multiple_turns_append(self, client, transcribed_project):
        for msg in ("first", "second", "third"):
            resp = client.post(
                f"/project/{transcribed_project}/chat",
                json={"message": msg, "history": []},
            )
            assert resp.status_code == 200

        meta = _read_meta(transcribed_project)
        history = meta["chat_history"]
        # 3 turns × 2 messages each.
        assert len(history) == 6
        user_contents = [m["content"] for m in history if m["role"] == "user"]
        assert user_contents == ["first", "second", "third"]

    def test_delete_clears_history(self, client, transcribed_project):
        client.post(
            f"/project/{transcribed_project}/chat",
            json={"message": "hello", "history": []},
        )
        assert len(_read_meta(transcribed_project)["chat_history"]) == 2

        resp = client.delete(f"/project/{transcribed_project}/chat")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "cleared"

        assert _read_meta(transcribed_project)["chat_history"] == []

    def test_delete_on_missing_project_is_404(self, client):
        resp = client.delete("/project/nonexistent/chat")
        assert resp.status_code == 404


class TestChatHistoryMultiProject:
    """Multi-project chat should NOT persist (ownership is ambiguous and we
    don't want to fork writes across multiple meta.json files)."""

    def test_multi_project_chat_does_not_persist(self, client, tmp_path, monkeypatch):
        # Create two projects with transcripts.
        pids = []
        for name in ("A", "B"):
            audio = tmp_path / f"{name}.wav"
            audio.write_bytes(b"RIFF____WAVE")
            resp = client.post("/create", json={
                "source_path": str(audio),
                "project_name": name,
            })
            pid = resp.get_json()["project_id"]
            meta_path = Path(app_module.app.config["PROJECTS_DIR"]) / pid / "meta.json"
            meta = json.loads(meta_path.read_text())
            meta["transcript"] = {
                "segments": [{"start": 0, "end": 1, "text": name, "speaker": "A"}],
                "language": "en",
            }
            meta_path.write_text(json.dumps(meta))
            pids.append(pid)

        def _fake_chat(transcript, message, history=None, project_name="", analysis=None,
                   profile_id=None, segment_vectors=None, paragraph_index=None):
            return "multi reply"
        monkeypatch.setattr("ai_analysis.chat_about_transcript", _fake_chat)

        combined = ",".join(pids)
        resp = client.post(
            f"/project/{combined}/chat",
            json={"message": "hi", "history": []},
        )
        assert resp.status_code == 200

        for pid in pids:
            meta = _read_meta(pid)
            # Chat history stays empty (or absent) on every project involved.
            assert not meta.get("chat_history")
