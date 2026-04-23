"""The AI Model modal shows per-variant estimates. When opened from a
project page it should surface full-analysis times scaled to that project's
transcript length; when opened from the dashboard it should fall back to a
generic per-call estimate labelled accordingly.

These tests lock in:
  - ``GET /api/ai-model/status`` with no project_id returns generic estimates
    and no ``project_context``.
  - ``GET /api/ai-model/status?project_id=X`` returns full-analysis estimates
    scaled to X's transcript and a populated ``project_context`` block the
    frontend uses to swap the chip label.
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


def _seed_project(client, name, duration_seconds):
    tmp_audio = Path(app_module.app.config["PROJECTS_DIR"]).parent / f"{name}.wav"
    tmp_audio.write_bytes(b"RIFF____WAVE")
    resp = client.post("/create", json={"source_path": str(tmp_audio), "project_name": name})
    pid = resp.get_json()["project_id"]
    meta_path = Path(app_module.app.config["PROJECTS_DIR"]) / pid / "meta.json"
    meta = json.loads(meta_path.read_text())
    # Fake a transcript whose last segment ends at duration_seconds.
    meta["transcript"] = {
        "segments": [
            {"start": 0, "end": duration_seconds / 2, "text": "first", "speaker": "A"},
            {"start": duration_seconds / 2, "end": duration_seconds, "text": "last", "speaker": "A"},
        ],
        "language": "en",
    }
    meta_path.write_text(json.dumps(meta))
    return pid


class TestStatusGeneric:
    def test_no_project_context(self, client):
        resp = client.get("/api/ai-model/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["project_context"] is None
        # Without project scope the casual estimate ends in "per call".
        small = next(v for v in data["variants"] if v["tier"] == "small")
        assert small["estimated_casual"].endswith("per call")


class TestStatusScopedToProject:
    def test_long_project_scales_estimates_up(self, client):
        pid = _seed_project(client, "Long", duration_seconds=100 * 60)
        resp = client.get(f"/api/ai-model/status?project_id={pid}")
        assert resp.status_code == 200
        data = resp.get_json()
        ctx = data["project_context"]
        assert ctx is not None
        assert ctx["project_id"] == pid
        assert ctx["project_name"] == "Long"
        assert ctx["duration_seconds"] == pytest.approx(100 * 60)

        # 100-min interview → ~7 chunks × 2 calls — casual estimates should
        # read in minutes for every variant (even the fastest) and be tagged
        # "for this project".
        for v in data["variants"]:
            est = v["estimated_casual"]
            assert "min" in est, f"{v['tier']} was {est}"
            assert est.endswith("for this project"), f"{v['tier']} was {est}"

    def test_short_project_estimate_contains_time_unit(self, client):
        pid = _seed_project(client, "Short", duration_seconds=300)  # 5 min
        resp = client.get(f"/api/ai-model/status?project_id={pid}")
        data = resp.get_json()
        small = next(v for v in data["variants"] if v["tier"] == "small")
        # Casual phrase is lowercase, starts with "About", and ends with the
        # project tag. Exact value depends on speed-table tps so don't pin it.
        est = small["estimated_casual"]
        assert est.startswith("About ")
        assert est.endswith("for this project")
        assert "sec" in est or "min" in est

    def test_variant_has_display_name_and_total_params(self, client):
        """Locks in the new card-friendly fields the modal renders."""
        pid = _seed_project(client, "Any", duration_seconds=600)
        resp = client.get(f"/api/ai-model/status?project_id={pid}")
        data = resp.get_json()
        for v in data["variants"]:
            assert v["display_name"], f"{v['tier']} missing display_name"
            # Line 1 must carry a parenthetical descriptor.
            assert "(" in v["display_name"] and ")" in v["display_name"]
            # Every tier, including dense 27B/32B, must emit a "Nn total" string.
            assert v["total_params"].endswith("total"), (
                f"{v['tier']} total_params was {v['total_params']!r}"
            )

    def test_unknown_project_falls_back_to_generic(self, client):
        resp = client.get("/api/ai-model/status?project_id=doesnotexist")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["project_context"] is None

    def test_project_without_transcript_is_generic(self, client, tmp_path):
        # Create a project but don't seed a transcript.
        tmp_audio = tmp_path / "no_transcript.wav"
        tmp_audio.write_bytes(b"RIFF____WAVE")
        resp = client.post("/create", json={"source_path": str(tmp_audio), "project_name": "NoTx"})
        pid = resp.get_json()["project_id"]
        resp = client.get(f"/api/ai-model/status?project_id={pid}")
        data = resp.get_json()
        assert data["project_context"] is None


