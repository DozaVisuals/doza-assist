"""Tests for the model-aware analysis HTTP timeout.

The old fixed 180s ceiling in the Ollama provider cut off the large local
models (gemma4:26b/31b ~7 tok/s) mid-generation, so analyses errored with
"Read timed out (180)" and the collection dashboard came back empty.
``recommended_analysis_timeout`` sizes the ceiling to the active variant +
hardware so the big models get the minutes they need while the small ones
stay at the fast 180s floor.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import model_config  # noqa: E402


_XHIGH_APPLE = {"arch": "arm64", "ram_gb": 96.0}


def _timeout_for_tier(monkeypatch, tier):
    monkeypatch.setattr(model_config, "load_model_config", lambda: {"tier": tier})
    return model_config.recommended_analysis_timeout(hw_info=dict(_XHIGH_APPLE))


def test_small_model_stays_near_floor(monkeypatch):
    # gemma4:e4b is fast — its calls finished fine under 180s, so the ceiling
    # should stay at the 180s floor (no behavior change for the default model).
    assert _timeout_for_tier(monkeypatch, "medium") == 180


def test_large_model_gets_minutes(monkeypatch):
    # gemma4:31b (xlarge) is ~7 tok/s on this hardware — it needs several
    # minutes, well above the old 180s that was killing it.
    t = _timeout_for_tier(monkeypatch, "xlarge")
    assert t > 600, f"expected a multi-minute ceiling for the 32B model, got {t}s"


def test_timeout_is_clamped_to_band(monkeypatch):
    # Even on a pathologically slow profile the ceiling never exceeds the cap,
    # and never drops below the floor.
    for tier in ("small", "medium", "large", "xlarge"):
        t = _timeout_for_tier(monkeypatch, tier)
        assert 180 <= t <= 1200


def test_larger_model_never_shorter_than_smaller(monkeypatch):
    # Monotonic: a slower (bigger) variant must get at least as much time.
    timeouts = [_timeout_for_tier(monkeypatch, tier)
                for tier in ("small", "medium", "large", "xlarge")]
    assert timeouts == sorted(timeouts)


def test_bad_config_falls_back_to_medium(monkeypatch):
    # A broken/missing model config must not raise — default to the balanced
    # tier's timeout rather than crashing the analysis call.
    def _boom():
        raise RuntimeError("no config")
    monkeypatch.setattr(model_config, "load_model_config", _boom)
    t = model_config.recommended_analysis_timeout(hw_info=dict(_XHIGH_APPLE))
    assert 180 <= t <= 1200
