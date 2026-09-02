"""NLE app discovery: every Final Cut SKU must be findable.

Since Jan 2026 Final Cut Pro exists as two coexisting Mac apps — the
one-time-purchase SKU (com.apple.FinalCut) and the Apple Creator Studio
subscription copy, a separate installation with its own bundle id. Discovery
therefore checks the exact purchase id first (a full install always wins) and
falls back to a com.apple.FinalCut* Spotlight glob so subscription-only and
trial-only editors still get "Send to Final Cut Pro" instead of a
"not installed" error.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(app_module, "_nle_path_cache", {})


def test_fcp_ids_exact_then_glob():
    ids = app_module._NLE_BUNDLE_IDS["fcp"]
    assert ids[0] == "com.apple.FinalCut"
    assert "com.apple.FinalCut*" in ids
    assert ids.index("com.apple.FinalCut") < ids.index("com.apple.FinalCut*")


def test_exact_hit_short_circuits_glob(monkeypatch):
    queried = []

    def fake_mdfind(bundle_id):
        queried.append(bundle_id)
        if bundle_id == "com.apple.FinalCut":
            return ["/Applications/Final Cut Pro.app"]
        return []

    monkeypatch.setattr(app_module, "_mdfind_app_by_bundle_id", fake_mdfind)
    assert app_module._find_nle_app_path("fcp") == "/Applications/Final Cut Pro.app"
    assert queried == ["com.apple.FinalCut"]  # glob never needed


def test_subscription_only_found_via_glob(monkeypatch):
    def fake_mdfind(bundle_id):
        # No purchase SKU installed; the glob finds the Creator Studio copy
        # (whatever Apple named its id — the glob doesn't care).
        if bundle_id == "com.apple.FinalCut*":
            return ["/Applications/Final Cut Pro.app"]
        return []

    monkeypatch.setattr(app_module, "_mdfind_app_by_bundle_id", fake_mdfind)
    monkeypatch.setattr(app_module, "_app_short_version", lambda p: (12, 3))
    assert app_module._find_nle_app_path("fcp") == "/Applications/Final Cut Pro.app"


def test_glob_hits_are_ranked(monkeypatch):
    def fake_mdfind(bundle_id):
        if bundle_id == "com.apple.FinalCut*":
            # Stray copy outside /Applications must lose to the real install.
            return ["/Users/x/Applications/SpliceKit/Final Cut Pro.app",
                    "/Applications/Final Cut Pro.app"]
        return []

    monkeypatch.setattr(app_module, "_mdfind_app_by_bundle_id", fake_mdfind)
    monkeypatch.setattr(app_module, "_app_short_version", lambda p: (12, 3))
    assert app_module._find_nle_app_path("fcp") == "/Applications/Final Cut Pro.app"


def test_nothing_installed_falls_to_known_paths(monkeypatch):
    monkeypatch.setattr(app_module, "_mdfind_app_by_bundle_id", lambda b: [])
    monkeypatch.setattr(os.path, "isdir",
                        lambda p: p == "/Applications/Final Cut Pro Trial.app")
    assert app_module._find_nle_app_path("fcp") == "/Applications/Final Cut Pro Trial.app"
