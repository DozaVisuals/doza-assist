"""Studio tier gating in the OSS core.

The wrapper resolves entitlement + the feature registry and passes the result
as DOZA_TIER / DOZA_FEATURES. The core must:

  - default to tier 'core' with NO features when the env is absent (OSS,
    older wrappers, Pro builds that resolved everything off), so nothing
    Studio-only renders;
  - expose feature_on() / doza_tier to templates exactly as given.
"""
import os
import sys

import pytest
from flask import render_template_string

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module

TPL = "{{ doza_tier }}|{{ 'ON' if feature_on('studio.panel') else 'off' }}|{{ 'RO' if doza_studio_readonly else 'rw' }}|{{ doza_upgrade_url }}"


def _render(monkeypatch, **env):
    for k in ('DOZA_TIER', 'DOZA_FEATURES', 'DOZA_STUDIO_READONLY', 'DOZA_UPGRADE_URL'):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with app_module.app.test_request_context('/'):
        return render_template_string(TPL)


def test_absent_env_is_core_with_nothing_on(monkeypatch):
    out = _render(monkeypatch)
    assert out.startswith('core|off|rw|')
    assert out.endswith('https://doza.ai/buy?plan=studio')


def test_pro_tier_resolves_only_what_the_wrapper_listed(monkeypatch):
    out = _render(monkeypatch, DOZA_TIER='pro', DOZA_FEATURES='upgrade.cta')
    assert out.startswith('pro|off|rw|')


def test_studio_active_turns_listed_features_on(monkeypatch):
    out = _render(monkeypatch, DOZA_TIER='studio_active',
                  DOZA_FEATURES='studio.panel,share.publish', DOZA_UPGRADE_URL='https://doza.ai/buy?plan=x')
    assert out == 'studio_active|ON|rw|https://doza.ai/buy?plan=x'


def test_lapsed_marks_readonly(monkeypatch):
    out = _render(monkeypatch, DOZA_TIER='studio_lapsed', DOZA_FEATURES='studio.panel', DOZA_STUDIO_READONLY='1')
    assert out.startswith('studio_lapsed|ON|RO|')


def test_feature_names_are_exact_not_prefix(monkeypatch):
    out = _render(monkeypatch, DOZA_TIER='studio_active', DOZA_FEATURES='studio.panel.extra, studio.settings')
    assert '|off|' in out


@pytest.mark.parametrize('page', ['/'])
def test_dashboard_renders_without_studio_markers_by_default(monkeypatch, tmp_path, page):
    """No Studio DOM markers leak into Pro/OSS renders."""
    for k in ('DOZA_TIER', 'DOZA_FEATURES'):
        monkeypatch.delenv(k, raising=False)
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    os.makedirs(app_module.app.config['PROJECTS_DIR'], exist_ok=True)
    app_module.app.config['TESTING'] = True
    html = app_module.app.test_client().get(page).get_data(as_text=True)
    assert 'data-studio' not in html
    assert 'studio-panel' not in html
