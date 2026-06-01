"""Tests for the global error handler (_handle_provider_error).

It's registered as @app.errorhandler(Exception), so it catches *everything*
— including Werkzeug HTTPExceptions. It used to ``raise e`` for anything
that wasn't a ProviderError, which turned a plain 404 (a poll of a missing
route, e.g. the setup assistant's /api/status hitting the main app after
setup) into a 500 + full traceback in the log. Real HTTP errors must keep
their own status; only genuinely unexpected errors should 500.
"""

import os
import sys

import pytest
from werkzeug.exceptions import NotFound, MethodNotAllowed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module  # noqa: E402
from ai_providers import ProviderError  # noqa: E402


@pytest.fixture
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def test_missing_route_returns_clean_404_not_500(client):
    # The bug this guards: a missing route was logged as a 500 with a scary
    # traceback. It must come back as a plain 404.
    assert client.get('/api/status').status_code == 404
    assert client.get('/definitely/not/a/real/route').status_code == 404


def test_handler_preserves_http_exception_status():
    # HTTPExceptions are returned (with their own status), not re-raised.
    assert app_module._handle_provider_error(NotFound()).code == 404
    assert app_module._handle_provider_error(MethodNotAllowed()).code == 405


def test_handler_returns_400_for_provider_error():
    with app_module.app.app_context():
        body, status = app_module._handle_provider_error(
            ProviderError('bad key', code='invalid_key')
        )
    assert status == 400
    assert body.get_json()['code'] == 'invalid_key'


def test_handler_reraises_genuine_errors():
    # A real bug must still surface as a 500 (Flask logs the re-raised error).
    with pytest.raises(ValueError):
        app_module._handle_provider_error(ValueError('boom'))
