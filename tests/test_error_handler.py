"""Tests for the global ``@app.errorhandler(Exception)`` handler.

The handler at ``app._handle_provider_error`` is on the hot path for every
unhandled exception in every route. Two regression guards live here:

1. ``ProviderError`` instances must come back as JSON 400, not 500 — that's
   what surfaces the "fix your API key" UI in the frontend.

2. Every other exception type must be re-raised with its identity intact,
   so Flask's normal 500 logging records the ORIGINAL traceback. The
   handler must NEVER swallow or substitute the exception — that was the
   bug in issue #25, where a lazy ``from ai_providers import ProviderError``
   inside the handler raised ``ModuleNotFoundError`` and masked the real
   root cause in ``server.log``.
"""

import os
import sys
import tempfile

import pytest


@pytest.fixture(scope='module')
def app_module():
    # Point DOZA_DATA_DIR at a tempdir so importing app.py doesn't create
    # projects/ / exports/ inside the repo.
    tmp = tempfile.mkdtemp(prefix='doza-test-handler-')
    os.environ['DOZA_DATA_DIR'] = tmp
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import app
    return app


def test_provider_error_returns_400_json(app_module):
    """A ProviderError flowing through the handler becomes JSON 400."""
    with app_module.app.test_request_context():
        result = app_module._handle_provider_error(
            app_module.ProviderError('missing key', code='no_key')
        )
    resp, status = result
    assert status == 400
    body = resp.get_json()
    assert body['code'] == 'no_key'
    assert body['error'] == 'missing key'
    assert body['settings_url'] == '/settings'


def test_non_provider_error_is_reraised(app_module):
    """Any non-ProviderError exception must propagate unchanged.

    The class identity AND the message must survive so Flask's normal
    error logger records the real traceback.
    """
    original = ValueError('original-error-marker-xyz')
    with app_module.app.test_request_context():
        with pytest.raises(ValueError) as excinfo:
            app_module._handle_provider_error(original)
    # Same instance, same message — handler did NOT substitute a new error.
    assert excinfo.value is original
    assert 'original-error-marker-xyz' in str(excinfo.value)


def test_provider_error_class_imported_at_top_level(app_module):
    """Guard against the lazy-import regression from issue #25.

    ``ProviderError`` must be a module-level attribute of ``app`` so that
    a missing ``ai_providers`` package causes server startup to fail
    loudly (in server.log, before Flask binds the port), rather than
    starting Flask and masking every 500 response with a secondary
    ``ModuleNotFoundError`` from inside the handler.
    """
    assert hasattr(app_module, 'ProviderError'), (
        "ProviderError must be imported at the top of app.py — "
        "see issue #25 for why."
    )
    from ai_providers import ProviderError as PE
    assert app_module.ProviderError is PE
