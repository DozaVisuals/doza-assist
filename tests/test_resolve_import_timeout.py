"""Regression: Resolve scripting calls must not block the request forever.

The Send-to-Resolve "beachball": ImportTimelineFromFile has no built-in
timeout, so a hung Resolve blocked the Flask request thread indefinitely.
_call_with_timeout caps any such call so the export always returns.
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exporters.resolve_import import _call_with_timeout


def test_returns_value_when_fast():
    assert _call_with_timeout(lambda x: x + 1, 41, timeout=5.0) == 42


def test_raises_timeout_when_slow():
    def _hang():
        time.sleep(5)
        return 'never'
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        _call_with_timeout(_hang, timeout=0.3)
    # Returned promptly at the cap, not after the full 5s hang.
    assert time.monotonic() - t0 < 2.0


def test_propagates_callee_exception():
    def _boom():
        raise ValueError('resolve said no')
    with pytest.raises(ValueError):
        _call_with_timeout(_boom, timeout=5.0)
