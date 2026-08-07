"""memory_budget governor: RAM tiers, policies, stage gate, mlx purge.

The governor exists because stacked AI components exhaust unified memory
on 8-16 GB Apple Silicon and corrupt the display compositor (FX tester,
M1 16 GB, macOS 15.7.8 — Metal kIOGPUCommandBufferCallbackErrorOutOfMemory
4s into Parakeet chunk 1 while gemma warmed and pyannote sat preloaded).
These tests pin the tier boundaries and the one-heavy-component-RESIDENT
contract (adversarial-review hardened: verified eviction, atomic prewarm
admission, bounded gate waits) so a future tweak can't silently regress
small-RAM machines.
"""

import os
import sys
import threading
import time
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import memory_budget  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_ram_class(monkeypatch):
    """Each test picks its own forced RAM; never leak the cached class."""
    memory_budget._reset_ram_class_cache_for_tests()
    yield
    monkeypatch.delenv('DOZA_FORCE_RAM_GB', raising=False)
    memory_budget._reset_ram_class_cache_for_tests()


def _force(monkeypatch, gb):
    monkeypatch.setenv('DOZA_FORCE_RAM_GB', str(gb))
    memory_budget._reset_ram_class_cache_for_tests()


# ── tier boundaries ──────────────────────────────────────────────────

@pytest.mark.parametrize('gb,expected', [
    (8, 'tight'),
    (11.99, 'tight'),
    (12, 'low'),
    (16, 'low'),
    (23.99, 'low'),
    (24, 'comfortable'),
    (32, 'comfortable'),
    (96, 'comfortable'),
])
def test_ram_class_boundaries(monkeypatch, gb, expected):
    _force(monkeypatch, gb)
    assert memory_budget.ram_class() == expected


def test_force_env_reaches_model_config(monkeypatch):
    _force(monkeypatch, 8)
    assert memory_budget.total_ram_gb() == 8.0


def test_unreadable_ram_degrades_to_comfortable(monkeypatch):
    monkeypatch.setattr(memory_budget, 'total_ram_gb',
                        lambda: (_ for _ in ()).throw(OSError('boom')))
    memory_budget._reset_ram_class_cache_for_tests()
    assert memory_budget.ram_class() == 'comfortable'


# ── policy matrix ────────────────────────────────────────────────────

def test_tight_policies(monkeypatch):
    _force(monkeypatch, 8)
    assert memory_budget.allow_boot_preload() is False
    assert memory_budget.allow_prewarm_now() is False
    assert memory_budget.release_models_after_use() is True
    assert memory_budget.whisper_model_prefs() == ('small', 'base')
    assert memory_budget.mlx_memory_limit_mb() == 2200
    assert memory_budget.parakeet_chunk_sec() == 30
    assert memory_budget.ollama_keep_alive() == '2m'
    assert memory_budget.chat_num_ctx_ceiling() == 16384
    assert memory_budget.analysis_num_ctx_ceiling() == 16384
    assert memory_budget.long_chat_seconds() == 20 * 60


def test_low_policies(monkeypatch):
    _force(monkeypatch, 16)
    assert memory_budget.allow_boot_preload() is False
    assert memory_budget.allow_prewarm_now() is True  # idle: prewarm OK
    assert memory_budget.release_models_after_use() is True
    assert memory_budget.whisper_model_prefs() == ('turbo', 'large-v3', 'base')
    assert memory_budget.mlx_memory_limit_mb() == 3584
    assert memory_budget.parakeet_chunk_sec() == 60
    assert memory_budget.ollama_keep_alive() == '10m'
    assert memory_budget.chat_num_ctx_ceiling() == 32768
    assert memory_budget.analysis_num_ctx_ceiling() == 32768
    assert memory_budget.long_chat_seconds() == 60 * 60


def test_comfortable_policies_preserve_today(monkeypatch):
    _force(monkeypatch, 32)
    assert memory_budget.allow_boot_preload() is True
    assert memory_budget.allow_prewarm_now() is True
    assert memory_budget.release_models_after_use() is False
    assert memory_budget.whisper_model_prefs() == ('turbo', 'large-v3', 'base')
    assert memory_budget.mlx_memory_limit_mb() == 0
    assert memory_budget.parakeet_chunk_sec() == 60
    assert memory_budget.ollama_keep_alive() == '30m'
    assert memory_budget.chat_num_ctx_ceiling() == 32768
    assert memory_budget.analysis_num_ctx_ceiling() == 32768
    assert memory_budget.long_chat_seconds() == 60 * 60


# ── heavy-stage gate ─────────────────────────────────────────────────

def test_heavy_stage_serializes_on_low(monkeypatch):
    _force(monkeypatch, 16)
    monkeypatch.setattr(memory_budget, 'evict_ollama_models', lambda **kw: 0)
    order = []
    entered_first = threading.Event()
    release_first = threading.Event()

    def first():
        with memory_budget.heavy_stage('transcribe'):
            order.append('first-in')
            entered_first.set()
            release_first.wait(timeout=5)
        order.append('first-out')

    def second():
        entered_first.wait(timeout=5)
        with memory_budget.heavy_stage('diarize'):
            order.append('second-in')

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start(); t2.start()
    entered_first.wait(timeout=5)
    assert memory_budget.heavy_stage_active() is True
    assert memory_budget.active_stage_name() == 'transcribe'
    time.sleep(0.15)                       # give t2 a chance to (wrongly) enter
    assert 'second-in' not in order
    release_first.set()
    t1.join(timeout=5); t2.join(timeout=5)
    # Partial order only: the gate guarantees second-in happens after
    # first RELEASES, but 'first-out' (user code after the with-block) may
    # legitimately interleave after the waiter wakes — asserting the exact
    # 3-element order was ~13% flaky (review-caught).
    assert order[0] == 'first-in'
    assert set(order) == {'first-in', 'first-out', 'second-in'}
    assert memory_budget.heavy_stage_active() is False


def test_heavy_stage_noop_on_comfortable(monkeypatch):
    _force(monkeypatch, 96)
    called = []
    monkeypatch.setattr(memory_budget, 'evict_ollama_models',
                        lambda **kw: called.append(1))
    with memory_budget.heavy_stage('transcribe'):
        # No gate, no eviction — today's big-machine behavior exactly.
        assert memory_budget.heavy_stage_active() is False
    assert called == []


def test_heavy_stage_evicts_llm_only_when_asked(monkeypatch):
    _force(monkeypatch, 8)
    evictions = []
    monkeypatch.setattr(memory_budget, 'evict_ollama_models',
                        lambda **kw: evictions.append(kw.get('reason')))
    with memory_budget.heavy_stage('transcribe'):
        pass
    with memory_budget.heavy_stage('analyze', evict_llm=False):
        pass
    assert evictions == ['transcribe']


def test_gate_released_after_stage_exception(monkeypatch):
    _force(monkeypatch, 8)
    monkeypatch.setattr(memory_budget, 'evict_ollama_models', lambda **kw: 0)
    with pytest.raises(RuntimeError):
        with memory_budget.heavy_stage('transcribe'):
            raise RuntimeError('engine died')
    assert memory_budget.heavy_stage_active() is False
    assert memory_budget.active_stage_name() is None


def test_on_wait_fires_only_when_contended(monkeypatch):
    _force(monkeypatch, 8)
    monkeypatch.setattr(memory_budget, 'evict_ollama_models', lambda **kw: 0)
    waits = []
    with memory_budget.heavy_stage('a', on_wait=lambda h: waits.append(h)):
        pass
    assert waits == []                      # uncontended: no callback
    entered = threading.Event()
    release = threading.Event()

    def holder():
        with memory_budget.heavy_stage('transcribe'):
            entered.set()
            release.wait(timeout=5)

    t = threading.Thread(target=holder)
    t.start()
    entered.wait(timeout=5)

    def waiter():
        with memory_budget.heavy_stage('diarize',
                                       on_wait=lambda h: waits.append(h)):
            pass

    t2 = threading.Thread(target=waiter)
    t2.start()
    time.sleep(0.2)
    release.set()
    t.join(timeout=5); t2.join(timeout=5)
    assert waits == ['transcribe']          # fired once, with holder name


def test_wedged_holder_is_abandoned(monkeypatch):
    _force(monkeypatch, 8)
    monkeypatch.setattr(memory_budget, 'evict_ollama_models', lambda **kw: 0)
    monkeypatch.setattr(memory_budget, 'GATE_WEDGE_TIMEOUT_SEC', 0.2)
    entered = threading.Event()
    release = threading.Event()

    def wedged():
        with memory_budget.heavy_stage('transcribe'):
            entered.set()
            release.wait(timeout=10)

    t = threading.Thread(target=wedged)
    t.start()
    entered.wait(timeout=5)
    ran = []
    with memory_budget.heavy_stage('diarize'):
        ran.append(True)                    # proceeded UNGATED after timeout
    assert ran == [True]
    assert memory_budget.heavy_stage_active() is True   # wedged holder keeps it
    release.set()
    t.join(timeout=5)
    assert memory_budget.heavy_stage_active() is False


def test_llm_blocked_semantics(monkeypatch):
    _force(monkeypatch, 16)
    monkeypatch.setattr(memory_budget, 'evict_ollama_models', lambda **kw: 0)
    assert memory_budget.llm_blocked() is False
    with memory_budget.heavy_stage('transcribe'):            # evicting stage
        assert memory_budget.llm_blocked() is True
    with memory_budget.heavy_stage('analyze', evict_llm=False):
        assert memory_budget.llm_blocked() is False          # LLM stage: chat may queue behind ollama itself
    assert memory_budget.llm_blocked() is False
    _force(monkeypatch, 96)
    assert memory_budget.llm_blocked() is False


# ── prewarm admission ────────────────────────────────────────────────

def test_prewarm_slot_tight_never_admits(monkeypatch):
    _force(monkeypatch, 8)
    with memory_budget.prewarm_slot() as admitted:
        assert admitted is False


def test_prewarm_slot_comfortable_admits_without_gate(monkeypatch):
    _force(monkeypatch, 96)
    with memory_budget.prewarm_slot() as admitted:
        assert admitted is True
        assert memory_budget.heavy_stage_active() is False


def test_prewarm_slot_low_holds_gate_while_admitted(monkeypatch):
    _force(monkeypatch, 16)
    with memory_budget.prewarm_slot() as admitted:
        assert admitted is True
        # The slot HOLDS the gate — a Transcribe click at this instant
        # serializes behind the warm-load instead of racing it (the field
        # failure's exact window, review-caught).
        assert memory_budget.heavy_stage_active() is True
        assert memory_budget.active_stage_name() == 'prewarm'
    assert memory_budget.heavy_stage_active() is False


def test_prewarm_slot_low_denied_while_stage_active(monkeypatch):
    _force(monkeypatch, 16)
    monkeypatch.setattr(memory_budget, 'evict_ollama_models', lambda **kw: 0)
    with memory_budget.heavy_stage('transcribe'):
        with memory_budget.prewarm_slot() as admitted:
            assert admitted is False


# ── ollama eviction (verified) ───────────────────────────────────────

class _Resp:
    def __init__(self, payload):
        self._payload = payload
    def read(self):
        return self._payload
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def test_evict_verifies_models_left(monkeypatch):
    _force(monkeypatch, 8)
    state = {'unloads': 0}
    invalidated = []

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, 'full_url') else str(req)
        if url.endswith('/api/ps'):
            if state['unloads'] >= 2:      # both unload requests landed
                return _Resp(b'{"models": []}')
            return _Resp(b'{"models": [{"name": "gemma4:e4b"}, {"model": "gemma4:e2b"}]}')
        state['unloads'] += 1
        return _Resp(b'{}')

    monkeypatch.setitem(sys.modules, 'ollama_url',
                        types.SimpleNamespace(ollama_base_url=lambda: 'http://127.0.0.1:11434'))
    monkeypatch.setitem(sys.modules, 'ai_analysis',
                        types.SimpleNamespace(invalidate_prewarm=lambda pn: invalidated.append(pn)))
    monkeypatch.setattr(memory_budget.urllib.request, 'urlopen', fake_urlopen)
    evicted = memory_budget.evict_ollama_models(reason='test')
    assert evicted == 2
    assert invalidated == [None]           # global clear-all, correct arity


def test_evict_deadline_expires_loudly_not_forever(monkeypatch, capsys):
    _force(monkeypatch, 16)                # low: 30s deadline
    clock = {'t': 0.0}
    monkeypatch.setattr(memory_budget.time, 'monotonic', lambda: clock['t'])

    def fake_sleep(s):
        clock['t'] += s
    monkeypatch.setattr(memory_budget.time, 'sleep', fake_sleep)

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, 'full_url') else str(req)
        if url.endswith('/api/ps'):
            # A chat generate in flight keeps the runner alive forever.
            return _Resp(b'{"models": [{"name": "gemma4:e4b"}]}')
        return _Resp(b'{}')

    monkeypatch.setitem(sys.modules, 'ollama_url',
                        types.SimpleNamespace(ollama_base_url=lambda: 'http://127.0.0.1:11434'))
    monkeypatch.setattr(memory_budget.urllib.request, 'urlopen', fake_urlopen)
    evicted = memory_budget.evict_ollama_models(reason='test')
    assert evicted == 0                    # nothing confirmed gone
    assert clock['t'] >= 30                # waited the full deadline, then gave up
    assert 'could not evict' in capsys.readouterr().out


def test_evict_transient_ps_failure_is_not_success(monkeypatch):
    """A mid-poll /api/ps timeout must NOT read as 'all evicted' — on a
    memory-starved machine that timeout is plausible at exactly the moment
    eviction runs (review-caught). The poll keeps the last real observation
    and keeps trying until the deadline."""
    _force(monkeypatch, 16)
    clock = {'t': 0.0}
    monkeypatch.setattr(memory_budget.time, 'monotonic', lambda: clock['t'])
    monkeypatch.setattr(memory_budget.time, 'sleep',
                        lambda s: clock.__setitem__('t', clock['t'] + s))
    calls = {'ps': 0}

    def fake_urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, 'full_url') else str(req)
        if url.endswith('/api/ps'):
            calls['ps'] += 1
            if calls['ps'] == 1:
                return _Resp(b'{"models": [{"name": "gemma4:e4b"}]}')
            raise TimeoutError('starved')      # every verify poll times out
        return _Resp(b'{}')

    monkeypatch.setitem(sys.modules, 'ollama_url',
                        types.SimpleNamespace(ollama_base_url=lambda: 'http://127.0.0.1:11434'))
    monkeypatch.setattr(memory_budget.urllib.request, 'urlopen', fake_urlopen)
    evicted = memory_budget.evict_ollama_models(reason='test')
    assert evicted == 0                        # unknown ≠ evicted
    assert clock['t'] >= 30                    # polled to the deadline


def test_evict_survives_dead_ollama(monkeypatch):
    _force(monkeypatch, 8)
    monkeypatch.setitem(sys.modules, 'ollama_url',
                        types.SimpleNamespace(ollama_base_url=lambda: 'http://127.0.0.1:1'))

    def refuse(*a, **kw):
        raise ConnectionRefusedError()
    monkeypatch.setattr(memory_budget.urllib.request, 'urlopen', refuse)
    assert memory_budget.evict_ollama_models() == 0  # never raises


# ── mlx phantom-module purge (transcribe.py) ─────────────────────────

def test_purge_mlx_modules_removes_phantom_namespace():
    # Simulate the macOS-14 poisoning: 'mlx' present, core unloadable.
    sys.modules['mlx'] = types.ModuleType('mlx')
    sys.modules['mlx.nn'] = types.ModuleType('mlx.nn')
    sys.modules['mlxfake'] = types.ModuleType('mlxfake')  # must survive
    try:
        from transcribe import _purge_mlx_modules
        _purge_mlx_modules()
        assert 'mlx' not in sys.modules
        assert 'mlx.nn' not in sys.modules
        assert 'mlxfake' in sys.modules       # prefix match is exact-dot only
        _purge_mlx_modules()                  # idempotent
    finally:
        sys.modules.pop('mlxfake', None)
        sys.modules.pop('mlx', None)
        sys.modules.pop('mlx.nn', None)


def test_purge_is_structurally_confined_to_the_failed_probe_branch():
    """The purge call must live ONLY in the probe's except handler.

    Purge-then-reimport of a successfully loaded mlx aborts the interpreter
    (its native core registers process-wide Metal state that can't init
    twice) — the full suite caught exactly that when the purge briefly ran
    in a finally. A runtime assert can't pin this (the failure mode is a
    crash, not a red test), so pin the AST: every call to
    _purge_mlx_modules in transcribe.py sits inside an except handler.
    """
    import ast
    import transcribe
    src_path = transcribe.__file__.replace('.pyc', '.py')
    tree = ast.parse(open(src_path).read())
    calls_in_handlers, calls_elsewhere = 0, 0

    class V(ast.NodeVisitor):
        def __init__(self):
            self.in_handler = 0
        def visit_ExceptHandler(self, node):
            self.in_handler += 1
            self.generic_visit(node)
            self.in_handler -= 1
        def visit_Call(self, node):
            nonlocal calls_in_handlers, calls_elsewhere
            name = getattr(node.func, 'id', '')
            if name == '_purge_mlx_modules':
                if self.in_handler:
                    calls_in_handlers += 1
                else:
                    calls_elsewhere += 1
            self.generic_visit(node)

    V().visit(tree)
    assert calls_in_handlers >= 1     # the failed-probe purge exists
    assert calls_elsewhere == 0       # and nowhere else — never on success


# ── model_config steering ────────────────────────────────────────────

def test_gemma_steering_tight_gets_e2b(monkeypatch):
    _force(monkeypatch, 8)
    import model_config
    hw = model_config.detect_hardware_tier()
    assert hw['tier'] == 'small'
    assert hw['variant'] == 'gemma4:e2b'


def test_gemma_steering_16gb_keeps_e4b(monkeypatch):
    _force(monkeypatch, 16)
    import model_config
    hw = model_config.detect_hardware_tier()
    assert hw['tier'] == 'medium'
    assert hw['variant'] == 'gemma4:e4b'


# ── ai_analysis integration points ───────────────────────────────────

def test_invalidate_prewarm_none_clears_all():
    import ai_analysis
    with ai_analysis._PREWARM_LOCK:
        ai_analysis._PREWARM_STATE['proj-a'] = ('x',)
        ai_analysis._PREWARM_STATE['proj-b'] = ('y',)
    ai_analysis.invalidate_prewarm('proj-a')
    with ai_analysis._PREWARM_LOCK:
        assert 'proj-a' not in ai_analysis._PREWARM_STATE
        assert 'proj-b' in ai_analysis._PREWARM_STATE
    ai_analysis.invalidate_prewarm(None)
    with ai_analysis._PREWARM_LOCK:
        assert not ai_analysis._PREWARM_STATE
