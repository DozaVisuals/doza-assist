"""Unified-memory budget governor for small-RAM Apple Silicon Macs.

Apple Silicon shares one memory pool between CPU and GPU. When the app's
resident AI components (Parakeet-MLX, Whisper, pyannote-MPS, Ollama/gemma)
stack up on an 8-16 GB machine, WindowServer itself is starved of pages and
the *display* corrupts (field report: FX tester, M1 16 GB, macOS 15.7.8 —
Metal "Insufficient Memory" killing Parakeet while gemma warmed and pyannote
sat boot-preloaded).

This module is the single policy source for "how much may we keep resident".
Everything keys off total RAM:

- ``tight``       (< 12 GB, i.e. 8 GB Macs)
- ``low``         (12-24 GB, i.e. 16 GB Macs)
- ``comfortable`` (>= 24 GB — today's behavior, nothing changes)

On tight/low the rules are: exactly one heavy background component RESIDENT
(not merely running) at a time — releases happen inside the gate, eviction
is verified against /api/ps, and prewarm holds the same gate it checks so
there is no check-then-act window. Proactive preload/prewarm paths that
exist purely as latency optimizations are skipped where they can't pay off.

Contract: stdlib-only at import time (mirrors model_config); anything
heavier is imported lazily inside functions. Every consumer treats this
module as optional AND heavy_stage itself must never raise — an internal
governor error degrades to ungoverned (today's) behavior, never a failed
user job.

``DOZA_FORCE_RAM_GB`` (handled in model_config._get_ram_gb, honored by
main.js too) forces the detected RAM for testing the tight/low paths on a
big development machine.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from contextlib import contextmanager

TIGHT_MAX_GB = 12.0
LOW_MAX_GB = 24.0

# A stage that holds the gate longer than this is presumed wedged and the
# waiter proceeds UNGATED (the governor is best-effort by contract — a
# permanently starved pipeline is worse than a memory race). 45 min covers
# a CPU diarization of a very long interview.
GATE_WEDGE_TIMEOUT_SEC = 45 * 60

_ram_class_cache = None
_ram_class_lock = threading.Lock()


def total_ram_gb() -> float:
    from model_config import _get_ram_gb
    return _get_ram_gb()


def ram_class() -> str:
    """'tight' | 'low' | 'comfortable' — cached for the process lifetime.
    Never raises: an unreadable RAM size degrades to 'comfortable'
    (= ungoverned, today's behavior)."""
    global _ram_class_cache
    if _ram_class_cache is None:
        with _ram_class_lock:
            if _ram_class_cache is None:
                try:
                    gb = total_ram_gb()
                    if gb < TIGHT_MAX_GB:
                        cls = 'tight'
                    elif gb < LOW_MAX_GB:
                        cls = 'low'
                    else:
                        cls = 'comfortable'
                    print(f"[mem-budget] {gb:.0f} GB RAM -> class '{cls}'",
                          flush=True)
                except Exception:
                    cls = 'comfortable'
                _ram_class_cache = cls
    return _ram_class_cache


def _reset_ram_class_cache_for_tests():
    global _ram_class_cache
    _ram_class_cache = None


# ── policies ─────────────────────────────────────────────────────────

def allow_boot_preload() -> bool:
    """May multi-GB models be loaded at app boot as a latency optimization?"""
    return ram_class() == 'comfortable'


def allow_prewarm_now() -> bool:
    """Advisory peek — prefer prewarm_slot(), which holds the gate and has
    no check-then-act window. Kept for cheap early-outs and tests."""
    cls = ram_class()
    if cls == 'comfortable':
        return True
    if cls == 'low':
        return not heavy_stage_active()
    return False


def release_models_after_use() -> bool:
    """Drop model caches (whisper, pyannote) when a stage finishes?"""
    return ram_class() != 'comfortable'


def whisper_model_prefs() -> tuple:
    """Whisper fallback preference order, largest-first that fits the budget.

    turbo (large-v3-turbo) runs ~5 GB in FP32 on CPU — fine at 16 GB when
    it is the only resident component, never fine at 8 GB. 'small' keeps
    multilingual quality well above 'base' at ~1 GB resident.
    """
    if ram_class() == 'tight':
        return ('small', 'base')
    return ('turbo', 'large-v3', 'base')


def mlx_memory_limit_mb() -> int:
    """Cap for mx.set_memory_limit in the Parakeet worker. 0 = leave MLX
    defaults (which allow ~1.5x device working set — that default is the
    reason an uncapped worker can push an 8 GB machine over)."""
    cls = ram_class()
    if cls == 'tight':
        return 2200
    if cls == 'low':
        return 3584
    return 0


def mlx_cache_limit_mb() -> int:
    return 256 if ram_class() != 'comfortable' else 0


def parakeet_chunk_sec() -> int:
    """Shorter decode chunks on 8 GB: smaller per-chunk command buffers and
    activation peaks (the 60s figure was itself a Metal-error mitigation)."""
    return 30 if ram_class() == 'tight' else 60


def ollama_keep_alive() -> str:
    """How long the gemma runner stays resident after the last call."""
    cls = ram_class()
    if cls == 'tight':
        return '2m'
    if cls == 'low':
        return '10m'
    return '30m'


def chat_num_ctx_ceiling() -> int:
    """Upper rung for chat num_ctx. The KV cache at 32K is a real slice of
    an 8 GB pool; 16K halves worst-case KV. Interviews whose payload can't
    fit 16K are routed to the chunked-search path via long_chat_seconds()
    instead of silently letting Ollama evict the transcript (the 1.0.32
    bug class)."""
    return 16384 if ram_class() == 'tight' else 32768


def analysis_num_ctx_ceiling() -> int:
    """Upper rung for /api/generate analysis calls — same rationale as the
    chat ceiling; chunked analysis prompts fit well under 10K tokens."""
    return 16384 if ram_class() == 'tight' else 32768


def long_chat_seconds() -> int:
    """Interview length above which chat routes to the chunked-search path
    (which never needs the whole transcript in one window). On tight the
    16K ceiling fits roughly a 20-25 min interview's full-transcript chat
    prefix — route earlier so the clamp can never silently evict."""
    return 20 * 60 if ram_class() == 'tight' else 60 * 60


# ── heavy-stage serialization ────────────────────────────────────────

_gate = threading.Lock()
_active_stage = {'name': None, 'evict_llm': False}


def heavy_stage_active() -> bool:
    return _gate.locked()


def active_stage_name():
    return _active_stage['name']


def llm_blocked() -> bool:
    """True while a gate-holding stage required LLM eviction (tight/low):
    an interactive chat generate would reload gemma alongside that stage's
    model — the exact stack the governor forbids."""
    if ram_class() == 'comfortable':
        return False
    return _gate.locked() and bool(_active_stage['evict_llm'])


@contextmanager
def heavy_stage(name: str, evict_llm: bool = True, on_wait=None):
    """Serialize memory-heavy background stages on tight/low machines.

    Stages: transcribe (MLX/whisper), diarize (pyannote-MPS), analyze /
    collection-build / batch phases (Ollama). Pass ``evict_llm=False`` for
    stages that themselves need the LLM. ``on_wait(holder_name)`` fires
    once if the gate is contended, so callers can surface a queued state.
    Interactive chat is not gated here — app.py gives it a bounded-wait
    busy path via llm_blocked().

    Contract hardening (review-caught): never raises from governor
    internals; a wedged holder is abandoned after GATE_WEDGE_TIMEOUT_SEC
    and the waiter proceeds ungated (loudly). Model-cache releases belong
    INSIDE the with-block — residency, not just execution, is what the
    gate serializes.

    On comfortable machines this is a no-op so today's concurrency (and
    its latency wins) is preserved exactly.
    """
    try:
        governed = ram_class() != 'comfortable'
    except Exception:
        governed = False
    if not governed:
        yield
        return

    acquired = _gate.acquire(blocking=False)
    if not acquired:
        holder = _active_stage['name']
        print(f"[mem-budget] stage '{name}' waiting for gate "
              f"(active: {holder})", flush=True)
        if on_wait is not None:
            try:
                on_wait(holder)
            except Exception:
                pass
        acquired = _gate.acquire(timeout=GATE_WEDGE_TIMEOUT_SEC)
        if not acquired:
            print(f"[mem-budget] WARNING: gate holder "
                  f"'{_active_stage['name']}' presumed wedged after "
                  f"{GATE_WEDGE_TIMEOUT_SEC}s — running '{name}' UNGATED",
                  flush=True)
            yield
            return
    _active_stage['name'] = name
    _active_stage['evict_llm'] = bool(evict_llm)
    print(f"[mem-budget] stage '{name}' running", flush=True)
    try:
        if evict_llm:
            try:
                evict_ollama_models(reason=name)
            except Exception:
                pass
        yield
    finally:
        _active_stage['name'] = None
        _active_stage['evict_llm'] = False
        _gate.release()
        print(f"[mem-budget] stage '{name}' done", flush=True)


@contextmanager
def prewarm_slot():
    """Atomic prewarm admission: claims the heavy-stage gate (non-blocking)
    for the duration of the warm-load, closing the check-then-act window
    where a prewarm in flight when Transcribe is clicked still loads gemma
    mid-transcription (the field failure's exact shape).

    Yields True when the prewarm may proceed (comfortable: always, gate
    untouched; low: gate claimed for the duration). Yields False when it
    must be skipped (tight always — the model would just be evicted by the
    next stage; low when any heavy stage is running or queued on the gate).
    """
    try:
        cls = ram_class()
    except Exception:
        cls = 'comfortable'
    if cls == 'comfortable':
        yield True
        return
    if cls == 'tight':
        yield False
        return
    if not _gate.acquire(blocking=False):
        yield False
        return
    _active_stage['name'] = 'prewarm'
    _active_stage['evict_llm'] = False
    try:
        yield True
    finally:
        _active_stage['name'] = None
        _active_stage['evict_llm'] = False
        _gate.release()


# ── ollama eviction ──────────────────────────────────────────────────

def _list_resident_models(base: str):
    """Names of models Ollama currently holds resident, or None if Ollama
    is unreachable (nothing to evict)."""
    try:
        req = urllib.request.Request(f'{base}/api/ps',
                                     headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            loaded = json.loads(resp.read().decode('utf-8')).get('models') or []
        return [m.get('name') or m.get('model')
                for m in loaded if m.get('name') or m.get('model')]
    except Exception:
        return None


def _send_unload(base: str, name: str) -> bool:
    body = json.dumps({'model': name, 'keep_alive': 0,
                       'prompt': ''}).encode('utf-8')
    req = urllib.request.Request(f'{base}/api/generate', data=body,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30):
            pass
        return True
    except Exception as e:
        print(f"[mem-budget] unload request for {name} failed: {e}", flush=True)
        return False


def evict_ollama_models(reason: str = '') -> int:
    """Ask Ollama to unload every resident model and VERIFY it left.

    keep_alive:0 is a request, not a guarantee — a generate in flight
    (interactive chat is deliberately ungated) keeps the runner alive, and
    a new call can re-load it. So: send unloads, then poll /api/ps
    (re-sending keep_alive:0 to survivors) until empty or a bounded
    deadline, and log loudly if models remain — an observable overlap
    beats a silent one. Returns the number of models confirmed gone.
    Never raises; a dead/absent Ollama means nothing to evict.
    """
    try:
        from ollama_url import ollama_base_url
        base = ollama_base_url()
    except Exception:
        return 0
    initial = _list_resident_models(base)
    if not initial:
        return 0
    for name in initial:
        _send_unload(base, name)
    deadline = time.monotonic() + (60 if ram_class() == 'tight' else 30)
    # A transient /api/ps failure (None) must NOT read as "all evicted" —
    # on a memory-starved machine a 3s timeout is plausible exactly now
    # (review-caught). Keep the last successful observation and keep
    # polling; only an actual empty list ends the wait early.
    remaining = _list_resident_models(base)
    if remaining is None:
        remaining = list(initial)
    while remaining and time.monotonic() < deadline:
        time.sleep(1.0)
        observed = _list_resident_models(base)
        if observed is not None:
            remaining = observed
        if remaining:
            # Survivor: an in-flight or brand-new call re-extended it.
            # Re-expire so it unloads the moment that call finishes.
            for name in remaining:
                _send_unload(base, name)
    gone = [n for n in initial if n not in remaining]
    for n in gone:
        print(f"[mem-budget] evicted ollama model {n}"
              f"{' for ' + reason if reason else ''}", flush=True)
    if remaining:
        print(f"[mem-budget] WARNING: could not evict {remaining} within "
              f"deadline{' for ' + reason if reason else ''} — proceeding "
              f"with models still resident", flush=True)
    if gone:
        try:
            from ai_analysis import invalidate_prewarm
            invalidate_prewarm(None)   # None = clear every project's cooldown
        except Exception:
            pass
    return len(gone)
