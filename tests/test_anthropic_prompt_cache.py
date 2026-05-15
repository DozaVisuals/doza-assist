"""Prompt-cache wiring tests for the Anthropic BYO API path.

Two layers:

  1. Unit tests (no network). Verify the sentinel-extraction helpers,
     structured-system block construction, and chat-message preparation
     produce the exact shapes Anthropic's prompt cache expects.

  2. Integration test (network, skipped without an API key). Fires two
     identical ``_call_ai`` requests with a >5K-token transcript and
     asserts the second response has ``cache_read_input_tokens > 0``,
     proving the cache actually hit end-to-end.

Run the live test with the app's bundled Python so the ``anthropic``
SDK is on the import path, plus either an env-var key or whatever the
running app has stored in the macOS Keychain::

    APP_PY="$HOME/Library/Application Support/DozaAssist/venv/bin/python3"
    "$APP_PY" -m pytest core/tests/test_anthropic_prompt_cache.py \
        -k integration -s

The fixture below resolves the key in this order:
    1. ``ANTHROPIC_API_KEY`` env var
    2. ``ai_providers.config.load_provider_config()`` (same Keychain
       lookup the app uses; will prompt the first time)
    3. skip the test

``ANTHROPIC_TEST_TRANSCRIPT`` can override the synthetic transcript with
a real one for ad-hoc runs.
"""
import logging
import os
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest


HERE = Path(__file__).resolve()
CORE_DIR = HERE.parent.parent
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))


# ---------------------------------------------------------------------------
# Unit layer: sentinels + structured-system construction
# ---------------------------------------------------------------------------

def test_sentinel_extraction_round_trips_text():
    from ai_analysis import (
        CACHE_TX_START, CACHE_TX_END, CACHE_DNA_START, CACHE_DNA_END,
        _split_cacheable_prompt, _strip_cache_sentinels,
    )

    transcript = "[00:00:00] Speaker: this is a long transcript " * 200
    dna = "MY STYLE CONTEXT — punchy intercuts, never trail off"
    raw_prompt = (
        f"PROJECT: demo\n\nTRANSCRIPT:\n"
        f"{CACHE_TX_START}{transcript}{CACHE_TX_END}\n\n"
        f"STYLE:\n{CACHE_DNA_START}{dna}{CACHE_DNA_END}\n\n"
        "Return ONLY valid JSON."
    )

    clean, found_tx, found_dna = _split_cacheable_prompt(raw_prompt)
    assert found_tx == transcript
    assert found_dna == dna
    # The cleaned prompt drops the sentinels and the wrapped text, replacing
    # it with the hoist notice so the model still has a reference to it.
    assert CACHE_TX_START not in clean
    assert CACHE_TX_END not in clean
    assert "TRANSCRIPT provided" in clean
    assert "EDITORIAL STYLE provided" in clean
    # _strip_cache_sentinels keeps the wrapped content but drops markers.
    stripped = _strip_cache_sentinels(raw_prompt)
    assert CACHE_TX_START not in stripped and CACHE_DNA_END not in stripped
    assert transcript in stripped and dna in stripped


def test_build_cached_system_blocks_marks_long_segments():
    """With a transcript present, layout is [transcript, dna?, system].
    The transcript carries the 1h-TTL cache marker; the system text comes
    AFTER it without a cache marker so it can vary between sibling calls
    without busting the transcript prefix."""
    from ai_providers.anthropic_provider import build_cached_system_blocks

    system = "S" * (1024 * 4 + 10)  # comfortably over the 1024-token min
    dna = "D" * (1024 * 4 + 10)  # long enough to qualify for its own cache marker
    transcript = "T" * (1024 * 4 + 10)

    blocks = build_cached_system_blocks(system, dna_block=dna, transcript_block=transcript)
    assert len(blocks) == 3
    tx, dna_block, sys_block = blocks

    assert tx["type"] == "text" and tx["text"].startswith("T")
    assert tx.get("cache_control") == {"type": "ephemeral", "ttl": "1h"}

    assert dna_block["type"] == "text" and dna_block["text"].startswith("D")
    # 1h TTL throughout so Anthropic's longer-must-precede-shorter
    # ordering rule never trips (mixed TTLs across system + messages
    # were rejecting cloud chat requests with HTTP 400).
    assert dna_block.get("cache_control") == {"type": "ephemeral", "ttl": "1h"}

    assert sys_block["type"] == "text" and sys_block["text"].startswith("S")
    # System text is intentionally uncached when a transcript precedes it.
    assert "cache_control" not in sys_block


def test_build_cached_system_blocks_no_transcript_caches_combined_stable_text():
    """Without a transcript (chat-style call), the system + DNA get joined
    into one cached block at 1h TTL, matching the transcript block's TTL
    in the messages array. Uniform TTL avoids the HTTP 400 ordering
    error from mixing 5m and 1h cache_control markers in one request."""
    from ai_providers.anthropic_provider import build_cached_system_blocks

    system = "S" * (1024 * 4 + 10)
    dna = "D" * 200  # short DNA, merges into the stable block

    blocks = build_cached_system_blocks(system, dna_block=dna)
    assert len(blocks) == 1
    stable = blocks[0]
    assert stable["text"].startswith("S") and dna in stable["text"]
    assert stable.get("cache_control") == {"type": "ephemeral", "ttl": "1h"}


def test_build_cached_system_blocks_transcript_first_independent_of_system():
    """Regression for the 0.8.5 bug: two calls with the same transcript
    but a DIFFERENT system_prompt must produce the same leading bytes,
    so Anthropic's prefix-based cache lookup hits on the second call.
    """
    from ai_providers.anthropic_provider import build_cached_system_blocks

    transcript = "T" * (1024 * 4 + 10)
    sys_a = "soundbites system prompt body" * 50  # different text
    sys_b = "overview system prompt body" * 50    # different text

    blocks_a = build_cached_system_blocks(sys_a, transcript_block=transcript)
    blocks_b = build_cached_system_blocks(sys_b, transcript_block=transcript)
    # The block carrying cache_control must be byte-identical across
    # both calls, otherwise the prefix differs and the cache misses.
    assert blocks_a[0]["text"] == blocks_b[0]["text"]
    assert blocks_a[0]["cache_control"] == blocks_b[0]["cache_control"]


def test_build_cached_system_blocks_skips_cache_on_small_blocks():
    """Below 1024 tokens the API refuses to cache, so we omit cache_control
    rather than send a malformed request."""
    from ai_providers.anthropic_provider import build_cached_system_blocks

    blocks = build_cached_system_blocks("short system", transcript_block="short tx")
    assert all("cache_control" not in b for b in blocks)


def test_prepare_chat_messages_hoists_transcript_to_user_block_list():
    """Anthropic chat path: the transcript message becomes a content-list
    with the transcript text carrying cache_control, while the system
    block list carries the chat system prompt (and any DNA examples)."""
    from ai_analysis import (
        _prepare_chat_messages_for_provider,
        CACHE_TX_START, CACHE_TX_END, CACHE_DNA_START, CACHE_DNA_END,
    )

    transcript = "T" * (1024 * 4 + 100)
    dna = "D" * (1024 * 4 + 100)
    style_msg = {
        "role": "user",
        "content": (
            "STYLE CONTEXT (active My Style profile):\n\n"
            f"{CACHE_DNA_START}{dna}{CACHE_DNA_END}"
        ),
    }
    transcript_msg = {
        "role": "user",
        "content": (
            "Here is the loaded project.\n\n"
            "PROJECT: demo\n\nTRANSCRIPT:\n"
            f"{CACHE_TX_START}{transcript}{CACHE_TX_END}\n"
        ),
    }
    ack = {"role": "assistant", "content": "Transcript loaded."}
    current = {"role": "user", "content": "find me three powerful soundbites"}

    messages = [style_msg, transcript_msg, ack, current]
    sys_param, msg_param = _prepare_chat_messages_for_provider(
        "anthropic", "chat system prompt", messages,
    )

    # System is now a structured list with chat-system + DNA cached
    # together. TTL must match the transcript's 1h marker in messages
    # — mixing 5m here with 1h in messages causes Anthropic to reject
    # the request with HTTP 400 (longer-TTL-must-precede-shorter rule).
    assert isinstance(sys_param, list)
    assert sys_param[0]["text"].startswith("chat system prompt")
    assert dna in sys_param[0]["text"]
    assert sys_param[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    # The orphan style header message was dropped; the transcript message
    # carries the cache_control on its transcript block.
    assert all(m.get("role") != "user" or
               m.get("content") != "STYLE CONTEXT (active My Style profile):"
               for m in msg_param)
    transcript_message = msg_param[0]
    assert isinstance(transcript_message["content"], list)
    cached_block = [b for b in transcript_message["content"] if "cache_control" in b]
    assert len(cached_block) == 1
    assert cached_block[0]["text"] == transcript
    assert cached_block[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_chat_path_uses_uniform_cache_ttl():
    """Regression for the cloud-chat bug where Anthropic rejected
    requests with HTTP 400:

      messages.0.content.1.cache_control.ttl: a ttl='1h' cache_control
      block must not come after a ttl='5m' cache_control block.

    The chat path puts a cache marker on the system block AND on the
    transcript in messages. Mixed TTLs (5m on system, 1h on transcript)
    fail Anthropic's processing-order rule. Uniform 1h TTL across all
    markers sidesteps the rule entirely.
    """
    from ai_analysis import (
        _prepare_chat_messages_for_provider,
        CACHE_TX_START, CACHE_TX_END, CACHE_DNA_START, CACHE_DNA_END,
    )

    transcript = "T" * (1024 * 4 + 100)
    dna = "D" * (1024 * 4 + 100)
    style_msg = {
        "role": "user",
        "content": (
            "STYLE CONTEXT (active My Style profile):\n\n"
            f"{CACHE_DNA_START}{dna}{CACHE_DNA_END}"
        ),
    }
    transcript_msg = {
        "role": "user",
        "content": f"PROJECT:\nTRANSCRIPT:\n{CACHE_TX_START}{transcript}{CACHE_TX_END}\n",
    }
    sys_param, msg_param = _prepare_chat_messages_for_provider(
        "anthropic", "chat system prompt", [style_msg, transcript_msg],
    )

    ttls_seen: list = []

    def _collect(block):
        cc = block.get("cache_control") if isinstance(block, dict) else None
        if cc:
            ttls_seen.append(cc.get("ttl") or "5m")

    if isinstance(sys_param, list):
        for b in sys_param:
            _collect(b)
    for m in msg_param or []:
        if isinstance(m, dict):
            c = m.get("content")
            if isinstance(c, list):
                for b in c:
                    _collect(b)

    assert ttls_seen, "expected at least one cached block in the chat payload"
    assert all(t == "1h" for t in ttls_seen), (
        f"chat path must use a uniform 1h TTL across all cache_control "
        f"markers (Anthropic rejects mixed TTLs with HTTP 400). Saw: {ttls_seen}"
    )


def test_prepare_chat_messages_passes_through_ollama():
    """Ollama path: sentinels stripped, no cache_control on anything."""
    from ai_analysis import (
        _prepare_chat_messages_for_provider,
        CACHE_TX_START, CACHE_TX_END,
    )
    msg = {
        "role": "user",
        "content": f"TRANSCRIPT:\n{CACHE_TX_START}body{CACHE_TX_END}",
    }
    sys_param, msg_param = _prepare_chat_messages_for_provider(
        "ollama", "sys", [msg, {"role": "user", "content": "find clips"}],
    )
    assert sys_param == "sys"
    assert msg_param[0]["content"] == "TRANSCRIPT:\nbody"
    assert msg_param[1]["content"] == "find clips"


def test_log_usage_warns_on_cache_miss_for_seen_prefix(caplog):
    """If the same prefix re-creates a cache instead of reading one, the
    provider logs a WARNING so we notice broken cache_control placement."""
    from ai_providers.anthropic_provider import (
        _log_usage, _seen_cache_prefixes,
    )

    class FakeUsage:
        cache_creation_input_tokens = 5000
        cache_read_input_tokens = 0
        input_tokens = 200
        output_tokens = 50

    _seen_cache_prefixes.add("test-signature")
    try:
        with caplog.at_level(logging.WARNING, logger="ai_providers.anthropic_provider"):
            _log_usage(FakeUsage(), "claude-opus-4", "unit-test", "test-signature")
        assert any("cache-miss-on-repeat" in r.message for r in caplog.records)
    finally:
        _seen_cache_prefixes.discard("test-signature")


# ---------------------------------------------------------------------------
# Integration layer: real Anthropic call (skipped without a key)
# ---------------------------------------------------------------------------

def _make_long_transcript(min_tokens: int = 5000) -> str:
    """Generate a synthetic but well-formed transcript that comfortably
    exceeds the 1024-token cache minimum and looks realistic enough for
    the model to engage with."""
    overridden = os.environ.get("ANTHROPIC_TEST_TRANSCRIPT")
    if overridden and Path(overridden).exists():
        return Path(overridden).read_text(encoding="utf-8")

    paragraph = textwrap.dedent("""\
    [00:{m:02d}:{s:02d}] Speaker: I think the moment that changed things for me
    was when I realized the story we were telling wasn't the story I'd lived.
    We had this draft on the wall that read like a press release, and the
    truth was messier. There was a long pause that morning, and somebody
    asked the only question that mattered, which was: do we want this to be
    accurate or do we want this to be safe. We chose accurate.
    """).strip() + "\n"
    chunks = []
    minute = 0
    while sum(len(c) for c in chunks) // 4 < min_tokens:
        for s in range(0, 60, 15):
            chunks.append(paragraph.format(m=minute, s=s))
        minute += 1
    return "".join(chunks)


def _resolve_anthropic_key() -> str:
    """Pull the Anthropic key from env first, then the app's own config
    loader (which reads the macOS Keychain via ``keyring``). Returns ''
    if neither source has a key. We do not shell out to ``security`` so
    no separate credential prompt happens beyond keyring's own UI."""
    env_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if env_key and env_key.lower() != "sk-ant-..." and not env_key.endswith("..."):
        return env_key
    try:
        from ai_providers.config import load_provider_config
        cfg = load_provider_config() or {}
        return ((cfg.get("anthropic") or {}).get("api_key") or "").strip()
    except Exception:
        return ""


@pytest.mark.integration
def test_repeat_call_reads_cache():
    """End-to-end: fire two identical analysis prompts and assert the
    second response shows a non-zero cache_read_input_tokens. Skipped
    when no Anthropic key is reachable (env var or Keychain) so CI
    without credentials stays green."""
    api_key = _resolve_anthropic_key()
    if not api_key:
        pytest.skip(
            "No Anthropic key found in ANTHROPIC_API_KEY or "
            "ai_providers.config.load_provider_config(); skipping live test"
        )

    from ai_providers.anthropic_provider import (
        AnthropicProvider,
        build_cached_system_blocks,
        _seen_cache_prefixes,
    )

    # Clear any cross-test state so we're measuring this run only.
    _seen_cache_prefixes.clear()

    transcript = _make_long_transcript(min_tokens=5500)
    system_prompt = (
        "You are an expert documentary film editor. Output JSON only. "
        "No markdown, no fences, no commentary. Copy HH:MM:SS timecodes "
        "exactly from the transcript without rounding."
    )
    system_param = build_cached_system_blocks(
        system_prompt, transcript_block=transcript,
    )
    user_prompt = (
        "Return a JSON object with this shape only: "
        '{"strongest_soundbites": [{"text": "...", "start": "00:00:00", '
        '"end": "00:00:00", "why": "..."}]}\n'
        "Find ONE soundbite from the transcript above. JSON only, no prose."
    )

    provider = AnthropicProvider(api_key=api_key)

    # Capture usage for both calls by patching _log_usage to also save
    # the numbers in a list we can assert on.
    captured = []
    real_log = provider.__class__.__module__

    from ai_providers import anthropic_provider as ap_mod
    original_log = ap_mod._log_usage

    def capturing_log(usage, model, call_site, signature):
        captured.append({
            "cache_create": int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            "cache_read": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            "input": int(getattr(usage, "input_tokens", 0) or 0),
            "output": int(getattr(usage, "output_tokens", 0) or 0),
            "site": call_site,
        })
        original_log(usage, model, call_site, signature)

    with mock.patch.object(ap_mod, "_log_usage", capturing_log):
        first = provider.generate(
            system_param, user_prompt, task_type="analysis",
            max_tokens=400, call_site="integration_first",
        )
        second = provider.generate(
            system_param, user_prompt, task_type="analysis",
            max_tokens=400, call_site="integration_second",
        )

    assert first and isinstance(first, str)
    assert second and isinstance(second, str)
    assert len(captured) == 2
    first_usage, second_usage = captured
    # First call should populate the cache (or hit a prior cache from a
    # recent run; either way cache_create + cache_read > 0).
    assert first_usage["cache_create"] + first_usage["cache_read"] > 0, (
        "First call did not exercise the cache at all: %s" % (first_usage,)
    )
    # Second call must read from cache.
    assert second_usage["cache_read"] > 0, (
        "Second call returned cache_read=0; cache_control did not apply. "
        "first=%s second=%s" % (first_usage, second_usage)
    )
    hit_rate = (
        second_usage["cache_read"] /
        max(1, second_usage["cache_read"] + second_usage["input"])
    )
    print(
        "\n[anthropic-cache] first: %s\n[anthropic-cache] second: %s\n"
        "[anthropic-cache] second-call hit rate: %.2f"
        % (first_usage, second_usage, hit_rate)
    )
    # >70% is the product goal.
    assert hit_rate > 0.7, f"Cache hit rate {hit_rate:.2f} below 0.70 target"
