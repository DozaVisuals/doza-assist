"""
Hardware-tiered Gemma 4 variant selection for Doza Assist.
Uses only stdlib — safe to import from setup_assistant.py and ai_analysis.py.
"""

import json
import os
import shutil
import subprocess
import sys
import time

SUPPORT_DIR = os.path.expanduser("~/Library/Application Support/DozaAssist")
MODEL_CONFIG_FILE = os.path.join(SUPPORT_DIR, "model_config.json")

# Real Gemma 4 tags from ollama.com/library/gemma4 (verified 2026-04-22
# against /api/tags + /api/show on the user's machine).
#
# The ``e2b`` and ``e4b`` tags are "effective" / mixture-of-experts variants:
# they run at the speed of a 2B / 4B model (only that many parameters active
# per forward pass) but carry the full 5.1B / 8B weights in memory. So the
# download size is the total parameter count; the speed table below reflects
# the smaller effective count.
#
# tier -> (ollama_tag, download_size, description)
GEMMA4_VARIANTS = {
    'small':  ('gemma4:e2b', '8.4 GB',  'Gemma 4 2B · 5B total (smallest, fastest)'),
    'medium': ('gemma4:e4b', '11.5 GB', 'Gemma 4 4B · 8B total (balanced)'),
    'large':  ('gemma4:26b', '18 GB',   'Gemma 4 27B (high quality)'),
    'xlarge': ('gemma4:31b', '19 GB',   'Gemma 4 32B (highest quality)'),
}

VALID_TIERS = ('small', 'medium', 'large', 'xlarge')


def _get_ram_gb():
    """Return total system RAM in GB."""
    try:
        result = subprocess.run(
            ['sysctl', '-n', 'hw.memsize'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    return 8.0


def _get_arch():
    """Return CPU architecture string (e.g. arm64, x86_64)."""
    try:
        result = subprocess.run(['uname', '-m'], capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except Exception:
        import platform
        return platform.machine()


def _get_available_disk_gb():
    """Return available disk space in GB at the Ollama models directory."""
    path = os.path.expanduser("~/.ollama/models")
    while path and not os.path.exists(path):
        path = os.path.dirname(path)
    if not path:
        path = os.path.expanduser("~")
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except Exception:
        return 100.0


def detect_hardware_tier():
    """
    Detect system hardware and return the appropriate Gemma 4 tier.

    Returns dict with keys: tier, variant, download_size, description,
                             ram_gb, arch, disk_gb, reason
    """
    ram_gb = _get_ram_gb()
    arch = _get_arch()
    disk_gb = _get_available_disk_gb()

    if ram_gb < 16:
        tier = 'small'
        reason = (
            f"{ram_gb:.0f} GB RAM detected (< 16 GB) — using smallest variant "
            "to ensure it fits in memory. Quality may be reduced."
        )
    elif ram_gb < 32:
        tier = 'medium'
        reason = (
            f"{ram_gb:.0f} GB RAM detected (16–32 GB) — using mid-tier variant "
            "for the best balance of speed and quality."
        )
    elif ram_gb < 64:
        tier = 'large'
        reason = (
            f"{ram_gb:.0f} GB RAM detected (32–64 GB) — using larger variant "
            "for higher quality."
        )
    else:
        tier = 'xlarge'
        reason = (
            f"{ram_gb:.0f} GB RAM detected (64 GB+) — using largest variant "
            "for maximum quality."
        )

    variant, download_size, description = GEMMA4_VARIANTS[tier]
    return {
        'tier': tier,
        'variant': variant,
        'download_size': download_size,
        'description': description,
        'ram_gb': ram_gb,
        'arch': arch,
        'disk_gb': disk_gb,
        'reason': reason,
    }


def load_model_config():
    """Load saved model config. Returns None if not found or invalid."""
    try:
        with open(MODEL_CONFIG_FILE) as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get('gemma4_variant'):
            return data
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return None


def save_model_config(tier, variant, auto_selected=True):
    """Persist model config to disk so users can inspect or override it."""
    os.makedirs(SUPPORT_DIR, exist_ok=True)
    data = {
        'gemma4_variant': variant,
        'tier': tier,
        'auto_selected': auto_selected,
        'selected_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(MODEL_CONFIG_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    return data


def get_cli_tier_override():
    """Parse --model-tier flag from sys.argv. Returns tier string or None."""
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == '--model-tier' and i + 1 < len(args):
            t = args[i + 1].lower()
            if t in VALID_TIERS:
                return t
        elif arg.startswith('--model-tier='):
            t = arg.split('=', 1)[1].lower()
            if t in VALID_TIERS:
                return t
    return None


def get_gemma4_variant(cli_tier=None):
    """
    Return the Gemma 4 variant to use. Priority order:
      1. cli_tier argument or --model-tier CLI flag
      2. Saved config (when user manually overrode auto-selection)
      3. Auto-detect from hardware

    Returns dict with: tier, variant, download_size, description, reason, source
    """
    if cli_tier is None:
        cli_tier = get_cli_tier_override()

    if cli_tier and cli_tier in VALID_TIERS:
        variant, download_size, description = GEMMA4_VARIANTS[cli_tier]
        return {
            'tier': cli_tier,
            'variant': variant,
            'download_size': download_size,
            'description': description,
            'reason': f'forced via --model-tier={cli_tier}',
            'source': 'cli',
        }

    config = load_model_config()
    if config and not config.get('auto_selected', True):
        tier = config.get('tier', 'medium')
        if tier in GEMMA4_VARIANTS:
            variant, download_size, description = GEMMA4_VARIANTS[tier]
            return {
                'tier': tier,
                'variant': variant,
                'download_size': download_size,
                'description': description,
                'reason': 'manually configured by user',
                'source': 'config',
            }

    hw = detect_hardware_tier()
    save_model_config(hw['tier'], hw['variant'], auto_selected=True)
    return {
        'tier': hw['tier'],
        'variant': hw['variant'],
        'download_size': hw['download_size'],
        'description': hw['description'],
        'reason': hw['reason'],
        'ram_gb': hw.get('ram_gb'),
        'arch': hw.get('arch'),
        'disk_gb': hw.get('disk_gb'),
        'source': 'auto',
    }


def format_selection_message(info):
    """Return a user-facing string describing the selected variant and why."""
    return '\n'.join([
        f"  Selected variant : {info['variant']}  ({info['description']})",
        f"  Why              : {info['reason']}",
        f"  Download size    : ~{info['download_size']}",
        "",
        "  To override, run with:  --model-tier small|medium|large|xlarge",
        f"  Or edit config at:      {MODEL_CONFIG_FILE}",
    ])


# ---------------------------------------------------------------------------
# Hardware-aware speed / quality estimates
# ---------------------------------------------------------------------------
#
# Decode tokens/sec estimates across common hardware/variant combinations.
# Calibrated against observed wall-clock on a 96 GB Apple Silicon machine
# running gemma4:e2b / gemma4:e4b / gemma4:31b — previous values were roughly
# 2× too optimistic because they didn't account for prefill on long inputs
# or the MoE "effective vs total" parameter distinction. Adjusted downward
# to match reality; the generic "per call" chip number now matches what
# users actually observe on a cold chat turn.
#
# Keys: (arch_class, ram_class) → {tier: tokens_per_second}.
#   arch_class: 'apple_silicon' (arm64) or 'x86_64'
#   ram_class:  'low'   (<16 GB)
#               'mid'   (16–32 GB)
#               'high'  (32–64 GB)
#               'xhigh' (64 GB+)
_SPEED_TABLE = {
    ('apple_silicon', 'low'):   {'small': 25, 'medium': 12, 'large': 2.5, 'xlarge': 1.5},
    ('apple_silicon', 'mid'):   {'small': 35, 'medium': 20, 'large': 5.0, 'xlarge': 3.0},
    ('apple_silicon', 'high'):  {'small': 50, 'medium': 30, 'large': 8.0, 'xlarge': 5.5},
    ('apple_silicon', 'xhigh'): {'small': 60, 'medium': 35, 'large': 10.0, 'xlarge': 7.0},
    ('x86_64', 'low'):          {'small': 7,  'medium': 3,  'large': 0.6, 'xlarge': 0.3},
    ('x86_64', 'mid'):          {'small': 12, 'medium': 5,  'large': 1.2, 'xlarge': 0.7},
    ('x86_64', 'high'):         {'small': 18, 'medium': 8,  'large': 1.8, 'xlarge': 1.1},
    ('x86_64', 'xhigh'):        {'small': 22, 'medium': 10, 'large': 2.2, 'xlarge': 1.5},
}

# Typical output length for a full transcript-analysis pass — used for the
# "estimated wait" copy. The analyze_transcript / segment-vectors prompts emit
# structured JSON in roughly this range; conservative upper bound.
_REPRESENTATIVE_OUTPUT_TOKENS = 2000

def _classify_ram(ram_gb: float) -> str:
    if ram_gb < 16:
        return 'low'
    if ram_gb < 32:
        return 'mid'
    if ram_gb < 64:
        return 'high'
    return 'xhigh'


def _classify_arch(arch: str) -> str:
    return 'apple_silicon' if arch.lower().startswith('arm') else 'x86_64'


def _ollama_installed_models() -> set:
    """Return the set of Ollama model tags already pulled on this machine.

    Queries the Ollama HTTP API at ``localhost:11434/api/tags`` first — that
    works anywhere the Ollama app/daemon is running, including inside a
    ``.app`` bundle launched from Finder where ``/usr/local/bin`` isn't on
    PATH and ``shutil.which('ollama')`` can't find the binary. Falls back to
    shelling out for dev setups where the API isn't reachable but the CLI is.

    Empty set on both failing — the caller treats "unknown" and "not
    installed" the same (show "needs download"), which is the safe default.
    """
    # Primary: HTTP API. Stdlib-only per the module-level contract.
    try:
        import urllib.request
        req = urllib.request.Request(
            'http://localhost:11434/api/tags',
            headers={'Accept': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        tags = set()
        for m in data.get('models') or []:
            name = m.get('name') or m.get('model')
            if name:
                tags.add(name)
        if tags:
            return tags
    except Exception:
        # Silent fall-through — the CLI path below might still work.
        pass

    # Fallback: shell out to `ollama list`.
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return set()
        tags = set()
        for line in result.stdout.splitlines()[1:]:  # skip header row
            parts = line.split()
            if parts:
                tags.add(parts[0])
        return tags
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return set()


def _split_model_description(description: str) -> tuple:
    """Split a GEMMA4_VARIANTS description into ``(display_name, total_params)``.

    Examples:
      "Gemma 4 4B · 8B total (balanced)" → ("Gemma 4 4B (balanced)", "8B total")
      "Gemma 4 27B (high quality)"       → ("Gemma 4 27B (high quality)", "27B total")

    For dense models (no " · " segment) we pull the params from the name tail
    so the metadata row still has a total-params value — keeping the card
    layout consistent across variants.
    """
    import re
    paren_match = re.search(r'\s*\(([^)]+)\)\s*$', description)
    paren = paren_match.group(1) if paren_match else ''
    prefix = description[:paren_match.start()].strip() if paren_match else description.strip()

    if '·' in prefix:
        parts = [p.strip() for p in prefix.split('·')]
        name = parts[0]
        total = parts[1] if len(parts) > 1 else ''
    else:
        name = prefix
        size_match = re.search(r'(\d+\.?\d*\s*B)\s*$', name)
        total = f"{size_match.group(1).replace(' ', '')} total" if size_match else ''

    display_name = f"{name} ({paren})" if paren else name
    return display_name, total


def _format_casual_estimate(seconds: float, has_project_context: bool) -> str:
    """Cast a seconds estimate into a loosely-rounded, lowercase phrase.

    Scaled examples:
      - 30 s  → "About 30 sec ..."
      - 75 s  → "About 1 min ..."
      - 4 min → "About 3–5 min ..."
      - 27 min → "About 25–30 min ..."
      - 75 min → "About 60–90 min ..."

    The range is intentionally wide: hardware speed tables are approximate and
    the model's output length varies, so a single precise number misleads more
    than a loose range helps.
    """
    tail = 'for this project' if has_project_context else 'per call'
    if not seconds or seconds <= 0:
        return f"Time varies {tail}"

    if seconds < 45:
        s = max(5, int(round(seconds / 5.0) * 5))
        return f"About {s} sec {tail}"

    minutes = seconds / 60.0
    if minutes < 1.25:
        return f"About 1 min {tail}"
    if minutes < 2.25:
        return f"About 2 min {tail}"
    if minutes < 10:
        low = max(1, int(round(minutes - 1)))
        high = int(round(minutes + 1))
        if high <= low:
            high = low + 1
        return f"About {low}–{high} min {tail}"

    def _r5(x):
        return max(5, int(round(x / 5.0)) * 5)

    low = _r5(minutes * 0.85)
    high = _r5(minutes * 1.15)
    if high <= low:
        high = low + 5
    return f"About {low}–{high} min {tail}"


_ANALYSIS_CHUNK_MINUTES = 15  # keep in sync with ai_analysis.CHUNK_MINUTES
_OVERHEAD_SECONDS_PER_CALL = 3  # mirror ai_analysis._ESTIMATE_OVERHEAD_SECONDS_PER_CALL

# Each analysis call prefills ~10k input tokens (a 15-minute transcript
# slice + the analysis system prompt). On Apple Silicon prefill runs roughly
# 4–6× the decode token rate; we use 5× as the typical ratio. Ignoring this
# used to make long-interview estimates half what they should be.
_ANALYSIS_INPUT_TOKENS_PER_CALL = 10000
_PREFILL_VS_DECODE_RATIO = 5.0


def _chunks_for_duration(duration_seconds: float) -> int:
    """How many AI-analysis chunks a transcript of this length produces.

    Mirrors the chunking threshold in ``ai_analysis._iter_transcript_chunks``:
    short transcripts get one pass, anything 15 min or longer is split into
    ceil(duration / 15min) slices.
    """
    if not duration_seconds or duration_seconds <= 0:
        return 0
    chunk_seconds = _ANALYSIS_CHUNK_MINUTES * 60
    if duration_seconds < chunk_seconds:
        return 1
    import math
    return max(1, math.ceil(duration_seconds / chunk_seconds))


def _full_analysis_seconds(duration_seconds: float, tokens_per_sec: float) -> int:
    """Wall-clock estimate for a full ``analyze_transcript`` run.

    Accounts for both prefill (reading the ~10k input tokens per chunk at
    ~5× decode speed) and decode (emitting ~2k output tokens at the variant's
    decode rate). Previous iteration only counted decode, which made the
    estimate half the real wall-clock on long interviews.
    """
    chunks = _chunks_for_duration(duration_seconds)
    if chunks <= 0 or tokens_per_sec <= 0:
        return 0
    calls = chunks * 2  # story + social per chunk (UI default)
    prefill_per_call = _ANALYSIS_INPUT_TOKENS_PER_CALL / (tokens_per_sec * _PREFILL_VS_DECODE_RATIO)
    decode_per_call = _REPRESENTATIVE_OUTPUT_TOKENS / float(tokens_per_sec)
    per_call = prefill_per_call + decode_per_call + _OVERHEAD_SECONDS_PER_CALL
    return int(round(calls * per_call))


def get_variant_estimates(hw_info: dict = None, total_seconds: float = None) -> list:
    """Return per-variant info for the AI Model picker UI.

    Each entry has:
      - tier, variant, description (raw from GEMMA4_VARIANTS)
      - display_name ("Gemma 4 4B (balanced)") — for the card title
      - total_params ("8B total") — for the metadata row
      - download_size ("11.5 GB") — for the metadata row and download button
      - tokens_per_sec (float) — internal, not displayed
      - estimated_casual ("About 25–30 min for this project") — for the card
      - downloaded (bool)

    If ``total_seconds`` is supplied (typically a real project's transcript
    duration), ``estimated_casual`` reflects a full analysis run for that
    length — chunks × 2 AI calls × per-call time. Without it, we fall back to
    a single representative call and the suffix changes to "per call".
    """
    if hw_info is None:
        hw_info = detect_hardware_tier()

    arch_class = _classify_arch(hw_info.get('arch', ''))
    ram_class = _classify_ram(float(hw_info.get('ram_gb', 8.0)))
    speeds = _SPEED_TABLE.get((arch_class, ram_class), _SPEED_TABLE[('apple_silicon', 'mid')])

    installed = _ollama_installed_models()
    has_project_context = total_seconds is not None and total_seconds > 0

    rows = []
    for tier in VALID_TIERS:
        variant, download_size, description = GEMMA4_VARIANTS[tier]
        tps = speeds.get(tier, 0.0)
        is_downloaded = variant in installed
        if not is_downloaded and 'gemma4:latest' in installed and tier == 'medium':
            is_downloaded = True

        if has_project_context:
            estimated_seconds = _full_analysis_seconds(total_seconds, tps)
        elif tps > 0:
            estimated_seconds = _REPRESENTATIVE_OUTPUT_TOKENS / tps
        else:
            estimated_seconds = 0

        display_name, total_params = _split_model_description(description)

        rows.append({
            'tier': tier,
            'variant': variant,
            'description': description,
            'display_name': display_name,
            'total_params': total_params,
            'download_size': download_size,
            'tokens_per_sec': tps,
            'estimated_casual': _format_casual_estimate(estimated_seconds, has_project_context),
            'downloaded': is_downloaded,
        })
    return rows


def set_variant_manually(tier: str) -> dict:
    """Save a user-chosen tier to config (auto_selected=False) and return info.

    Raises ValueError for an invalid tier so the HTTP handler can surface a
    400 without special-casing.
    """
    if tier not in GEMMA4_VARIANTS:
        raise ValueError(f"invalid tier {tier!r}; must be one of {list(VALID_TIERS)}")
    variant, download_size, description = GEMMA4_VARIANTS[tier]
    save_model_config(tier, variant, auto_selected=False)
    return {
        'tier': tier,
        'variant': variant,
        'download_size': download_size,
        'description': description,
        'source': 'manual',
    }
