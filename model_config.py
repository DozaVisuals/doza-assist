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

# Real Gemma 4 tags from ollama.com/library/gemma4 (verified 2026-04-19)
# tier -> (ollama_tag, download_size, description)
GEMMA4_VARIANTS = {
    'small':  ('gemma4:e2b', '7.2 GB', 'Gemma 4 2B (smallest, fastest)'),
    'medium': ('gemma4:e4b', '9.6 GB', 'Gemma 4 4B (balanced)'),
    'large':  ('gemma4:26b', '18 GB',  'Gemma 4 27B (high quality)'),
    'xlarge': ('gemma4:31b', '20 GB',  'Gemma 4 32B (highest quality)'),
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
