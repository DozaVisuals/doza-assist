#!/usr/bin/env python3
"""
Prints the detected hardware tier and selected Gemma 4 variant.
Run once to confirm hardware detection works on your system.

Usage:
  python test_hardware_tier.py
  python test_hardware_tier.py --model-tier small   # test a specific override
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_config import (
    detect_hardware_tier,
    get_gemma4_variant,
    format_selection_message,
    GEMMA4_VARIANTS,
    VALID_TIERS,
    MODEL_CONFIG_FILE,
)


def main():
    print("=" * 62)
    print("  Doza Assist — Hardware Tier Detection")
    print("=" * 62)
    print()

    hw = detect_hardware_tier()
    print(f"  RAM          : {hw['ram_gb']:.1f} GB")
    print(f"  Architecture : {hw['arch']}")
    print(f"  Disk (free)  : {hw['disk_gb']:.1f} GB")
    print()

    info = get_gemma4_variant()
    print(format_selection_message(info))
    print()

    print("  All available tiers:")
    for tier, (tag, size, desc) in GEMMA4_VARIANTS.items():
        marker = "  <-- selected" if tier == info['tier'] else ""
        print(f"    {tier:8s}: {tag:16s}  {size:6s}  {desc}{marker}")
    print()

    print(f"  Config file  : {MODEL_CONFIG_FILE}")
    print()

    # If the user passed a tier override on the command line, show what that would do
    for arg in sys.argv[1:]:
        t = arg.lstrip('-').split('=')[-1].lower()
        if t in VALID_TIERS:
            override = get_gemma4_variant(cli_tier=t)
            print(f"  Override --model-tier={t}:")
            print(f"    Variant: {override['variant']}  ({override['download_size']})")
            print()


if __name__ == '__main__':
    main()
