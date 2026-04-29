"""Storytelling foundation injection for AI system prompts.

Loads ``docs/storytelling-foundation-oss.md`` (or a path given via the
``DOZA_STORYTELLING_PATH`` environment variable) and prepends its
contents to the system prompt sent to the local LLM. This gives the
model a stable operating manual of decision rules, anti-patterns, and
self-check questions so its choices stay consistent across sessions.

The injection is wrapped in an XML-style ``<storytelling_foundation>``
tag for readability and so downstream tooling can detect / strip it.

Skipped if:
  - the system prompt already contains ``<storytelling_foundation>``
    (signalling an upstream caller has already injected one),
  - the environment variable ``DOZA_STORYTELLING_DISABLED`` is set, or
  - the storytelling document cannot be located or read.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


# docs/ sits one level up from this file (editorial_dna/ -> repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DOC_PATH = _REPO_ROOT / "docs" / "storytelling-foundation-oss.md"


def _resolve_path() -> Optional[Path]:
    env = os.environ.get("DOZA_STORYTELLING_PATH")
    if env:
        candidate = Path(env)
        if candidate.exists():
            return candidate
    return _DEFAULT_DOC_PATH if _DEFAULT_DOC_PATH.exists() else None


def inject_storytelling_foundation(system_prompt: str) -> str:
    """Prepend a ``<storytelling_foundation>`` block to ``system_prompt``.

    Idempotent: returns the prompt unchanged if a storytelling block is
    already present, if the disable env var is set, or if the document
    cannot be loaded. Errors during file read are swallowed so the LLM
    call still proceeds without the foundation rather than failing.
    """
    if not system_prompt:
        return system_prompt
    if "<storytelling_foundation>" in system_prompt:
        return system_prompt
    if os.environ.get("DOZA_STORYTELLING_DISABLED"):
        return system_prompt

    path = _resolve_path()
    if path is None:
        return system_prompt

    try:
        body = path.read_text(encoding="utf-8").strip()
    except OSError as e:
        print(f"[storytelling] could not read {path}: {e}")
        return system_prompt

    if not body:
        return system_prompt

    block = (
        "<storytelling_foundation>\n"
        f"{body}\n"
        "</storytelling_foundation>\n\n"
    )
    return block + system_prompt
