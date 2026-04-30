"""Storytelling foundation injection for AI system prompts.

Loads ``docs/storytelling-foundation-oss.md`` (or a path given via the
``DOZA_STORYTELLING_PATH`` environment variable) and prepends its
contents to the system prompt sent to the local LLM. This gives the
model a stable operating manual of decision rules, anti-patterns, and
self-check questions so its choices stay consistent across sessions.

Per-task section routing
------------------------
The foundation document is divided into named sections. Different LLM
call sites need different subsets of the document, both for relevance
and to avoid overflowing the model's context window:

  - AI Chat default ............. Decision Rules, Anti-Patterns, Self-Check
  - AI Analysis ................. Anatomy of a Great Clip, Reading Spoken
                                  Word, Format-Specific Clip Logic, Decision
                                  Rules, Anti-Patterns, Self-Check
  - Story Builder ............... Foundational Story Frameworks, Emotional
                                  Arc, Theme and Through-Line, Narrative
                                  Reordering Principles
  - Internal subroutines ........ NONE. The chunked search and segment
                                  vector classifier each run hundreds of
                                  small LLM calls per long transcript;
                                  prepending the foundation to every one
                                  of them blows past the context window
                                  with no benefit (these calls don't make
                                  user-facing reasoning decisions).

Routing is based on a content-pattern detection of the system_prompt
the caller passed in. If detection misses, defaults to the lightest set
(AI Chat default) — better to under-include than to overflow.

For the OSS lighter doc (which only contains the operational sections
9, 10, 12 of the master), most task-specific section requests fall
through to "return the whole doc" since the requested section titles
aren't in the lighter doc. The lighter doc is small enough that this
is fine.

Skipped entirely if:
  - the system prompt already contains ``<storytelling_foundation>``
    (signalling an upstream caller has already injected one),
  - the environment variable ``DOZA_STORYTELLING_DISABLED`` is set, or
  - the storytelling document cannot be located or read.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional


# docs/ sits one level up from this file (editorial_dna/ -> repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DOC_PATH = _REPO_ROOT / "docs" / "storytelling-foundation-oss.md"


# Per-task section title routing. Keys are detected task names. Values
# are case-insensitive substrings to match against ``## ...`` headers
# in the foundation document. Empty list = skip the foundation entirely.
_TASK_SECTIONS: dict[str, list[str]] = {
    "ai_chat": [
        "Anatomy of a Great Clip",
        "Format-Specific Clip Logic",
        "Decision Rules and Heuristics",
        "Anti-Patterns and Failure Modes",
        "Self-Check Questions",
    ],
    "ai_analysis": [
        "Anatomy of a Great Clip",
        "Reading Spoken Word in Transcript Form",
        "Format-Specific Clip Logic",
        "Decision Rules and Heuristics",
        "Anti-Patterns and Failure Modes",
        "Self-Check Questions",
    ],
    "story_builder": [
        "Foundational Story Frameworks",
        "Emotional Arc and Story Shape",
        "Theme and Through-Line",
        "Narrative Reordering Principles",
        "Decision Rules and Heuristics",
    ],
    # Long-transcript chunked search runs many times per query. We can't
    # afford the full ai_chat section set on each call, but we DO need
    # format-aware reasoning so platform hints in the user's query
    # ("instagram reels", "linkedin", "podcast", "broadcast") are
    # interpreted as format requests rather than literal keyword
    # searches against the transcript.
    "chunked_search": [
        "Anatomy of a Great Clip",
        "Format-Specific Clip Logic",
    ],
    # Mechanical classifier subroutine (segment vectors). No storytelling
    # reasoning needed.
    "internal_skip": [],
}


def _detect_task(system_prompt: str) -> str:
    """Map a system_prompt to a routing task key by content pattern.

    Patterns are anchored on the exact opening lines OSS uses, so matches
    are stable across paraphrasing of the body. Unknown prompts fall
    back to "ai_chat" (lightest set) — better to under-include than to
    overflow the context window.
    """
    head = (system_prompt or "")[:1500].lower()

    # Long-transcript chunked search: many calls per query, but they
    # need format awareness to interpret platform hints in the user's
    # question. Lightweight section subset (Anatomy + Format Logic).
    if "scanning excerpt" in head:
        return "chunked_search"

    # Mechanical classifier subroutines — skip foundation entirely.
    if "documentary story analyst" in head and "segment vectors" in head:
        return "internal_skip"
    if "documentary story analyst" in head:
        return "internal_skip"

    # Story Builder paths
    if "story editor building a narrative sequence" in head:
        return "story_builder"
    if "documentary story editor" in head and "menu of pre-classified" in head:
        return "story_builder"

    # AI Analysis paths
    if "expert documentary film editor" in head:
        return "ai_analysis"
    if "social media content strategist" in head:
        return "ai_analysis"

    # AI Chat
    if "expert editorial consultant" in head and "hard output contract" in head:
        return "ai_chat"

    return "ai_chat"


def _select_sections(text: str, requested_titles: Iterable[str]) -> str:
    """Return the preamble plus only the sections whose ``## `` header
    matches one of ``requested_titles`` (case-insensitive substring).

    If no headers exist in the document, returns the whole document
    (the lighter OSS doc has none and is small enough to ship whole).
    If headers exist but none match, returns just the preamble (rare —
    means the document doesn't have the requested topic and the caller
    is operating on a doc it didn't expect).
    """
    headers = list(re.finditer(r"^##\s+.*$", text, re.MULTILINE))
    if not headers:
        return text

    requested = [t.lower() for t in requested_titles]
    if not requested:
        # Caller asked for zero sections (internal_skip case). Return
        # empty so injection is suppressed entirely.
        return ""

    preamble = text[: headers[0].start()].rstrip()
    parts: list[str] = []
    if preamble:
        parts.append(preamble)

    matched_any = False
    for i, m in enumerate(headers):
        # Strip "## " and an optional "Section N:" prefix from the title.
        raw_title = m.group()[2:].strip()
        title = re.sub(r"^Section\s+\d+:\s*", "", raw_title).lower()
        if any(req in title for req in requested):
            start = m.start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            parts.append(text[start:end].rstrip())
            matched_any = True

    if not matched_any:
        # Document doesn't have the requested topic. Return whole doc —
        # for the OSS lighter doc this fallback applies when callers
        # request "Foundational Frameworks" etc. which aren't in it,
        # and the lighter doc is small enough to ship whole.
        return text

    return "\n\n".join(parts)


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

    task = _detect_task(system_prompt)
    sections_for_task = _TASK_SECTIONS.get(task, _TASK_SECTIONS["ai_chat"])

    # Internal subroutines explicitly skip the foundation — many small
    # LLM calls per long transcript, prepending the foundation to all
    # of them would overflow the context window.
    if task == "internal_skip":
        return system_prompt

    path = _resolve_path()
    if path is None:
        return system_prompt

    try:
        full_text = path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"[storytelling] could not read {path}: {e}")
        return system_prompt

    body = _select_sections(full_text, sections_for_task).strip()
    if not body:
        return system_prompt

    block = (
        f"<storytelling_foundation task=\"{task}\">\n"
        f"{body}\n"
        "</storytelling_foundation>\n\n"
    )
    return block + system_prompt
