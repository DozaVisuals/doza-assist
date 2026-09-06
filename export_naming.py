"""Names for exported timelines and their files (1.1).

One rule for every export path (raw media, round-trip, Story Builder,
markers, collections, batch):

    timeline = "{Project} – {Kind} {N}"      en dash with spaces
    kinds    = "Selects", "Markers", "Story: {title}"

``N`` is a per-project counter kept in meta.json under ``export_counts``,
keyed by kind, and advanced after each successful export of that kind, so
the first export is "Selects 1" and a re-export never lands on a same-named
timeline in Final Cut. Story exports carry no N on their first export (the
title tells them apart); a re-export of the same story appends N from the
second export on.

Events: round-trip keeps the editor's original event name; raw media uses the
project name. Fallbacks when a name is missing: "Doza Assist" for the event
and "Interview" for the project base. The word Doza never goes into a
timeline name; provenance is one "Doza Assist" keyword on each clip.

The filename is the timeline name plus the extension, made safe for the
filesystem with :func:`safe_filename`, so Finder and Final Cut show the same
name.
"""
from __future__ import annotations

import re

EN_DASH = ' – '
PROJECT_FALLBACK = 'Interview'
EVENT_FALLBACK = 'Doza Assist'
PROVENANCE_KEYWORD = 'Doza Assist'

KIND_SELECTS = 'selects'
KIND_MARKERS = 'markers'
KIND_STORY = 'story'


def safe_filename(name: str, fallback: str = 'Export') -> str:
    """Make a user/AI-supplied string safe to use as a single filename.

    Replaces '/' and ':' (the legacy HFS separator, which Finder displays as
    '/') with '-', turns NULs and other control characters into spaces,
    collapses whitespace, and returns ``fallback`` when nothing displayable
    survives. For filenames only; never feed the result back into
    user-visible text.
    """
    cleaned = []
    for ch in str(name or ''):
        if ch in '/:':
            cleaned.append('-')
        elif ord(ch) < 32 or ch == '\x7f':
            cleaned.append(' ')
        else:
            cleaned.append(ch)
    out = ' '.join(''.join(cleaned).split())
    return out or fallback


def project_base(project) -> str:
    """The project's display name, or the fallback base."""
    name = ((project or {}).get('name') if isinstance(project, dict) else project) or ''
    name = re.sub(r'\s+', ' ', str(name)).strip()
    return name or PROJECT_FALLBACK


def event_name_for(project) -> str:
    """Raw-media exports: the event carries the project name, nothing added."""
    name = ((project or {}).get('name') if isinstance(project, dict) else project) or ''
    name = re.sub(r'\s+', ' ', str(name)).strip()
    return name or EVENT_FALLBACK


def story_kind(story_title: str) -> str:
    """Counter key for one story title."""
    title = re.sub(r'\s+', ' ', str(story_title or 'Story')).strip().lower()
    return f'{KIND_STORY}:{title}'


def export_count(project: dict, kind: str) -> int:
    """How many exports of ``kind`` this project has done so far."""
    counts = (project or {}).get('export_counts') or {}
    try:
        return int(counts.get(kind, 0) or 0)
    except (TypeError, ValueError):
        return 0


def next_count(project: dict, kind: str) -> int:
    return export_count(project, kind) + 1


def timeline_name(project, kind: str, n: int | None = None,
                  story_title: str | None = None, override: str | None = None) -> str:
    """The Final Cut timeline (project) name for one export.

    ``override`` (the Export tab's Timeline name field) wins when non-empty:
    whitespace-collapsed, counter-free. ``n`` is the export number for this
    kind: Selects and Markers always show it; a story shows it only from 2.
    """
    if override and str(override).strip():
        return re.sub(r'\s+', ' ', str(override)).strip()
    base = project_base(project)
    if kind == KIND_SELECTS:
        label = 'Selects'
    elif kind == KIND_MARKERS:
        label = 'Markers'
    elif kind == KIND_STORY or kind.startswith(KIND_STORY + ':'):
        title = re.sub(r'\s+', ' ', str(story_title or 'Story')).strip() or 'Story'
        name = f'{base}{EN_DASH}Story: {title}'
        if n and n >= 2:
            name += f' {n}'
        return name
    else:
        label = str(kind).strip().capitalize() or 'Export'
    name = f'{base}{EN_DASH}{label}'
    if n:
        name += f' {n}'
    return name


def filename_for(timeline: str, ext: str = '.fcpxml') -> str:
    """The file that carries ``timeline``: same name, made safe, plus ext."""
    ext = ext if ext.startswith('.') else '.' + ext
    return safe_filename(timeline, fallback='Export') + ext
