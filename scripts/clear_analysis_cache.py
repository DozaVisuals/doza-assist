#!/usr/bin/env python3
"""Clear cached AI-analysis results from project meta.json files.

The v0.5.x analysis pipeline could silently drop most of a long interview's
analysis when a chunk-merge produced a non-dict response (Gemma 4b in
particular). Once the bug fix lands, those broken results stay cached on
disk under ``meta.json -> analysis`` and ``meta.json -> analysis_cache``
until the editor explicitly re-runs analysis with ``force=true``. This
script wipes both fields so the next /analyze call regenerates from
scratch.

Usage:
    python3 clear_analysis_cache.py <project_id> [<project_id> ...]
    python3 clear_analysis_cache.py --all

Without arguments, prints a list of projects that have cached analyses
and exits without writing anything (dry run).

Safe to re-run; missing or already-cleared projects are skipped quietly.
The transcript itself is not touched, only the AI-analysis fields.
"""

import json
import os
import sys

SUPPORT_DIR = os.path.expanduser("~/Library/Application Support/DozaAssist")
PROJECTS_DIR = os.path.join(SUPPORT_DIR, "projects")


def list_projects_with_analysis():
    """Return a list of (project_id, name) for every project whose
    meta.json carries cached analysis data."""
    rows = []
    if not os.path.isdir(PROJECTS_DIR):
        return rows
    for pid in sorted(os.listdir(PROJECTS_DIR)):
        meta_path = os.path.join(PROJECTS_DIR, pid, "meta.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            continue
        has_analysis = bool(meta.get("analysis"))
        has_cache = bool(meta.get("analysis_cache"))
        if has_analysis or has_cache:
            rows.append((pid, meta.get("name", "?"), has_analysis, has_cache))
    return rows


def clear(project_id):
    """Clear analysis fields for a single project. Returns True on success,
    False if the project doesn't exist or has nothing to clear."""
    meta_path = os.path.join(PROJECTS_DIR, project_id, "meta.json")
    if not os.path.isfile(meta_path):
        print(f"  {project_id}: meta.json not found — skipping")
        return False
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception as e:
        print(f"  {project_id}: read failed ({e}) — skipping")
        return False
    had_analysis = bool(meta.get("analysis"))
    had_cache = bool(meta.get("analysis_cache"))
    if not (had_analysis or had_cache):
        print(f"  {project_id}: already clear")
        return False
    meta["analysis"] = None
    meta["analysis_cache"] = {}
    try:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"  {project_id}: write failed ({e}) — skipping")
        return False
    bits = []
    if had_analysis:
        bits.append("analysis")
    if had_cache:
        bits.append("cache")
    print(f"  {project_id}: cleared ({', '.join(bits)}) — {meta.get('name', '?')}")
    return True


def main(argv):
    if not argv:
        rows = list_projects_with_analysis()
        if not rows:
            print("No projects with cached analysis found.")
            return 0
        print("Projects with cached analysis (dry run — pass IDs or --all to clear):")
        for pid, name, has_analysis, has_cache in rows:
            tags = []
            if has_analysis:
                tags.append("analysis")
            if has_cache:
                tags.append("cache")
            print(f"  {pid}  [{', '.join(tags)}]  {name}")
        return 0

    if argv[0] == "--all":
        targets = [pid for pid, *_ in list_projects_with_analysis()]
        if not targets:
            print("Nothing to clear.")
            return 0
        print(f"Clearing {len(targets)} project(s):")
    else:
        targets = argv

    cleared = 0
    for pid in targets:
        if clear(pid):
            cleared += 1
    print(f"Done. Cleared {cleared} project(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
