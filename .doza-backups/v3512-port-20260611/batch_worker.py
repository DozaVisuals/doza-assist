"""Per-item pipeline runner.

Calls into core/'s pure functions (all callable today, except
``create_project_from_path`` which is added by the OSS plugin-loader commit):

  - core: ``transcribe.extract_audio``
  - core: ``transcribe.transcribe_file``
  - core: ``ai_analysis.analyze_transcript``
  - core: ``ai_analysis.generate_segment_vectors``
  - core: ``doza_assist.retrieval.build_paragraph_index`` + ``save_index``
  - core: ``app.save_project``
  - core (NEW from OSS commit): ``app.create_project_from_path``
  - core: ``editorial_dna.profiles`` for the profile save/restore dance

If ``create_project_from_path`` isn't available yet (OSS commit not landed),
each item fails with a clear startup-style error so the editor knows what's
missing rather than getting a confusing "no such function" deep in the trace.

My Style profile injection happens via temporarily mutating the active-profile
set for the duration of the batch run. ``ai_analysis`` reads the active set
internally; we save the user's prior selection on entry and restore it after
the last item finishes (or after cancel). Trade-off: if the user opens
single-file mode while a batch is running with an overridden profile, that
single-file run picks up the batch's profile. Documented in the spec — users
walk away from a batch.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import traceback
from datetime import datetime

from .models import (
    BatchJob, BatchItem,
    JOB_RUNNING, JOB_COMPLETED, JOB_COMPLETED_WITH_ERRORS, JOB_CANCELLED,
    ITEM_EXTRACTING, ITEM_TRANSCRIBING, ITEM_ANALYZING,
    ITEM_GENERATING_SELECTS, ITEM_COMPLETE, ITEM_FAILED,
)


def run_batch(
    job: BatchJob,
    cancel_event: threading.Event,
    manager,
    projects_dir: str,
    exports_dir: str,
) -> None:
    """Worker entry point. Runs in a daemon thread until the batch completes,
    cancels, or every item has hit a terminal state.
    """
    job.status = JOB_RUNNING

    # Snapshot + override active profile set for the batch duration.
    # ``my_style_profile_id`` semantics live on BatchJob; see models.py.
    prior_active_ids = _snapshot_and_set_profile(job)

    try:
        for item in job.items:
            # Cancel checks happen between items only — the in-flight item
            # always finishes (or fails) cleanly.
            if cancel_event.is_set() or job.status == JOB_CANCELLED:
                break
            _run_item(item, job, projects_dir)
    finally:
        _restore_profile(prior_active_ids)
        completed_at = datetime.now().isoformat()
        if job.status != JOB_CANCELLED:
            job.status = (
                JOB_COMPLETED_WITH_ERRORS if job.failed_items > 0 else JOB_COMPLETED
            )
        manager._finish_job(job, completed_at)
        _fire_completion_notification(job)


def _run_item(item: BatchItem, job: BatchJob, projects_dir: str) -> None:
    item.started_at = datetime.now().isoformat()
    try:
        # Lazy imports keep the worker robust against partial OSS rollouts.
        # If core/ doesn't expose create_project_from_path yet, raise with a
        # message the editor can understand from the failed-item card.
        try:
            from app import create_project_from_path, save_project
        except ImportError as e:
            raise RuntimeError(
                'Batch needs the OSS plugin-loader / create_project_from_path '
                'commit to be landed in core/. Update Doza Assist and retry.'
            ) from e

        from transcribe import transcribe_file, extract_audio
        from ai_analysis import analyze_transcript, generate_segment_vectors
        from doza_assist.retrieval import build_paragraph_index, save_index

        # 1. Project creation. Expected signature:
        #       create_project_from_path(source_path, project_name=None) -> str
        #    where None defaults to filename-without-extension.
        project_name = os.path.splitext(item.file_name)[0] or item.file_name
        project_id = create_project_from_path(item.file_path, project_name)
        item.project_id = project_id
        project_dir = os.path.join(projects_dir, project_id)
        item.results_path = project_dir

        # 2. Audio extraction. Pure function, writes <project_dir>/audio.wav.
        item.status = ITEM_EXTRACTING
        item.progress_percent = 5
        extract_audio(item.file_path, project_dir=project_dir)

        # 3. Transcription. The slowest step on a typical interview.
        item.status = ITEM_TRANSCRIBING
        item.progress_percent = 15
        transcript = transcribe_file(item.file_path, project_dir=project_dir)

        # Persist the transcript to meta.json the same way core/app.py does
        # after its single-file /transcribe route.
        project = _load_meta(project_dir)
        project['transcript'] = transcript
        project['status'] = 'transcribed'
        save_project(project_id, project)

        # Auto-build paragraph index so the project's Chat tab works the
        # moment the editor opens it. Mirrors core/app.py:1509-1520.
        try:
            idx = build_paragraph_index(transcript)
            save_index(idx, os.path.join(project_dir, 'paragraph_index.json'))
        except Exception as e:
            print(
                f'[batch {job.id}] paragraph_index build failed for {project_id}: {e}',
                flush=True,
            )

        item.progress_percent = 50

        # 4. AI analysis. Profile injection happens implicitly via the
        # active-profile override set up in run_batch().
        item.status = ITEM_ANALYZING

        def _on_analysis_progress(step, total, current):
            # Map analyzer progress (0..total) to per-item percent (50..90).
            try:
                if total and total > 0:
                    pct = 50 + int(40 * (int(step) / float(total)))
                    item.progress_percent = max(item.progress_percent, min(90, pct))
            except Exception:
                pass

        analysis = analyze_transcript(
            transcript,
            project_name=project.get('name') or project_name,
            analysis_type='all',
            progress_callback=_on_analysis_progress,
        )

        # 5. "Selects" — per Q2, this IS the analysis output (story_beats,
        # strongest_soundbites, social_clips). No separate computation.
        # Persist the analysis on meta.json identically to /analyze.
        item.status = ITEM_GENERATING_SELECTS
        item.progress_percent = 92
        project['analysis'] = analysis
        project['status'] = 'analyzed'
        save_project(project_id, project)

        # Segment vectors give the Chat tab semantic ranking. Best-effort.
        try:
            vectors = generate_segment_vectors(
                transcript, project_name=project.get('name') or project_name,
            )
            if vectors:
                with open(os.path.join(project_dir, 'segment_vectors.json'), 'w') as f:
                    json.dump(vectors, f, indent=2)
        except Exception as e:
            print(
                f'[batch {job.id}] segment_vectors failed for {project_id}: {e}',
                flush=True,
            )

        # 6. Done.
        item.status = ITEM_COMPLETE
        item.progress_percent = 100
        item.completed_at = datetime.now().isoformat()

    except Exception as e:
        item.status = ITEM_FAILED
        item.error_message = (str(e) or e.__class__.__name__).strip()
        item.completed_at = datetime.now().isoformat()
        # Full traceback to stdout so it surfaces in the Electron child log.
        print(f'[batch {job.id}] item {item.file_name!r} failed:', flush=True)
        traceback.print_exc()


# ── Profile save/restore ─────────────────────────────────────────────────

def _snapshot_and_set_profile(job: BatchJob) -> list[str] | None:
    """Returns the prior active-id list (to restore later), or None if we
    chose not to touch it.

    Semantics on ``job.my_style_profile_id``:
      - None (omitted): leave the active set alone — return None.
      - '' (empty string): clear the active set for the batch.
      - '<id>': activate exactly that profile for the batch.
    """
    sentinel = job.my_style_profile_id
    if sentinel is None:
        return None
    try:
        from editorial_dna.profiles import get_active_profile_ids, set_active_ids
        prior = list(get_active_profile_ids())
        set_active_ids([sentinel] if sentinel else [])
        return prior
    except Exception as e:
        print(f'[batch {job.id}] could not override active profile: {e}', flush=True)
        return None


def _restore_profile(prior_active_ids: list[str] | None) -> None:
    if prior_active_ids is None:
        return
    try:
        from editorial_dna.profiles import set_active_ids
        set_active_ids(prior_active_ids)
    except Exception as e:
        print(f'[batch] could not restore active profile: {e}', flush=True)


# ── Misc helpers ─────────────────────────────────────────────────────────

def _load_meta(project_dir: str) -> dict:
    with open(os.path.join(project_dir, 'meta.json'), 'r') as f:
        return json.load(f)


def _fire_completion_notification(job: BatchJob) -> None:
    """Best-effort macOS notification. Never raises — a failed notify can't
    break the batch state.
    """
    try:
        if job.status == JOB_CANCELLED:
            msg = (
                f'Batch cancelled: {job.completed_items} of {job.total_items} processed'
            )
        elif job.failed_items:
            msg = (
                f'Batch complete: {job.completed_items} of {job.total_items} '
                'files processed'
            )
        else:
            msg = f'Batch complete: all {job.total_items} files processed'
        # Quote escaping: AppleScript strings are double-quoted; project names
        # don't appear in this message so simple substitution is safe.
        subprocess.run(
            [
                'osascript', '-e',
                f'display notification "{msg}" with title "Doza Assist"',
            ],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass
