"""Background diarization worker.

A daemon worker thread pulls project IDs from a queue. For each project it:

1. Reads ``meta.json`` to confirm a transcript exists and locate the audio file.
2. Runs pyannote diarization (long; CPU runtime ~0.75x audio duration).
3. Re-reads ``meta.json`` (in case the transcript was edited mid-run) and
   merges speaker labels onto each segment via the alignment module.
4. Writes the updated ``meta.json`` and a top-level ``diarization`` block
   recording model/elapsed/speakers.

Progress is persisted to ``projects/<id>/diarization_status.json`` so the
frontend can poll without holding HTTP connections open.

The worker is intentionally single-threaded (one diarization at a time) —
pyannote is CPU-bound and parallelism would just thrash. Queue is FIFO.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from .alignment import align_speakers
from .pipeline import DEFAULT_MODEL_ID, DiarizationError, run_diarization

logger = logging.getLogger(__name__)


STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"
STATUS_NOT_STARTED = "not_started"


def status_path(project_dir: Path) -> Path:
    return Path(project_dir) / "diarization_status.json"


def read_status(project_dir: Path) -> dict:
    path = status_path(project_dir)
    if not path.exists():
        return {"status": STATUS_NOT_STARTED}
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.warning("could not parse %s: %s", path, e)
        return {"status": STATUS_NOT_STARTED}


def write_status(project_dir: Path, **fields) -> None:
    path = status_path(project_dir)
    existing = read_status(project_dir)
    if existing.get("status") == STATUS_NOT_STARTED:
        existing = {}
    existing.update(fields)
    existing["updated_at"] = datetime.now().isoformat()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write so a concurrent reader never sees a half-written file.
        fd, tmp = tempfile.mkstemp(prefix=".diarization_status_", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(existing, f, indent=2)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise
    except Exception as e:
        logger.warning("could not write status %s: %s", path, e)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


class DiarizationWorker:
    """Singleton background worker. Project-id queue, one job at a time."""

    def __init__(self, projects_dir: Path) -> None:
        self.projects_dir = Path(projects_dir)
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._stop_event = threading.Event()
        self._current: Optional[str] = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._loop, name="diarization-worker", daemon=True
        )
        self._thread.start()
        logger.info("DiarizationWorker started (projects_dir=%s)", self.projects_dir)

    @property
    def current_project(self) -> Optional[str]:
        with self._lock:
            return self._current

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def enqueue(self, project_id: str) -> None:
        if not project_id:
            return
        project_dir = self.projects_dir / project_id
        write_status(
            project_dir,
            status=STATUS_QUEUED,
            started_at=None,
            completed_at=None,
            error=None,
        )
        self._queue.put(project_id)
        logger.info("diarization enqueued: %s", project_id)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            project_id = self._queue.get()
            with self._lock:
                self._current = project_id
            try:
                self._process(project_id)
            except Exception as e:
                logger.error("diarization worker crash on %s: %s", project_id, e, exc_info=True)
                try:
                    write_status(
                        self.projects_dir / project_id,
                        status=STATUS_ERROR,
                        error=f"worker crashed: {type(e).__name__}: {e}",
                    )
                except Exception:
                    pass
            finally:
                with self._lock:
                    self._current = None
                self._queue.task_done()

    def _process(self, project_id: str) -> None:
        project_dir = self.projects_dir / project_id
        meta_path = project_dir / "meta.json"
        if not meta_path.exists():
            write_status(project_dir, status=STATUS_ERROR, error="meta.json missing")
            return

        try:
            meta = json.loads(meta_path.read_text())
        except Exception as e:
            write_status(project_dir, status=STATUS_ERROR, error=f"meta.json unreadable: {e}")
            return

        transcript = meta.get("transcript") or {}
        segments = transcript.get("segments") or []
        if not segments:
            logger.info("diarization: project %s has no transcript segments; skipping", project_id)
            write_status(project_dir, status=STATUS_SKIPPED, error="no transcript segments")
            return

        audio_path = self._resolve_audio_path(project_dir, meta)
        if audio_path is None:
            write_status(project_dir, status=STATUS_ERROR, error="audio file not found")
            return

        write_status(
            project_dir,
            status=STATUS_RUNNING,
            started_at=datetime.now().isoformat(),
            error=None,
            audio_path=str(audio_path),
            segment_count=len(segments),
        )

        try:
            t0 = time.perf_counter()
            diar_segments = run_diarization(audio_path)
            elapsed = time.perf_counter() - t0
        except DiarizationError as e:
            logger.error("diarization config error for %s: %s", project_id, e)
            write_status(project_dir, status=STATUS_ERROR, error=str(e))
            return
        except Exception as e:
            logger.error("diarization run error for %s: %s", project_id, e, exc_info=True)
            write_status(project_dir, status=STATUS_ERROR, error=f"{type(e).__name__}: {e}")
            return

        if not diar_segments:
            write_status(
                project_dir,
                status=STATUS_ERROR,
                error="diarization returned no segments",
                elapsed_seconds=round(elapsed, 1),
            )
            return

        # Re-read meta in case the user edited the transcript mid-run.
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as e:
            write_status(project_dir, status=STATUS_ERROR, error=f"meta.json unreadable after run: {e}")
            return

        segments = (meta.get("transcript") or {}).get("segments") or []
        if not segments:
            write_status(project_dir, status=STATUS_SKIPPED, error="transcript cleared during run")
            return

        align_speakers(segments, diar_segments)
        meta.setdefault("transcript", {})["segments"] = segments
        speakers = sorted({s.speaker for s in diar_segments})
        meta["diarization"] = {
            "model": os.environ.get("DIARIZATION_MODEL", DEFAULT_MODEL_ID),
            "device": (os.environ.get("DIARIZATION_DEVICE") or "mps").lower(),
            "completed_at": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "speakers": speakers,
            "diar_segment_count": len(diar_segments),
        }

        try:
            _atomic_write_json(meta_path, meta)
        except Exception as e:
            write_status(project_dir, status=STATUS_ERROR, error=f"could not save meta.json: {e}")
            return

        # Persist the raw exclusive-diarization segments next to meta.json.
        # The spot-check helper consumes this to play windows around speaker
        # changes; meta.json's per-transcript-segment speaker field is the
        # aligned/merged view, not the raw boundaries from pyannote.
        diar_segments_path = project_dir / "diarization_segments.json"
        try:
            _atomic_write_json(
                diar_segments_path,
                {
                    "model": meta["diarization"]["model"],
                    "completed_at": meta["diarization"]["completed_at"],
                    "exclusive_diarization": [
                        {"start": s.start, "end": s.end, "speaker": s.speaker}
                        for s in diar_segments
                    ],
                },
            )
        except Exception as e:
            # Non-fatal: meta.json is already saved with aligned speakers, so
            # the user-facing surface is correct. Only the spot-check helper
            # loses its native input — it can still walk meta.json segments.
            logger.warning("could not save raw diar segments to %s: %s", diar_segments_path, e)

        write_status(
            project_dir,
            status=STATUS_DONE,
            completed_at=datetime.now().isoformat(),
            elapsed_seconds=round(elapsed, 1),
            speakers=speakers,
            error=None,
        )
        logger.info(
            "diarization done: %s — %d speakers, %.1fs elapsed",
            project_id, len(speakers), elapsed,
        )

        # Auto-name the speakers now that diarization is done. This also
        # runs as a post-AI-Analysis step in the core analyze worker, but
        # that step skips if diarization hadn't finished yet — and never
        # retries. When diarization finishes AFTER analysis (the common
        # race), this is the call that actually populates speaker_names.
        # auto_name_speakers is itself a no-op when speaker_names already
        # has entries, so running it from both places is safe and
        # idempotent. Never let a naming failure mark diarization as
        # errored — it's already done and saved above.
        try:
            from .auto_naming import auto_name_speakers
            auto_name_speakers(project_id, project_dir.parent)
        except Exception as e:
            logger.warning("auto_name_speakers after diarization failed for %s: %s",
                           project_id, e)

    def _resolve_audio_path(self, project_dir: Path, meta: dict) -> Optional[Path]:
        # Prefer local WAVs already decoded during transcription — they're
        # 16 kHz mono PCM, so pyannote only has to resample (no codec decode,
        # no encoder-delay ambiguity).
        for name in ("timeline_audio.wav", "audio.wav"):
            candidate = project_dir / name
            if candidate.exists():
                return candidate
        source = meta.get("source_path") or meta.get("filepath")
        if source:
            p = Path(source)
            if p.exists():
                return p
        return None


_worker_lock = threading.Lock()
_worker: Optional[DiarizationWorker] = None


def get_worker(projects_dir: Path) -> DiarizationWorker:
    """Return (and lazy-create) the singleton worker for a given projects_dir."""
    global _worker
    if _worker is not None:
        return _worker
    with _worker_lock:
        if _worker is None:
            _worker = DiarizationWorker(Path(projects_dir))
        return _worker
