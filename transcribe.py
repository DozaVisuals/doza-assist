"""
Transcription engine for Doza Assist.
Uses WhisperX for transcription + speaker diarization.
Falls back to standard Whisper if WhisperX is not available.
"""

import os
import ssl
import sys
import json
import shutil
import time
import subprocess
import tempfile
import certifi

# Fix macOS Python SSL certificates (needed for Whisper model downloads)
_cert_file = certifi.where()
os.environ.setdefault('SSL_CERT_FILE', _cert_file)
os.environ.setdefault('REQUESTS_CA_BUNDLE', _cert_file)
_original_create_context = ssl.create_default_context
def _create_ssl_context(*args, **kwargs):
    ctx = _original_create_context(*args, **kwargs)
    ctx.load_verify_locations(_cert_file)
    return ctx
ssl.create_default_context = _create_ssl_context


# ── mlx startup probe ────────────────────────────────────────────────
# Detect the macOS Metal SDK drift class of regression at app boot
# instead of mid-transcribe. A bundled mlx wheel compiled against the
# macOS 26 (Tahoe) Metal SDK emits a .metallib at Shader Language
# version 4.0, which only loads on macOS 26. Users still on macOS 15
# (Sequoia) hit "Failed to load the default metallib" and the
# transcribe path silently falls through to Whisper-CPU (~10x slower,
# ~7 GB RAM). Surfacing the failure here makes the support flow obvious
# instead of looking like a generic "transcription is slow" ticket.
try:
    import mlx  # noqa: F401
    import mlx.nn  # triggers metallib load
    _mlx_version = getattr(mlx, "__version__", "?")
    print(f"mlx OK: {_mlx_version}", flush=True)
except Exception as _mlx_err:
    print(
        f"WARN mlx unavailable at startup: {_mlx_err}. "
        "Parakeet path will be disabled; transcription will fall back "
        "to Whisper. If the message above mentions 'language version 4', "
        "the bundled mlx wheel is incompatible with this macOS version "
        "(macOS 26 build host shipping wheels that need macOS 26 runtime); "
        "pin mlx to a version compiled against an older Metal SDK in "
        "requirements-bundle.txt and rebuild the bundle.",
        flush=True,
    )


def _ensure_ffmpeg_on_path():
    """Ensure ffmpeg is discoverable on PATH (needed by whisper internally).

    Resolution order:
      1. ``DOZA_FFMPEG_DIR`` env var — set by the 0.7.0 Electron shell to
         the path of the bundled LGPL ffmpeg build inside the .app. Wins
         so the user gets the deterministic version we shipped.
      2. Existing PATH entry (e.g. Homebrew install on a dev machine).
      3. Common Homebrew prefixes as a last resort.
    """
    bundled = os.environ.get('DOZA_FFMPEG_DIR')
    if bundled and os.path.isfile(os.path.join(bundled, 'ffmpeg')):
        os.environ['PATH'] = bundled + ':' + os.environ.get('PATH', '')
        return
    if shutil.which('ffmpeg'):
        return
    for bin_dir in ['/opt/homebrew/bin', '/usr/local/bin']:
        if os.path.isfile(os.path.join(bin_dir, 'ffmpeg')):
            os.environ['PATH'] = bin_dir + ':' + os.environ.get('PATH', '')
            return


# Run once at import time so whisper/whisperx can find ffmpeg
_ensure_ffmpeg_on_path()


# ── Model cache ──
# Transcription models are expensive to load (5–15s) and large (~600MB for
# Parakeet-TDT, ~3GB for Whisper large-v3). Loading once per process and
# reusing across transcriptions saves that cost for every file after the
# first. Memory stays high while the app runs — this is a desktop app,
# which is fine. A single process lock serializes the first load so two
# concurrent transcriptions don't race to load the same model.
import threading

_model_lock = threading.Lock()
_parakeet_model = None
_whisperx_model = None          # (model, device, compute_type)
_whisperx_align_cache = {}      # {lang_code: (model_a, metadata, device)}
_whisper_cache = {}             # {model_name: model}


def _find_ffmpeg():
    """Find the ffmpeg binary.

    Same resolution order as ``_ensure_ffmpeg_on_path`` (bundled first,
    PATH, then Homebrew prefixes).
    """
    bundled_dir = os.environ.get('DOZA_FFMPEG_DIR')
    if bundled_dir:
        candidate = os.path.join(bundled_dir, 'ffmpeg')
        if os.path.isfile(candidate):
            return candidate
    path = shutil.which('ffmpeg')
    if path:
        return path
    for candidate in ['/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg']:
        if os.path.isfile(candidate):
            return candidate
    return 'ffmpeg'  # fall back, will error if not found


# ── Whisper progress hook ──
# whisper.transcribe creates `tqdm.tqdm(total=content_frames, ...)` and calls
# `pbar.update(n)` after each ~30s decode window — even when the bar is
# disabled (verbose=None). Swapping the `tqdm` reference inside the
# whisper.transcribe module for a subclass turns those updates into a real
# progress signal, replacing the wall-clock estimator's guesswork during
# the long CPU decode. ContextVar routing keeps concurrent jobs from
# cross-reporting. Fail-open: if a future whisper drops tqdm, the
# estimator fallback still runs. [Ported from OSS v3.5.12 / 61c8ae5.]
import contextvars

_whisper_progress_cb = contextvars.ContextVar('whisper_progress_cb', default=None)


def _install_whisper_progress_hook():
    """Install the tqdm shim into whisper.transcribe (idempotent).
    Returns True when the hook is active."""
    try:
        # NOT `import whisper.transcribe as _wt`: whisper/__init__ rebinds
        # the `transcribe` attribute to the FUNCTION, so attribute-style
        # import grabs that instead of the module. import_module returns
        # the real module from sys.modules.
        import importlib
        _wt = importlib.import_module('whisper.transcribe')
        if getattr(_wt, '_doza_progress_hooked', False):
            return True
        if not hasattr(_wt, 'tqdm') or not hasattr(_wt.tqdm, 'tqdm'):
            return False
        import tqdm as _tqdm_mod

        class _ProgressTqdm(_tqdm_mod.tqdm):
            def update(self, n=1):
                try:
                    self._doza_seen = getattr(self, '_doza_seen', 0) + (n or 0)
                    cb = _whisper_progress_cb.get()
                    if cb and self.total:
                        cb(min(self._doza_seen / self.total, 1.0))
                except Exception:
                    pass
                return super().update(n)

        class _TqdmShim:
            tqdm = _ProgressTqdm

        _wt.tqdm = _TqdmShim
        _wt._doza_progress_hooked = True
        return True
    except Exception:
        return False


def _audio_stream_plan(filepath):
    """Probe the source's audio streams once per extraction.

    Returns (stream_count, extra_ffmpeg_args).

    - 0 streams -> the caller raises a clear "no audio track" error instead
      of surfacing ffmpeg's raw "Output file does not contain any stream".
    - 1 stream  -> no extra args (ffmpeg's default selection is correct).
    - N streams -> mix them all: camera MXF/MOV records the real interview
      mic on track 2+ with scratch on track 1, and ffmpeg's default picks
      exactly ONE stream — the wrong-mic (or near-empty) transcript class
      of bug. amix is enabled in the bundled LGPL ffmpeg build.
    """
    from exporters.media_probe import _find_ffprobe
    ffprobe = _find_ffprobe()
    count = 1  # fail open: assume one stream if probing is impossible
    if ffprobe:
        try:
            result = subprocess.run(
                [ffprobe, '-v', 'quiet', '-select_streams', 'a',
                 '-show_entries', 'stream=index', '-of', 'csv=p=0', filepath],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                count = len([ln for ln in result.stdout.split() if ln.strip()])
        except Exception:
            count = 1
    if count <= 1:
        return count, []
    pads = ''.join(f'[0:a:{i}]' for i in range(count))
    return count, [
        '-filter_complex', f'{pads}amix=inputs={count}:normalize=0[aout]',
        '-map', '[aout]',
    ]


# Extraction recipe version. Bump whenever the ffmpeg invocation below
# changes in a way that affects WAV content (e.g. the multi-stream amix
# mixdown, or any change to the Pro bundled-ffmpeg invocation). Cached WAVs
# carrying an older recipe — or no sidecar at all, which covers everything
# extracted by pre-port builds including truncated non-atomic-era files and
# silent wrong-stream picks — are discarded and re-extracted once.
# [Ported from OSS v3.5.12 / 61c8ae5.]
_EXTRACT_RECIPE = 2


def _audio_meta_path(audio_path):
    return audio_path + '.meta.json'


def _wav_duration_seconds(audio_path):
    """Duration of OUR extracted WAV (16 kHz mono s16) from its byte size.

    No ffprobe dependency — extraction must keep working on machines where
    a standalone ffmpeg resolves but ffprobe doesn't. The 44-byte canonical
    header is noise at this precision."""
    try:
        return max(os.path.getsize(audio_path) - 44, 0) / 32000.0
    except OSError:
        return 0.0


def _cached_audio_valid(audio_path, filepath):
    """True iff the cached WAV was produced by the CURRENT extraction recipe
    from the CURRENT source file and still matches the duration recorded at
    write time. Anything else — no sidecar (pre-port builds), older recipe,
    source replaced, suspiciously tiny file, or a WAV that shrank since the
    sidecar was written (truncation) — is treated as poisoned."""
    from doza_assist.jsonio import load_json
    meta = load_json(_audio_meta_path(audio_path))
    if not isinstance(meta, dict) or meta.get('recipe') != _EXTRACT_RECIPE:
        return False
    try:
        st = os.stat(filepath)
    except OSError:
        return False
    if meta.get('source_size') != st.st_size or meta.get('source_mtime') != int(st.st_mtime):
        return False
    try:
        if os.path.getsize(audio_path) <= 1024:
            return False
    except OSError:
        return False
    wav_dur = _wav_duration_seconds(audio_path)
    if wav_dur < 0.5:
        return False
    # Compare against the duration recorded when THIS wav was written —
    # self-consistent, needs no source probe, and sidesteps sources whose
    # audio track is legitimately shorter than the video container.
    recorded = meta.get('wav_duration')
    if isinstance(recorded, (int, float)) and recorded > 0 and wav_dur < 0.9 * recorded:
        return False
    return True


def extract_audio(filepath, project_dir=None):
    """
    Extract / convert any media file to a 16 kHz mono WAV for processing.

    Always produces a WAV so the browser's ``<audio>`` element plays the
    exact same decoded PCM that Parakeet used for timestamping.  Without
    this, compressed formats (MP3, AAC …) are decoded independently by
    the browser and by ffmpeg, and differences in encoder-delay handling
    or VBR frame timing cause progressive drift between the transcript
    timestamps and the audible playback position.

    If *project_dir* is provided the WAV is written to
    ``projects/<id>/audio.wav``; otherwise it lands next to the source.

    The cache is recipe-versioned and self-healing: WAVs from older builds
    (truncated, or silent wrong-stream picks from before the amix mixdown)
    are detected and re-extracted automatically — no uninstall needed.
    """
    # Determine output path for extracted audio
    if project_dir:
        audio_path = os.path.join(project_dir, 'audio.wav')
    else:
        audio_path = filepath.rsplit('.', 1)[0] + '_audio.wav'

    # Same-path guard: callers may hand us our own previous OUTPUT as the
    # input (My Style import extracts first, then transcribe_file re-enters
    # here with the WAV). The sidecar records the ORIGINAL source's
    # size/mtime, so validating the WAV against itself is a guaranteed
    # mismatch — and the self-heal below would delete its own input. When
    # input and output are the same file there is nothing to extract; fall
    # through to the 16k-mono passthrough.
    if os.path.exists(audio_path) and \
            os.path.abspath(filepath) != os.path.abspath(audio_path):
        if _cached_audio_valid(audio_path, filepath):
            return audio_path
        # Stale, truncated, wrong-recipe, or silent-wrong-stream cache —
        # remove and re-extract. This is the no-uninstall-needed self-heal.
        for stale in (audio_path, _audio_meta_path(audio_path)):
            try:
                os.remove(stale)
            except OSError:
                pass

    # If the source is already a 16 kHz mono WAV we can just reference it
    # directly — no decode ambiguity is possible for uncompressed PCM.
    ext = filepath.rsplit('.', 1)[-1].lower()
    if ext == 'wav':
        try:
            import wave
            with wave.open(filepath, 'rb') as wf:
                if (wf.getnchannels() == 1
                        and wf.getframerate() == 16000
                        and wf.getsampwidth() == 2):
                    return filepath
        except Exception:
            pass  # not a valid WAV or wrong format — fall through to ffmpeg

    ffmpeg = _find_ffmpeg()
    # Write to a temp path and os.replace into place only on success, so an
    # interrupted extraction (crash, force-quit, disk full) can't leave a
    # truncated audio.wav that the existence check above would later serve as
    # complete — silently cutting every future transcript short. `-y` moves
    # BEFORE the output path: as a trailing arg it was a no-op, so a stale
    # .part from a prior crash wasn't even overwritten.
    stream_count, mix_args = _audio_stream_plan(filepath)
    if stream_count == 0:
        raise RuntimeError('This file has no audio track to transcribe.')
    # Unique temp suffix: the background transcribe job, /media/audio's
    # on-the-fly extraction, and the batch worker can all extract the same
    # project concurrently — two writers on one tmp path meant the second
    # os.replace exploded on a vanished file. -nostdin keeps a confused
    # ffmpeg from blocking on a TTY that isn't there.
    tmp_path = f'{audio_path}.part-{os.getpid()}-{threading.get_ident()}.wav'
    try:
        result = subprocess.run([
            ffmpeg, '-nostdin', '-y', '-i', filepath,
            '-vn', *mix_args, '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            tmp_path,
        ], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[:500]}")
        os.replace(tmp_path, audio_path)
    finally:
        # Failure or interruption: never leave a partial file behind.
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # Empty-audio guard: a source whose audio decodes to (near-)nothing
    # produces a valid-but-empty WAV, which used to sail into the engines
    # and crash Whisper with "cannot reshape tensor of 0 elements". The
    # zero-STREAM case was caught above; this catches zero-CONTENT.
    wav_dur = _wav_duration_seconds(audio_path)
    if wav_dur < 0.1:
        for stale in (audio_path, _audio_meta_path(audio_path)):
            try:
                os.remove(stale)
            except OSError:
                pass
        raise RuntimeError(
            'Extracted audio is empty — the file may have no usable audio '
            'track. Check that the source plays sound in QuickTime.'
        )

    try:
        from doza_assist.jsonio import atomic_write_json
        st = os.stat(filepath)
        atomic_write_json(_audio_meta_path(audio_path), {
            'recipe': _EXTRACT_RECIPE,
            'source_size': st.st_size,
            'source_mtime': int(st.st_mtime),
            'wav_duration': wav_dur,
        })
    except OSError:
        # Sidecar write failure just means re-extraction next time.
        pass
    return audio_path


def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def transcribe_file(filepath, project_dir=None, speaker_labels=None, num_speakers=2,
                    language='en', progress_cb=None):
    """
    Transcribe an audio/video file.

    Tries engines in order: Parakeet MLX (fastest) → WhisperX → Whisper.
    For non-English languages, skips Parakeet (English-only) and uses WhisperX/Whisper.

    ``progress_cb`` is an optional ``callable(dict)`` invoked at major
    milestones. The dict carries ``{"phase": str, "pct": int, "engine":
    str}``. Phases land in this order:

      - ``extract_audio`` (pct=0)
      - ``load_model`` with the engine name (pct=5)
      - ``transcribing`` with periodic pct updates as the engine works
      - ``finalize`` (pct=90, downstream code can use 90..100 for
        diarization or post-processing)

    Pct caps at 90 so the caller can reserve the last decile for
    whatever post-transcription work it queues.

    Returns:
        dict with 'segments' list, each containing:
            - start, end (float seconds)
            - text (str)
            - speaker (str)
            - start_formatted, end_formatted (str HH:MM:SS.mmm)
            - words (list of {start, end, word})
    """
    def _emit(phase, pct, engine=None, **extra):
        """Best-effort progress emit. A bad callback never breaks the
        actual transcription run."""
        if progress_cb is None:
            return
        try:
            event = {"phase": phase, "pct": int(pct)}
            if engine is not None:
                event["engine"] = engine
            event.update(extra)
            progress_cb(event)
        except Exception:
            pass

    _emit("extract_audio", 0)
    # Extract audio first — needed for all engines (video files are too large for direct processing)
    audio_path = extract_audio(filepath, project_dir=project_dir)

    # Try Parakeet MLX first (fastest on Apple Silicon) — English only
    # NOTE: the engines receive the RAW progress_cb (an event-dict callable),
    # NOT the milestone helper _emit(phase, pct, ...) above. Passing _emit
    # here was the bug that killed all engine progress: every engine calls
    # progress_cb(event_dict), which raised a swallowed TypeError inside
    # _emit, so the status file never saw a 'transcribing' phase and the bar
    # parked at 5% for entire runs.
    if language == 'en':
        try:
            _emit("load_model", 5, engine="parakeet-mlx")
            return _transcribe_parakeet(audio_path, speaker_labels, progress_cb=progress_cb)
        except ImportError:
            print("Parakeet MLX not available, trying Whisper...", flush=True)
        except Exception as e:
            import traceback
            err_text = str(e)
            # macOS 13/14: bundled mlx.metallib is compiled against Metal
            # SL 4.0 which only loads on macOS 15+. The user-visible
            # symptom is Parakeet failing and the app silently falling
            # to Whisper-CPU (10x slower, ~7 GB RAM). Annotate the log
            # with a clear cause so the support flow doesn't waste time
            # diagnosing a "transcription is slow" ticket.
            if "default metallib" in err_text or "language version 4" in err_text:
                print(
                    "Parakeet failed: Metal shader library is incompatible "
                    "with this macOS version. The bundled mlx package needs "
                    "macOS 15 (Sequoia) or newer; this Mac will use the "
                    "slower Whisper fallback for now. Upgrade macOS for "
                    "the fast transcription path.",
                    flush=True,
                )
            else:
                print(f"Parakeet failed: {e}", flush=True)
            traceback.print_exc()
            print("Falling back to Whisper...", flush=True)
    else:
        print(f"Language '{language}' selected — skipping Parakeet (English-only), using Whisper...", flush=True)

    # Try WhisperX
    try:
        _emit("load_model", 5, engine="whisperx", slow_mode=True)
        return _transcribe_whisperx(audio_path, speaker_labels, language=language, progress_cb=progress_cb)
    except ImportError:
        print("WhisperX not available, trying standard Whisper...")

    # Fall back to standard Whisper
    try:
        _emit("load_model", 5, engine="whisper", slow_mode=True)
        return _transcribe_whisper(
            audio_path, speaker_labels, num_speakers=num_speakers,
            language=language, progress_cb=progress_cb,
        )
    except ImportError:
        raise RuntimeError(
            "No transcription engine found. Install one of:\n"
            "  pip install parakeet-mlx\n"
            "  pip install openai-whisper"
        )


def _transcribe_parakeet(filepath, speaker_labels=None, progress_cb=None):
    """Transcribe using Parakeet MLX in an isolated subprocess.

    The decode is moved into ``parakeet_worker.py`` and spawned as a child
    Python process. Rationale (issue #23, ported from OSS v3.5.4): on some
    M1 systems MLX/Metal raises a C++ exception inside the Metal completion
    handler that Python cannot catch — it propagates to ``std::terminate``
    and SIGABRTs the whole process. In-process, that killed the entire
    backend mid-job; the Electron shell restarted it, the status file went
    permanently stale, and the user got a frozen "Transcribing…" card with
    no retry. With isolation the SIGABRT kills the worker only and this
    function raises, so transcribe_file falls through to the bundled
    Whisper engine like any other Parakeet failure.

    Cost: the model reloads per call (~5-15s) — the warm in-process cache
    can't survive process isolation. The per-chunk progress contract is
    preserved: the worker streams ``DOZA_PROGRESS {json}`` lines (same
    phase/pct/audio_sec events the in-process version emitted) and this
    parent forwards them to ``progress_cb``.

    First-run import check up front so a missing parakeet_mlx surfaces as
    ImportError (the signal transcribe_file uses to skip to Whisper)
    rather than a worker crash.
    """
    import importlib.util
    if importlib.util.find_spec('parakeet_mlx') is None:
        raise ImportError('parakeet_mlx is not installed')

    def _emit(event):
        if progress_cb is None:
            return
        try:
            progress_cb(event)
        except Exception:
            pass

    worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'parakeet_worker.py')
    out_fd, out_path = tempfile.mkstemp(prefix='parakeet_result_', suffix='.json')
    os.close(out_fd)
    err_fd, err_path = tempfile.mkstemp(prefix='parakeet_stderr_', suffix='.log')
    os.close(err_fd)

    default_speaker = 'Speaker'
    if speaker_labels:
        default_speaker = speaker_labels.get('SPEAKER_00', 'Speaker')

    try:
        with open(err_path, 'w') as err_file:
            proc = subprocess.Popen(
                [sys.executable, worker_path,
                 '--audio', filepath,
                 '--output', out_path,
                 '--speaker', default_speaker],
                stdout=subprocess.PIPE,
                stderr=err_file,
                text=True,
                bufsize=1,
            )
            # Stream worker stdout live: DOZA_PROGRESS lines become
            # progress events (keeping the wrapper's per-chunk bar);
            # everything else is forwarded to the app log verbatim.
            for line in proc.stdout:
                line = line.rstrip('\n')
                if line.startswith('DOZA_PROGRESS '):
                    try:
                        _emit(json.loads(line[len('DOZA_PROGRESS '):]))
                    except (ValueError, json.JSONDecodeError):
                        pass
                elif line:
                    print(f"[parakeet-worker] {line}", flush=True)
            proc.wait()

        try:
            with open(err_path, 'r') as f:
                err_text = f.read()
        except OSError:
            err_text = ''
        if err_text.strip():
            print(f"[parakeet-worker:stderr] {err_text[-2000:]}", flush=True)

        if proc.returncode != 0:
            # Include stderr in the raised message: transcribe_file's
            # metallib detection ("default metallib" / "language version 4")
            # keys off this text to pick its fallback messaging.
            detail = ''
            try:
                with open(out_path, 'r') as f:
                    payload = json.load(f)
                detail = payload.get('error', '')
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            raise RuntimeError(
                f"Parakeet worker exited with {proc.returncode}"
                f"{' (likely Metal/MLX crash)' if proc.returncode < 0 or proc.returncode == 134 else ''}: "
                f"{detail or err_text[-500:] or 'no error detail'}"
            )

        with open(out_path, 'r') as f:
            result = json.load(f)
        if 'error' in result and 'segments' not in result:
            raise RuntimeError(f"Parakeet worker failed: {result['error']}")
        return result
    finally:
        for tmp in (out_path, err_path):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _transcribe_whisperx(audio_path, speaker_labels=None, language='en', progress_cb=None):
    """Transcribe using WhisperX with word-level timestamps and diarization.

    ``progress_cb`` emits a single ``transcribing`` event when the
    model starts (slow_mode=True) so the UI can switch to an
    indeterminate barber-pole. WhisperX doesn't expose per-segment
    progress, so we can't drive a determinate bar from inside.
    """
    if progress_cb is not None:
        try:
            progress_cb({
                "phase": "transcribing", "pct": 10, "engine": "whisperx",
                "slow_mode": True,
            })
        except Exception:
            pass
    global _whisperx_model
    import whisperx
    import torch

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "cpu"  # WhisperX MPS support is limited, CPU is more reliable

    # Use float32 on CPU for best quality (int8 quantization significantly degrades
    # non-English transcription, especially for morphologically rich languages like
    # Czech, Polish, Russian). float16 is only supported on CUDA.
    compute_type = "float16" if device == "cuda" else "float32"

    with _model_lock:
        cache_key = (device, compute_type)
        if _whisperx_model is None or _whisperx_model[1] != cache_key:
            print(f"Loading WhisperX model (compute_type={compute_type})...")
            model = whisperx.load_model("large-v3", device, compute_type=compute_type)
            _whisperx_model = (model, cache_key)
        else:
            print("Using cached WhisperX model.")
            model = _whisperx_model[0]

    print(f"Transcribing (language: {language})...")
    audio = whisperx.load_audio(audio_path)
    # Pass language to avoid auto-detection when user has specified it
    transcribe_kwargs = {"batch_size": 16}
    if language != 'auto':
        transcribe_kwargs["language"] = language
    result = model.transcribe(audio, **transcribe_kwargs)

    # Align whisper output for word-level timestamps
    lang_code = result["language"]
    with _model_lock:
        align_entry = _whisperx_align_cache.get(lang_code)
        if align_entry is None or align_entry[2] != device:
            print(f"Loading align model for '{lang_code}'...")
            model_a, metadata = whisperx.load_align_model(language_code=lang_code, device=device)
            _whisperx_align_cache[lang_code] = (model_a, metadata, device)
        else:
            print(f"Using cached align model for '{lang_code}'.")
            model_a, metadata = align_entry[0], align_entry[1]
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)

    # Speaker diarization.
    # When the Pro `diarization` extension is active (DIARIZATION_ENABLED=true,
    # which the extension sets at load time) we skip the inline WhisperX
    # diarization here — the extension owns speaker labeling for all engines
    # via a separate post-transcription worker using a newer pyannote model.
    # When DIARIZATION_ENABLED is false/unset, the legacy inline path runs.
    _diar_ext_active = os.environ.get('DIARIZATION_ENABLED', '').strip().lower() in ('1', 'true', 'yes', 'on')
    hf_token = os.environ.get('HF_TOKEN', '')
    if hf_token and not _diar_ext_active:
        print("Running speaker diarization (WhisperX inline)...")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
    elif _diar_ext_active:
        print("Skipping inline WhisperX diarization — Pro extension will handle speakers.")

    # Format output
    segments = []
    for seg in result.get("segments", []):
        speaker = seg.get("speaker", "SPEAKER_00")
        if speaker_labels and speaker in speaker_labels:
            speaker = speaker_labels[speaker]

        words = []
        for w in seg.get("words", []):
            words.append({
                'start': round(w.get('start', 0), 3),
                'end': round(w.get('end', 0), 3),
                'word': w.get('word', ''),
            })

        segments.append({
            'start': round(seg['start'], 3),
            'end': round(seg['end'], 3),
            'text': seg['text'].strip(),
            'speaker': speaker,
            'start_formatted': format_timestamp(seg['start']),
            'end_formatted': format_timestamp(seg['end']),
            'words': words,
        })

    return {
        'segments': segments,
        'language': result.get('language', 'en'),
        'duration': segments[-1]['end'] if segments else 0,
        'engine': 'whisperx',
    }


def _transcribe_lightning(audio_path, speaker_labels=None):
    """Transcribe using Lightning Whisper MLX (fastest on Apple Silicon)."""
    from lightning_whisper_mlx import LightningWhisperMLX

    print("Loading Lightning Whisper MLX (distil-large-v3)...")
    whisper = LightningWhisperMLX(model="distil-large-v3", batch_size=12, quant=None)

    print("Transcribing...")
    result = whisper.transcribe(audio_path)

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            'start': round(seg['start'], 3),
            'end': round(seg['end'], 3),
            'text': seg['text'].strip(),
            'speaker': speaker_labels.get('SPEAKER_00', 'Speaker') if speaker_labels else 'Speaker',
            'start_formatted': format_timestamp(seg['start']),
            'end_formatted': format_timestamp(seg['end']),
            'words': [],
        })

    return {
        'segments': segments,
        'language': 'en',
        'duration': segments[-1]['end'] if segments else 0,
        'engine': 'lightning-whisper-mlx',
        'note': 'Speaker diarization requires WhisperX. Install with: pip install whisperx',
    }


def _transcribe_whisper(audio_path, speaker_labels=None, num_speakers=2, language='en',
                        progress_cb=None):
    """Transcribe using OpenAI Whisper. Speaker assignment done manually by user.

    Uses 'turbo' (Whisper large-v3-turbo, 1.62GB) — the same model MacWhisper uses.
    This is the distilled large-v3 with 4 decoder layers, ~8x faster than large-v3
    with comparable quality including strong non-English support (Czech, Polish,
    Russian, etc.). Previously used 'base' (74M params), which produced unusable
    output for non-English languages.

    ``progress_cb`` emits ``transcribing`` events on a wallclock timer
    while Whisper runs. CPU-bound Whisper has no per-segment hook, so
    we estimate progress from elapsed seconds against an assumed
    realtime multiplier (~1.0x for turbo on Apple Silicon CPU).
    """
    import whisper
    import threading

    def _emit(phase, pct, **extra):
        if progress_cb is None:
            return
        try:
            event = {"phase": phase, "pct": int(pct), "engine": "whisper", "slow_mode": True}
            event.update(extra)
            progress_cb(event)
        except Exception:
            pass

    # Try turbo first (best quality/speed balance, matches MacWhisper)
    # Fall back to large-v3 or base if turbo unavailable (older whisper versions)
    model = None
    with _model_lock:
        # Reuse any previously loaded Whisper model — first cache hit wins.
        for cached_name, cached_model in _whisper_cache.items():
            print(f"Using cached Whisper model ({cached_name}).")
            model = cached_model
            break

        if model is None:
            for model_name in ("turbo", "large-v3", "base"):
                try:
                    print(f"Loading Whisper model ({model_name})...")
                    model = whisper.load_model(model_name)
                    _whisper_cache[model_name] = model
                    print(f"Loaded Whisper {model_name}")
                    break
                except Exception as e:
                    print(f"Could not load {model_name}: {e}")
                    continue
    if model is None:
        raise RuntimeError("Could not load any Whisper model")

    print(f"Transcribing (language: {language})...")
    transcribe_kwargs = {"word_timestamps": True}
    if language != 'auto':
        transcribe_kwargs["language"] = language

    # Whisper-CPU has no per-segment progress hook. Wall-clock estimator
    # thread: ticks every 2s, bumps pct toward 88 based on elapsed time
    # vs the audio duration's assumed realtime multiplier. Caps at 88
    # so the bar visibly nudges forward even if Whisper takes longer
    # than estimated, but doesn't claim "Done!" before model.transcribe
    # actually returns.
    audio_duration_sec = 0
    try:
        import soundfile as _sf
        _info = _sf.info(audio_path)
        audio_duration_sec = max(1.0, float(_info.frames) / max(1, _info.samplerate))
    except Exception:
        audio_duration_sec = 60.0  # safe baseline

    _stop_estimator = threading.Event()
    # Once the tqdm hook delivers a real decode fraction, the wall-clock
    # estimator goes silent — real data beats the 1.0x-realtime guess that
    # camped at 88% for the whole back half on memory-pressured machines.
    _real_progress_seen = threading.Event()

    def _on_real_progress(fraction):
        _real_progress_seen.set()
        # Map 0–1 decode fraction into the 10–88 band; 90–100 stays
        # reserved for finalize + downstream (diarization) phases.
        pct = min(88, 10 + int(78 * fraction))
        _emit("transcribing", pct, audio_sec=int(audio_duration_sec))

    def _estimator():
        import time as _t
        # Fail-open fallback when the tqdm hook is unavailable: Whisper
        # turbo on Apple Silicon CPU runs at ~1.0x realtime in FP32. Aim
        # the bar at the estimated finish but cap at 88 so it never
        # claims completion.
        REALTIME_MULTIPLIER = 1.0
        start = _t.time()
        estimated_total = audio_duration_sec * REALTIME_MULTIPLIER
        _emit("transcribing", 10, audio_sec=int(audio_duration_sec))
        last_pct = 10
        while not _stop_estimator.wait(2.0):
            if _real_progress_seen.is_set():
                continue  # the engine is reporting truth; stay quiet
            elapsed = _t.time() - start
            ratio = min(1.0, elapsed / max(1.0, estimated_total))
            pct = min(88, int(10 + 78 * ratio))
            if pct > last_pct:
                _emit("transcribing", pct, audio_sec=int(audio_duration_sec))
                last_pct = pct

    estimator_thread = threading.Thread(target=_estimator, daemon=True)
    estimator_thread.start()
    _cb_token = None
    if _install_whisper_progress_hook():
        _cb_token = _whisper_progress_cb.set(_on_real_progress)
    try:
        result = model.transcribe(audio_path, **transcribe_kwargs)
    finally:
        if _cb_token is not None:
            _whisper_progress_cb.reset(_cb_token)
        _stop_estimator.set()
        try:
            estimator_thread.join(timeout=1.0)
        except Exception:
            pass
    _emit("finalize", 90)

    # Default speaker name
    default_speaker = 'Speaker'
    if speaker_labels:
        default_speaker = speaker_labels.get('SPEAKER_00', 'Speaker')

    segments = []
    for seg in result.get("segments", []):
        words = []
        for w in seg.get("words", []):
            words.append({
                'start': round(w.get('start', 0), 3),
                'end': round(w.get('end', 0), 3),
                'word': w.get('word', ''),
            })

        segments.append({
            'start': round(seg['start'], 3),
            'end': round(seg['end'], 3),
            'text': seg['text'].strip(),
            'speaker': default_speaker,
            'start_formatted': format_timestamp(seg['start']),
            'end_formatted': format_timestamp(seg['end']),
            'words': words,
        })

    out = {
        'segments': segments,
        'language': result.get('language', 'en'),
        'duration': segments[-1]['end'] if segments else 0,
        'engine': 'whisper',
    }
    if num_speakers and num_speakers > 1:
        # Engine honesty: Whisper cannot tell speakers apart, so every
        # segment above carries one label. On Pro the pyannote extension
        # usually rewrites these minutes later; the UI banner keys on the
        # diarization status + distinct labels, and uses this flag to know
        # the single label is an engine limitation rather than reality.
        out['diarization'] = 'unavailable'
        out['note'] = (
            f'The Whisper engine cannot separate speakers — all segments '
            f'are labeled "{default_speaker}". Speaker identification runs '
            f'separately; you can also click any speaker name in the '
            f'transcript to reassign that paragraph.'
        )
    return out


