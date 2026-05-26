"""
Transcription engine for Doza Assist.
Uses WhisperX for transcription + speaker diarization.
Falls back to standard Whisper if WhisperX is not available.
"""

import os
import ssl
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
    """
    # Determine output path for extracted audio
    if project_dir:
        audio_path = os.path.join(project_dir, 'audio.wav')
    else:
        audio_path = filepath.rsplit('.', 1)[0] + '_audio.wav'

    # Skip extraction if audio already exists in the project dir
    if os.path.exists(audio_path):
        return audio_path

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
    result = subprocess.run([
        ffmpeg, '-i', filepath,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', '16000', '-ac', '1',
        audio_path, '-y'
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[:500]}")
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
    if language == 'en':
        try:
            _emit("load_model", 5, engine="parakeet-mlx")
            return _transcribe_parakeet(audio_path, speaker_labels, progress_cb=_emit)
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
        return _transcribe_whisperx(audio_path, speaker_labels, language=language, progress_cb=_emit)
    except ImportError:
        print("WhisperX not available, trying standard Whisper...")

    # Fall back to standard Whisper
    try:
        _emit("load_model", 5, engine="whisper", slow_mode=True)
        return _transcribe_whisper(
            audio_path, speaker_labels, num_speakers=num_speakers,
            language=language, progress_cb=_emit,
        )
    except ImportError:
        raise RuntimeError(
            "No transcription engine found. Install one of:\n"
            "  pip install parakeet-mlx\n"
            "  pip install openai-whisper"
        )


def _transcribe_parakeet(filepath, speaker_labels=None, progress_cb=None):
    """Transcribe using Parakeet MLX — fastest on Apple Silicon.

    Chunks long audio into 5-minute segments to avoid Metal GPU memory limits.

    ``progress_cb`` is invoked per chunk with phase=transcribing and a
    pct in [10, 90] derived from chunk_end / total_samples. The 10-90
    band is the engine's working share; the wrapper reserves 0-10 for
    audio extraction + model load and 90-100 for post-processing.
    """
    global _parakeet_model
    import numpy as np
    from parakeet_mlx.audio import load_audio

    def _emit(phase, pct, **extra):
        if progress_cb is None:
            return
        try:
            event = {"phase": phase, "pct": int(pct), "engine": "parakeet-mlx"}
            event.update(extra)
            progress_cb(event)
        except Exception:
            pass

    with _model_lock:
        if _parakeet_model is None:
            from parakeet_mlx import from_pretrained
            print("Loading Parakeet TDT model...", flush=True)
            _emit("load_model", 5)
            _parakeet_model = from_pretrained('mlx-community/parakeet-tdt-0.6b-v2')
        else:
            print("Using cached Parakeet TDT model.", flush=True)
        model = _parakeet_model

    print("Loading audio...", flush=True)
    _emit("load_audio", 8)
    audio_data = load_audio(filepath, model.preprocessor_config.sample_rate)

    sr = model.preprocessor_config.sample_rate
    total_samples = len(audio_data)
    total_duration = total_samples / sr

    # Chunk into ~5 minute segments with 1s overlap to avoid cutting words
    chunk_sec = 300  # 5 minutes
    overlap_sec = 1
    chunk_samples = int(chunk_sec * sr)
    overlap_samples = int(overlap_sec * sr)

    default_speaker = 'Speaker'
    if speaker_labels:
        default_speaker = speaker_labels.get('SPEAKER_00', 'Speaker')

    all_segments = []
    chunk_start = 0
    chunk_idx = 0

    while chunk_start < total_samples:
        chunk_end = min(chunk_start + chunk_samples, total_samples)
        chunk = audio_data[chunk_start:chunk_end]
        time_offset = chunk_start / sr

        chunk_idx += 1
        # Emit AFTER chunk_end is known so the bar tracks actual
        # progress (which sample range we're about to decode).
        pct = 10 + int(80 * chunk_end / max(1, total_samples))
        _emit("transcribing", pct, audio_sec=int(total_duration))
        print(f"Transcribing chunk {chunk_idx} ({time_offset:.0f}s - {chunk_end/sr:.0f}s)...", flush=True)

        # Save chunk as temp WAV (parakeet.transcribe expects a file path)
        import soundfile as sf
        tmp_path = os.path.join(tempfile.gettempdir(), f'parakeet_chunk_{chunk_idx}.wav')
        sf.write(tmp_path, np.array(chunk), sr)

        result = model.transcribe(tmp_path)
        os.remove(tmp_path)

        for sent in result.sentences:
            if not sent.text.strip():
                continue

            # Merge subword tokens into full words
            # Parakeet uses BPE: tokens starting with space begin a new word
            words = []
            for tok in sent.tokens:
                tok_text = tok.text
                tok_start = round(tok.start + time_offset, 3)
                tok_end = round(tok.end + time_offset, 3)

                if tok_text.startswith(' ') or not words:
                    # New word
                    words.append({
                        'start': tok_start,
                        'end': tok_end,
                        'word': tok_text,
                    })
                else:
                    # Continuation of previous word — merge
                    words[-1]['word'] += tok_text
                    words[-1]['end'] = tok_end

            seg_start = (sent.tokens[0].start if sent.tokens else 0) + time_offset
            seg_end = (sent.tokens[-1].end if sent.tokens else 0) + time_offset

            all_segments.append({
                'start': round(seg_start, 3),
                'end': round(seg_end, 3),
                'text': sent.text.strip(),
                'speaker': default_speaker,
                'start_formatted': format_timestamp(seg_start),
                'end_formatted': format_timestamp(seg_end),
                'words': words,
            })

        # Advance past this chunk, minus overlap
        chunk_start = chunk_end - overlap_samples
        if chunk_end >= total_samples:
            break

    # Remove duplicate segments from overlap regions
    if len(all_segments) > 1:
        deduped = [all_segments[0]]
        for seg in all_segments[1:]:
            # Skip if this segment starts before the previous one ends (overlap duplicate)
            if seg['start'] < deduped[-1]['end'] - 0.5:
                continue
            deduped.append(seg)
        all_segments = deduped

    print(f"Parakeet done: {len(all_segments)} segments in {total_duration:.0f}s of audio", flush=True)

    return {
        'segments': all_segments,
        'language': 'en',
        'duration': all_segments[-1]['end'] if all_segments else 0,
        'engine': 'parakeet-mlx',
    }


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

    def _estimator():
        import time as _t
        # Whisper turbo on Apple Silicon CPU runs at ~1.0x realtime in
        # FP32. A 10-min interview takes ~10 min. We aim the bar at the
        # estimated finish but cap at 88 so it never claims completion.
        REALTIME_MULTIPLIER = 1.0
        start = _t.time()
        estimated_total = audio_duration_sec * REALTIME_MULTIPLIER
        _emit("transcribing", 10, audio_sec=int(audio_duration_sec))
        last_pct = 10
        while not _stop_estimator.wait(2.0):
            elapsed = _t.time() - start
            ratio = min(1.0, elapsed / max(1.0, estimated_total))
            pct = min(88, int(10 + 78 * ratio))
            if pct > last_pct:
                _emit("transcribing", pct, audio_sec=int(audio_duration_sec))
                last_pct = pct

    estimator_thread = threading.Thread(target=_estimator, daemon=True)
    estimator_thread.start()
    try:
        result = model.transcribe(audio_path, **transcribe_kwargs)
    finally:
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

    return {
        'segments': segments,
        'language': result.get('language', 'en'),
        'duration': segments[-1]['end'] if segments else 0,
        'engine': 'whisper',
    }


