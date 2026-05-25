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


def _ensure_ffmpeg_on_path():
    """Ensure ffmpeg is discoverable on PATH (needed by whisper internally)."""
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
# Parakeet runs in a subprocess (see _transcribe_parakeet); no parent-side
# model cache. Whisper variants stay in-process and share the lock.
_whisperx_model = None          # (model, device, compute_type)
_whisperx_align_cache = {}      # {lang_code: (model_a, metadata, device)}
_whisper_cache = {}             # {model_name: model}


def _find_ffmpeg():
    """Find the ffmpeg binary, checking common Homebrew paths if not on PATH."""
    path = shutil.which('ffmpeg')
    if path:
        return path
    for candidate in ['/opt/homebrew/bin/ffmpeg', '/usr/local/bin/ffmpeg']:
        if os.path.isfile(candidate):
            return candidate
    return 'ffmpeg'  # fall back, will error if not found


def extract_audio(filepath, project_dir=None):
    """
    Extract audio from video files to WAV for processing.

    If project_dir is provided, the extracted WAV is written there
    (projects/<id>/audio.wav) instead of next to the source file.
    This avoids copying huge video files -- we only create a small
    16kHz mono WAV (~10MB per hour of audio).
    """
    ext = filepath.rsplit('.', 1)[-1].lower()
    if ext in ('wav', 'mp3', 'aac', 'm4a', 'flac', 'aif', 'aiff'):
        return filepath

    # Determine output path for extracted audio
    if project_dir:
        audio_path = os.path.join(project_dir, 'audio.wav')
    else:
        audio_path = filepath.rsplit('.', 1)[0] + '_audio.wav'

    # Skip extraction if audio already exists in the project dir
    if os.path.exists(audio_path):
        return audio_path

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


def transcribe_file(filepath, project_dir=None, speaker_labels=None, num_speakers=2, language='en'):
    """
    Transcribe an audio/video file.

    Tries engines in order: Parakeet MLX (fastest) → WhisperX → Whisper.
    For non-English languages, skips Parakeet (English-only) and uses WhisperX/Whisper.
    For multi-speaker requests, also skips Parakeet — it has no diarization, so
    every segment would collapse onto SPEAKER_00 and the user would see one
    speaker even though they configured several (issue #28).

    Returns:
        dict with 'segments' list, each containing:
            - start, end (float seconds)
            - text (str)
            - speaker (str)
            - start_formatted, end_formatted (str HH:MM:SS.mmm)
            - words (list of {start, end, word})
    """
    # Extract audio first — needed for all engines (video files are too large for direct processing)
    audio_path = extract_audio(filepath, project_dir=project_dir)

    # Try Parakeet MLX first (fastest on Apple Silicon) — English, single-speaker only.
    # Parakeet doesn't diarize; on a multi-speaker project it would label every
    # word as the first speaker, so we route those to WhisperX instead even
    # though Parakeet is the faster engine.
    if language == 'en' and num_speakers <= 1:
        try:
            return _transcribe_parakeet(audio_path, speaker_labels)
        except ImportError:
            print("Parakeet MLX not available, trying Whisper...", flush=True)
        except Exception as e:
            import traceback
            print(f"Parakeet failed: {e}", flush=True)
            traceback.print_exc()
            print("Falling back to Whisper...", flush=True)
    elif language != 'en':
        print(f"Language '{language}' selected — skipping Parakeet (English-only), using Whisper...", flush=True)
    else:
        print(f"num_speakers={num_speakers} — skipping Parakeet (no diarization), using WhisperX for speaker labels...", flush=True)

    # Try WhisperX
    try:
        return _transcribe_whisperx(audio_path, speaker_labels, language=language, num_speakers=num_speakers)
    except ImportError:
        print("WhisperX not available, trying standard Whisper...")

    # Fall back to standard Whisper
    try:
        return _transcribe_whisper(audio_path, speaker_labels, num_speakers=num_speakers, language=language)
    except ImportError:
        raise RuntimeError(
            "No transcription engine found. Install one of:\n"
            "  pip install parakeet-mlx\n"
            "  pip install openai-whisper"
        )


def _transcribe_parakeet(filepath, speaker_labels=None):
    """Transcribe using Parakeet MLX in an isolated subprocess.

    The Parakeet decode is moved into ``parakeet_worker.py`` and spawned
    as a child Python process. Rationale: on some M1 systems MLX/Metal
    raises a C++ exception inside the Metal completion handler that
    Python literally cannot catch — it propagates to ``std::terminate``
    and aborts the whole process (issue #23). With this isolation the
    SIGABRT kills the worker, not the Flask server, and the outer
    ``transcribe_file`` falls through to the WhisperX / Whisper engines.

    Chunking and the per-chunk MLX cache flush both live inside the
    worker — see ``parakeet_worker.transcribe``. The model is re-loaded
    on every call (no shared cache across files) which is the cost of
    process isolation; first-load is ~5–15 s. If batch throughput
    becomes a concern later, switch to a long-lived worker with a
    request pipe — keeping it per-call for now is the simplest shape
    that fixes the crash.
    """
    default_speaker = 'Speaker'
    if speaker_labels:
        default_speaker = speaker_labels.get('SPEAKER_00', 'Speaker')

    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parakeet_worker.py')
    if not os.path.isfile(worker):
        raise RuntimeError(f"parakeet_worker.py not found next to transcribe.py at {worker}")

    fd, out_path = tempfile.mkstemp(prefix='parakeet_result_', suffix='.json')
    os.close(fd)
    try:
        cmd = [
            sys.executable, worker,
            '--audio', filepath,
            '--output', out_path,
            '--speaker', default_speaker,
        ]
        # Stream child stdout/stderr forward so the app log keeps
        # showing per-chunk progress lines. Combine streams so the
        # interleaving stays in causal order.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end='', flush=True)
        proc.wait()

        if proc.returncode != 0:
            # On Unix a process killed by signal N reports returncode -N.
            # We surface either the negative signal or the positive exit
            # code; if the worker wrote {"error": ...} before exiting we
            # also include that for diagnostics. A SIGABRT (issue #23)
            # leaves the file empty / absent so this just falls through.
            extra = ''
            try:
                if os.path.getsize(out_path) > 0:
                    with open(out_path) as f:
                        err = (json.load(f) or {}).get('error')
                        if err:
                            extra = f' -- {err}'
            except Exception:
                pass
            raise RuntimeError(
                f"Parakeet worker exited with code {proc.returncode}{extra}"
            )

        with open(out_path) as f:
            return json.load(f)
    finally:
        try:
            os.remove(out_path)
        except FileNotFoundError:
            pass


def _transcribe_whisperx(audio_path, speaker_labels=None, language='en', num_speakers=2):
    """Transcribe using WhisperX with word-level timestamps and diarization."""
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
    # Requires HF_TOKEN env var + the HF account having accepted the pyannote
    # speaker-diarization model terms. Without the token diarization silently
    # no-ops and every segment ends up as SPEAKER_00, which on a multi-speaker
    # project surfaces to the user as "I asked for 2 speakers, got 1" (issue
    # #28). Log the missing-token case loudly so the support flow is obvious.
    print("Running speaker diarization...")
    hf_token = os.environ.get('HF_TOKEN', '') or os.environ.get('HUGGINGFACE_TOKEN', '')
    if hf_token:
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        # Hint pyannote with the expected speaker count from the project so it
        # doesn't auto-detect a different number. Bounds are inclusive.
        if num_speakers and num_speakers > 1:
            diarize_segments = diarize_model(audio, min_speakers=num_speakers, max_speakers=num_speakers)
        else:
            diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
    elif num_speakers and num_speakers > 1:
        print(
            "WARNING: HF_TOKEN env var not set — speaker diarization will be "
            "skipped and all segments will be labeled with the first speaker. "
            "To enable multi-speaker labeling: "
            "1) create a HuggingFace account, "
            "2) accept the model terms at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1, "
            "3) generate a read token, "
            "4) set HF_TOKEN=<your-token> before launching the app.",
            flush=True,
        )

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


def _transcribe_whisper(audio_path, speaker_labels=None, num_speakers=2, language='en'):
    """Transcribe using OpenAI Whisper. Speaker assignment done manually by user.

    Uses 'turbo' (Whisper large-v3-turbo, 1.62GB) — the same model MacWhisper uses.
    This is the distilled large-v3 with 4 decoder layers, ~8x faster than large-v3
    with comparable quality including strong non-English support (Czech, Polish,
    Russian, etc.). Previously used 'base' (74M params), which produced unusable
    output for non-English languages.
    """
    import whisper

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
    result = model.transcribe(audio_path, **transcribe_kwargs)

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


