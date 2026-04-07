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


def transcribe_file(filepath, project_dir=None, speaker_labels=None, num_speakers=2):
    """
    Transcribe an audio/video file with speaker diarization.

    Args:
        filepath: Path to the source audio/video file.
        project_dir: Project directory where extracted audio should be stored.
        speaker_labels: Dict mapping speaker IDs to names.

    Returns:
        dict with 'segments' list, each segment containing:
            - start: float (seconds)
            - end: float (seconds)
            - text: str
            - speaker: str
            - start_formatted: str (HH:MM:SS.mmm)
            - end_formatted: str (HH:MM:SS.mmm)
            - words: list of word-level timestamps
    """
    audio_path = extract_audio(filepath, project_dir=project_dir)

    # Try WhisperX first (best quality + diarization)
    try:
        return _transcribe_whisperx(audio_path, speaker_labels)
    except ImportError:
        print("WhisperX not available, trying lightning-whisper-mlx...")

    # Try Lightning Whisper MLX (fastest on Apple Silicon)
    try:
        return _transcribe_lightning(audio_path, speaker_labels)
    except ImportError:
        print("Lightning Whisper MLX not available, trying standard Whisper...")

    # Fall back to standard Whisper
    try:
        return _transcribe_whisper(audio_path, speaker_labels, num_speakers=num_speakers)
    except ImportError:
        raise RuntimeError(
            "No transcription engine found. Install one of:\n"
            "  pip install whisperx\n"
            "  pip install lightning-whisper-mlx\n"
            "  pip install openai-whisper"
        )


def _transcribe_whisperx(audio_path, speaker_labels=None):
    """Transcribe using WhisperX with word-level timestamps and diarization."""
    import whisperx
    import torch

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "cpu"  # WhisperX MPS support is limited, CPU is more reliable

    compute_type = "int8"

    print("Loading WhisperX model...")
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)

    print("Transcribing...")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)

    # Align whisper output for word-level timestamps
    print("Aligning timestamps...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)

    # Speaker diarization
    print("Running speaker diarization...")
    hf_token = os.environ.get('HF_TOKEN', '')
    if hf_token:
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

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


def _transcribe_whisper(audio_path, speaker_labels=None, num_speakers=2):
    """Transcribe using OpenAI Whisper. Speaker assignment done manually by user."""
    import whisper

    print("Loading Whisper model (base)...")
    model = whisper.load_model("base")

    print("Transcribing...")
    result = model.transcribe(audio_path, word_timestamps=True)

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


