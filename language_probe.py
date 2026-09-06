"""Decide the engine for an Auto-detect transcription (1.1).

Only ``language == 'en'`` takes the fast Parakeet path; Auto-detect used to
go straight to Whisper, so an editor who picked Auto-detect once (the picker
was sticky) pushed every English interview through the slow engine.

The probe asks Whisper's language identification about the first 30 seconds
of the extracted audio. English routes the whole file to Parakeet; another
language is passed to Whisper explicitly (better than letting it guess per
window); no verdict, or Whisper not installed, keeps Auto-detect exactly as
before, including the install prompt.

A first version scored Parakeet's own output for "Englishness". That was
wrong: Parakeet is an English-only model and turns Norwegian or German speech
into fluent, invented English sentences, so the score passed. Only a real
language identifier can make this call. transcribe.py is not touched: the
job in app.py calls this before it picks the engine.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import tempfile
import wave
from typing import Callable

PROBE_SECONDS = 30
MIN_PROBABILITY = 0.5
# How far into the file to look for the first speech.
SPEECH_SEARCH_SECONDS = 600
# A window whose peak is below this is too quiet to judge (same rule the
# transcribe job uses to call a track silent).
QUIET_PEAK_DBFS = -55.0

# Smallest Whisper models first: identification only needs a rough listen,
# and 'base' loads in a second or two. Anything the machine already cached
# for transcription works as a fallback.
LID_MODEL_PREFERENCE = ('base', 'small', 'medium', 'turbo', 'large-v3')


def whisper_available() -> bool:
    try:
        return importlib.util.find_spec('whisper') is not None
    except (ImportError, ValueError):
        return False


# Speech-start detection is done in pure Python on the extracted WAV: the
# bundled ffmpeg is a stripped LGPL build with no silencedetect filter.
FRAME_SECONDS = 0.1
SPEECH_PEAK_DBFS = -35.0     # a frame louder than this counts as sound
SPEECH_RUN_SECONDS = 1.0     # this much continuous sound means talking


def first_speech_offset(audio_path: str, ffmpeg: str | None = None,
                        search_seconds: int = SPEECH_SEARCH_SECONDS) -> float:
    """Seconds into a 16 kHz mono PCM16 WAV where sound first holds for
    SPEECH_RUN_SECONDS. 0.0 when it opens on sound, -1.0 when the searched
    span never does, 0.0 when the file cannot be read (fall back to the
    top of the file rather than give up)."""
    import math
    try:
        with wave.open(audio_path, 'rb') as w:
            rate = w.getframerate() or 16000
            channels = w.getnchannels() or 1
            width = w.getsampwidth()
            if width != 2:
                return 0.0
            frame_len = max(1, int(rate * FRAME_SECONDS))
            total_frames = int(min(w.getnframes(), search_seconds * rate))
            need = max(1, int(round(SPEECH_RUN_SECONDS / FRAME_SECONDS)))
            run = 0
            pos = 0
            limit = 32768.0 * (10 ** (SPEECH_PEAK_DBFS / 20))
            while pos < total_frames:
                raw = w.readframes(frame_len)
                if not raw:
                    break
                n = len(raw) // 2
                if n == 0:
                    break
                import array
                samples = array.array('h', raw[: n * 2])
                peak = max(abs(v) for v in samples)
                if peak >= limit:
                    run += 1
                    if run >= need:
                        start_frame = pos - (need - 1) * frame_len
                        return max(0.0, start_frame / rate)
                else:
                    run = 0
                pos += frame_len
            return -1.0
    except Exception:
        return 0.0


def trim_head(audio_path: str, ffmpeg: str, seconds: int = PROBE_SECONDS,
              offset: float = 0.0) -> str:
    """``seconds`` of audio from ``offset`` as a 16 kHz mono WAV temp file
    (caller removes it)."""
    fd, out = tempfile.mkstemp(prefix='doza_langprobe_', suffix='.wav')
    os.close(fd)
    cmd = [ffmpeg, '-y', '-v', 'error']
    if offset and offset > 0:
        cmd += ['-ss', f'{offset:.2f}']
    cmd += ['-i', audio_path, '-t', str(seconds), '-ac', '1', '-ar', '16000',
            '-acodec', 'pcm_s16le', out]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return out


def peak_dbfs(wav_path: str) -> float:
    """Peak level of a PCM16 WAV in dBFS (-inf for digital silence)."""
    import math
    import numpy as np
    audio = _read_wav_float32(wav_path)
    if audio.size == 0:
        return float('-inf')
    peak = float(np.max(np.abs(audio)))
    return 20 * math.log10(peak) if peak > 0 else float('-inf')


def _read_wav_float32(path: str):
    """16 kHz mono PCM16 WAV to float32 in [-1, 1] (Whisper's input), read
    with the standard library: the bundled ffmpeg is not on PATH, so
    whisper.load_audio cannot be used here."""
    import numpy as np
    with wave.open(path, 'rb') as w:
        frames = w.readframes(w.getnframes())
        width = w.getsampwidth()
        channels = w.getnchannels()
    if width != 2:
        raise ValueError(f'expected 16-bit PCM, got {width * 8}-bit')
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio


def _load_lid_model(cache: dict | None = None):
    """The smallest Whisper model that loads. ``cache`` (transcribe's own
    model cache when available) is reused and filled so the model is not
    loaded twice in one process."""
    import whisper
    cache = cache if cache is not None else {}
    for name in LID_MODEL_PREFERENCE:
        if name in cache:
            return cache[name]
    last = None
    for name in LID_MODEL_PREFERENCE:
        try:
            model = whisper.load_model(name)
            cache[name] = model
            return model
        except Exception as exc:  # missing download, out of memory, bad file
            last = exc
            continue
    raise RuntimeError(f'no Whisper model could be loaded for language identification: {last}')


def whisper_identify(head_wav: str, cache: dict | None = None) -> tuple[str, float]:
    """(language code, probability) for the head clip via Whisper LID."""
    import whisper
    model = _load_lid_model(cache)
    audio = whisper.pad_or_trim(_read_wav_float32(head_wav))
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    _, probs = model.detect_language(mel)
    if isinstance(probs, list):
        probs = probs[0]
    code = max(probs, key=probs.get)
    return str(code), float(probs[code])


def probe_language(audio_path: str, ffmpeg: str | None,
                   identify: Callable[[str], tuple[str, float]],
                   seconds: int = PROBE_SECONDS,
                   min_probability: float = MIN_PROBABILITY) -> dict:
    """Run the probe. Returns {'language': code | None, 'probability',
    'error'}; ``language`` is None whenever nothing confident came back,
    which the caller treats as "keep Auto-detect"."""
    out: dict = {'language': None, 'probability': 0.0, 'error': None, 'method': 'whisper-lid',
                 'offset': 0.0}
    if not audio_path or not os.path.exists(audio_path):
        out['error'] = 'no audio to probe'
        return out
    tmp = None
    try:
        head = audio_path
        if ffmpeg:
            # Listen where the talking starts, not at 0:00: a silent or
            # music-only opening would give Whisper a coin flip.
            offset = first_speech_offset(audio_path)
            if offset < 0:
                out['error'] = 'no speech found in the opening minutes'
                return out
            out['offset'] = round(offset, 2)
            tmp = trim_head(audio_path, ffmpeg, seconds, offset)
            head = tmp
        level = peak_dbfs(head)
        out['peak_dbfs'] = round(level, 1) if level != float('-inf') else None
        if level < QUIET_PEAK_DBFS:
            out['error'] = 'window too quiet to judge'
            return out
        code, prob = identify(head)
        out['probability'] = round(float(prob), 3)
        code = (code or '').strip().lower()
        if code and prob >= min_probability:
            out['language'] = code
    except Exception as exc:  # any failure keeps Auto-detect
        out['error'] = str(exc) or exc.__class__.__name__
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass
    return out
