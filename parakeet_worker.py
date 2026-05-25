"""Isolated subprocess worker for Parakeet MLX transcription.

Runs in a child Python process spawned by ``transcribe._transcribe_parakeet``.
The chunked Parakeet decode and MLX cache flush live here, not in the
Flask server, so a hard Metal/MLX crash (issue #23:
``mlx::core::gpu::check_error`` raises a C++ exception inside Metal's
``addCompletedHandler``, which has no catch and aborts via
``std::terminate``) takes down this worker only. The parent sees the
nonzero exit and falls through to the WhisperX / Whisper engines.

Contract:

    parakeet_worker.py --audio <path> --output <jsonfile> --speaker <name>

  - On success: writes the transcript dict to ``--output`` as JSON and
    exits 0.
  - On a Python-level exception: writes ``{"error": "<message>"}`` to
    ``--output`` and exits 1.
  - On SIGABRT from MLX: the process dies; the file is empty or absent.
    The parent treats any nonzero exit as a worker crash regardless.

Progress is streamed to stdout (one line per phase / chunk). The parent
forwards those lines so the app log still shows what the engine is doing.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import tempfile

import certifi


def _setup_ssl() -> None:
    """Mirror the SSL bundle setup that transcribe.py does at import.

    Whisper model downloads + huggingface fetches use requests/urllib, both
    of which need the certifi bundle on macOS Python builds that don't
    ship a system trust store.
    """
    cert_file = certifi.where()
    os.environ.setdefault('SSL_CERT_FILE', cert_file)
    os.environ.setdefault('REQUESTS_CA_BUNDLE', cert_file)
    _orig = ssl.create_default_context

    def _ctx(*args, **kwargs):
        ctx = _orig(*args, **kwargs)
        ctx.load_verify_locations(cert_file)
        return ctx

    ssl.create_default_context = _ctx


def _format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _clear_mlx_cache() -> None:
    """Flush pending Metal work and release cached GPU buffers.

    Wrapped because the exact MLX cache API has moved between versions —
    try the current top-level call, fall back to the older metal
    namespace, and silently skip if neither exists rather than turn
    cleanup into a new failure mode.
    """
    try:
        import mlx.core as mx
        if hasattr(mx, 'synchronize'):
            mx.synchronize()
        if hasattr(mx, 'clear_cache'):
            mx.clear_cache()
        elif hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
            mx.metal.clear_cache()
    except Exception:
        pass


def transcribe(audio_path: str, speaker_name: str) -> dict:
    import numpy as np
    import soundfile as sf
    from parakeet_mlx import from_pretrained
    from parakeet_mlx.audio import load_audio

    print("Loading Parakeet TDT model...", flush=True)
    model = from_pretrained('mlx-community/parakeet-tdt-0.6b-v2')

    print("Loading audio...", flush=True)
    audio_data = load_audio(audio_path, model.preprocessor_config.sample_rate)

    sr = model.preprocessor_config.sample_rate
    total_samples = len(audio_data)
    total_duration = total_samples / sr

    # 60s chunks + 1s overlap. Matches the v3.4.1 mitigation: smaller per-chunk
    # command buffers reduce the odds of hitting Metal's error path. Even
    # with subprocess isolation we keep the smaller chunks so the worker
    # itself is less likely to crash and have to be restarted.
    chunk_sec = 60
    overlap_sec = 1
    chunk_samples = int(chunk_sec * sr)
    overlap_samples = int(overlap_sec * sr)

    all_segments: list[dict] = []
    chunk_start = 0
    chunk_idx = 0

    while chunk_start < total_samples:
        chunk_end = min(chunk_start + chunk_samples, total_samples)
        chunk = audio_data[chunk_start:chunk_end]
        time_offset = chunk_start / sr

        chunk_idx += 1
        print(
            f"Transcribing chunk {chunk_idx} "
            f"({time_offset:.0f}s - {chunk_end/sr:.0f}s)...",
            flush=True,
        )

        # mkstemp so parallel transcriptions in separate workers can't
        # collide on chunk_idx-named files in /tmp.
        fd, tmp_path = tempfile.mkstemp(prefix='parakeet_chunk_', suffix='.wav')
        os.close(fd)
        sf.write(tmp_path, np.array(chunk), sr)

        try:
            result = model.transcribe(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass

        _clear_mlx_cache()

        for sent in result.sentences:
            if not sent.text.strip():
                continue

            # Parakeet uses BPE: tokens starting with space begin a new word;
            # otherwise we merge the subword onto the previous word.
            words: list[dict] = []
            for tok in sent.tokens:
                tok_text = tok.text
                tok_start = round(tok.start + time_offset, 3)
                tok_end = round(tok.end + time_offset, 3)
                if tok_text.startswith(' ') or not words:
                    words.append({
                        'start': tok_start,
                        'end': tok_end,
                        'word': tok_text,
                    })
                else:
                    words[-1]['word'] += tok_text
                    words[-1]['end'] = tok_end

            seg_start = (sent.tokens[0].start if sent.tokens else 0) + time_offset
            seg_end = (sent.tokens[-1].end if sent.tokens else 0) + time_offset
            all_segments.append({
                'start': round(seg_start, 3),
                'end': round(seg_end, 3),
                'text': sent.text.strip(),
                'speaker': speaker_name,
                'start_formatted': _format_timestamp(seg_start),
                'end_formatted': _format_timestamp(seg_end),
                'words': words,
            })

        chunk_start = chunk_end - overlap_samples
        if chunk_end >= total_samples:
            break

    # Drop overlap duplicates: any segment that starts before the previous
    # one ended (with a 0.5s tolerance) is the second half of the overlap.
    if len(all_segments) > 1:
        deduped = [all_segments[0]]
        for seg in all_segments[1:]:
            if seg['start'] < deduped[-1]['end'] - 0.5:
                continue
            deduped.append(seg)
        all_segments = deduped

    print(
        f"Parakeet done: {len(all_segments)} segments "
        f"in {total_duration:.0f}s of audio",
        flush=True,
    )

    return {
        'segments': all_segments,
        'language': 'en',
        'duration': all_segments[-1]['end'] if all_segments else 0,
        'engine': 'parakeet-mlx',
    }


def main() -> None:
    _setup_ssl()
    ap = argparse.ArgumentParser(description="Parakeet MLX subprocess worker")
    ap.add_argument('--audio', required=True, help="Path to the audio file to transcribe")
    ap.add_argument('--output', required=True, help="Path to write result JSON")
    ap.add_argument('--speaker', default='Speaker', help="Speaker label for all segments")
    args = ap.parse_args()

    try:
        result = transcribe(args.audio, args.speaker)
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            with open(args.output, 'w') as f:
                json.dump({'error': f'{type(e).__name__}: {e}'}, f)
        except Exception:
            pass
        sys.exit(1)

    with open(args.output, 'w') as f:
        json.dump(result, f)
    sys.exit(0)


if __name__ == '__main__':
    main()
