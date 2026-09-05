"""The Parakeet worker streams the WAV chunk by chunk instead of decoding
the whole file into one array.

A 36-hour English file used to need roughly 20 GB of unified memory before
the first chunk ran (bytes + int16 + two float32 copies of every sample); the
worker died and the job silently fell back to Whisper. The same whole-file
array also hit MLX's int32 shape limit past 2,236 minutes at 16 kHz."""

import json
import os
import sys
import types

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import parakeet_worker  # noqa: E402


class _Tok:
    def __init__(self, text, start, end):
        self.text, self.start, self.end = text, start, end


class _Sentence:
    def __init__(self, text, tokens):
        self.text, self.tokens = text, tokens


class _Result:
    def __init__(self, sentences):
        self.sentences = sentences


class _Config:
    sample_rate = 16000


class _FakeModel:
    """Records the length of every chunk WAV it is handed and answers one
    sentence at 0.5 s into the chunk."""
    preprocessor_config = _Config()

    def __init__(self):
        self.chunk_frames = []

    def transcribe(self, path):
        info = sf.info(path)
        self.chunk_frames.append((info.frames, info.samplerate, info.subtype))
        return _Result([_Sentence("hello", [_Tok(" hello", 0.5, 1.0)])])


def _install_fake_parakeet(monkeypatch, model, load_audio):
    pkg = types.ModuleType("parakeet_mlx")
    pkg.from_pretrained = lambda name: model
    audio = types.ModuleType("parakeet_mlx.audio")
    audio.load_audio = load_audio
    pkg.audio = audio
    monkeypatch.setitem(sys.modules, "parakeet_mlx", pkg)
    monkeypatch.setitem(sys.modules, "parakeet_mlx.audio", audio)
    monkeypatch.setattr(parakeet_worker, "_budget_chunk_sec", lambda default=60: 60)
    monkeypatch.setattr(parakeet_worker, "_apply_mlx_memory_caps", lambda: None)
    monkeypatch.setattr(parakeet_worker, "_clear_mlx_cache", lambda: None)


def _progress_events(captured_out):
    out = []
    for line in captured_out.splitlines():
        if line.startswith("DOZA_PROGRESS "):
            out.append(json.loads(line[len("DOZA_PROGRESS "):]))
    return out


def test_streams_matching_wav_without_whole_file_load(tmp_path, monkeypatch, capsys):
    wav = tmp_path / "audio.wav"
    sf.write(str(wav), np.zeros(180 * 16000, dtype=np.int16), 16000, subtype="PCM_16")
    model = _FakeModel()

    def forbidden_load_audio(*a, **k):
        raise AssertionError("whole-file load_audio must not run on a matching WAV")

    _install_fake_parakeet(monkeypatch, model, forbidden_load_audio)
    result = parakeet_worker.transcribe(str(wav), "Speaker")

    # 60 s chunks with a 1 s overlap over 180 s: starts at 0, 59, 118, 177.
    assert [round(s["start"], 3) for s in result["segments"]] == [0.5, 59.5, 118.5, 177.5]
    assert all(s["speaker"] == "Speaker" and s["words"] for s in result["segments"])
    # Every chunk handed to the model was read straight from disk as PCM_16.
    assert [f for f, _, _ in model.chunk_frames] == [960000, 960000, 960000, 48000]
    assert all(sr == 16000 and sub == "PCM_16" for _, sr, sub in model.chunk_frames)
    assert result["engine"] == "parakeet-mlx" and result["language"] == "en"

    events = _progress_events(capsys.readouterr().out)
    load_audio = [e for e in events if e["phase"] == "load_audio"]
    assert load_audio and load_audio[0]["audio_sec"] == 180
    assert all(e["audio_sec"] == 180 for e in events if e["phase"] == "transcribing")


def test_mismatched_rate_falls_back_to_whole_file_decode(tmp_path, monkeypatch):
    wav = tmp_path / "native.wav"
    sf.write(str(wav), np.zeros(5 * 44100, dtype=np.int16), 44100, subtype="PCM_16")
    model = _FakeModel()
    calls = []

    def fake_load_audio(path, sr):
        calls.append((path, sr))
        return np.zeros(5 * sr, dtype=np.float32)

    _install_fake_parakeet(monkeypatch, model, fake_load_audio)
    result = parakeet_worker.transcribe(str(wav), "S")
    assert calls == [(str(wav), 16000)]
    assert len(result["segments"]) == 1 and result["segments"][0]["start"] == 0.5


def test_open_chunk_source_reports_frames_before_any_read(tmp_path):
    wav = tmp_path / "a.wav"
    sf.write(str(wav), np.zeros(16000 * 7, dtype=np.int16), 16000, subtype="PCM_16")
    read_chunk, total, sr, mode = parakeet_worker._open_chunk_source(str(wav), 16000)
    assert (total, sr, mode) == (16000 * 7, 16000, "stream")
    chunk = read_chunk(16000, 16000 * 3)
    assert len(chunk) == 16000 * 2 and chunk.dtype == np.int16
