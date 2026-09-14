---
title: "How to Transcribe Video Locally on a Mac (No Cloud, No Subscription)"
slug: transcribe-video-locally-mac
date: 2026-09-16
author: Chris Doza
description: "A practical guide to offline video transcription on Apple Silicon: which models to use, what hardware you need, how to get word-level timestamps, and how to do it free with Doza Assist Core."
category: Guides
---

# How to Transcribe Video Locally on a Mac

Every transcription service on the market wants two things from you: your footage and a monthly fee. If you cut interviews for a living, neither is a great deal. The footage is often under NDA, and the fee scales with exactly the thing you want to do more of, which is transcribe everything.

An M-series Mac can transcribe long-form interview audio entirely on its own, faster than real time, with word-level timestamps, for free. This guide covers how, what to expect from the two best open models, and what hardware you actually need.

## What "local transcription" means

Local means the speech model's weights are downloaded once and run on your machine. The audio is decoded, processed, and turned into text without a network connection. Nothing is uploaded, nothing is metered, and it works on a plane.

On Apple Silicon this is practical because of unified memory and Apple's MLX framework, which runs machine learning models on the Mac's GPU cores without a separate graphics card.

## The two models worth using

**NVIDIA Parakeet TDT 0.6B** is the fast option for English. It's a 600-million-parameter model that runs comfortably alongside an editing app and chews through an hour of interview audio in a few minutes on a base M-series machine. Accuracy on clean, single-speaker or small-panel dialogue is excellent, good enough that the transcript is usable for finding story beats without a cleanup pass. It produces word-level timestamps, which is what lets you click a word and jump to that frame. Run through MLX, it's the default engine in Doza Assist.

**WhisperX large-v3** is the multilingual option. It covers 99+ languages and adds forced alignment on top of Whisper's base output, so you still get accurate word timing. It's heavier and slower than Parakeet, so use it when the footage isn't in English or when you need speaker diarization that Parakeet doesn't do.

Both are free to download and run.

## What hardware you need

Any Apple Silicon Mac will transcribe. The variable is how much local AI you want to run beyond transcription.

| Machine | Transcription | Local story analysis |
|---|---|---|
| M1 or M2 with 8 to 16 GB | Parakeet runs well | Small language model (about 8 GB download) |
| M-series with 24 to 32 GB | Parakeet and WhisperX both fine | Mid-size model, the balanced default |
| M-series Pro, Max, or Ultra with 64 GB+ | Everything | 27B-class model for the highest-quality analysis |

Doza Assist detects your RAM at first launch and picks the right model tier automatically. You can override it if you want to trade speed for quality.

## Step by step with Doza Assist Core

1. **Install.** Doza Assist Core is free and open source under MIT. Clone the repo and run the install script. First launch downloads Parakeet and the local language model, which takes about a minute on a normal connection.
2. **Drop in the footage.** MP4, MOV, WAV, MP3, MXF, or a Final Cut FCPXML from a multicam or synchronized clip. The app reads the enabled audio angle straight out of the FCPXML, so you don't have to export a separate audio file.
3. **Wait a few minutes.** A 60-minute interview typically finishes in under five on a base machine.
4. **Assign speakers.** Click a speaker label and name it. Paragraphs are grouped by speaker.
5. **Highlight to make clips.** Drag across words to create a select, the same way you'd highlight a document. Five renamable color labels keep the selects organized.
6. **Export.** SRT subtitles, plain text, JSON, or a pre-cut timeline for Final Cut Pro, Premiere Pro, or DaVinci Resolve.

That's the whole workflow. No account, no upload, no per-minute charge.

## When cloud is still the right call

Two cases. If the audio is in a language Parakeet doesn't handle and WhisperX is too slow on your machine, a cloud service will be faster. And if you need a human-verified transcript for legal or broadcast captioning, local models are your first draft, not your last.

For the everyday job of finding the story in a stack of interviews, local is now the better tool, not just the cheaper one.

## FAQ

### Can I transcribe video offline on a Mac for free?
Yes. NVIDIA Parakeet and WhisperX are free open models that run on Apple Silicon. Doza Assist Core wraps them in a free, open-source app with a transcript viewer and NLE export.

### How accurate is local transcription compared to cloud services?
On clean interview audio in English, Parakeet TDT is competitive with commercial services and produces word-level timestamps. Heavy accents, crosstalk, and noisy rooms lower accuracy for every model, local or cloud.

### Does local transcription support languages other than English?
Yes, through WhisperX large-v3, which covers 99+ languages. It's slower than Parakeet and benefits from a machine with more memory.

### How long does it take to transcribe an hour of video locally?
A few minutes with Parakeet on a base M-series Mac. WhisperX takes longer, roughly real time or slower depending on the machine.

---

*Chris Doza is a documentary filmmaker and the creator of Doza Assist. Get [Doza Assist Core on GitHub](https://github.com/DozaVisuals/doza-assist).*
