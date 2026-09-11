---
title: "Sovereign AI Shouldn't Require a Data Center"
slug: sovereign-ai-without-the-data-center
date: 2026-09-11
author: Chris Doza
description: "IBC 2026 is talking about sovereign AI for media: owned models, owned infrastructure, footage that never leaves the building. Doza Assist has run that way since day one, on a Mac."
category: Perspectives
---

# Sovereign AI Shouldn't Require a Data Center

The phrase you'll hear on every stage at IBC this week is "sovereign AI for media." Owned models. Owned infrastructure. Footage that never leaves the building.

It's the right conversation. Broadcasters and post houses have spent three years being told to upload their raw interviews to somebody else's cloud so a model they don't control can transcribe them, and the bill comes back metered by the minute. For newsrooms, documentary teams, and anyone working under an NDA, that was never a comfortable trade.

But watch how the conversation usually ends. Sovereignty gets defined as a rack of GPUs, a private cluster, an enterprise contract, and a procurement cycle. The implicit message is that owning your AI is something only a large organization can afford.

I don't think that's true. Doza Assist has been sovereign in exactly this sense since day one, and it runs on a laptop.

## What "sovereign" actually looks like on a Mac

Doza Assist transcribes interviews with NVIDIA's Parakeet speech model. Not through NVIDIA's cloud, not through ours. The model weights (parakeet-tdt-0.6b-v2) download once and run locally on Apple Silicon through MLX, Apple's array framework, using the same unified memory and GPU cores that already sit inside every M-series Mac.

That gives you the three things the sovereign AI panels are asking for:

- **Owned model.** The weights live on your disk. Nobody can deprecate them, reprice them, or change their behavior underneath you.
- **Owned infrastructure.** Your infrastructure is the machine you're already editing on. No GPU rack, no colocation, no VPN into a private cluster.
- **Footage that never leaves the building.** Audio is decoded and transcribed on the same box it was ingested on. There is no upload step to audit because there is no upload.

Story analysis follows the same rule. The app ships with a local Gemma 4 model and picks the variant that fits your hardware, from an 8 GB download on a base machine up to a 27B-parameter model on a high-memory Mac Studio. If you choose to route analysis through Anthropic or OpenAI with your own API key, that's an explicit, opt-in decision, and your footage still never touches a Doza server. The default is local.

## No metering

The second half of the sovereignty argument is economic, and it gets less airtime. Cloud transcription is billed per minute of audio. That pricing model quietly shapes editorial behavior: you transcribe the selects, not the whole interview, because the whole interview costs money.

Local inference flips that. Once Parakeet is on your machine, the marginal cost of transcribing your 400th hour of footage is zero. You transcribe everything, which is the only way an assistant can actually find the moment you forgot was in there.

## Why a 0.6B model is the right size

There's a reflex in the industry to equate sovereignty with size: if you're going to own the model, own the biggest one. For speech-to-text on interview footage, that's backwards.

Parakeet TDT at 600 million parameters is small enough to fit comfortably alongside your NLE, fast enough to chew through long-form interviews on a laptop, and accurate enough on clean, single-speaker or small-panel audio that the transcript is usable for finding story beats without a cleanup pass. It doesn't need a data center because the job doesn't need a data center.

## What this means for the IBC crowd

If you're walking the floor this weekend looking at sovereign AI proposals, a few questions worth asking every vendor:

1. Where do the model weights physically live, and can I keep running them if you go away?
2. What leaves my network during a normal job, and can I verify that with a packet capture?
3. What does the meter look like at hour 1,000?

For Doza Assist the answers are: on your Mac, nothing, and there is no meter.

Doza Assist Core is free and open source under MIT, so you can read the transcription pipeline yourself rather than take my word for it. The commercial Mac app adds multi-interview and press-ready workflow features on top of the same local foundation.

Sovereignty shouldn't require a data center. It should require a download.

---

*Chris Doza is a documentary filmmaker and the creator of Doza Assist. Try it at [doza.ai](https://doza.ai) or read the source on [GitHub](https://github.com/DozaVisuals/doza-assist).*
