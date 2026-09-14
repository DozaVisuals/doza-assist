---
title: "AI That Edits Like You: What 'My Style' Learns From Your Cuts"
slug: ai-that-edits-like-you-my-style
date: 2026-09-17
author: Chris Cardoza
description: "Most AI video editing tools sound generic because they've never seen your work. My Style in Doza Assist builds an editorial profile from your finished cuts, locally, and filters every suggestion through it."
category: Product
---

# AI That Edits Like You

Ask any AI editing assistant to find the best moments in an interview and you'll get a competent, forgettable answer. The hook it picks is the obvious one. The structure is the structure every explainer video uses. It's not wrong. It just isn't you.

That's not a model quality problem. It's a data problem. The assistant has never seen a single thing you've cut, so it falls back on the average of everything it was trained on. My Style, the feature at the center of Doza Assist, fixes that by learning from your finished work and applying what it learns to every suggestion after that.

## What it reads

You feed My Style the pieces you've already finished. It accepts an FCPXML export of a completed edit, so the app can hear exactly what audio survived the cut, in timeline order. That transcript of the finished piece is the raw material. It's the closest thing to a record of your editorial decisions that exists: what you kept, what you dropped, where you put it.

## What it learns

From a set of finished pieces, My Style builds a profile with four layers.

- **Narrative patterns.** How long you typically hold on a speaker before cutting away. How you open. How you resolve. Where the emotional peak tends to land in the runtime.
- **Thematic fingerprint.** The subjects and angles that recur across your portfolio. Whether your pieces center the protagonist or the community around them. How much vulnerability you let into the frame.
- **Structural habits.** Cold opens versus scene-setting. Chronological versus intercut. Button endings versus open-ended ones.
- **Voice characteristics.** Tone, formality, whether you lean on narration or let subjects carry the story.

On top of those it writes a short prose summary of your sensibility. The point of the summary is that it's specific. If it reads like "cinematic, measured, observational," it failed, and the profile builder is tuned to avoid that kind of filler.

## How it shows up while you edit

When a profile is active, a green STYLE: ON pill sits in both Chat and Story Builder, showing which profile is applied. Every AI call runs through it: clip finding, story building, chat questions. Toggle it off with one click when you want a neutral read.

Ask Story Builder for a three-minute piece about a subject's move from athlete to coach, and it assembles clips in your arc, not the generic one. The editorial notes it attaches to each clip explain the placement in your terms.

## Multiple profiles

You don't cut a feature documentary the way you cut a 60-second social piece. My Style supports as many profiles as you need: a doc profile, a social cuts profile, a corporate testimonial profile, each learned from the finished work you import into it. A dropdown in Chat and Story Builder switches profiles per session without changing the default. Profiles can be renamed, deleted, or toggled independently.

## Evolution and refinement

As you add more finished work, the profile updates and keeps snapshots, so you can see how your habits shift over time. If the summary gets something wrong, you can correct it in plain language and the correction carries forward.

## It stays yours

Everything about My Style runs locally. Your finished pieces are analyzed on your Mac and the profile is stored there. It's never uploaded, and Doza Visuals never sees it. If you connect a frontier model through your own API key, the profile shapes the prompt, but your finished work itself doesn't leave the machine.

And it's in the free, open-source Doza Assist Core. Not a paid tier. The commercial app adds multi-interview and press-ready workflow features on top, but the editorial voice engine is the thing we most want editors to have, so it's the thing we gave away.

## FAQ

### What does My Style need from me to work?
Finished edits, exported from Final Cut Pro as FCPXML, or the media files of completed pieces. Three to five representative pieces produce a usable profile; more improves it.

### Is my finished work uploaded anywhere?
No. Profile building runs on your Mac and the profile is stored locally. Nothing is sent to Doza Visuals.

### Can I have different styles for different kinds of work?
Yes. Create as many profiles as you need and switch between them per session from a dropdown in Chat or Story Builder.

### Is My Style available in the free version?
Yes. It's fully included in Doza Assist Core under the MIT license.

---

*Chris Cardoza is a documentary filmmaker and the creator of Doza Assist. Try My Style in [Doza Assist Core](https://github.com/DozaVisuals/doza-assist).*
