---
title: "Why FCPXML Round-Trips Break, and How We Fixed It (Patent Pending)"
slug: fcpxml-round-trip-multicam-patent-pending
date: 2026-09-15
author: Chris Cardoza
description: "Most tools that write FCPXML lose your multicam angles and bookmarks, so selects come back into Final Cut Pro with audio but no video. Here's why, and how Doza Assist's patent-pending round-trip keeps the container intact."
category: Engineering
---

# Why FCPXML Round-Trips Break, and How We Fixed It

If you've ever sent a Final Cut Pro project through a third-party tool and gotten it back with the audio in place and a black hole where the video should be, you've hit the FCPXML round-trip problem. It's the most common complaint in every transcription and AI-selects tool that claims Final Cut support, and it's the reason a lot of editors keep those tools at arm's length.

Doza Assist has a round-trip that doesn't break, and we've filed a patent on the method. This post explains what goes wrong in the usual approach and what ours does differently, at the level of detail we already publish in the open-source documentation.

## What FCPXML actually contains

FCPXML is Final Cut's interchange format. A typical export from an interview edit has two halves that matter here.

The first is the **resources** block. It lists every asset the project touches, each with an ID, and for every asset it carries a base64 security-scoped bookmark. That bookmark is how Final Cut finds the media on re-import. It is not a path. It's an opaque blob that macOS resolves back to a file, and Final Cut is strict about it: if the bookmark doesn't match what it expects, the media comes in offline or the import fails.

The second is the **timeline**. For an interview shot on two or three cameras, the timeline usually isn't a plain clip. It's a multicam clip, an `mc-clip` element that references a multicam container, and inside the container is the set of angles with one of them flagged as the enabled audio source. A synchronized clip is the simpler cousin: an `asset-clip` with an audio role tag.

## Why most round-trips lose the video

Here's what a typical tool does. It parses the FCPXML far enough to find an audio file, transcribes it, lets you mark selects, and then writes a brand new FCPXML with a flat timeline of asset-clips pointing at that audio file.

Two things go wrong at once. The new timeline references the audio asset only, so the multicam container and its angle enablement are gone. Final Cut imports your selects as audio-only clips. And the tool generates a fresh resources block, so the bookmark blobs are either missing or regenerated, and Final Cut refuses to relink cleanly. You end up manually reconnecting media and rebuilding multicam relationships for every select. At that point the tool cost you more time than it saved.

## What Doza Assist does instead

The round-trip in Doza Assist treats the original FCPXML as the source of truth and never rebuilds what it doesn't need to.

On import, it reads the multicam clip on the timeline, finds the angle flagged as the enabled audio source, and resolves that angle back to the underlying audio asset. For synchronized clips it reads the dialogue-role asset-clip directly. It handles FCPXML 1.13 and 1.14, both the loose `.fcpxml` file and the `.fcpxmld` bundle Final Cut produces by default, and it decodes URL-encoded paths with spaces and special characters before touching the filesystem. The detected audio path is shown at the top of the project so you can confirm it found the right file before transcription starts. Your source media stays on your edit drive, untouched.

On export, after you've made selects, the FCPXML Round-Trip panel gives you two modes:

- **Selects as a new project.** Each select becomes an `mc-clip` (or an `asset-clip` for sync-clip sources) on a fresh timeline. It reuses the same reference ID, the same angle enablement, and the same resources block as the original. Import it and your selects drop in as a new project against the original multicam, with video and synced audio intact.
- **Markers on the existing timeline.** A copy of your original timeline with a marker at each select's in-point. Marker style encodes the select type: completion markers for the strongest picks, standard markers for supporting material, to-do markers for open questions.

The detail that makes it land: the original resources block, including the asset IDs and the base64 bookmark blobs, is preserved byte for byte in the output. Final Cut sees exactly the bookmarks it wrote, so relinking is a non-event.

That combination, reading the enabled angle out of the container on the way in and writing selects back as container-aware clips over a byte-identical resources block on the way out, is what we've filed for patent protection on. It's the difference between a tool that exports "to Final Cut" and one that actually round-trips.

## One thing to watch

Doza Assist also has a plain "Export FCPXML" button that writes a standard pre-cut timeline with keyword ranges and chapter markers. That's the right choice for single-camera footage you imported as a media file. For anything that came in as an FCPXML with multicam angles or sync clips, use the FCPXML Round-Trip section. The plain export writes against the audio file only, which is exactly the failure mode described above.

## FAQ

### Why do my selects import into Final Cut with audio but no video?
The tool that wrote the FCPXML built a flat timeline against the audio file and dropped the multicam container. Final Cut has no angle information, so it imports audio-only clips. Doza Assist's round-trip writes selects back as multicam clips with the original angle enablement.

### Does Doza Assist support multicam FCPXML?
Yes. It reads the enabled audio angle out of the multicam clip on import and writes selects back as `mc-clip` elements on export, reusing the original references and resources block.

### Which FCPXML versions does Doza Assist read?
FCPXML 1.13 and 1.14, as a loose `.fcpxml` file or a `.fcpxmld` bundle.

### What does "patent pending" cover?
The method of extracting the enabled audio angle from a multicam or synchronized-clip container for transcription, and writing selections back as container-aware clips over a preserved resources block so Final Cut relinks without intervention. The open-source parser and the plain FCPXML exporter remain MIT licensed.

---

*Chris Cardoza is a documentary filmmaker and the creator of Doza Assist. The FCPXML parser is part of [Doza Assist Core](https://github.com/DozaVisuals/doza-assist), free and open source.*
