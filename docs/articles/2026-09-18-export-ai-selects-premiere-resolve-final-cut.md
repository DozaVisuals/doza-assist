---
title: "Premiere, Resolve, or Final Cut: Getting AI Selects Into Your NLE Without Re-Cutting"
slug: export-ai-selects-premiere-resolve-final-cut
date: 2026-09-18
author: Chris Cardoza
description: "A practical comparison of FCPXML, Premiere XML, and CMX 3600 EDL for moving selects from a transcription tool into Final Cut Pro, Premiere Pro, and DaVinci Resolve. What each format carries and what it loses."
category: Guides
---

# Getting AI Selects Into Your NLE Without Re-Cutting

Finding the moments is half the job. The other half is getting them onto a timeline without retyping timecodes. Every editing app speaks a different interchange dialect, and the differences decide whether your selects arrive as a usable rough cut or a list of numbers.

This is a field guide to the three formats Doza Assist exports, what each one carries into Final Cut Pro, Premiere Pro, and DaVinci Resolve, and what gets lost along the way.

## The three formats at a glance

| Target | Format | Carries | Loses |
|---|---|---|---|
| Final Cut Pro | FCPXML 1.11+ | Cuts as real edits, keyword ranges, chapter markers, multicam round-trip | Nothing significant |
| Premiere Pro | Final Cut Pro 7 XML (xmeml v5) | Cuts on V1 + A1/A2, clip names | Keyword ranges, marker styles |
| DaVinci Resolve | CMX 3600 EDL | Cuts, clip names and notes as comments | Multicam relationships, color labels, rich notes |

Doza Assist has an "Edit in" selector in the project header. Pick your platform once and every export button in the project switches to the right format. The choice persists per project and sets the default for new ones.

## Final Cut Pro: FCPXML

This is the richest path. The export writes a pre-cut timeline where each select is an actual edit referencing your source media, plus keyword ranges on the source clip so you can filter the browser by select label, plus chapter markers for navigation. Import it and you have an Event with the media and a Project with the clips in order.

If the footage came in as a multicam or synchronized clip from an FCPXML export, use the FCPXML Round-Trip section instead of the plain export. It preserves the original multicam container and bookmarks so selects come back with video and synced audio intact. We covered why that matters in [Tuesday's post on the round-trip](/news/fcpxml-round-trip-multicam-patent-pending/).

## Premiere Pro: Final Cut Pro 7 XML

Premiere doesn't read FCPXML. What it reads well is the older Final Cut Pro 7 XML format, which Adobe still recommends for third-party round-tripping. Doza Assist writes xmeml version 5 with integer frame timing, the rounded timebase (24, 25, 30) and the NTSC flag set for fractional rates like 23.976 and 29.97, exactly the way FCP7 itself produced it. Premiere has accepted that for years.

Import via File, then Import, and the selects land on V1 with A1 and A2. File references use absolute paths, so keep the media where it was when you transcribed, or relink once on import.

What you lose: keyword ranges and marker styles don't survive, because the format has no equivalent. Clip names do.

## DaVinci Resolve: CMX 3600 EDL

EDL is the oldest format of the three and the most universal. It's plain text. Resolve rebuilds cuts cleanly from it as long as the source media is in the project bin and the reel name matches. Import via File, Import, Timeline, then Pre-Conformed EDL.

Doza Assist writes clip names and editorial notes as comment lines, so the context survives in a readable form even though EDL has no field for it. The format's limits are real, though: no multicam relationships, no color labels, and reel names are truncated to 32 characters. The app surfaces each of those as a warning at export time rather than silently dropping data.

## Which to pick when you have a choice

If you're platform-agnostic, Final Cut's FCPXML carries the most. If you live in Premiere, the FCP7 XML path is solid and has been for a decade. If you're in Resolve, EDL works, and the tradeoff is that you'll re-apply labels by hand.

Whichever you use, the clips arrive in story order. Story Builder's one-click export puts a full sequence on the timeline, ordered the way the story was assembled, ready to refine. That's the point: the AI's job ends where your cut begins.

## FAQ

### How do I import AI-generated clips into Premiere Pro?
Export as Final Cut Pro 7 XML, then in Premiere use File, Import. The clips land on V1 with A1 and A2 tracks. Doza Assist writes this format when "Edit in" is set to Premiere Pro.

### Can DaVinci Resolve import selects from a transcription tool?
Yes, through a CMX 3600 EDL. In Resolve use File, Import, Timeline, Pre-Conformed EDL, with the source media already in the bin. Clip names and notes arrive as EDL comments.

### Does FCPXML preserve multicam when exporting selects?
Only if the tool writes container-aware clips over the original resources block. Doza Assist's FCPXML Round-Trip does; a plain FCPXML export writes against the audio file and doesn't.

### What frame rates do the exports handle?
Integer and fractional NTSC rates, including 23.976, 24, 25, 29.97, 30, 59.94, and 60. Fractional rates use the 1001 frame duration in FCPXML and the NTSC flag in Premiere XML.

---

*Chris Cardoza is a documentary filmmaker and the creator of Doza Assist. All three exporters are open source in [Doza Assist Core](https://github.com/DozaVisuals/doza-assist).*
