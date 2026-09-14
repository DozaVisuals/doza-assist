---
title: "Local First, Frontier When You Want It: Why Doza Assist Gives Editors a Choice of AI Model"
slug: local-first-frontier-models-mcp
date: 2026-09-14
author: Chris Cardoza
description: "Doza Assist runs a local model by default and lets you switch to Claude or GPT with your own key. With the MCP connector, Claude Desktop can read your transcripts too. Here's why choice matters."
category: Perspectives
---

# Local First, Frontier When You Want It

Last week's post was about sovereignty: owned models, owned hardware, footage that never leaves the building. A few people wrote back with the same fair question. If you believe in local AI so strongly, why does Doza Assist have a settings panel for Anthropic and OpenAI keys?

Because sovereignty is about who decides, not about which model you use. The whole point of owning your setup is that nobody else gets to make that call for you. Sometimes the right call is a 4-billion-parameter model running on your laptop. Sometimes it's the most capable frontier model on the planet, for one hard job, at your own expense. An editor should be able to make that choice per project, per task, and change their mind tomorrow.

So here is how Doza Assist actually handles it, and what the new MCP connector adds.

## Local by default, always

Out of the box, every AI feature in Doza Assist runs on your Mac. Transcription uses NVIDIA's Parakeet speech model through MLX. Story analysis, clip finding, chat, and the My Style profile all run on a local Gemma 4 model that the app sizes to your hardware, from a 2B variant on a base machine to a 27B variant on a high-memory Mac Studio.

No account. No usage fee. Works on a plane. Your footage, your transcript, and your editorial profile never leave the machine. That's the default, and the default is what most people never change. We think that's the correct default for anyone handling interview footage, and we're not going to move it.

## Frontier models, if you want them, on your terms

If you want sharper analysis, open AI Settings and paste in your own Anthropic or OpenAI API key. From that point, the parts of the app you choose route through Claude or GPT instead of the local model.

A few things about how this is built, because the details are where trust lives:

- **Your key, your bill.** The API call goes from your Mac directly to Anthropic or OpenAI. Doza Visuals is not in the middle. There's no proxy, no markup, and no way for us to see what you send. A typical interview costs a few cents, billed by the provider.
- **Keys live in the macOS Keychain**, not in a config file on disk.
- **Footage still never leaves.** Only transcript text and the prompt go to the API. Your video and audio stay local no matter which model is selected.
- **The model is matched to the job.** When Claude is selected, deep work like story briefs, editorial analysis, and building a My Style profile routes to the stronger Opus tier. Lighter work like chat routes to Sonnet. You get the frontier where it matters and don't pay for it where it doesn't.
- **One flip to go back.** Remove the key and everything returns to local. Nothing about your project changes.

This is what choice looks like in practice. Not a cloud product with a "local mode" checkbox that quietly does less, and not a local product that pretends the frontier doesn't exist.

## The MCP connector: bring your own assistant

The Doza Assist app for Mac now ships a connector for Claude Desktop, built on the Model Context Protocol (MCP). MCP is the open standard that lets an AI assistant call tools in other apps. Ours is deliberately narrow.

Once you install it, you can ask Claude things like:

- "Read the transcript of the Maria interview and summarize the story in five beats."
- "Search it for every time she mentions her father."
- "Mark the strongest 20 seconds about leaving home as a select."
- "Build a stringout of all the selects."

Selects that Claude creates show up in your Clip Library within seconds, labeled "AI assistant," so you always know which ones were yours.

What makes this consistent with everything above is the gate. Nothing is visible to Claude until you open a specific project from its gear menu and switch on "Claude can read this project." Switch it off and Claude sees nothing. The connector runs entirely on your Mac and talks to the app over the loopback interface only. It can't see your media files, can't read your My Style profiles, can't touch projects you haven't opened, and there is no delete tool at all.

We built it this way because "connect your AI to everything" is the wrong instinct for an editor with client footage. "Connect your AI to this one interview, for this one afternoon" is the right one.

## Why we won't pick for you

There's commercial pressure in this industry to pick a side. Cloud-only tools want the recurring revenue. Local-only purists want the ideological clarity. Both positions are easier to market than "it depends."

But it does depend. A one-person doc shop cutting a sensitive interview under NDA should never send a word of it anywhere, and with Doza Assist they never have to. A broadcast team on a deadline with a public-figure interview might happily pay eight cents to have Opus find the story beats in ninety seconds. Same app, same project format, different call, made by the person who understands the footage.

Local first, always. Frontier when you want it. Your key, your gate, your decision.

## FAQ

### Does Doza Assist require an API key?
No. The app ships with a local Gemma 4 model and NVIDIA Parakeet transcription that run entirely on your Mac with no account and no usage fees. An Anthropic or OpenAI key is optional.

### If I add a Claude or OpenAI key, does my footage get uploaded?
No. Only transcript text and the prompt are sent, directly from your Mac to the provider you chose. Video and audio files never leave your machine, and Doza Visuals never sees the request.

### What is the Doza Assist MCP connector?
It's a Claude Desktop extension that lets Claude read transcripts and selects, search a transcript, create selects, and build a stringout, for projects you explicitly switch on. It runs on your Mac over loopback and is MIT licensed. It requires the Doza Assist app for Mac.

### Does the MCP connector work with Doza Assist Core?
Not yet. The open-source Core doesn't have the assistant-access panel that gates which projects are visible. The connector works with the Doza Assist app for Mac, which includes a free trial.

---

*Chris Cardoza is a documentary filmmaker and the creator of Doza Assist. The MCP connector is open source at [github.com/DozaVisuals/doza-assist-claude-extension](https://github.com/DozaVisuals/doza-assist-claude-extension).*
