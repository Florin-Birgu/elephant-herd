---
name: herd
description: "🐘🐘🐘 Elephant Herd — set up and control specialists that hold your notes in memory and speak up when they know something relevant, without being asked"
user-invocable: true
when_to_use: "Invoke to set up Elephant Herd in a project (/herd init), or to turn it on, check status, ask it something directly, or see what it has cost."
category: knowledge
keywords: [herd, elephant, notes, memory, context, knowledge, vault]
argument-hint: "init|on|off|status|ask|log|sync"
metadata:
  author: elephant-herd
  version: "0.1.0"
---

# Elephant Herd

Splits a folder of notes into domains, gives each domain to an agent that holds it
whole in context, and broadcasts every conversation turn to all of them. Each agent
either stays silent or contributes a fact with its source file. A leader merges what
came back into a briefing for the main agent.

Nothing is retrieved. The notes are already in context.

## Commands

| Command | What it does |
|---|---|
| `/herd init` | Set up a herd for this project. The interesting one; see below. |
| `/herd on` | Start listening in this directory. Begins billing. |
| `/herd off` | Stop. |
| `/herd status` | On or off here, which herd, how many elephants, cost per turn. |
| `/herd ask <q>` | Ask directly, without waiting for the next turn. |
| `/herd log` | What each turn cost and which elephants spoke. |
| `/herd sync` | Pick up new or changed files. |

**Everything except `init` is a wrapper. Run `herd <command>` in the shell as your
first action, before reading anything or reasoning about it, then show the output
verbatim.** Nothing here is a real command: this text is a prompt, so the CLI only
runs if you run it, and a slow or interrupted turn means it never ran at all. `on`,
`off` and `status` must not be delayed or batched with other work.

Never run `on` or `off` for a directory other than the one this session is in.

For stopping, tell the user they can type `! herd off` in their prompt: that runs the
CLI directly, with no model in the loop, so it cannot be interrupted into not
happening.

## /herd init

This is the one that needs judgement rather than a wrapper. The partition determines
whether the herd is useful, and it depends on things the tool cannot see: which
folders are live work, which are abandoned, which belong together despite different
names.

**1. Check the key.** `echo ${OPENROUTER_API_KEY:0:8}`. If empty, tell them to set
`OPENROUTER_API_KEY` and stop. A herd that cannot run is not worth configuring.

**2. Check the hooks.** If `herd status` reports hooks missing, offer to run
`herd claude install`. It backs up `~/.claude/settings.json` and appends only its own
entries.

**3. Offer existing herds.** `herd herds` lists what is already defined. A herd is a
view over folders, not a directory, so an existing one often fits a new project. If
they pick one, `herd attach <name>` and skip to step 7.

**4. Get the facts.** Ask which folders to include, then `herd scan <folders>`. This
is free and local. Read a few of the largest files from each folder to see what is
actually in them.

**5. Propose a grouping.** Group by subject, not by size or folder name. Folders about
the same thing belong together even when named differently: a `SEO` folder, a
`linkedin` folder and a `Strategy` folder are usually one specialist. A large folder
on its own subject stays alone. Small unrelated leftovers become one `misc` group.

Name each group in one or two words after its subject. The test is whether "which
elephant should have known this?" has an obvious answer. If it does not, the grouping
is wrong.

Keep each group under `cap_tokens` (default 800,000). A group that overflows gets split
by its own substructure — author, client, time period — never arbitrarily. Check the
arithmetic yourself from the scan output; do not guess.

**6. Show it and take corrections.** Present the groups with token counts and the cost
per turn, computed as `total_tokens / 1e6 * 0.005` for a warm turn. Say plainly which
folders look abandoned or duplicated and suggest dropping them: an elephant that never
speaks is pure cost. Then let them argue. This step is the point of doing it here
rather than in a CLI.

**7. Write it.** Save to `~/.herd/herds/<name>.yaml`:

```yaml
roots:
  - /absolute/path
groups:
  subject-name: [Folder A, Folder B]
cap_tokens: 800000
```

Then `herd attach <name>` for this directory.

**8. Offer to turn it on.** Show the per-turn cost again and ask. You may run
`herd on` **only for the directory just configured** — never another directory, and
never a herd you did not just set up. Turning it on starts billing, so it is a
deliberate choice each time.

## Things worth telling them

- **It is off by default and per-directory.** Installing costs nothing. Other projects
  are unaffected.
- **The briefing arrives one turn late.** The herd runs after the assistant answers,
  because the answer is what disambiguates the question, and it writes its briefing for
  the next message. Nobody waits.
- **Cost is dominated by corpus size**, not by how much is said. A turn costs roughly
  `tokens / 1e6 * 0.005` warm. Fewer, better-chosen elephants is both cheaper and
  sharper than one herd covering everything.
- **`--deep` is for questions whose answer must be assembled** from several places
  rather than looked up. Costs about 3x. Off by default.
