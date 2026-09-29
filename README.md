# Elephant Herd

Your notes, held in memory by agents that listen to your conversations and speak up
when they know something.

Not retrieval. Nothing gets searched. The notes are already in context.

## The problem

A folder of notes grows for months. The assistant you're working with sees whatever a
retriever pulled for the current question and nothing else. You wrote the decision
down in March. Nobody has read it since.

Retrieval only runs when someone asks. Most of the time nobody asks.

## How it works

Split your notes into domains. Give each domain to its own agent, an **elephant**,
which holds the whole thing in context. A **herd** is a set of elephants plus a
**herd leader**.

Every turn of your conversation goes to all of them. Each elephant either says
`<silent/>` or gives you a fact with the file it came from. The leader turns whatever
came back into a short briefing and hands it to the agent you're actually talking to.

Most turns, most elephants say nothing. That's the point.

```
turn ─► [projects] [research] [notes] [clients] ...   elephants, each holding a domain
              └───────────┬────────────┘
                     herd leader ─► briefing for the main agent
```

## Why it works now

Two things got cheap. Small models got good, and prompt caching means you pay for the
corpus once instead of every turn.

| Setup | $/turn on a 1.19M-token corpus |
|---|---|
| Frontier model, no caching | 5.95 |
| Frontier model, cached | 0.60 |
| Small model, no caching | 0.089 |
| **Small model, cached** | **0.018** |

Cheap models account for about 67x of that. Caching another 5x. Two years ago this
design was absurd. Now a turn costs two cents.

Caching matters more than the price suggests. Reading the corpus again is nearly free,
so the herd can go round more than once, and that turns out to be what makes multi-hop
questions work.

## What we measured

On RULER (NVIDIA, Apache 2.0), 131K-token contexts, the same model in every arm so the
comparison is about splitting the corpus and not about model quality.

| Task | one model, whole context | herd, 7 partitions |
|---|---|---|
| find one fact | 100% | 100% |
| find one fact among distractors | 100% | 100% |
| find four scattered facts | 100% | 100% |
| follow a chain of references | 100% | 100% |

Splitting the corpus costs nothing, including when the answer has to be assembled from
links that no single elephant can see whole.

We also checked whether splitting *helps* recall as the context grows. It does not: at
131K, 256K and 512K tokens, one model reading everything and seven elephants reading a
seventh each both score 100%, at the same cost. Split because a real corpus does not fit
in any context window, and because elephants that map to real subjects can tell you
which one should have known something. Not because it remembers better.

### Which models can do the listening

Three tasks, 128K tokens, 7 partitions, one round, 10 samples per cell:

| model | one fact | several facts | chain | chain, old prompt | $/sample |
|---|---|---|---|---|---|
| deepseek-v4.1-flash | 10/10 | 10/10 | 10/10 | 0/10 | $0.011 |
| gemini-3.7-flash | 10/10 | 10/10 | 10/10 | 0/10 | $0.086 |
| claude-sonnet-5 | 10/10 | 10/10 | 9/10 | 0/10 | $0.357 |
| glm-5.3-flash | 10/10 | 10/10 | 9/10 | 1/10 | $0.016 |
| qwen3.7-flash | 10/10 | 10/10 | 9/10 | 0/10 | $0.004 |
| xiaomi/mimo-v2.5 | 10/10 | 8/10 | 2/10 | 0/9 | $0.012 |
| gemini-2.5-flash-lite | 10/10 | 9/10 | 0/10 | 0/10 | $0.016 |

Cheap models do this job: four of them match Sonnet 5 at up to a ninetieth of the price.
Two models fail on chains under both prompts, and we have not yet found out why.

The "old prompt" column is the first wording we used, with a 2,000-token reply cap. It
broke chains for every model, Sonnet included. See the first two notes below.

### A test that looks like real use

The puzzles above hide one fact in filler text. Real notes are different: many
documents on one subject, answers that changed over time, and questions asked in other
words than the text uses. So we built a test from a real company's public handbook:
GitLab's, pinned to one commit, eight sections, about 800K tokens.

96 questions and 20 overheard conversations, none using the handbook's own wording.
Claude wrote them from the handbook, a separate Claude pass checked every answer
against its source, and GPT-5.4 graded the herd's replies.

| listener | correct | full run |
|---|---|---|
| deepseek-v4.1-flash | 111/116 | $0.59 |
| glm-5.3-flash | 111/116 | $4.16 |

By kind, for DeepSeek: answer held by one elephant 45/45, more than one right answer
8/8, answer changed over time 10/10, answer spread over two or three elephants 17/19,
no answer in the handbook 12/14, overheard conversations 19/20. DeepSeek is the
default listener because of this table.

The weak spot is noise: about a third of replies come from elephants that do not hold
the answer, often to say so. In normal use the leader filters them out, but they still
cost.

Everything needed to rerun it is in `bench/handbook/`: the questions with their accepted
answers and sources, the model, date and commit they were written from, and the exact
prompts used. `python -m herd.handbook_bench --model deepseek/deepseek-v4.1-flash
--provider deepinfra/fp8` runs it; the handbook is fetched at the pinned commit.

## Use it

```bash
git clone https://github.com/Florin-Birgu/elephant-herd
cd elephant-herd && pip install -e .        # not on PyPI yet
export OPENROUTER_API_KEY=sk-or-...
herd claude install          # hooks + the /herd:init skill. Nothing runs yet.
```

Then in a Claude Code session, inside the project you want it in:

```
/herd:init
```

It reads your folders, proposes which ones belong together, shows you the token
counts and what a turn will cost, and takes your corrections. Grouping is a judgement
call that needs to see the content and be argued with, so it happens in the
conversation rather than in a wizard.

After that:

```bash
herd on                      # this directory only. Now it bills.
herd ask "what did we decide about pricing?"
herd ask --deep "where does this value end up?"
herd status                  # what is attached here, and what it costs per turn
herd off
```

Off by default and per-directory, so installing costs nothing and other projects are
untouched. Herd definitions live in `~/.herd/herds/`, attachments in
`~/.herd/attachments.json`. Your paths never enter the repo.

Not using Claude Code? `herd init --auto` groups by folder name without asking
anything. It is worse, because folder names do not tell you what is inside them.

## Things that cost us a day

- **Pin the provider.** The same model is served by ~30 hosts at cache-read prices
  spanning 10x, at quantizations from fp4 to bf16, and some with no caching at all. If
  you don't pin, your results change between calls and you won't know why.
- **Don't trust listed cache prices.** One provider billed 4.2x its advertised rate,
  because a 5-minute TTL means calls inside the window re-create the cache while still
  reporting a 100% cache hit. Measure one warm call before planning anything.
- **Watch `finish_reason`.** An agent that runs out of output budget mid-thought sends
  back nothing, which looks identical to staying silent. We scored real misses as
  deliberate silence until we caught it.
- **Ask for material, not answers.** An elephant told "answer or stay silent" will
  correctly stay silent when its slice holds only part of the answer, and the fact never
  surfaces. Ask for anything relevant to the question instead. With the first wording
  (and a small reply cap, next note) seven models scored 0 or 1 out of 10 on chained
  facts, Sonnet 5 included; with the second, five of them scored 9 or 10.
- **Give replies room.** Reasoning comes out of the same token budget as the answer. At
  2,000 tokens a careful model thinks, runs out, and returns an empty string, which is
  scored as silence. 20,000 fixed it.
- **Leave reasoning on.** Turning it off saves about 200 output tokens a turn, worth
  $0.0001, and loses correct answers.

## The conversation is an elephant too

Domain elephants know facts. None of them knows which client "the client" is.
That knowledge is in the conversation, so the conversation becomes an elephant like the rest.

It speaks first, so the others know what the turn is about before deciding whether
they hold something relevant. Then it stays in the pool, because something you said
three days ago and never wrote down exists nowhere else.

It stays cheap for a reason worth knowing: conversations only grow at the end, so
appending a turn leaves the cached prefix intact. A 5.7MB transcript filters to about
52K tokens once tool output and thinking are stripped, which makes it one of the
smaller members of the herd.

## Status

Works, used daily by one person, not yet used by anyone else.

Done: the runtime, both Claude Code hooks, the conversation elephant, `--deep` with
the halting fix, named herds, the `/herd:init` skill, the benchmark harness, and the
RULER results above. The listener model was chosen by measurement, not by price list:
DeepSeek-V4.1-Flash answered 111 of 116 questions about GitLab's public handbook, the
same as GLM-5.3-Flash, at a seventh of the cost.

Not done, and worth knowing before you rely on it:

- No RAG baseline, so "better than retrieval" is untested. The measured comparison is
  partitioned-versus-whole-context, which is a different question.
- No test of whether grouping by subject beats grouping at random. That is the central
  design claim and it needs a question set that only real use produces.
- The overhearing case is unmeasured. Every benchmark number here comes from direct
  questions, which is not the setting this was built for.
- Barely any tests, and nothing has been run end to end since the named-herd refactor.

Write-up in `docs/paper-draft.md`.

## Licence

Apache 2.0. Includes a patent grant.

Benchmark corpora are fetched, never committed. The repo ships manifests and download
scripts. RULER is Apache 2.0 and unrestricted. NoLiMa is Adobe Research, non-commercial
research only, so its numbers belong in a paper and not on a pricing page.

---

*Elephants never forget, and a herd is led by its matriarch, the one who remembers
where the water was. A good elephant also knows when what it holds is simply
irrelephant.*
