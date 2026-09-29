# Herd-Bench — generative benchmark design

Status: design agreed 2026-08-27, **not yet built**. Sequencing note at the bottom.

The goal is not "our 50 questions". It is a command anyone can point at any corpus — including their own private one — that generates a benchmark, runs the arms, and prints a table. That reusability is the artifact contribution of the paper.

## Pipeline

```
fetch  ->  partition  ->  plant  ->  generate  ->  verify  ->  run  ->  score
```

Every stage is a command; every stage is seeded; the whole thing is reproducible from a manifest.

### 1. fetch

Ship a fetcher, never the corpus. Standard practice (LOFT, RULER, ∞Bench) and it sidesteps redistribution entirely — we point at sources, we do not republish them.

Determinism is the requirement, or numbers are not comparable across people:
- manifest of **exact arXiv IDs** and a **pinned Stack Exchange dump snapshot** (dated, on archive.org) — never a live query, which returns different data every week
- **SHA256 per file**, verified on download
- **fixed seed** recorded in the manifest

Repo ships `bench/corpora/<name>.json` (ids + hashes + seed) plus the fetcher.

**Licensing:** arXiv's default licence is *non-exclusive distribution* — it permits arXiv to distribute, not us. Filter to **CC-BY / CC-BY-SA / CC0** via the metadata field. Stack Exchange dumps are CC BY-SA throughout. Filter to **post-cutoff dates** so memorisation cannot substitute for retrieval.

### 2. partition

Two partitionings over the *same* corpus, which makes "does specialisation matter?" an experiment rather than an assumption:

- **domain** — arXiv categories (`cs.CL`, `math.PR`, `q-bio`) or Stack Exchange sites. Real domains, no hand-waving about what a partition means.
- **random control** — same file count and token budget per partition, membership shuffled by seed.

If the random control performs as well as domain partitioning, specialisation is theatre and that is a finding. If it does not, the design is empirically justified rather than merely asserted.

### 3. plant

Synthetic facts inserted into documents. Not a convenience — the only way to prove an elephant recalled something from resident context rather than from training data. Every corpus worth using is already memorised.

**Random position**, recorded. Position bias (lost-in-the-middle) is the best-documented long-context failure; without recording position you cannot distinguish "the elephant missed it" from "everything at 700K gets missed".

**Three difficulty rungs**, same random positions:

| Rung | Needle | Purpose |
|---|---|---|
| 1 | random token (`X7QF-2291`) | sanity check — plumbing works. Herd *and* RAG should score ~100%; a failure here means a bug, not a finding |
| 2 | plausible fact, question shares its vocabulary | the ordinary case |
| 3 | plausible fact, question shares almost none of its vocabulary | needs a semantic hop — where the idea lives or dies |

The **gap between rung 1 and rung 3** is the result. Both approaches acing rung 1 while only the herd survives rung 3 is a clean, hard-to-dispute finding.

### 4. generate — abstract-blind

The trap that would sink everything: generating questions from document text makes the question inherit the document's vocabulary, so retrieval finds it trivially and the benchmark measures nothing. This is why NoLiMa exists.

**Fix (structural, not a filter): the generator sees only the abstract, never the body.** It cannot leak body vocabulary because it never saw it. arXiv already separates the two, so this is free.

This also produces genuine buried facts for free: the abstract says what the paper is *about*, the body holds the detail, so a question generated from the abstract and answerable only from section 4.3 is buried by construction rather than by contrivance.

For corpora without abstracts (Stack Exchange, a personal vault): generate a short summary of the document with one model, then generate from the summary with a second model that never sees the source.

### 5. verify

Two checks, both cheap, both mechanical:

- **answerable from the body?** The generator works blind, so it will sometimes ask what the paper never says. A validator that *does* see the body confirms the answer exists and records where.
- **answerable from the abstract alone?** If yes, discard — abstracts rank near the top for any retriever, so RAG finds it trivially and the question tests nothing.

Keep only: topic visible in the abstract, answer buried in the body.

Also record the **lexical-overlap distribution** between questions and gold passages. Overlap will not be zero (abstract and body share terminology) and should not be — "that paper on sparse attention, what did they use for the ablation?" is how people actually talk. Reporting the distribution is the evidence that generation was not rigged.

### 6. conversations, with topic shift

Eavesdrop fixtures are multi-turn and **deliberately change subject mid-conversation**. A single-topic conversation cannot exercise the dynamics that make a herd interesting:

- **wake on shift** — a quiet elephant speaks when the topic turns toward its domain
- **sleep on shift** — an elephant that contributed on turn 3 goes silent by turn 7 instead of continuing out of momentum. This *stickiness* is a real failure mode and is unmeasured in the literature
- **handoff** — one domain hands to another mid-conversation; structurally impossible for a single agent to do well
- **router breakage** — a router that chose partitions from turn 1 is wrong by turn 7, while broadcast is unaffected. This is the strongest available argument for broadcast over routing, and it only appears in shifting conversations

Turns are a mix of trigger turns (a planted fact is genuinely relevant), substantive-but-irrelevant turns (silence is correct — catches false contributions), and filler (`ok`, `thanks` — the turn gate should drop these before the herd sees them).

### 7. run & score

Arms: RAG · single long-context · herd broadcast · herd routed. Metrics per arm, per rung, per tag: recall@gold, correctness, false-contribution rate, silent-false-negative rate, stickiness rate, $/turn, latency.

## Why this is the contribution

Gate 1 concluded the architecture is not novel. This pipeline is: a reproducible, corpus-agnostic generator for *conversational* knowledge-recall benchmarks, with memorisation controlled by construction, difficulty laddered, and specialisation testable rather than assumed. Nobody has published that.

## Sequencing

This is the **full benchmark**, not the pilot. The pilot (`plans/260827-1108-herd-pilot/`) stays deliberately small — private vault, hand-authored fixtures, ~$10 — and exists only to answer "is there any signal here at all". Building this generator before the pilot answers that would be exactly the mistake the gate structure exists to prevent.

Build order: pilot → go/no-go → this.

## Open questions

1. Does the abstract-blind generator produce enough *usable* questions after the two verify filters, or is the discard rate so high it becomes expensive? Measure on 20 papers before committing.
2. Stack Exchange threads have no abstract — is one-model summarisation a strong enough blind, or does the summariser leak vocabulary too?
3. How many topic shifts per conversation is realistic rather than artificial?
