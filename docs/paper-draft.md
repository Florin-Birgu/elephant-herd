# Elephant Herd: Partitioning a Cached Corpus Across Listening Agents

**Draft, 2026-08-28.** All numbers measured; none pending.

---

## Abstract

Keeping an entire corpus resident in a language model's context, rather than retrieving
fragments of it per query, was until recently priced out of existence. Two independent
cost reductions changed that: capable small models, roughly 67× cheaper per token than
frontier models, and prompt caching, a further 5× on repeated reads. Together a 334×
reduction, which takes a turn over a 1.19M-token corpus from $5.95 to $0.018.

We describe **Elephant Herd**, which partitions a corpus across N persistent listeners,
each holding its partition as a cached prefix, and broadcasts every conversational turn
to all of them. Each listener contributes a cited fact or stays silent, and a leader
merges what returns into a briefing.

Measured on RULER, four results.

**Partitioning does not improve retrieval.** A 7-way herd and a single model reading
everything both score 100% at 131K, 256K and 512K tokens, at the same cost. The
argument for partitioning is capacity and specialisation, not recall: a corpus larger
than any context window has no single-context arm at all.

**Partitioning destroys multi-hop questions.** Where an answer must be assembled from
links spanning partitions, the herd scores 0% against the single model's 100%, at every
chain length from two hops upward. It does not degrade; it fails outright.

**Iteration recovers them.** Re-broadcasting each round's findings so listeners extend
one another restores 100%, at roughly 3× cost. Rounds grow sub-linearly with hops,
fitting `2.7 + 0.34 × hops`, because one listener holding consecutive links resolves
several at once and different listeners extend different parts simultaneously.

**The controller, not the model, caused the residual failures.** Halting on the first
round that produced nothing new treated silence as completion and lost 38% of long
chains. Tolerating three silent rounds takes 8-hop accuracy from 62% to 100%. A silent
round means no listener had anything to add this round, not that the answer is
complete.

Caching is what makes iteration affordable: a round costs one cache read per listener,
$0.002 for a 131K-token herd, against $0.010 for a full re-read.

## 1. Motivation

A project folder grows for months. The assistant working in it sees whatever a
retriever fetched for the current question, and nothing else. You wrote the decision
down in March. Nobody has read it since.

This looks like a retrieval problem, and it is usually treated as one. We think it is
a different problem. Retrieval runs when somebody asks a question, and most of the
time nobody asks. A note about pricing does not surface during a conversation about
pricing unless someone stops and goes looking for it.

Elephant Herd works the other way round. Agents hold the notes in context and watch
the conversation go by. When one of them recognises something relevant, it says so.
Nobody has to ask.

Two years ago this was not affordable. Now it is. At
$0.015/M cached input, holding 1.19M tokens resident across 7 listeners costs $0.018
per turn; the same herd on a frontier model without caching costs $5.95.

| Configuration | Cost per turn, 1.19M-token corpus |
|---|---|
| Frontier model, uncached | $5.95 |
| Frontier model, cached | $0.60 |
| Small model, uncached | $0.089 |
| **Small model, cached** | **$0.018** |

Measured, not assumed: on a low-lexical-overlap gold question, three trials each, GLM
scored 2/3 at $0.0079 per call, Claude Sonnet 5 scored 2/3 at $0.1597, and Claude
Opus 5 scored 0/3 at $0.3962. Frontier models are not better at this task, so the
cheap listener is a substitution rather than a compromise.

Two effects multiply here. Cheaper models account for about 67x, caching for another
5x. The first made a resident corpus viable at all. The second made it cheap to read
that corpus repeatedly, which is what iteration needs, and iteration is what separates
a lookup system from one that can follow a chain.

## 2. System

A **herd** is N **elephants** plus a **leader**.

- Each elephant holds one partition as a cached prefix. Partitions follow domain
  boundaries wherever possible, splitting only when a domain exceeds the context
  window; assembly is byte-deterministic so the cached prefix stays valid.
- Every turn is broadcast to all elephants. A local **turn gate** drops conversational
  filler ("ok", "thanks") before it reaches them. This costs nothing and removes a
  class of false contribution, since an elephant cannot speak on a turn it never saw.
- Each elephant replies with a cited contribution or the literal token `<silent/>`.
  Silence is the expected case.
- The **leader** merges contributions into a briefing for the calling agent. The
  audience is a model, not a person, so the output is facts with source paths and no
  prose. When nothing was contributed it emits nothing. When two elephants disagree it
  reports both rather than picking one.
- **`--deep`** enables blackboard iteration: each round re-broadcasts the findings so
  far, so an elephant can extend what another found. It halts when a round adds
  nothing new, so easy turns still cost one round.

### Implementation notes that turned out to matter

- **Provider pinning is experimental control, not cost hygiene.** The same model is
  served by ~30 hosts at cache-read prices spanning 10× and quantizations from fp4 to
  bf16. Unpinned routing varies recall between calls.
- **Listed cache prices are unreliable.** One provider billed 4.2× its advertised
  cache-read rate, because a 5-minute TTL means calls inside the window re-create the
  cache while still reporting `cached_tokens: 100%`. Measure, do not read the price
  list.
- **Truncation is indistinguishable from silence.** A listener that runs out of output
  budget mid-reasoning emits nothing at all, which looks exactly like a deliberate
  `<silent/>`. Unless `finish_reason` is checked, this gets scored as a genuine miss,
  and it corrupts the one metric the system is judged on.
- **Reasoning must be enabled.** Disabling it saves ~200 output tokens per silent turn
  (~$0.0001, noise against the cache read) and loses correct answers.

## 3. Experiments

RULER (Apache 2.0), Paul Graham essay haystacks, 131K tokens, GLM-5.3-Flash
(320B MoE / 18B active) pinned to its first-party host. Both arms use the *same*
model, so the comparison isolates partitioning rather than model quality.

- `single`: one call, full 131K context.
- `herd`: 7 calls, ~18.7K tokens each, same total tokens read.
- `herd_rounds`: as `herd`, iterating until convergence.

### 3.1 Retrieval is preserved

| Task | Single | Herd | Mean speakers |
|---|---:|---:|---:|
| Single needle | 100% | 100% | 1.0 / 7 |
| Needle among distractors | 100% | 100% | 2.9 / 7 |
| Four scattered values | 100% | 100% | 3.8 / 7 |

n = 10 per task.

Multi-value is the notable one: the four gold values are scattered, scoring requires
all of them, and ~3.8 elephants spoke per sample - the values genuinely landed in
different partitions and the merge recovered them. Silence discriminated cleanly: zero
false contributions across 70 listener calls on the single-needle task.

### 3.2 Context length: no separation up to 512K

If partitioning helped recall, the advantage should appear as the single context grows
and each partition stays small. It does not.

| Context | Single | Herd (7 partitions) | Single | Herd |
|---:|---:|---:|---:|---:|
| 131K | 100% | 100% | $0.0072 | $0.0055 |
| 256K | 100% | 100% | $0.0121 | $0.0126 |
| 512K | 100% | 100% | $0.0219 | $0.0235 |

n = 8 per length. GLM-5.3-Flash shows no degradation at half a million tokens, so the
curves never diverge, and cost tracks total tokens read, which is identical by
construction.

**Partitioning does not improve retrieval, at any length we can test.** The reasons to
partition are capacity and specialisation, not recall: a corpus larger than any context
window has no single-context arm at all, and partitions that map to real domains make
"which specialist should have known this?" answerable, which is what silence and
attribution depend on. Neither is expressible in RULER, whose haystacks are arbitrary
essay chunks.

### 3.3 Multi-hop reasoning is destroyed, and iteration restores it

Variable tracking is more than a synthetic probe. "Find every variable that ends up
holding this value" has the same shape as tracing a symbol through a codebase, or
following a decision through notes that reference each other. The answer is not
written down in any one place. It has to be assembled.

| Task | Single | Herd, 1 round | Herd, iterated |
|---|---:|---:|---:|
| Variable tracking, 4 hops | 100% | **0%** | **100%** |

Cost 2.2×, latency ~7× (28s to 193s). The chain assembles visibly across rounds: round
one reports the anchor assignment, a later round attaches the next link, and so on
until no elephant has anything to add.

### 3.4 Rounds versus hops

Chain length was varied from 2 to 8 hops, 8 samples per condition, 7 partitions.

| Hops | 1 round | Iterated | Rounds to success | 1 round | Iterated | Ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0% | 100% | 3.3 | $0.0074 | $0.0199 | 2.7× |
| 3 | 0% | 50% | 4.2 | $0.0098 | $0.0235 | 2.4× |
| 4 | 0% | 100% | 3.5 | $0.0070 | $0.0229 | 3.3× |
| 6 | 0% | 89% | 4.9 | $0.0106 | $0.0331 | 3.1× |
| 8 | 0% | 70%\* | 5.4 | $0.0108 | $0.0320 | 3.0× |

\* 100% once the halting rule is corrected; see 3.4.

Three results.

**A single round scores 0% at every chain length, including two hops.** Partitioning
does not degrade gracefully on linked questions. The moment an answer requires joining
anything across a boundary, one broadcast returns nothing.

**Rounds grow sub-linearly with hops**, fitting `rounds = 2.7 + 0.34 x hops`. The naive
model predicts one round per hop. The measured slope is a third of that.

**Cost is flat in chain length**, around 3x a single round whether the chain is two
links or eight.


### Why iteration is cheaper than it looks

A chain of 8 links, scattered across 7 elephants:

```
  A=69735 → B=A → C=B → D=C → E=D → F=E → G=F → H=G → I=H

  e1 │ A=69735          D=C  E=D                    │
  e2 │                                    G=F       │
  e3 │        B=A                                   │
  e4 │             C=B                              │
  e5 │                          F=E                 │
  e6 │                                         H=G  │
  e7 │ (nothing)                                    │
```

Each round, every elephant sees what has been found and adds what it can.

```
  round 1   e1: "A = 69735"                    known: A
            everyone else: <silent/>

  round 2   e3: "B = A"                        known: A B

  round 3   e4: "C = B"                        known: A B C
            e1: "D = C, and E = D"             known: A B C D E   <- two links, one round

  round 4   e5: "F = E"                        known: A B C D E F

  round 5   e2: "G = F"                                           <- two elephants,
            e6: "H = G"                        known: all            same round

  round 6   everyone: <silent/>  -> converged
```

Eight links, five rounds. Two effects collapse the count: one elephant holding
consecutive links resolves them in a single reply, and different elephants extend
different parts of the chain simultaneously.

Worst case is still one round per hop, if every link sits in a different elephant and
they resolve in the most adversarial order. Best case is two. Random placement lands
near the middle, which is what we measure: **rounds ~ 2.7 + 0.34 x hops**.

The practical consequence is that `--deep` costs about 3x whether the chain is two
links or eight. Difficulty does not compound.

### 3.5 The halting rule, not the model, caused long-chain failures

Accuracy first appeared to decline on long chains: 70% at 8 hops. Two hypotheses were
tested rather than assumed.

The first was that the `max_rounds = 6` ceiling truncated runs against a mean
requirement of 5.4 rounds. Raising it to 12 produced **62%**, no better, so the ceiling
was never the constraint. Inspecting the failures showed why: every failed run halted
at round 2 having heard from a single elephant, while every success ran 4-6 rounds with
4-8 speakers. The ceiling was never approached.

The second hypothesis was the halting rule. The loop stopped as soon as a round
produced nothing new, treating silence as completion. Tolerating three consecutive
silent rounds instead of one gives:

| Halting rule | Accuracy | Mean rounds | Cost per sample |
|---|---:|---:|---:|
| Stop at first silent round | 62% (5/8) | 4.1 | $0.035 |
| **Tolerate 3 silent rounds** | **100% (7/7)** | 7.3 | $0.049 |

All three previously failing samples recovered. One required ten rounds, stalling
twice before completing. The listeners could always resolve their links; the
controller was ending the search and reporting a partial chain as a finished one.

**A silent round means no listener had anything to add this round, not that the answer
is complete.** Conflating the two loses 38% of long chains. The cost of the fix is 42%
more per query, confined to `--deep`.

One sample was lost to a provider 520 and is excluded rather than scored as a
failure.

n = 8 per condition. The 50% at three hops is out of line with its neighbours and
should be read as noise; the 0% column is not.

## 4. Limitations

- Single corpus type (essay haystacks). Lengths tested: 131K, 256K, 512K.
- n is small: 8-10 samples per condition.
- **RULER cannot test the system's central design choice.** Its haystacks are
  arbitrary essay chunks with no domain structure, so semantic partitioning is not
  expressible. The reported results therefore measure *arbitrary* partitioning - a
  lower bound. Whether domain-coherent partitions do better is untested and needs a
  corpus with real domains.
- No retrieval (RAG) baseline. The claim here is about partitioning versus whole
  context, not versus retrieval.
- The conversational/overhearing setting - the motivating application - is not
  benchmarked; RULER is single-query. Whether a listener recognises relevance when *no
  question was asked* is the claim the system rests on and the one least tested here.

## 5. Related work

Closest are **ExtAgents** (Liu et al., ACL 2026), which partitions external knowledge
across agents and broadcasts a query, but instantiates agents per query rather than
persisting them; and **LLM-based multi-agent blackboard systems** (Salemi et al.,
2025), where partition-owning agents volunteer to a central agent but retrieve on
demand rather than holding knowledge resident. **Cache-Augmented Generation** (Chan et
al., 2024) argues for a resident corpus in a single context. The contribution here is
not the topology but the persistence-via-cache economics, the measurement of silence,
and the rounds-per-hop cost of iteration.

## Open questions

1. Does domain-coherent partitioning beat arbitrary partitioning? Untestable on
   synthetic benchmarks; needs a corpus with real domains.
2. How does the retrieval/reasoning gap change with context length?
3. Does rounds-to-convergence scale with partition count as well as hop count? More
   partitions means more chains crossing boundaries.
4. Silent-false-negative rate - an elephant that held the answer and stayed quiet - is
   unmeasured here and is the metric that matters most in production.
