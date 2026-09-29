"""RULER NIAH, run two ways: one model reading everything, vs a herd reading parts.

RULER (NVIDIA, Apache 2.0) assumes a single model reads one long context. This runner
adds the arm that matters for us - the same haystack partitioned across N listeners,
each holding its slice, all asked the same question.

The comparison is only honest if the *model* is held constant. Otherwise "herd beats
GPT-4o" conflates two things: partitioning, and whichever model the herd happens to
use. So both arms run the same listener by default:

    single : 1 call,  131K tokens of context
    herd   : N calls, 131K/N tokens each, same total tokens read

What differs is calls, not tokens. That is the trade being measured - and the obvious
objection (each elephant has an easier job) is the point, not a flaw, as long as it is
reported rather than hidden.

    python -m herd.ruler --data /tmp/ruler_data/niah_single_2/validation.jsonl --n 7
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from herd import config
from herd.providers.openrouter import OpenRouterClient

# RULER's own instruction wording is kept verbatim so scores stay comparable to
# published numbers; only the context each caller sees changes between arms.
# Each task ends with its own question phrasing.
QUESTION_MARKERS = (
    "\nWhat is the special magic",
    "\nWhat are all the special magic",
    "\nQuestion: Find all variables",
    "\nQuestion:",
)


def split_sample(raw: str) -> tuple[str, str, str]:
    """RULER input -> (instruction, context, question)."""
    q_at = max((raw.rfind(m) for m in QUESTION_MARKERS), default=-1)
    if q_at == -1:
        return "", raw, ""
    head, question = raw[:q_at], raw[q_at:].strip()
    first_nl = head.find("\n")
    instruction, context = (head[:first_nl], head[first_nl + 1:]) if first_nl != -1 else ("", head)
    return instruction.strip(), context.strip(), question


def chunk(text: str, n: int) -> list[str]:
    """Split on whitespace into n roughly equal parts, preserving word boundaries."""
    words = text.split(" ")
    per = len(words) // n + 1
    return [" ".join(words[i:i + per]) for i in range(0, len(words), per)][:n]


ROUND_SYSTEM = (
    "You are given part of a longer text, plus findings other readers reported from\n"
    "their parts. Your job is to EXTEND their findings using your part.\n"
    "Report anything in your part that connects to what they found - for example if\n"
    "they report a value and your part assigns that value onward, report that link.\n"
    "Report only NEW information, not what they already said.\n"
    "If your part adds nothing new, reply with exactly: <silent/>"
)

# Two listener wordings, kept side by side because the difference between them moves
# the score as much as the model does, and a result that does not say which was used
# cannot be compared with anything.
#
# "answer" asks for the answer. For a fact chained across partitions that is
# unanswerable from any single part - an elephant holding "B = A" has no answer - so
# silence is the correct reading, and a model that follows the instruction faithfully
# scores zero while one that ignores it and volunteers its lines scores well.
#
# "material" asks for the lines instead, and leaves the joining up to the merge.
LISTENER_PROMPTS = {
    "answer": (
        "You are given part of a longer text. Answer only from what you were given.\n"
        "If the answer is present, reply with it and nothing else. If the question asks "
        "for ALL matching items, list every one you can find in your part - other parts "
        "are held by others, so report yours completely.\n"
        "If it is not present in your part, reply with exactly: <silent/>"
    ),
    # Multi-matching retrieval - "report every match", not "find the one" - is a
    # documented collapse point: accuracy falls towards zero as the number of matching
    # items grows, for large models as well as small (arXiv 2410.04422). The paper's
    # remedy is to force explicit steps instead of a single-shot answer, so this asks
    # for a walk through the part rather than a summary of it.
    "material_cot": (
        "You hold part of a longer text. Work through it from beginning to end and "
        "write out every line that could bear on the question, exactly as written, in "
        "the order you meet them - including lines that only point at something else "
        "rather than answering.\n"
        "Do not stop at the first match: there are usually several, and a missed line "
        "breaks the answer for everyone. Do not summarise, explain or conclude.\n"
        "When you reach the end of your part, write DONE on its own line.\n"
        "If your part contains nothing of the kind, reply with exactly: <silent/>"
    ),
    # Same instruction as "material" in plain words. "Bear on the question" is not
    # how anyone speaks, and a stiff instruction is a worse instruction - the model has
    # to work out what was meant before it can follow it.
    "plain": (
        "You hold part of a longer text. Report anything in your part that is relevant "
        "to the question, quoted exactly.\n"
        "Include things that only point at something else instead of answering it.\n"
        "Others hold the rest of the text, so do not try to work out the final answer.\n"
        "Nothing relevant? Reply: <silent/>"
    ),
    # "plain" plus one line. GLM's own reasoning on a silent reply showed it saw its link
    # ("VAR AMBYO = VAR COWQQ"), read the instruction to include pointers, then decided
    # the pointer was irrelevant because the other half was not in its part. That line
    # takes the judgement away: the other half is exactly what someone else holds.
    "plain_unsure": (
        "You hold part of a longer text. Report anything in your part that is relevant "
        "to the question, quoted exactly.\n"
        "Include things that only point at something else instead of answering it.\n"
        "If you are not sure whether a line matters, report it anyway: someone else may "
        "hold the other half.\n"
        "Others hold the rest of the text, so do not try to work out the final answer.\n"
        "Nothing relevant? Reply: <silent/>"
    ),
    # Firmer than plain_unsure, which only lifted GLM from 1/8 to 4/8 on the same slice:
    # it still judged a pointer irrelevant when the other end was out of sight.
    "report_not_judge": (
        "You hold part of a longer text. Others hold the rest. Your job is to report, "
        "not to judge.\n"
        "Quote every line in your part that could connect to the question, even through "
        "a chain you cannot see. A line that points at something outside your part still "
        "counts: report it.\n"
        "Do not try to work out the final answer.\n"
        "Reply <silent/> only if no line in your part could connect at all."
    ),
    "material": (
        "You hold part of a longer text. Report every line from your part that could bear "
        "on the question, exactly as written - including lines that only point at something "
        "else rather than answering.\n"
        "Other parts are held by others and the answer may span several of them, so report "
        "yours completely and do not try to finish the reasoning.\n"
        "If your part contains nothing of the kind, reply with exactly: <silent/>"
    ),
}

@dataclass
class Result:
    # Recorded on every row: a results file that does not say which model produced
    # it cannot be traced back to anything, and arms are only comparable when the
    # model is held constant - which only the row itself can prove.
    model: str
    provider: str | None
    arm: str
    index: int
    correct: bool
    answer: str
    n_parts: int
    cost_usd: float
    latency_s: float
    prompt_tokens: int
    speakers: int
    rounds: int = 1
    # One entry per model call: what it cost, how long it thought, how much it wrote,
    # why it stopped, and whether it stayed silent. Totals per sample hid every
    # failure so far - an empty reply, a cut-off think, a dropped link all look alike.
    calls: list = field(default_factory=list)
    # The thinking text per call, written to a separate .thoughts.jsonl file.
    thoughts: list = field(default_factory=list, repr=False)


LISTENER_SYSTEM = LISTENER_PROMPTS["plain"]   # rebound by main() from --prompt


def _is_silent(text: str) -> bool:
    return "<silent/>" in text.lower().replace(" ", "")


def _log(responses) -> tuple[list, list]:
    """Per-call stats and thinking text, in call order (slice order for the herd)."""
    calls, thoughts = [], []
    for k, r in enumerate(responses):
        u = r.usage
        text = r.text.strip()
        calls.append({
            "slice": k, "cost_usd": u.cost_usd, "prompt_tokens": u.prompt_tokens,
            "reasoning_tokens": u.reasoning_tokens,
            "reply_tokens": max(u.completion_tokens - u.reasoning_tokens, 0),
            "finish": u.finish_reason, "latency_s": round(u.latency_s, 1),
            "silent": _is_silent(text), "empty": not text,
        })
        msg = (r.raw.get("choices") or [{}])[0].get("message") or {}
        thoughts.append(msg.get("reasoning") or "")
    return calls, thoughts

# Reasoning is on, and reasoning tokens come out of this same budget before any visible
# output. At 2000 an elephant asked to work through its part thinks until the budget is
# gone and returns an empty reply, which is indistinguishable from silence and was
# scored as one. The product path already scales its budget for this reason; this
# runner did not, which is what made the step-forcing prompt look like a failure.
MAX_TOKENS = 20_000
# Separate cap on thinking, so a long think cannot leave the reply empty. None = no cap.
REASONING_MAX = None


def score(text: str, expected: list[str]) -> bool:
    """RULER scores by substring presence, and ALL gold strings must appear.

    This matters for the herd arm: on multi-value tasks the answers can be spread
    across partitions, so no single elephant sees them all. Scoring "any" would hide
    exactly the failure mode partitioning is most likely to cause.
    """
    low = (text or "").lower()
    return all(str(e).lower() in low for e in expected)


def run_single(sample: dict, client: OpenRouterClient, model: str, provider: str | None,
               idx: int) -> Result:
    instruction, context, question = split_sample(sample["input"])
    resp = client.chat(
        model=model, provider=provider, max_tokens=MAX_TOKENS, reasoning_max_tokens=REASONING_MAX,
        messages=[
            {"role": "system", "content": LISTENER_SYSTEM},
            {"role": "user", "content": f"{instruction}\n{context}\n{question}"},
        ],
    )
    return Result(
        model, provider, "single", idx, score(resp.text, sample["outputs"]), resp.text.strip()[:80],
        1, resp.usage.cost_usd, resp.usage.latency_s, resp.usage.prompt_tokens,
        speakers=1, calls=_log([resp])[0], thoughts=_log([resp])[1],
    )


def run_herd(sample: dict, client: OpenRouterClient, model: str, provider: str | None,
             idx: int, n: int) -> Result:
    instruction, context, question = split_sample(sample["input"])
    parts = chunk(context, n)

    def ask(part: str):
        return client.chat(
            model=model, provider=provider, max_tokens=MAX_TOKENS, reasoning_max_tokens=REASONING_MAX,
            messages=[
                {"role": "system", "content": LISTENER_SYSTEM},
                {"role": "user", "content": f"{instruction}\n{part}\n{question}"},
            ],
        )

    # Elephants are independent: wall-clock is the slowest, not the sum.
    with ThreadPoolExecutor(max_workers=n) as pool:
        responses = list(pool.map(ask, parts))

    # An empty reply is not speaking: it is a cut-off or a failed call, and counting it
    # as a speaker is what once hid a budget bug behind a healthy-looking number.
    spoke = [r for r in responses if r.text.strip() and not _is_silent(r.text)]
    # No leader needed for NIAH: the herd is correct if any elephant surfaced the fact.
    # A synthesis step could only lose information here, so leaving it out measures the
    # listeners themselves rather than the leader's summarising.
    merged = " | ".join(r.text.strip() for r in spoke)
    return Result(
        model, provider, "herd", idx, score(merged, sample["outputs"]), merged[:80], n,
        sum(r.usage.cost_usd for r in responses),
        max(r.usage.latency_s for r in responses),
        sum(r.usage.prompt_tokens for r in responses),
        speakers=len(spoke), calls=_log(responses)[0], thoughts=_log(responses)[1],
    )


def run_herd_rounds(sample: dict, client: OpenRouterClient, model: str,
                    provider: str | None, idx: int, n: int, max_rounds: int = 6,
                    patience: int = 1) -> Result:
    """Blackboard-style iteration: broadcast, then re-broadcast what was found.

    One round finds facts that sit whole inside a single partition. A chain spanning
    partitions needs one round per hop: the elephant holding `B = A` can only speak
    once someone has reported A. Iterate until a round adds nothing new - the classic
    blackboard termination condition - which also bounds cost on easy questions,
    since those converge after round one.
    """
    instruction, context, question = split_sample(sample["input"])
    parts = chunk(context, n)
    found: list[str] = []
    cost = 0.0
    latency = 0.0
    rounds = 0
    speakers = 0
    # `patience` is how many consecutive silent rounds we tolerate before concluding
    # the chain is complete. At 1 (halt on first silence) a single elephant failing to
    # recognise its link ends the search, and no round budget can recover it, because
    # the loop has already stopped.
    quiet = 0
    all_calls: list = []
    all_thoughts: list = []

    for rnd in range(max_rounds):
        rounds = rnd + 1
        board = "\n".join(f"- {f}" for f in found) or "(nothing yet)"
        # Switch to "extend what others found" only once there is something to
        # extend. With an empty board that instruction is incoherent, and a model
        # that follows it correctly answers <silent/> every round - which the
        # halting rule then reads as "the chain is complete". A model that ignores
        # the instruction and talks anyway is rewarded, which is backwards.
        system = LISTENER_SYSTEM if (rnd == 0 or not found) else ROUND_SYSTEM

        def ask(part: str):
            user = f"{instruction}\n{part}\n{question}"
            if rnd:
                user = (f"{instruction}\n{part}\n\n"
                        f"<findings_so_far>\n{board}\n</findings_so_far>\n{question}")
            return client.chat(model=model, provider=provider, max_tokens=MAX_TOKENS, reasoning_max_tokens=REASONING_MAX,
                               messages=[{"role": "system", "content": system},
                                         {"role": "user", "content": user}])

        with ThreadPoolExecutor(max_workers=n) as pool:
            responses = list(pool.map(ask, parts))
        c, t = _log(responses)
        all_calls += [dict(x, round=rounds) for x in c]
        all_thoughts += t
        cost += sum(r.usage.cost_usd for r in responses)
        latency += max(r.usage.latency_s for r in responses)

        new = []
        for r in responses:
            t = r.text.strip()
            if not t or "<silent/>" in t.lower().replace(" ", ""):
                continue
            if t not in found:
                new.append(t)
        speakers += len(new)
        if not new:
            quiet += 1
            if quiet >= patience:
                break      # converged
            continue       # give the stalled link another chance
        quiet = 0
        found.extend(new)

    merged = " | ".join(found)
    return Result(model, provider, "herd_rounds", idx, score(merged, sample["outputs"]), merged[:80],
                  n, cost, latency, 0, speakers, rounds=rounds,
                  calls=all_calls, thoughts=all_thoughts)


def main() -> int:
    global LISTENER_SYSTEM, MAX_TOKENS, REASONING_MAX
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--n", type=int, default=7, help="elephants in the herd arm")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--patience", type=int, default=1,
                    help="consecutive silent rounds tolerated before halting")
    ap.add_argument("--model", help="override listener from config")
    ap.add_argument("--provider", help="host slug to pin when --model is given")
    ap.add_argument("--prompt", choices=sorted(LISTENER_PROMPTS), default="plain",
                    help="listener wording: 'answer' (ask for the answer) or "
                         "'material' (ask for the lines). Recorded in the results.")
    ap.add_argument("--reasoning-max", type=int, default=None,
                    help="cap on thinking tokens alone. Recorded in the results.")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                    help="reply budget, reasoning included. Recorded in the results.")
    # choices, so an unknown name fails at once: an unrecognised arm used to fall
    # through to the iterated arm, which is slower and costs several times more.
    ap.add_argument("--arms", nargs="*", default=["single", "herd"],
                    choices=["single", "herd", "herd_rounds"])
    ap.add_argument("--out", type=Path, default=Path("runs/ruler-niah.jsonl"))
    args = ap.parse_args()

    LISTENER_SYSTEM = LISTENER_PROMPTS[args.prompt]
    MAX_TOKENS = args.max_tokens
    REASONING_MAX = args.reasoning_max

    cfg = config.load()
    model = args.model or cfg.listener
    provider = args.provider if args.model else cfg.listener_provider

    samples = [json.loads(l) for l in args.data.read_text().splitlines() if l.strip()][: args.limit]
    ctx_tokens = samples[0]["length"]
    print(f"{len(samples)} samples @ ~{ctx_tokens:,} tokens | model {model}")
    print(f"herd arm: {args.n} elephants, ~{ctx_tokens // args.n:,} tokens each\n")

    client = OpenRouterClient(session_id="ruler")
    results: list[Result] = []
    for i, s in enumerate(samples):
        for arm in args.arms:
            try:
                if arm == "single":
                    r = run_single(s, client, model, provider, i)
                elif arm == "herd":
                    r = run_herd(s, client, model, provider, i, args.n)
                else:
                    r = run_herd_rounds(s, client, model, provider, i, args.n,
                                        max_rounds=args.max_rounds,
                                        patience=args.patience)
            except Exception as exc:
                print(f"  [{i}] {arm:<6} FAILED {str(exc)[:60]}")
                continue
            results.append(r)
            print(f"  [{i}] {arm:<12} {'OK ' if r.correct else 'MISS'} "
                  f"${r.cost_usd:.4f} {r.latency_s:5.1f}s r={r.rounds} "
                  f"spoke={r.speakers}/{r.n_parts}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        # Header row: what was run, against what, when. Without it a stale results
        # file is indistinguishable from a fresh one.
        fh.write(json.dumps({
            "_run": {
                "model": model, "provider": provider, "n_parts": args.n,
                "prompt": args.prompt, "max_tokens": args.max_tokens,
                "reasoning_max": args.reasoning_max,
                "data": str(args.data), "context_tokens": ctx_tokens,
                "samples": len(samples), "arms": args.arms,
                "max_rounds": args.max_rounds, "patience": args.patience,
                "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        }) + "\n")
        for r in results:
            row = asdict(r)
            row.pop("thoughts")
            fh.write(json.dumps(row) + "\n")
    with args.out.with_suffix(".thoughts.jsonl").open("w") as fh:
        for r in results:
            for k, t in enumerate(r.thoughts):
                fh.write(json.dumps({"index": r.index, "arm": r.arm, "call": k, "thinking": t}) + "\n")

    print("\n" + "=" * 74)
    print(f"{'arm':<10}{'accuracy':>12}{'$/sample':>11}{'p50 s':>9}{'tokens':>12}")
    print("-" * 74)
    for arm in args.arms:
        rs = [r for r in results if r.arm == arm]
        if not rs:
            continue
        acc = sum(r.correct for r in rs) / len(rs)
        print(f"{arm:<10}{acc:>11.0%} {statistics.mean(r.cost_usd for r in rs):>10.4f}"
              f"{statistics.median(r.latency_s for r in rs):>9.1f}"
              f"{int(statistics.mean(r.prompt_tokens for r in rs)):>12,}")
    print("-" * 74)
    print(f"model: {model} @ {provider or 'unpinned'} | prompt: {args.prompt}")
    print(f"total spend: ${sum(r.cost_usd for r in results):.4f}   -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
