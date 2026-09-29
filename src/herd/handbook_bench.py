"""Handbook test: the real herd, on a real company's notes, graded by a judge.

Eight elephants, one per handbook section, each holding its section as a cached prefix
through the same Elephant class the product uses, with the product's own listener
prompt. Every question (and every overheard conversation) goes to all eight; whatever
they contribute is joined and graded twice:

- text match: every entry of answer_parts must appear (case-insensitive substring)
- judge: a model from a third family reads the question, accepted answers, source
  quotes and the reply, and says right or wrong. Text matching cannot read "not";
  the judge can. Disagreements are listed for a human to look at.

No leader: it would add a frontier model to every turn and grade its summary rather
than what the elephants found.

    python -m herd.handbook_bench --model deepseek/deepseek-v4.1-flash --provider wafer
    python -m herd.handbook_bench --model z-ai/glm-5.3-flash --provider z-ai/fp8 --limit 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from herd.agents import Elephant
from herd.corpus import assemble, scan
from herd.providers.openrouter import OpenRouterClient

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "bench" / "handbook"
COMMIT = "7a2e264cb798305fd5c92b586ef9fe633a730244"
ROOT = REPO / f"corpora/handbook/handbook-{COMMIT}-content-handbook/content/handbook"
SECTIONS = ["product", "finance", "total-rewards", "leadership", "hiring",
            "communication", "values", "ceo"]

JUDGE_MODEL = "openai/gpt-5.4"
# Version 2. Version 1 read the joined reply as one voice, so a specialist writing "my
# part does not mention this" next to another specialist's correct answer was scored
# as a self-contradiction, and so was true extra detail. 14 of 116 DeepSeek replies and
# 21 of 116 GLM replies disagreed with text matching, almost all for that reason.
JUDGE_VERSION = 2
JUDGE_SYSTEM = """You grade one reply from a knowledge system against a known answer.
The reply is several specialists' notes joined together, each tagged [section]. You
get the question, the accepted answers, the source text the answer comes from, and
the reply. Decide whether the reply, taken as a whole, gives a correct answer.

Rules:
- Correct if every required part is stated by at least one specialist, in any wording.
- A specialist saying its own material does not contain something, or adding side
  information, is not a contradiction and does not make the reply wrong.
- Extra detail is fine, including true exceptions and conditions.
- Wrong if a required part is stated by no specialist, or if the reply gives a
  different value for a required part and never gives the correct one.
- Negation matters: "not all team members" is not "all team members".
- For a question marked NO ANSWER EXPECTED: correct if the reply is empty, says there
  is nothing or it does not know, or gives only information that does not answer the
  question. Wrong if it states an answer to the question.
- For a conversation marked NOTHING RELEVANT: correct only if the reply is empty.

Reply with JSON only: {"correct": true or false, "reason": "one short sentence"}"""


def load(name: str) -> list[dict]:
    return [json.loads(l) for l in (BENCH / name).read_text().splitlines() if l.strip()]


def text_match(reply: str, parts: list[list[str]]) -> bool:
    low = reply.lower()
    return all(any(a.lower() in low for a in part) for part in parts)


def judge(client: OpenRouterClient, item: dict, reply: str) -> dict:
    kind = item.get("type") or ("overhearing" if "conversation" in item else "")
    prompt = item.get("question") or f"(overheard conversation)\n{item['conversation']}"
    if kind == "no_answer":
        expected = "NO ANSWER EXPECTED"
    elif "conversation" in item and not item.get("relevant"):
        expected = "NOTHING RELEVANT"
    else:
        expected = json.dumps(item["answer_parts"])
    sources = "\n".join(f"- {s['quote']}" for s in item.get("sources", [])) or "(none)"
    msg = (f"QUESTION:\n{prompt}\n\nACCEPTED ANSWERS (every part required):\n{expected}\n\n"
           f"SOURCE TEXT:\n{sources}\n\nREPLY:\n{reply or '(empty - the system said nothing)'}")
    r = client.chat(model=JUDGE_MODEL, max_tokens=4000, reasoning_max_tokens=1000,
                    messages=[{"role": "system", "content": JUDGE_SYSTEM},
                              {"role": "user", "content": msg}])
    m = re.search(r"\{.*\}", r.text, re.S)
    try:
        out = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        out = {}
    return {"correct": bool(out.get("correct")), "reason": out.get("reason", r.text[:200]),
            "cost_usd": r.usage.cost_usd}


def _listen(e, turn, client, max_tokens):
    """One elephant's call. A host error is recorded as that elephant saying nothing,
    rather than ending the run: one failed call out of eight is data, not a crash."""
    try:
        return e.listen(turn, client, max_tokens=max_tokens)
    except Exception as exc:  # noqa: BLE001
        from herd.providers.openrouter import Response, Usage
        print(f"    {e.name}: call failed ({str(exc)[:60]})", flush=True)
        return None, Response(text="", usage=Usage(finish_reason="error"))


def rejudge(path: Path) -> int:
    """Regrade saved replies with the current judge. The herd is not called again."""
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    head, rows = lines[0], lines[1:]
    items = {i["id"]: i for i in load("questions.jsonl") + load("overhearing_final.jsonl")}
    client = OpenRouterClient(session_id="hb-rejudge")
    with ThreadPoolExecutor(8) as pool:
        verdicts = list(pool.map(lambda r: judge(client, items[r["id"]], r["reply"]), rows))
    for r, j in zip(rows, verdicts):
        r.update(judge=j["correct"], judge_reason=j["reason"], agree=r["text_match"] == j["correct"])
    head["_run"].update(judge_version=JUDGE_VERSION)
    path.write_text("".join(json.dumps(x) + "\n" for x in [head] + rows))
    print(f"{'type':<14}{'n':>4}{'judge':>8}{'text':>8}")
    for t in sorted({r["type"] for r in rows}):
        rs = [r for r in rows if r["type"] == t]
        print(f"{t:<14}{len(rs):>4}{sum(r['judge'] for r in rs):>8}{sum(r['text_match'] for r in rs):>8}")
    print(f"total: judge {sum(r['judge'] for r in rows)}/{len(rows)}, "
          f"disagreements {sum(not r['agree'] for r in rows)}, "
          f"regrade ${sum(j['cost_usd'] for j in verdicts):.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N questions only")
    ap.add_argument("--no-overhearing", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=60000)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rejudge", type=Path, default=None,
                    help="regrade a saved run with the current judge; no herd calls")
    args = ap.parse_args()
    if args.rejudge:
        return rejudge(args.rejudge)

    elephants = []
    for s in SECTIONS:
        docs = scan(ROOT, s)
        elephants.append(Elephant(name=s, corpus=assemble(docs), model=args.model,
                                  provider=args.provider))
    tag = args.model.replace("/", "-").replace(".", "-")
    # One client per elephant: session_id keeps each cached prefix on one host.
    clients = {e.name: OpenRouterClient(session_id=f"hb-{tag}-{e.name}") for e in elephants}
    judge_client = OpenRouterClient(session_id="hb-judge")

    items = load("questions.jsonl")[: args.limit]
    if not args.no_overhearing and args.limit is None:
        items += load("overhearing_final.jsonl")
    out = args.out or REPO / f"runs/handbook-{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(items)} items x {len(elephants)} elephants | {args.model} @ {args.provider}")

    rows = []
    for n, item in enumerate(items, 1):
        turn = item.get("question") or item["conversation"]
        t0 = time.monotonic()
        with ThreadPoolExecutor(len(elephants)) as pool:
            res = list(pool.map(lambda e: _listen(e, turn, clients[e.name], args.max_tokens),
                                elephants))
        spoke, calls = {}, []
        for e, (contrib, resp) in zip(elephants, res):
            u = resp.usage
            calls.append({"elephant": e.name, "cost_usd": u.cost_usd,
                          "prompt_tokens": u.prompt_tokens, "cached_tokens": u.cached_tokens,
                          "reasoning_tokens": u.reasoning_tokens,
                          "reply_tokens": max(u.completion_tokens - u.reasoning_tokens, 0),
                          "finish": u.finish_reason, "latency_s": round(u.latency_s, 1),
                          "spoke": bool(contrib and contrib.text and not contrib.truncated),
                          "truncated": bool(contrib and contrib.truncated)})
            if contrib and contrib.text and not contrib.truncated:
                spoke[e.name] = contrib.text
        reply = "\n".join(f"[{k}] {v}" for k, v in spoke.items())

        no_answer = item.get("type") == "no_answer" or ("conversation" in item and not item.get("relevant"))
        tm = (not spoke) if no_answer else text_match(reply, item["answer_parts"])
        j = judge(judge_client, item, reply)
        want = set(item.get("sections", [])) | set(item.get("also_ok_sections", []))
        row = {"id": item["id"], "type": item.get("type", "overhearing"),
               "difficulty": item.get("difficulty"), "sections": item.get("sections", []),
               "spoke": sorted(spoke), "right_speakers": sorted(set(spoke) & want),
               "wrong_speakers": sorted(set(spoke) - want),
               "text_match": tm, "judge": j["correct"], "judge_reason": j["reason"],
               "agree": tm == j["correct"], "reply": reply,
               "cost_usd": sum(c["cost_usd"] for c in calls), "judge_cost_usd": j["cost_usd"],
               "latency_s": round(time.monotonic() - t0, 1), "calls": calls}
        rows.append(row)
        cache = sum(c["cached_tokens"] for c in calls) / max(sum(c["prompt_tokens"] for c in calls), 1)
        print(f"  [{n:>3}] {item['id']:<8} {row['type']:<12} judge={'OK ' if j['correct'] else 'MISS'} "
              f"text={'OK ' if tm else 'MISS'} spoke={len(spoke)} ${row['cost_usd']:.4f} "
              f"cache={cache:.0%} {row['latency_s']:.0f}s", flush=True)

    with out.open("w") as fh:
        fh.write(json.dumps({"_run": {
            "model": args.model, "provider": args.provider, "judge": JUDGE_MODEL, "judge_version": JUDGE_VERSION,
            "prompt": "agents.LISTENER_SYSTEM (product)", "max_tokens": args.max_tokens,
            "corpus_commit": COMMIT, "sections": SECTIONS, "items": len(rows),
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}}) + "\n")
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print(f"\n{'type':<14}{'n':>4}{'judge':>8}{'text':>8}")
    for t in sorted({r['type'] for r in rows}):
        rs = [r for r in rows if r["type"] == t]
        print(f"{t:<14}{len(rs):>4}{sum(r['judge'] for r in rs):>8}{sum(r['text_match'] for r in rs):>8}")
    print(f"total: judge {sum(r['judge'] for r in rows)}/{len(rows)}, "
          f"disagreements {sum(not r['agree'] for r in rows)}, "
          f"herd ${sum(r['cost_usd'] for r in rows):.3f} + judge ${sum(r['judge_cost_usd'] for r in rows):.3f}"
          f" -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
