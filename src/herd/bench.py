"""Listener model comparison.

Which cheap model actually finds a buried fact in a resident corpus? Published
benchmarks cannot answer this - there are no NoLiMa-style (low-lexical-overlap)
scores for any of these models - so we measure on real gold questions.

Design notes learned the hard way:

- **Generous max_tokens.** Reasoning length varies enormously for identical input
  (observed 1,608 and 15,784 output tokens for the same model, corpus and question).
  A model cut off mid-reasoning emits nothing and is indistinguishable from one that
  chose silence, so a tight cap silently converts recall failures into fake silences.
- **Cold then warm.** The first call per model writes the cache; later calls read it.
  Reporting one blended number hides an order-of-magnitude difference.
- **Pin the host.** The same model is served at quantizations from fp4 to bf16;
  unpinned routing varies recall, not just price.

    python -m herd.bench --trials 1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path

from herd import config
from herd.agents import Elephant
from herd.corpus import assemble, scan
from herd.providers.openrouter import OpenRouterClient

CORPUS_FOLDER_KEY = 0  # index into cfg.groups; the corpus holding the gold answers

# (model, first-party-or-cheapest host slug)
CANDIDATES = [
    ("qwen/qwen3.7-flash", "alibaba"),
    ("z-ai/glm-5.3-flash", "z-ai/fp8"),
    ("google/gemini-2.5-flash-lite", "google-ai-studio"),
    ("google/gemini-3.7-flash", "google-ai-studio"),
    ("deepseek/deepseek-v4-flash-0731", "fireworks"),
    # Never actually ran in the first sweep. DeepSeek's own host is the cheapest
    # ($0.003/M cache read) but is dropped by the account's privacy guardrail, so
    # this pins the cheapest surviving host with a stated quantization. Still 3.6x
    # below GLM, so a hit here changes the cost line for the whole herd.
    ("deepseek/deepseek-v4.1-flash", "deepinfra/fp8"),
    ("xiaomi/mimo-v2.5", "xiaomi/fp8"),
    ("openai/gpt-4.1-nano", "openai"),
    ("qwen/qwen3.8-27b", "alibaba"),      # dense 27B - does size fix low-overlap recall?
    ("qwen/qwen3.8-flash", "alibaba"),
    # Frontier tier. The cost argument assumes a cheap listener is adequate; until
    # these are measured on the same question that is an assumption, not a result.
    ("anthropic/claude-sonnet-5", "anthropic"),
    ("anthropic/claude-opus-5", "anthropic"),
]

# Human-authored, answers known and verified in the corpus by hand.
@dataclass
class GoldQuestion:
    id: str
    question: str
    must_contain: list[str]  # any one of these counts as correct
    kind: str


def load_gold(cfg) -> list[GoldQuestion]:
    """Gold questions live in a gitignored JSONL - they quote private material."""
    if not cfg.gold_path or not cfg.gold_path.is_file():
        raise SystemExit(
            f"No gold questions at {cfg.gold_path}. Write a few questions you know "
            "the answer to, one JSON object per line: "
            '{"id":..,"question":..,"keywords":[..],"tag":..}'
        )
    out = []
    for line in cfg.gold_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(GoldQuestion(d["id"], d["question"], d["keywords"], d.get("tag", "")))
    return out

MAX_TOKENS = 24_000  # deliberately generous; truncation would fake a silence


@dataclass
class Row:
    model: str
    provider: str
    question: str
    trial: int
    correct: bool
    silent: bool
    truncated: bool
    cost_usd: float
    latency_s: float
    prompt_tokens: int
    output_tokens: int
    cache_hit_ratio: float
    excerpt: str


def run(models: list[tuple[str, str]], trials: int, out_path: Path,
        gold: list[GoldQuestion]) -> list[Row]:
    cfg = config.load()
    folder_name, folders = list(cfg.groups.items())[CORPUS_FOLDER_KEY]
    corpus = assemble([d for f in folders for d in scan(cfg.vault, f)])
    print(f"corpus: {folder_name}, ~{len(corpus)//3.58:,.0f} tokens")
    print(f"{len(models)} models x {len(gold)} questions x {trials} trial(s)\n")

    rows: list[Row] = []
    for model, provider in models:
        elephant = Elephant(name="cmp", corpus=corpus, model=model, provider=provider)
        # One client per model: session_id keeps its cache on one host.
        client = OpenRouterClient(session_id=f"bench-{model.replace('/', '-')}")
        print(f"--- {model} @ {provider}")
        for q in gold:
            for trial in range(trials):
                try:
                    contrib, resp = elephant.listen(q.question, client, max_tokens=MAX_TOKENS)
                except Exception as exc:
                    print(f"    {q.id:<14} FAILED {str(exc)[:60]}")
                    continue
                text = (contrib.text if contrib else "") or ""
                correct = any(k.lower() in text.lower() for k in q.must_contain)
                truncated = bool(contrib and contrib.truncated) or resp.usage.truncated
                row = Row(
                    model, provider, q.id, trial, correct,
                    silent=contrib is None, truncated=truncated,
                    cost_usd=resp.usage.cost_usd, latency_s=resp.usage.latency_s,
                    prompt_tokens=resp.usage.prompt_tokens,
                    output_tokens=resp.usage.completion_tokens,
                    cache_hit_ratio=resp.usage.cache_hit_ratio,
                    excerpt=text[:200].replace("\n", " "),
                )
                rows.append(row)
                mark = "OK  " if correct else ("TRUNC" if truncated else ("silent" if contrib is None else "miss"))
                print(
                    f"    {q.id:<14} {mark:<6} ${resp.usage.cost_usd:.4f} "
                    f"{resp.usage.latency_s:6.1f}s out={resp.usage.completion_tokens:>6,} "
                    f"cache={resp.usage.cache_hit_ratio:.0%}"
                )
        print(f"    spent on this model: ${client.total_cost_usd:.4f}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        # Header row: the corpus a gold question was scored against changes as notes
        # are written, so a result is only meaningful next to the size and date of
        # the corpus that produced it.
        fh.write(json.dumps({"_run": {
            "corpus_group": folder_name,
            "corpus_chars": len(corpus),
            "trials": trials,
            "questions": [q.id for q in gold],
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }}) + "\n")
        for r in rows:
            fh.write(json.dumps(asdict(r)) + "\n")
    return rows


def summarize(rows: list[Row]) -> None:
    print("=" * 96)
    print(f"{'model':<34}{'correct':>9}{'trunc':>7}{'silent':>8}{'warm $':>9}{'cold $':>9}{'p50 s':>8}")
    print("-" * 96)
    by_model: dict[str, list[Row]] = {}
    for r in rows:
        by_model.setdefault(r.model, []).append(r)
    for model, rs in sorted(by_model.items(), key=lambda kv: -sum(r.correct for r in kv[1])):
        warm = [r.cost_usd for r in rs if r.cache_hit_ratio > 0.5]
        cold = [r.cost_usd for r in rs if r.cache_hit_ratio <= 0.5]
        print(
            f"{model:<34}{sum(r.correct for r in rs):>4}/{len(rs):<4}"
            f"{sum(r.truncated for r in rs):>7}{sum(r.silent for r in rs):>8}"
            f"{(statistics.median(warm) if warm else 0):>9.4f}"
            f"{(statistics.median(cold) if cold else 0):>9.4f}"
            f"{statistics.median([r.latency_s for r in rs]):>8.1f}"
        )
    print("-" * 96)
    print(f"total spend: ${sum(r.cost_usd for r in rows):.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--models", nargs="*", help="substring filter on model id")
    ap.add_argument("--only", nargs="*", help="run only these gold question ids")
    ap.add_argument("--out", type=Path, default=Path("runs/model-comparison.jsonl"))
    args = ap.parse_args()

    models = CANDIDATES
    if args.models:
        models = [m for m in CANDIDATES if any(f in m[0] for f in args.models)]
    gold = load_gold(config.load())
    if args.only:
        gold = [q for q in gold if q.id in args.only]
    rows = run(models, args.trials, args.out, gold)
    summarize(rows)
    print(f"rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
