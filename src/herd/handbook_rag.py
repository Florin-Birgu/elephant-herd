"""RAG baseline for the handbook test: search, then answer from what the search found.

Same eight sections, same questions and overheard conversations, same answering model
and the same judge as herd.handbook_bench. The only difference is how the model gets
the text: here a search picks the top-k chunks for each turn, instead of every section
being held whole. So a difference in score is a difference in retrieval versus holding.

Two retrievers, so the baseline is not a straw man:
- embed: OpenAI text-embedding-3-large, cosine similarity (the usual RAG)
- bm25:  keyword search, the old-school baseline that is often hard to beat

    python -m herd.handbook_rag --retriever embed
    python -m herd.handbook_rag --retriever bm25 --limit 5
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from herd.corpus import scan
from herd.handbook_bench import (BENCH, COMMIT, JUDGE_MODEL, JUDGE_VERSION, REPO, ROOT,
                                 SECTIONS, judge, load, text_match)
from herd.providers.openrouter import OpenRouterClient, load_api_key

EMBED_MODEL = "openai/text-embedding-3-large"
CHUNK_TOKENS = 400        # ~1,400 characters: a common RAG chunk size
OVERLAP_TOKENS = 50
CHARS_PER_TOKEN = 3.58
CACHE = REPO / "corpora" / "handbook"

# Mirrors the herd's job description: report what bears on the turn, or stay silent.
# The herd listener prompt asks the same of each elephant over its whole section.
ANSWER_SYSTEM = """You are given excerpts from a company handbook, found by searching for
the turn below. The turn is either a question or an overheard conversation.

Report anything in the excerpts that answers the question, or that the people talking
should know. Be specific: names, numbers, rules. Cite the source path of each excerpt
you use.

If nothing in the excerpts is relevant, reply with exactly: <silent/>"""


# --- chunking -----------------------------------------------------------------


def chunks() -> list[dict]:
    """Split every document of the eight sections into overlapping chunks.

    Each chunk carries its file path and nearest heading, so the answering model can
    cite it and the run can check whether the gold source was retrieved.
    """
    size = int(CHUNK_TOKENS * CHARS_PER_TOKEN)
    step = size - int(OVERLAP_TOKENS * CHARS_PER_TOKEN)
    out = []
    for section in SECTIONS:
        for doc in scan(ROOT, section):
            text = doc.text
            for start in range(0, max(len(text), 1), step):
                piece = text[start:start + size]
                if not piece.strip():
                    continue
                heads = re.findall(r"^#+ .*$", text[:start + 1], re.M)
                out.append({"file": doc.path, "section": section,
                            "heading": heads[-1] if heads else "",
                            "text": piece})
                if start + size >= len(text):
                    break
    return out


# --- retrievers ---------------------------------------------------------------


def _embed(texts: list[str], key: str) -> tuple[np.ndarray, float]:
    import urllib.request

    vecs, cost = [], 0.0
    for i in range(0, len(texts), 96):
        body = json.dumps({"model": EMBED_MODEL, "input": texts[i:i + 96]}).encode()
        req = urllib.request.Request("https://openrouter.ai/api/v1/embeddings", data=body,
                                     headers={"Authorization": f"Bearer {key}",
                                              "Content-Type": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    out = json.loads(r.read())
                break
            except Exception:  # noqa: BLE001 - retry transient host errors
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt * 2)
        vecs += [d["embedding"] for d in out["data"]]
        cost += float((out.get("usage") or {}).get("cost", 0) or 0)
    m = np.array(vecs, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True), cost


class EmbedRetriever:
    def __init__(self, docs: list[dict], key: str) -> None:
        self.key = key
        cache = CACHE / f"rag-embed-{COMMIT[:8]}-{len(docs)}.npy"
        self.cost = 0.0
        if cache.is_file():
            self.m = np.load(cache)
        else:
            self.m, self.cost = _embed([f"{d['file']}\n{d['heading']}\n{d['text']}" for d in docs], key)
            np.save(cache, self.m)

    def search(self, query: str, k: int) -> tuple[list[int], float]:
        q, cost = _embed([query], self.key)
        return list(np.argsort(-(self.m @ q[0]))[:k]), cost


class BM25Retriever:
    """Okapi BM25, k1=1.5, b=0.75, lowercase word tokens."""

    def __init__(self, docs: list[dict]) -> None:
        self.toks = [self._tok(f"{d['file']} {d['heading']} {d['text']}") for d in docs]
        self.tf = [Counter(t) for t in self.toks]
        self.avg = sum(map(len, self.toks)) / len(self.toks)
        df = Counter(w for t in self.toks for w in set(t))
        n = len(self.toks)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.cost = 0.0

    @staticmethod
    def _tok(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def search(self, query: str, k: int) -> tuple[list[int], float]:
        q = [w for w in self._tok(query) if w in self.idf]
        scores = np.zeros(len(self.toks))
        for i, (tf, toks) in enumerate(zip(self.tf, self.toks)):
            norm = 1.5 * (1 - 0.75 + 0.75 * len(toks) / self.avg)
            scores[i] = sum(self.idf[w] * tf[w] * 2.5 / (tf[w] + norm) for w in q if w in tf)
        return list(np.argsort(-scores)[:k]), 0.0


# --- run ----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retriever", choices=["embed", "bm25"], required=True)
    ap.add_argument("--k", type=int, default=10, help="chunks retrieved per turn")
    ap.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    ap.add_argument("--provider", default="deepinfra/fp8")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    docs = chunks()
    key = load_api_key()
    ret = EmbedRetriever(docs, key) if args.retriever == "embed" else BM25Retriever(docs)
    print(f"{len(docs)} chunks from {len(SECTIONS)} sections | retriever {args.retriever} "
          f"k={args.k} | {args.model} @ {args.provider} | index cost ${ret.cost:.3f}")

    items = load("questions.jsonl") + load("overhearing_final.jsonl")
    items = items[: args.limit] if args.limit else items
    client = OpenRouterClient(session_id=f"rag-{args.retriever}")
    judge_client = OpenRouterClient(session_id="rag-judge")
    tag = args.model.replace("/", "-").replace(".", "-")
    out = args.out or REPO / f"runs/handbook-rag-{args.retriever}-{tag}.jsonl"
    rows = []
    for n, item in enumerate(items, 1):
        turn = item.get("question") or item["conversation"]
        idx, search_cost = ret.search(turn, args.k)
        found = [docs[i] for i in idx]
        excerpts = "\n\n".join(f'<excerpt source="{d["file"]}">\n{d["text"]}\n</excerpt>' for d in found)
        r = client.chat(model=args.model, provider=args.provider, max_tokens=60000,
                        messages=[{"role": "system", "content": ANSWER_SYSTEM},
                                  {"role": "user", "content": f"{excerpts}\n\n<turn>{turn}</turn>"}])
        reply = "" if "<silent/>" in r.text.replace(" ", "") else r.text.strip()
        no_answer = item.get("type") == "no_answer" or ("conversation" in item and not item.get("relevant"))
        tm = (not reply) if no_answer else text_match(reply, item["answer_parts"])
        j = judge(judge_client, item, reply)
        gold = {s["file"] for s in item.get("sources", [])}
        got = {d["file"] for d in found}
        row = {"id": item["id"], "type": item.get("type", "overhearing"),
               "difficulty": item.get("difficulty"), "sections": item.get("sections", []),
               "retrieved_files": sorted(got),
               "gold_retrieved": sorted(gold & got), "gold_missed": sorted(gold - got),
               "text_match": tm, "judge": j["correct"], "judge_reason": j["reason"],
               "agree": tm == j["correct"], "reply": reply,
               "cost_usd": r.usage.cost_usd + search_cost, "judge_cost_usd": j["cost_usd"],
               "prompt_tokens": r.usage.prompt_tokens, "reasoning_tokens": r.usage.reasoning_tokens,
               "finish": r.usage.finish_reason}
        rows.append(row)
        print(f"  [{n:>3}] {item['id']:<8} {row['type']:<12} judge={'OK ' if j['correct'] else 'MISS'} "
              f"text={'OK ' if tm else 'MISS'} gold-found={len(gold & got)}/{len(gold)} "
              f"${row['cost_usd']:.4f}", flush=True)

    with out.open("w") as fh:
        fh.write(json.dumps({"_run": {
            "arm": f"rag-{args.retriever}", "k": args.k, "chunk_tokens": CHUNK_TOKENS,
            "overlap_tokens": OVERLAP_TOKENS, "embed_model": EMBED_MODEL if args.retriever == "embed" else None,
            "model": args.model, "provider": args.provider, "judge": JUDGE_MODEL,
            "judge_version": JUDGE_VERSION, "corpus_commit": COMMIT, "sections": SECTIONS,
            "chunks": len(docs), "index_cost_usd": ret.cost, "items": len(rows),
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}}) + "\n")
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    print(f"\n{'type':<14}{'n':>4}{'judge':>8}{'text':>8}")
    for t in sorted({r['type'] for r in rows}):
        rs = [r for r in rows if r["type"] == t]
        print(f"{t:<14}{len(rs):>4}{sum(r['judge'] for r in rs):>8}{sum(r['text_match'] for r in rs):>8}")
    print(f"total: judge {sum(r['judge'] for r in rows)}/{len(rows)}, disagreements "
          f"{sum(not r['agree'] for r in rows)}, answer ${sum(r['cost_usd'] for r in rows):.3f} "
          f"+ index ${ret.cost:.3f} + judge ${sum(r['judge_cost_usd'] for r in rows):.3f} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
