"""Gold question generation from a private corpus.

The trap this module exists to avoid: if you generate a question from a document's
text, the question inherits that document's vocabulary, retrieval finds it trivially,
and the benchmark measures nothing. That is why NoLiMa exists.

Mitigations here:

1. The generator is explicitly instructed to avoid the document's distinctive
   wording - to ask the way a colleague would, from memory, days later.
2. Lexical overlap between question and source is *measured* and recorded, so the
   overlap distribution can be reported rather than asserted. Low-overlap questions
   are tagged `low_overlap` and are the ones that discriminate.
3. A validation pass (a second call, different model) checks the question is
   answerable from the source and NOT from general knowledge alone.

Questions are frozen to JSONL once generated. Regenerating after seeing results
would be tuning on the test set.

    python -m herd.questions --per-elephant 5 --out bench/questions/vault-gold.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from herd import config
from herd.corpus import Doc, scan
from herd.providers.openrouter import OpenRouterClient

# Models come from configs/local.yaml; these are only fallbacks.
DEFAULT_GEN = ("z-ai/glm-5.3-flash", "z-ai/fp8")


GEN_PROMPT = """Below is one document from someone's private work notes.

Write ONE question that this document answers, as the note's own author would ask it
months later, from memory - vague about wording, certain about the fact they want.

Rules that matter:
- Do NOT reuse the document's distinctive nouns, product names, or phrasing in the
  question. If the document names a specific vendor, do not name it in the question -
  describe it instead ("the tool we switched to for invoicing").
  The question must be findable by meaning, not by keyword match.
- The answer must be a specific fact IN this document: a name, number, price, date,
  decision. Not a summary, not an opinion.
- It must NOT be answerable from general knowledge. Someone without this document
  must be unable to answer it.

Reply as JSON only:
{"question": "...", "answer": "...", "keywords": ["...", "..."]}

`keywords` are 1-3 distinctive strings that MUST appear in any correct answer (a
name, number, or proper noun). They are used for exact scoring, so choose tokens that
cannot appear by luck."""

VALIDATE_PROMPT = """You are checking a benchmark question for defects.

Answer as JSON only:
{"answerable_from_source": true/false,
 "answerable_from_general_knowledge": true/false,
 "answer_is_specific": true/false,
 "verdict": "keep" or "drop",
 "why": "one short sentence"}

Drop it if the answer is not actually stated in the source, if a well-read person
could answer it without the source, or if the answer is vague/subjective."""

STOP = set(
    "the a an and or of to in for on with is are was were be been it its this that "
    "what which who whom how why when where do does did our we you i they them their "
    "use uses used using".split()
)


def lexical_overlap(question: str, source: str) -> float:
    """Share of the question's content words that appear in the source.

    Low overlap = the question cannot be answered by keyword matching = the case
    where retrieval struggles and residency might help.
    """
    qs = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in STOP and len(w) > 2}
    if not qs:
        return 0.0
    src = set(re.findall(r"[a-z0-9]+", source.lower()))
    return len(qs & src) / len(qs)


@dataclass
class GoldItem:
    id: str
    elephant: str
    question: str
    answer: str
    keywords: list[str]
    gold_doc_id: str
    gold_path: str
    lexical_overlap: float
    tag: str
    validation: dict


def _json_from(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def generate(docs: list[Doc], elephant: str, n: int, client: OpenRouterClient,
             rng: random.Random) -> list[GoldItem]:
    # Prefer substantial documents: a 200-token stub rarely holds a specific fact.
    pool = [d for d in docs if 1500 < len(d.text) < 60_000]
    rng.shuffle(pool)
    items: list[GoldItem] = []
    for doc in pool:
        if len(items) >= n:
            break
        try:
            resp = client.chat(
                model=DEFAULT_GEN[0], provider=DEFAULT_GEN[1], max_tokens=2000,
                messages=[
                    {"role": "system", "content": GEN_PROMPT},
                    {"role": "user", "content": f"<document path=\"{doc.path}\">\n{doc.text[:40000]}\n</document>"},
                ],
            )
        except Exception as exc:
            print(f"    gen failed on {doc.path[:50]}: {str(exc)[:60]}", file=sys.stderr)
            continue
        data = _json_from(resp.text)
        if not data or not data.get("question") or not data.get("keywords"):
            continue

        overlap = lexical_overlap(data["question"], doc.text)
        item = GoldItem(
            id=f"{elephant}-{len(items):02d}",
            elephant=elephant,
            question=data["question"].strip(),
            answer=str(data.get("answer", "")).strip(),
            keywords=[str(k) for k in data["keywords"]][:3],
            gold_doc_id=doc.doc_id,
            gold_path=doc.path,
            lexical_overlap=round(overlap, 3),
            tag="low_overlap" if overlap < 0.5 else "normal",
            validation={},
        )
        items.append(item)
        print(f"    [{item.tag:<11} ov={overlap:.2f}] {item.question[:66]}")
    return items


def validate(items: list[GoldItem], client: OpenRouterClient, docs_by_id: dict[str, Doc],
             validate_model: str, validate_provider: str | None) -> None:
    """Second pass, different model family - marks each item keep/drop in place."""
    for it in items:
        doc = docs_by_id.get(it.gold_doc_id)
        if not doc:
            continue
        try:
            resp = client.chat(
                model=validate_model, provider=validate_provider, max_tokens=800,
                messages=[
                    {"role": "system", "content": VALIDATE_PROMPT},
                    {"role": "user", "content":
                        f"QUESTION: {it.question}\nCLAIMED ANSWER: {it.answer}\n\n"
                        f"<source>\n{doc.text[:40000]}\n</source>"},
                ],
            )
        except Exception:
            continue
        it.validation = _json_from(resp.text) or {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-elephant", type=int, default=5)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", type=Path, default=Path("bench/questions/vault-gold.jsonl"))
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    cfg = config.load()
    rng = random.Random(args.seed)
    client = OpenRouterClient(session_id="qgen")
    all_items: list[GoldItem] = []
    docs_by_id: dict[str, Doc] = {}

    for elephant, folders in cfg.groups.items():
        docs = [d for f in folders for d in scan(cfg.vault, f)]
        docs_by_id.update({d.doc_id: d for d in docs})
        print(f"\n{elephant} ({len(docs)} docs)")
        all_items += generate(docs, elephant, args.per_elephant, client, rng)

    if not args.no_validate:
        print(f"\nvalidating {len(all_items)} questions with {cfg.judge}...")
        validate(all_items, client, docs_by_id, cfg.judge, cfg.judge_provider)

    kept = [i for i in all_items if i.validation.get("verdict", "keep") == "keep"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for it in kept:
            fh.write(json.dumps(asdict(it)) + "\n")

    low = sum(1 for i in kept if i.tag == "low_overlap")
    print(f"\ngenerated {len(all_items)}, kept {len(kept)} ({len(all_items)-len(kept)} dropped)")
    print(f"  low_overlap: {low}   normal: {len(kept)-low}")
    print(f"  spend: ${client.total_cost_usd:.4f}")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
