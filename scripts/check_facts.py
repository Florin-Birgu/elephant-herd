"""Check that every fact a reader wrote down is really in the handbook, word for word.

Readers are models, and a model can paraphrase a quote or invent one. A question built
on an invented quote has no right answer, so every quote is checked against its file
before any question is written from it.

    python3 scripts/check_facts.py            # all files in bench/handbook/facts/
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "corpora/handbook/handbook-7a2e264cb798305fd5c92b586ef9fe633a730244-content-handbook/content/handbook"
FACTS = REPO / "bench/handbook/facts"


def norm(text: str) -> str:
    # Whitespace and markdown emphasis vary between what a reader copies and the raw
    # file; neither changes the fact, so both are ignored when comparing.
    text = re.sub(r"[*_`]", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> int:
    total = bad = 0
    for path in sorted(FACTS.glob("*.jsonl")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            total += 1
            try:
                fact = json.loads(line)
                source = ROOT / fact["file"]
                found = source.is_file() and norm(fact["quote"]) in norm(source.read_text())
            except (json.JSONDecodeError, KeyError) as exc:
                found, fact = False, {"id": f"{path.name}:{n}", "file": f"unparseable ({exc})"}
            if not found:
                bad += 1
                print(f"NOT FOUND  {fact.get('id')}  {fact.get('file')}")
    print(f"{total - bad}/{total} quotes found verbatim")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
