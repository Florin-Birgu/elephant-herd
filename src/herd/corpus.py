"""Corpus scanning and assembly.

An elephant's corpus becomes a cached prefix, so assembly must be byte-identical
across runs or every cache hit is lost. That means: deterministic file ordering
(byte-order sort, not locale), no timestamps in the output, and stable document
ids so gold labels stay valid between runs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

TEXT_EXTENSIONS = {".md", ".markdown", ".txt"}
SKIP_DIRS = {"assets", "node_modules", ".git", "__pycache__", ".venv"}

# Measured against provider-reported prompt_tokens on real markdown: 1,892,926 chars
# came back as 528,373 tokens. Used only for pre-flight estimates and cap decisions;
# actual accounting always comes from the API response. Single source of truth - other
# modules import this rather than keeping their own copy, because two constants for one
# quantity means every size estimate is wrong by whichever one you happened to ask.
CHARS_PER_TOKEN = 3.58


@dataclass
class Doc:
    doc_id: str
    path: str  # relative to root
    text: str
    sha256: str

    @property
    def est_tokens(self) -> int:
        return int(len(self.text) / CHARS_PER_TOKEN)


def scan(root: Path, subdir: str | None = None) -> list[Doc]:
    """All text documents under root (optionally one subdir), deterministically ordered."""
    root = root.expanduser().resolve()
    base = root / subdir if subdir else root

    paths: list[Path] = []
    for p in base.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if any(part in SKIP_DIRS or part.startswith(".") for part in p.relative_to(root).parts):
            continue
        paths.append(p)

    # Byte-order sort on the relative path: stable across machines and locales.
    paths.sort(key=lambda p: str(p.relative_to(root)).encode())

    docs: list[Doc] = []
    for i, p in enumerate(paths):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not text.strip():
            continue
        rel = str(p.relative_to(root))
        docs.append(
            Doc(
                doc_id=f"D{i:05d}",
                path=rel,
                text=text,
                sha256=hashlib.sha256(text.encode()).hexdigest()[:16],
            )
        )
    return docs


def assemble(docs: list[Doc]) -> str:
    """Documents into one cacheable blob.

    Each document is delimited and labelled so a listener can cite precisely;
    the id and path are what a contribution must point back to.
    """
    parts = [
        f'<doc id="{d.doc_id}" path="{d.path}">\n{d.text.rstrip()}\n</doc>' for d in docs
    ]
    return "\n\n".join(parts)


def summarize(docs: list[Doc]) -> str:
    total = sum(d.est_tokens for d in docs)
    return f"{len(docs)} docs, ~{total:,} est tokens"


def split_to_cap(docs: list[Doc], cap_tokens: int) -> list[list[Doc]]:
    """Split one domain into as few partitions as fit under the cap.

    Only called when a domain genuinely exceeds one context window. Splitting is by
    the same byte-order path sort used everywhere else, so partitions are stable
    across runs and a given document always lands in the same one - required, since
    each partition is a separate cached prefix.

    Grouping stays semantic wherever possible: a domain is split only under duress,
    never for convenience, because "which elephant should have known this?" is only
    answerable when a partition corresponds to something nameable.
    """
    total = sum(d.est_tokens for d in docs)
    if total <= cap_tokens or not docs:
        return [docs]
    # Greedy fill, not equal-size targeting. An earlier version split at total/N and
    # stopped opening new parts once the planned count was reached, so with uneven
    # document sizes everything left over piled into the last partition and pushed it
    # over the cap - a partition larger than the context window, which is not a
    # partition. A test written for the open-source push caught it.
    parts: list[list[Doc]] = [[]]
    running = 0
    for doc in docs:
        if parts[-1] and running + doc.est_tokens > cap_tokens:
            parts.append([])
            running = 0
        parts[-1].append(doc)
        running += doc.est_tokens
    return [p for p in parts if p]
