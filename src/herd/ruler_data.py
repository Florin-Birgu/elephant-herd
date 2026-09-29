"""Fetch RULER task data, so a run can be reproduced rather than remembered.

The first round of RULER runs read files from /tmp, which a reboot deleted. The
numbers survived in the results files; the inputs that produced them did not. This
module makes the data a named, versioned download instead of a local accident.

Source: the RULER tasks prepared at 4k..1M by `self-long/RULER-llama3-1M` on the
Hugging Face hub, mirroring NVIDIA's RULER (Apache 2.0) synthetic tasks. Nothing is
redistributed here - the file is fetched on demand and cached under corpora/, which
is gitignored.

    python -m herd.ruler_data niah_single_2 --length 128k
    python -m herd.ruler_data vt --length 128k --limit 10
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "corpora" / "ruler"
DATASET = "self-long/RULER-llama3-1M"
URL = "https://huggingface.co/datasets/{ds}/resolve/main/{cfg}/validation-00000-of-00001.parquet"

LENGTHS = ("4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1M")
TASKS = (
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue",
    "vt", "cwe", "fwe", "qa_1", "qa_2",
)


def fetch(task: str, length: str = "128k", limit: int | None = None) -> Path:
    """Download one task at one length, cache it as jsonl, return the path.

    The cached file is what `herd.ruler --data` consumes: one JSON object per line
    with `input`, `outputs` and `length`, which is RULER's own shape.
    """
    if task not in TASKS:
        raise SystemExit(f"unknown task {task!r}. One of: {', '.join(TASKS)}")
    if length not in LENGTHS:
        raise SystemExit(f"unknown length {length!r}. One of: {', '.join(LENGTHS)}")

    out = CACHE / f"{task}_{length}.jsonl"
    if out.is_file() and out.stat().st_size:
        return out

    import pandas as pd

    url = URL.format(ds=DATASET, cfg=f"{task}_{length}")
    print(f"fetching {task} @ {length} ...", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=600) as resp:
        blob = resp.read()
    frame = pd.read_parquet(io.BytesIO(blob))
    if limit:
        frame = frame.head(limit)

    CACHE.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in frame.to_dict("records"):
            # The hub copy calls the gold column "answers"; RULER's own runner calls
            # it "outputs". Accept either, or the run scores every sample as a miss.
            answers = next((row[k] for k in ("outputs", "answers", "answer") if k in row), None)
            if hasattr(answers, "tolist"):
                answers = answers.tolist()
            elif isinstance(answers, str):
                answers = [answers]
            fh.write(json.dumps({
                "input": row["input"],
                "outputs": list(answers or []),
                "length": int(row.get("length") or len(str(row["input"])) // 4),
            }) + "\n")
    print(f"{out}  ({len(frame)} samples)", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", choices=TASKS)
    ap.add_argument("--length", default="128k", choices=LENGTHS)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    print(fetch(args.task, args.length, args.limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
