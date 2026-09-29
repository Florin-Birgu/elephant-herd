"""Named herds, and which directory uses which.

A herd is a view over folders, not a folder itself: it can pull the project you are
working in, the reference material two directories away, and last year's notes from
somewhere else entirely. So herds are defined once, globally, and directories attach
to them.

Attachment is per-directory for a reason that is about quality as much as cost. A
global herd covering everything you have ever written wakes fourteen specialists to
get two useful contributions, and "which elephant should have known this?" stops
having an answer once every elephant is just a slice of everything. A project-scoped
herd of three or four is both cheaper and sharper.

Layout:
    ~/.herd/herds/<name>.yaml     one herd definition
    ~/.herd/attachments.json      directory -> [herd names]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

HERD_DIR = Path.home() / ".herd"
HERDS_DIR = HERD_DIR / "herds"
ATTACHMENTS = HERD_DIR / "attachments.json"

# DeepSeek-V4.1-Flash on DeepInfra. Warm is measured (handbook test: $0.0037 for a
# ~806K-token turn, thinking included); cold is the uncached input price.
CACHE_READ_PER_M = 0.0046
CACHE_WRITE_PER_M = 0.14


@dataclass
class HerdDef:
    name: str
    roots: list[str]
    groups: dict[str, list[str]]
    options: dict

    @property
    def path(self) -> Path:
        return HERDS_DIR / f"{self.name}.yaml"

    def save(self) -> None:
        HERDS_DIR.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            yaml.safe_dump(
                {"roots": self.roots, "groups": self.groups, **self.options},
                sort_keys=False,
                width=100,
            )
        )


def load(name: str) -> HerdDef | None:
    p = HERDS_DIR / f"{name}.yaml"
    if not p.is_file():
        return None
    data = yaml.safe_load(p.read_text()) or {}
    groups = data.pop("groups", {}) or {}
    roots = data.pop("roots", []) or []
    return HerdDef(name, roots, groups, data)


def all_herds() -> list[HerdDef]:
    if not HERDS_DIR.is_dir():
        return []
    return [h for h in (load(p.stem) for p in sorted(HERDS_DIR.glob("*.yaml"))) if h]


# --- attachments -------------------------------------------------------------


def _read_attachments() -> dict:
    try:
        return json.loads(ATTACHMENTS.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def attached_to(cwd: str | Path | None = None) -> list[str]:
    """Herds for this directory, inheriting from an attached parent."""
    here = Path(cwd or Path.cwd()).resolve()
    names: list[str] = []
    for path, herds in _read_attachments().items():
        p = Path(path)
        if here == p or p in here.parents:
            names += [h for h in herds if h not in names]
    return names


def attach(name: str, cwd: str | Path | None = None) -> None:
    HERD_DIR.mkdir(parents=True, exist_ok=True)
    here = str(Path(cwd or Path.cwd()).resolve())
    data = _read_attachments()
    current = data.setdefault(here, [])
    if name not in current:
        current.append(name)
    ATTACHMENTS.write_text(json.dumps(data, indent=2))


def detach(name: str | None = None, cwd: str | Path | None = None) -> None:
    here = str(Path(cwd or Path.cwd()).resolve())
    data = _read_attachments()
    if here not in data:
        return
    data[here] = [] if name is None else [h for h in data[here] if h != name]
    if not data[here]:
        del data[here]
    ATTACHMENTS.write_text(json.dumps(data, indent=2))


# --- sizing ------------------------------------------------------------------


def size_groups(roots: list[str], groups: dict[str, list[str]]) -> dict[str, int]:
    """Estimated tokens per group. Local only, no API calls."""
    from herd.corpus import scan

    sizes = {}
    for name, folders in groups.items():
        total = 0
        for root in roots:
            for folder in folders:
                total += sum(d.est_tokens for d in scan(Path(root), folder))
        sizes[name] = total
    return sizes


def cost_line(total_tokens: int) -> str:
    warm = total_tokens / 1e6 * CACHE_READ_PER_M
    cold = total_tokens / 1e6 * CACHE_WRITE_PER_M
    return f"~${warm:.4f}/turn warm, ${cold:.2f} to warm up"


def propose_groups(root: Path, cap_tokens: int = 800_000,
                   group_below: int = 40_000) -> dict[str, list[str]]:
    """Suggest one group per top-level folder, bundling the small ones.

    Grouping follows the folders someone already made, because those encode a
    judgement about what belongs together that no clustering will recover. Small
    leftovers are bundled rather than each becoming an elephant that never speaks.
    """
    from herd.corpus import scan

    root = root.expanduser()
    sizes: dict[str, int] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        tokens = sum(d.est_tokens for d in scan(root, child.name))
        if tokens:
            sizes[child.name] = tokens

    groups: dict[str, list[str]] = {}
    small: list[str] = []
    for folder, tokens in sorted(sizes.items(), key=lambda kv: -kv[1]):
        if tokens < group_below:
            small.append(folder)
        else:
            groups[folder.lower().replace(" ", "-")] = [folder]
    if small:
        groups["misc"] = small
    return groups


def scan_report(roots: list[str], samples: int = 6) -> dict:
    """Folder-by-folder facts: file counts, token counts, a few filenames.

    Deliberately dumb. Deciding which folders belong together is a judgement call
    that needs to see content and be argued with, so it lives in the skill; this
    only supplies the numbers that judgement needs. Local, free, no API.
    """
    from herd.corpus import scan

    out: dict[str, dict] = {}
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            docs = scan(root, child.name)
            if not docs:
                continue
            tokens = sum(d.est_tokens for d in docs)
            biggest = sorted(docs, key=lambda d: -d.est_tokens)[:samples]
            out[child.name] = {
                "root": str(root),
                "files": len(docs),
                "tokens": tokens,
                "over_cap": tokens > 800_000,
                "sample": [Path(d.path).name for d in biggest],
            }
    return out
