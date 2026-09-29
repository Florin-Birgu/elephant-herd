"""Configuration loading.

Everything corpus-specific lives in `configs/local.yaml`, which is gitignored.
Folder names alone reveal a person's clients, projects and interests, so no real
path, domain name or gold question may appear in tracked source. The repo ships
`configs/example.yaml` with placeholders; the real file never leaves the machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CONFIG = REPO_ROOT / "configs" / "local.yaml"
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "example.yaml"


@dataclass
class Config:
    vault: Path
    groups: dict[str, list[str]]           # elephant name -> folders it holds
    listener: str
    listener_provider: str | None
    herd_leader: str
    herd_leader_provider: str | None
    judge: str
    judge_provider: str | None
    cap_tokens: int
    ceiling_usd: float
    gold_path: Path | None = None          # gitignored gold questions, if any
    raw: dict = field(default_factory=dict, repr=False)


def load(path: Path | None = None) -> Config:
    """Load configs/local.yaml, or the path given."""
    cfg_path = path or LOCAL_CONFIG
    if not cfg_path.is_file():
        raise SystemExit(
            f"No config at {cfg_path}.\n"
            f"Copy {EXAMPLE_CONFIG.relative_to(REPO_ROOT)} to "
            f"{cfg_path.relative_to(REPO_ROOT)} and point it at your own folders.\n"
            "It is gitignored - your paths and domain names stay local."
        )
    data = yaml.safe_load(cfg_path.read_text()) or {}
    models = data.get("models", {})
    partition = data.get("partition", {})
    budget = data.get("budget", {})

    roots = data.get("roots") or []
    if not roots:
        raise SystemExit(f"{cfg_path}: 'roots' is empty - nothing to load.")

    gold = data.get("gold_questions")
    return Config(
        vault=Path(roots[0]).expanduser(),
        groups=data.get("groups") or {},
        listener=models.get("listener", "deepseek/deepseek-v4.1-flash"),
        listener_provider=models.get("listener_provider", "deepinfra/fp8"),
        herd_leader=models.get("herd_leader", "anthropic/claude-sonnet-5"),
        herd_leader_provider=models.get("herd_leader_provider", "anthropic"),
        judge=models.get("judge", "google/gemini-2.5-flash-lite"),
        judge_provider=models.get("judge_provider", "google-ai-studio"),
        cap_tokens=int(partition.get("cap_tokens", 800_000)),
        ceiling_usd=float(budget.get("ceiling_usd", 15)),
        gold_path=(REPO_ROOT / gold) if gold else None,
        raw=data,
    )


def load_defaults() -> Config:
    """Model choices without requiring a corpus config.

    Named herds carry their own folders, so the only thing still needed from global
    config is which models to use. Falls back to the measured defaults when there is
    no local.yaml at all.
    """
    try:
        return load()
    except SystemExit:
        return Config(
            vault=Path.home(), groups={},
            listener="deepseek/deepseek-v4.1-flash", listener_provider="deepinfra/fp8",
            herd_leader="anthropic/claude-sonnet-5", herd_leader_provider="anthropic",
            judge="google/gemini-2.5-flash-lite", judge_provider="google-ai-studio",
            cap_tokens=800_000, ceiling_usd=15.0,
        )
