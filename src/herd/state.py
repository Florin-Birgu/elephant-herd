"""On/off state and the briefing handoff file.

Two tiny pieces of state, both in ~/.herd/:

- `state.json`  - is the herd enabled? Checked by the hook on every turn, so it must
  be cheap to read and safe to read when missing or corrupt.
- `pending.md`  - the briefing prepared during the *previous* turn, waiting to be
  injected on this one. The herd takes ~90s to answer, far too slow to block a
  conversation, so it always answers one turn behind.
"""

from __future__ import annotations

import json
from pathlib import Path

HERD_DIR = Path.home() / ".herd"
STATE_FILE = HERD_DIR / "state.json"
PENDING_FILE = HERD_DIR / "pending.md"


def _read() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        # Missing or corrupt state must never break a turn - default to off.
        return {}


def is_enabled(cwd: str | Path | None = None) -> bool:
    """Is the herd on for this directory?

    Per-directory, never global. A herd costs real money per turn, and most projects
    have nothing to do with the vault you pointed it at. Turning it on everywhere at
    once would bill you for every unrelated session.

    A directory inherits from an enabled parent, so switching it on at a project root
    covers its subdirectories.
    """
    enabled = _read().get("enabled_dirs") or []
    if not enabled:
        return False
    here = Path(cwd or Path.cwd()).resolve()
    return any(here == Path(d) or Path(d) in here.parents for d in enabled)


def set_enabled(enabled: bool, cwd: str | Path | None = None) -> None:
    HERD_DIR.mkdir(parents=True, exist_ok=True)
    here = str(Path(cwd or Path.cwd()).resolve())
    state = _read()
    dirs = set(state.get("enabled_dirs") or [])
    if enabled:
        dirs.add(here)
    else:
        dirs.discard(here)
    state["enabled_dirs"] = sorted(dirs)
    # Drop the global toggle an earlier version used. Nothing reads it now, but a
    # key that says "enabled" in a file that means the opposite is a trap.
    state.pop("enabled", None)
    STATE_FILE.write_text(json.dumps(state, indent=2))
    if not enabled:
        clear_pending()


def enabled_dirs() -> list[str]:
    return list(_read().get("enabled_dirs") or [])


def take_pending() -> str | None:
    """Read and consume the briefing left by the previous turn."""
    try:
        text = PENDING_FILE.read_text().strip()
    except OSError:
        return None
    clear_pending()
    return text or None


def put_pending(briefing: str) -> None:
    HERD_DIR.mkdir(parents=True, exist_ok=True)
    PENDING_FILE.write_text(briefing)


def clear_pending() -> None:
    PENDING_FILE.unlink(missing_ok=True)
