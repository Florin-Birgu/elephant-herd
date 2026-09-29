"""Claude Code integration.

Two hook points, and the split between them is the whole design:

- `Stop` fires after the assistant answers. That is when the herd runs, because the
  assistant's reply is the disambiguation. "What did we charge them?" is ambiguous;
  the reply naming the project is not. The herd works while the user is reading, and
  writes its briefing to disk.
- `UserPromptSubmit` fires on the next message. It prints whatever the previous turn
  prepared and returns immediately.

So the briefing is always one turn behind, which is what makes a 90-second herd usable
in a conversation: nobody ever waits for it.

Both hooks exit silently and successfully when the herd is off, not configured, or
broken. A knowledge system that blocks the conversation it is trying to help is worse
than no knowledge system.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from herd import state

SETTINGS = Path.home() / ".claude" / "settings.json"
REPO = Path(__file__).resolve().parents[2]
HERD_BIN = REPO / "bin" / "herd"
SKILLS_DIR = Path.home() / ".claude" / "skills" / "herd"

# Written as a trailing shell comment on every command we install, and matched by
# install, uninstall and status. It has to appear in the command itself: an earlier
# version matched the string "herd hook", which the command `"herd" hook stop` does
# not contain, so every install appended another copy and uninstall removed nothing.
MARKER = "elephant-herd"


def _skill_source() -> Path | None:
    """Locate SKILL.md in a checkout or an installed package."""
    for candidate in (Path(__file__).parent / "skill" / "SKILL.md",
                      REPO / "skill" / "SKILL.md"):
        if candidate.is_file():
            return candidate
    return None


def _is_ours(item: dict) -> bool:
    """Is this hook entry one of ours?

    Recognises entries written before the marker existed too, so an upgrade can clean
    up after the version that duplicated them.
    """
    if MARKER in json.dumps(item):
        return True
    command = " ".join(h.get("command", "") for h in item.get("hooks", []))
    return "herd" in command and (" hook stop" in command or " hook prompt" in command)


def installed() -> bool:
    """Are our hooks in settings? `herd status` asks this, and it must not lie.

    Checking for a substring in the raw file is what made it lie before: the file
    stores the command with its quotes escaped, so the literal never matched.
    """
    try:
        data = json.loads(SETTINGS.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    hooks = data.get("hooks") or {}
    return any(
        _is_ours(i) for event in ("UserPromptSubmit", "Stop") for i in hooks.get(event) or []
    )


# --- hook entry points -------------------------------------------------------


def hook_prompt() -> int:
    """UserPromptSubmit: print the briefing prepared last turn. Must be instant."""
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        payload = {}
    if not state.is_enabled(payload.get("cwd")):
        return 0
    briefing = state.take_pending()
    if briefing:
        print(briefing)
    return 0


def hook_stop() -> int:
    """Stop: run the herd on what was just discussed, for the next turn.

    Returns immediately. The work happens in a detached child, so the assistant is
    never blocked and the user never waits.
    """
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return 0
    if not state.is_enabled(payload.get("cwd")):
        return 0

    transcript = payload.get("transcript_path")
    if not transcript or not Path(transcript).is_file():
        return 0

    # The session's own directory travels with the job. This process runs from the
    # repo, so without it the herd would be assembled from the repo's attachments
    # rather than the ones belonging to the directory the conversation is in.
    cwd = str(payload.get("cwd") or "")

    # Detached, output discarded: this must not hold the conversation open.
    subprocess.Popen(
        [sys.executable, "-m", "herd.hooks", "_background", transcript, cwd],
        cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return 0


def run_background(transcript_path: str, cwd: str | None = None) -> int:
    """Ask the herd about the last exchange and leave the briefing for next turn.

    Re-checks that the herd is still on. Turning it off while a run is already in
    flight must stop that run, otherwise the off switch costs one more turn every
    time it is used, and leaves a briefing nobody will ever be shown.
    """
    if not state.is_enabled(cwd):
        return 0

    from herd.cli import build_herd
    from herd.conversation import read_transcript

    turns = read_transcript(Path(transcript_path), max_turns=6)
    if not turns:
        return 0
    last_user = next((t for t in reversed(turns) if t.role == "user"), None)
    if not last_user:
        return 0
    last_assistant = next((t for t in reversed(turns) if t.role == "assistant"), None)

    # The assistant's reply is what resolves the reference, so include it.
    turn = last_user.text
    if last_assistant:
        turn = f"{turn}\n\n<assistant_replied>{last_assistant.text[:1500]}</assistant_replied>"

    try:
        result = build_herd(cwd).hear(turn)
    except Exception:
        return 0     # never leave a broken state behind
    if result.briefing:
        state.put_pending(result.briefing)
    return 0


# --- install / uninstall -----------------------------------------------------


def _herd_command() -> str:
    """The `herd` executable, whether installed on PATH or run from a checkout."""
    found = shutil.which("herd")
    return found if found else str(HERD_BIN)


def _entry(subcommand: str) -> dict:
    # The trailing comment is the marker: the shell ignores it, and install,
    # uninstall and status all recognise the entry by it.
    return {"hooks": [{"type": "command",
                       "command": f'"{_herd_command()}" hook {subcommand} # {MARKER}'}]}


def install() -> int:
    """Append our two hooks to Claude Code settings, leaving existing ones alone."""
    if not SETTINGS.is_file():
        print(f"No {SETTINGS}. Run Claude Code once first.")
        return 1
    backup = SETTINGS.with_suffix(".json.herd-backup")
    shutil.copy2(SETTINGS, backup)

    data = json.loads(SETTINGS.read_text())
    hooks = data.setdefault("hooks", {})
    added = []
    for event, sub in (("UserPromptSubmit", "prompt"), ("Stop", "stop")):
        existing = hooks.setdefault(event, [])
        if any(_is_ours(item) for item in existing):
            continue
        existing.append(_entry(sub))
        added.append(event)

    SETTINGS.write_text(json.dumps(data, indent=2) + "\n")

    # The skill is what makes /herd:init possible. Grouping needs to see content and
    # be argued with, which a CLI cannot do, so that work lives here.
    src = _skill_source()
    skill_note = "SKILL.md not found; /herd:init will be unavailable"
    if src:
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, SKILLS_DIR / "SKILL.md")
        skill_note = f"skill  -> {SKILLS_DIR / 'SKILL.md'}"

    print(f"backup: {backup}")
    print(f"installed: {', '.join(added) if added else 'hooks already present'}")
    print(skill_note)
    print("\nInstalled but OFF everywhere. Nothing runs and nothing is billed until")
    print("you `herd on` inside a specific project directory.")
    print("Restart Claude Code first. The first briefing arrives on your second")
    print("message in that project, not your first.")
    return 0


def uninstall() -> int:
    skill = SKILLS_DIR / "SKILL.md"
    if skill.is_file():
        skill.unlink()
        try:
            SKILLS_DIR.rmdir()
        except OSError:
            pass
        print(f"removed {skill}")
    if not SETTINGS.is_file():
        return 0
    data = json.loads(SETTINGS.read_text())
    hooks = data.get("hooks", {})
    removed = 0
    for event in ("UserPromptSubmit", "Stop"):
        before = hooks.get(event, [])
        after = [i for i in before if not _is_ours(i)]
        removed += len(before) - len(after)
        if after:
            hooks[event] = after
        elif event in hooks:
            del hooks[event]
    SETTINGS.write_text(json.dumps(data, indent=2) + "\n")
    print(f"removed {removed} hook entr{'y' if removed == 1 else 'ies'}")
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    cmd = argv[0]
    if cmd == "prompt":
        return hook_prompt()
    if cmd == "stop":
        return hook_stop()
    if cmd == "_background":
        return run_background(argv[1], argv[2] if len(argv) > 2 and argv[2] else None)
    print(f"unknown hook command: {cmd}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
