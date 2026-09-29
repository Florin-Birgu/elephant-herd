"""The conversation elephant: a herd member whose corpus is the conversation itself.

Domain elephants know facts. None of them knows what "the client" refers to, or
what "give me more detail" points back at. That knowledge lives in the conversation,
so the conversation becomes an elephant like any other and answers those questions
through the same mechanism.

This works because of one property: conversations only grow at the end. Appending a
turn leaves the cached prefix intact, so keeping this elephant warm costs a cache read
plus the newest turns, no matter how long the session runs. A corpus edited in the
middle would invalidate on every turn and be unaffordable.

Sidechain entries (subagent traffic) are excluded. They are voluminous, they are not
what the user said, and including them would bury the actual conversation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"

# Long tool results and pasted files dominate a transcript by volume and carry almost
# none of the conversational meaning that reference-resolution needs.
MAX_BLOCK_CHARS = 2000


@dataclass
class Turn:
    role: str
    text: str
    ts: str


def _text_from(message: dict) -> str:
    """Flatten a message's content, dropping tool noise and thinking blocks."""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        # thinking is not what was said; tool_use/tool_result is machinery
        if block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(p.strip() for p in parts if p.strip())[:MAX_BLOCK_CHARS]


def read_transcript(path: Path, max_turns: int | None = None) -> list[Turn]:
    """Parse one Claude Code transcript into user/assistant turns, oldest first."""
    turns: list[Turn] = []
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return turns
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("type") not in ("user", "assistant"):
            continue
        if row.get("isSidechain") or row.get("isMeta"):
            continue
        message = row.get("message")
        if not isinstance(message, dict):
            continue
        text = _text_from(message)
        if not text:
            continue
        turns.append(Turn(message.get("role", row["type"]), text, row.get("timestamp", "")))
    return turns[-max_turns:] if max_turns else turns


def project_dir(cwd: str | Path) -> Path:
    """Claude Code encodes the working directory into the transcript folder name.

    Both separators and dots become dashes, so /Users/jane.doe/Projects/x becomes
    -Users-jane-doe-Projects-x.
    """
    encoded = str(cwd).replace("/", "-").replace(".", "-")
    direct = CLAUDE_PROJECTS / encoded
    if direct.is_dir():
        return direct
    # Fall back to a suffix match, so a slightly different encoding still resolves.
    tail = Path(cwd).name.replace(".", "-")
    for d in CLAUDE_PROJECTS.glob(f"*-{tail}"):
        if d.is_dir():
            return d
    return direct


def session_files(cwd: str | Path, limit: int = 5) -> list[Path]:
    """Most recent transcripts for a project, newest last so ordering is append-like."""
    d = project_dir(cwd)
    if not d.is_dir():
        return []
    files = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-limit:]


def build_corpus(cwd: str | Path, sessions: int = 3, max_turns_per: int = 400) -> str:
    """Assemble recent conversations into one cacheable, append-ordered blob.

    Oldest session first so that new material lands at the end and the prefix stays
    stable: the same reason document order is fixed for file corpora.
    """
    chunks = []
    for path in session_files(cwd, limit=sessions):
        turns = read_transcript(path, max_turns=max_turns_per)
        if not turns:
            continue
        body = "\n\n".join(f"[{t.role}] {t.text}" for t in turns)
        chunks.append(f'<session id="{path.stem[:8]}" turns="{len(turns)}">\n{body}\n</session>')
    return "\n\n".join(chunks)


CONVERSATION_SYSTEM = """You hold the recent conversations of the person being helped.

The other specialists know facts but do not know what this person is talking about.
Your job is to tell them, so their knowledge can be applied to the right thing.

Answer only what the conversation establishes:
- which project, product or client is being discussed
- what an ambiguous reference points to ("it", "that strategy", "the client")
- what has already been decided or ruled out, so nobody repeats it

Be brief and concrete. Name things.

If the current turn is unambiguous on its own and nothing in the history changes how it
should be read, reply with exactly: <silent/>

Otherwise:
<contribution>
<source path="conversation"/>
<text>What this turn refers to, in one or two sentences. Name the project and the
specific thing being referenced.</text>
</contribution>"""
