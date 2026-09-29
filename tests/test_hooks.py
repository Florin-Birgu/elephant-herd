"""Smoke tests for the money path: install, uninstall, on/off, and which herd runs.

The pure-math tests never touched any of this, which is how a marker matching nothing
shipped: hooks installed twice (every turn billed twice), `herd status` reporting "not
installed" while two copies ran, and uninstall removing nothing. Every test here runs
the real functions against a temporary settings file and a temporary herd directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import herd.cli as cli
from herd import herds, hooks, state


@pytest.fixture
def home(tmp_path: Path, monkeypatch) -> Path:
    """Redirect every file the hooks touch into a temp directory."""
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"hooks": {}}, indent=2))

    herd_dir = tmp_path / ".herd"
    monkeypatch.setattr(hooks, "SETTINGS", settings)
    monkeypatch.setattr(hooks, "SKILLS_DIR", tmp_path / ".claude" / "skills" / "herd")
    monkeypatch.setattr(state, "HERD_DIR", herd_dir)
    monkeypatch.setattr(state, "STATE_FILE", herd_dir / "state.json")
    monkeypatch.setattr(state, "PENDING_FILE", herd_dir / "pending.md")
    monkeypatch.setattr(herds, "HERD_DIR", herd_dir)
    monkeypatch.setattr(herds, "HERDS_DIR", herd_dir / "herds")
    monkeypatch.setattr(herds, "ATTACHMENTS", herd_dir / "attachments.json")
    return tmp_path


def _entries(event: str) -> list[dict]:
    return json.loads(hooks.SETTINGS.read_text()).get("hooks", {}).get(event) or []


def _seed(event: str, command: str) -> None:
    data = json.loads(hooks.SETTINGS.read_text())
    data.setdefault("hooks", {})[event] = [
        {"hooks": [{"type": "command", "command": command}]}
    ]
    hooks.SETTINGS.write_text(json.dumps(data, indent=2))


FOREIGN = {"hooks": [{"type": "command", "command": "echo not-ours"}]}


# --- install / uninstall -----------------------------------------------------


def test_install_writes_one_entry_per_event(home):
    hooks.install()
    assert len(_entries("UserPromptSubmit")) == 1
    assert len(_entries("Stop")) == 1


def test_install_is_idempotent(home):
    """The live bug: a second install appended a second pair, doubling every bill."""
    hooks.install()
    hooks.install()
    assert len(_entries("UserPromptSubmit")) == 1
    assert len(_entries("Stop")) == 1


def test_install_does_not_duplicate_an_entry_written_before_the_marker(home):
    _seed("Stop", '"/opt/bin/herd" hook stop')
    hooks.install()
    assert len(_entries("Stop")) == 1


def test_uninstall_removes_entries_written_before_the_marker(home):
    _seed("Stop", '"/opt/bin/herd" hook stop')
    _seed("UserPromptSubmit", '"/opt/bin/herd" hook prompt')
    hooks.uninstall()
    assert _entries("Stop") == []
    assert _entries("UserPromptSubmit") == []


def test_foreign_hooks_survive_install_and_uninstall(home):
    _seed("Stop", "echo not-ours")
    hooks.install()
    hooks.uninstall()
    assert _entries("Stop") == [FOREIGN]


def test_status_check_tells_the_truth(home):
    """It reported "NOT installed" while two copies ran, inviting a third."""
    assert not hooks.installed()
    hooks.install()
    assert hooks.installed()
    hooks.uninstall()
    assert not hooks.installed()


# --- on / off ----------------------------------------------------------------


def test_off_clears_a_pending_briefing(home):
    state.set_enabled(True, home)
    state.put_pending("facts from an earlier turn")
    state.set_enabled(False, home)
    assert state.take_pending() is None


def test_state_write_drops_the_legacy_global_toggle(home):
    state.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state.STATE_FILE.write_text(json.dumps({"enabled": True, "enabled_dirs": []}))
    state.set_enabled(True, home)
    assert "enabled" not in json.loads(state.STATE_FILE.read_text())


# --- the background run ------------------------------------------------------


def _transcript(tmp_path: Path) -> Path:
    p = tmp_path / "session.jsonl"
    p.write_text(
        json.dumps({"type": "user", "message": {"role": "user", "content": "what did we decide about the discount"}})
        + "\n"
        + json.dumps({"type": "assistant", "message": {"role": "assistant", "content": "It is 29"}})
        + "\n"
    )
    return p


class _FakeHerd:
    def __init__(self, seen: dict) -> None:
        self.seen = seen

    def hear(self, turn: str, **kwargs):
        self.seen["turn"] = turn
        return type("R", (), {"briefing": None})()


def test_background_run_aborts_when_the_herd_was_turned_off(home, monkeypatch):
    """A run already in flight must stop when the herd is turned off, or the off
    switch costs one more turn every time it is used."""
    built: list = []
    monkeypatch.setattr(cli, "build_herd", lambda *a, **k: built.append(a))
    assert hooks.run_background(str(_transcript(home)), str(home)) == 0
    assert built == []


def test_background_run_uses_the_session_directory(home, monkeypatch):
    """The hook runs from the repo, so it must be told which directory the
    conversation is in, or it assembles the wrong project's herd."""
    state.set_enabled(True, home)
    seen: dict = {}
    monkeypatch.setattr(cli, "build_herd", lambda cwd=None: _FakeHerd(seen))
    assert hooks.run_background(str(_transcript(home)), str(home)) == 0
    assert seen["turn"].startswith("what did we decide about the discount")


# --- which herd a directory gets ---------------------------------------------


def test_attachment_follows_the_directory(home):
    one, two = home / "one", home / "two"
    one.mkdir()
    two.mkdir()
    herds.attach("alpha", one)
    herds.attach("beta", two)
    assert herds.attached_to(one) == ["alpha"]
    assert herds.attached_to(two) == ["beta"]
