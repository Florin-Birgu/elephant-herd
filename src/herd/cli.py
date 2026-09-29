"""herd - command line control.

    herd on        wake the herd
    herd off       silence it (also drops any pending briefing)
    herd status    is it on, and is anything waiting
    herd ask "..." one-off question, prints the briefing (blocks ~90s)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from herd import state


def build_herd(cwd: str | Path | None = None):
    """Imported lazily - `herd on/off/status` must stay instant.

    Named herds attached to this directory win. The single configs/local.yaml is a
    fallback for setups that predate them.

    `cwd` is the directory whose attachments and conversations to use. The hook runs
    from the repo, so it passes the session's directory explicitly; without it the
    herd would be assembled from the wrong project.
    """
    from herd import config, herds
    from herd.agents import Elephant, Herd
    from herd.corpus import assemble, scan, split_to_cap

    here = Path(cwd) if cwd else Path.cwd()
    attached = herds.attached_to(here)
    if attached:
        return _build_from_herds(attached, here)

    cfg = config.load()
    elephants = []
    for name, folders in cfg.groups.items():
        docs = [d for f in folders for d in scan(cfg.vault, f)]
        parts = split_to_cap(docs, cfg.cap_tokens)
        for i, part in enumerate(parts):
            # Suffix only when a domain had to be split, so names stay meaningful.
            label = name if len(parts) == 1 else f"{name}-{chr(ord('a') + i)}"
            elephants.append(Elephant(
                name=label,
                corpus=assemble(part),
                model=cfg.listener,
                provider=cfg.listener_provider,
            ))
    # The conversation elephant holds recent sessions rather than files. It is what
    # lets the others resolve "the client" or "that strategy" to a real thing.
    context_elephant = None
    if cfg.raw.get("conversation", {}).get("enabled", True):
        from herd.conversation import CONVERSATION_SYSTEM, build_corpus

        convo = build_corpus(
            here, sessions=cfg.raw.get("conversation", {}).get("sessions", 3)
        )
        if convo.strip():
            context_elephant = Elephant(
                name="conversation",
                corpus=convo,
                model=cfg.listener,
                provider=cfg.listener_provider,
                system=CONVERSATION_SYSTEM,
            )

    return Herd(
        elephants,
        leader_model=cfg.herd_leader,
        leader_provider=cfg.herd_leader_provider,
        session_id=cfg.raw.get("herd", "herd"),
        context_elephant=context_elephant,
    )


def _build_from_herds(names: list[str], cwd: Path | None = None):
    """Assemble the union of every herd attached to this directory.

    A project can attach to more than one - its own docs plus shared reference
    material - and gets all of them. Names are prefixed when two herds would
    otherwise collide on a group name.
    """
    from herd import config, herds
    from herd.agents import Elephant, Herd
    from herd.corpus import assemble, scan, split_to_cap
    from herd.conversation import CONVERSATION_SYSTEM, build_corpus

    cfg = config.load_defaults()
    elephants = []
    for name in names:
        hd = herds.load(name)
        if hd is None:
            continue
        cap = int(hd.options.get("cap_tokens", 800_000))
        for group, folders in hd.groups.items():
            docs = [d for root in hd.roots for f in folders for d in scan(Path(root), f)]
            if not docs:
                continue
            label = group if len(names) == 1 else f"{name}:{group}"
            for i, part in enumerate(split_to_cap(docs, cap)):
                suffix = "" if i == 0 and len(split_to_cap(docs, cap)) == 1 else f"-{chr(ord('a') + i)}"
                elephants.append(Elephant(
                    name=f"{label}{suffix}",
                    corpus=assemble(part),
                    model=cfg.listener,
                    provider=cfg.listener_provider,
                ))

    context_elephant = None
    convo = build_corpus(cwd or Path.cwd(), sessions=3)
    if convo.strip():
        context_elephant = Elephant(
            name="conversation", corpus=convo, model=cfg.listener,
            provider=cfg.listener_provider, system=CONVERSATION_SYSTEM,
        )
    return Herd(
        elephants,
        leader_model=cfg.herd_leader,
        leader_provider=cfg.herd_leader_provider,
        session_id="-".join(names),
        context_elephant=context_elephant,
    )


def cmd_on() -> int:
    state.set_enabled(True)
    print(f"herd: ON for {Path.cwd()}")
    print("Other directories are unaffected. `herd off` here to stop.")
    return 0


def cmd_off() -> int:
    state.set_enabled(False)
    print(f"herd: OFF for {Path.cwd()}")
    return 0


def cmd_attach(name: str) -> int:
    from herd import herds

    if herds.load(name) is None:
        print(f"No herd called {name!r}. `herd herds` lists them.")
        return 1
    herds.attach(name)
    print(f"{Path.cwd()} -> {name}")
    print("Still off. `herd on` here when you want it running.")
    return 0


def cmd_detach() -> int:
    from herd import herds

    herds.detach()
    print(f"{Path.cwd()} detached")
    return 0


def cmd_scan(roots: list[str], as_json: bool) -> int:
    """Facts for whoever is deciding the partition. No API calls, no cost."""
    import json as _json

    from herd import herds

    report = herds.scan_report(roots)
    if as_json:
        print(_json.dumps(report, indent=2))
        return 0
    total = sum(v["tokens"] for v in report.values())
    print(f"{len(report)} folders, {total:,} est tokens, {herds.cost_line(total)}\n")
    for name, v in sorted(report.items(), key=lambda kv: -kv[1]["tokens"]):
        flag = "  needs splitting" if v["over_cap"] else ""
        print(f"  {name:<22}{v['files']:>5} files{v['tokens']:>10,} tk{flag}")
    return 0


def cmd_herds() -> int:
    from herd import herds

    defined = herds.all_herds()
    attached = herds.attached_to()
    if not defined:
        print("No herds yet. Run /herd init in a Claude session.")
        return 0
    for h in defined:
        sizes = herds.size_groups(h.roots, h.groups)
        total = sum(sizes.values())
        mark = " <- attached here" if h.name in attached else ""
        print(f"{h.name:<16}{len(h.groups):>3} elephants{total:>10,} tk   "
              f"{herds.cost_line(total)}{mark}")
    return 0


def cmd_status() -> int:
    from herd import config, herds, hooks

    on = state.is_enabled()
    print(f"herd    : {'ON' if on else 'OFF'} for {Path.cwd()}")
    others = [d for d in state.enabled_dirs() if d != str(Path.cwd())]
    if others:
        print(f"also on : {', '.join(others)}")

    attached = herds.attached_to()
    if attached:
        total = 0
        names = []
        for n in attached:
            hd = herds.load(n)
            if not hd:
                continue
            sizes = herds.size_groups(hd.roots, hd.groups)
            total += sum(sizes.values())
            names += list(hd.groups)
        print(f"herds   : {', '.join(attached)}")
        print(f"elephants: {len(names)} ({', '.join(names)})")
        print(f"cost    : {herds.cost_line(total)}")
    else:
        try:
            cfg = config.load()
            print(f"vault   : {cfg.vault}  (legacy config; run /herd init)")
            print(f"elephants: {len(cfg.groups)} ({', '.join(cfg.groups)})")
        except SystemExit:
            print("herds   : none attached here. Run /herd init")

    cfg = config.load_defaults()
    print(f"listener: {cfg.listener} @ {cfg.listener_provider}")
    print(f"hooks   : {'installed' if hooks.installed() else 'NOT installed - run: herd claude install'}")
    pending = state.PENDING_FILE.exists()
    print(f"pending : {'briefing waiting for next turn' if pending else 'none'}")
    return 0


def cmd_ask(question: str, rounds: int = 1) -> int:
    herd = build_herd()
    result = herd.hear(question, rounds=rounds)
    if result.gated:
        print(f"(herd not woken: {result.gate_reason})")
        return 0
    print(result.briefing or "(nothing relevant)")
    print(
        f"\n-- {len(result.contributions)} spoke, {len(result.silent)} silent"
        f" | ${result.cost_usd:.4f} | {result.latency_s:.0f}s",
        file=sys.stderr,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="herd", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    scan_p = sub.add_parser("scan", help="folder sizes and token counts (free, local)")
    scan_p.add_argument("roots", nargs="+")
    scan_p.add_argument("--json", action="store_true")
    sub.add_parser("herds", help="list named herds and what is attached here")
    claude = sub.add_parser("claude", help="Claude Code integration")
    claude.add_argument("action", choices=["install", "uninstall"])
    sub.add_parser("attach", help="attach this directory to a named herd").add_argument("name")
    sub.add_parser("detach", help="detach this directory")
    hook = sub.add_parser("hook", help="internal: invoked by Claude Code")
    hook.add_argument("event", nargs="+")
    sub.add_parser("on")
    sub.add_parser("off")
    sub.add_parser("status")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--deep", action="store_true",
                     help="blackboard mode: re-broadcast findings so elephants can "
                          "extend each other. Needed for facts chained across "
                          "domains; costs one extra pass per round.")
    ask.add_argument("--rounds", type=int, default=None, help="explicit round count")
    args = ap.parse_args()

    from herd import hooks

    return {
        "scan": lambda: cmd_scan(args.roots, args.json),
        "herds": cmd_herds,
        "claude": lambda: hooks.install() if args.action == "install" else hooks.uninstall(),
        "attach": lambda: cmd_attach(args.name),
        "detach": cmd_detach,
        "hook": lambda: hooks.main(args.event),
        "on": cmd_on,
        "off": cmd_off,
        "status": cmd_status,
        "ask": lambda: cmd_ask(args.question,
                               args.rounds or (4 if args.deep else 1)),
    }[args.cmd]()


if __name__ == "__main__":
    sys.exit(main())
