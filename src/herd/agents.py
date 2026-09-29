"""Elephants, the turn gate, and the herd leader.

The listener code here is what the product will wrap in a daemon; the benchmark
imports the same objects, so measured numbers come from the shipped path.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from herd.providers.openrouter import OpenRouterClient, Response

# Chosen by measurement. On the handbook test (GitLab's public handbook, 8 sections,
# ~806K tokens; 96 questions asked in other words than the text, across one to three
# sections, plus overheard conversations) DeepSeek and GLM-5.3-Flash both scored
# 111/116, but DeepSeek cost $0.59 for the whole run against GLM's $4.16. On RULER chains
# split seven ways DeepSeek stayed at 10/10 up to 512K while GLM fell to 8/10, dropping
# lines it judged irrelevant. GLM was the earlier default, picked on one hand-written
# question that later turned out to have two valid answers.
# Host: DeepInfra bills its cache discount (~$0.0046/M warm, measured). Wafer reports
# 100% cache hits but charges near full price. DeepInfra is slow when a model thinks at
# length; listener turns think little, so it has not mattered here.
LISTENER_MODEL = "deepseek/deepseek-v4.1-flash"
LISTENER_PROVIDER = "deepinfra/fp8"  # pinning controls price AND quantization (fp4..bf16)

SILENT = "<silent/>"

LISTENER_SYSTEM = """You are a knowledge specialist. Your entire domain corpus is below.

You are overhearing a conversation. Decide whether YOUR corpus contains information
that materially changes how this turn should be answered.

Search the whole corpus before deciding - the answer may be in any document, and
the wording of the turn may not match the wording in your corpus.

Stay SILENT unless your corpus genuinely holds something specific. General knowledge,
restating the turn, or "this seems related" are all silence.

If you have nothing, reply with exactly:
<silent/>

If you do, reply with ONLY this, no preamble:
<contribution>
<source path="..."/>
<text>1-3 sentences of what your corpus actually says. Be specific: names, numbers,
quotes. No advice, no padding.</text>
</contribution>

Cite only documents that exist in your corpus."""

ROUND_SYSTEM = """You are a knowledge specialist. Your corpus is below, along with
what other specialists already reported from theirs.

Your job is to EXTEND their findings from your own corpus - a fact that connects to,
qualifies, or contradicts something they reported. Report only what is NEW.

If your corpus adds nothing beyond what they already said, reply exactly: <silent/>

Otherwise use the same format:
<contribution>
<source path="..."/>
<text>what your corpus adds, and how it connects.</text>
</contribution>"""

LEADER_SYSTEM = """You write a briefing for an AI assistant that is mid-conversation
with its user. Below are contributions from knowledge specialists that overheard the
current turn.

Output ONLY the relevant facts, as bullets, each ending with its source path in
backticks. No preamble, no "based on the contributions", no restating the question,
no advice.

If two specialists disagree, give both bullets and let the assistant weigh them.
If nothing is genuinely relevant, output exactly: NOTHING

Never assert that something does not exist. The specialists only report what they
found; a specialist that stayed silent may still hold the answer. Absence of a
contribution is not evidence of absence in the corpus.

Format:
## From your knowledge base
- <fact> — `path/to/file.md`"""


# --- turn gate ---------------------------------------------------------------
# Filler must never reach the elephants: it costs a cache read per elephant and,
# empirically, invites protocol violations (one elephant answered "ok thanks" with
# "You're welcome!"). Local, no API call, free.

_FILLER = {
    "ok", "okay", "thanks", "thank you", "ty", "yes", "no", "yep", "yeah", "nope",
    "sure", "cool", "nice", "great", "perfect", "got it", "sounds good", "do it",
    "go ahead", "continue", "next", "stop", "wait", "hmm", "lol", "k", "fine",
}


def should_wake_herd(turn: str) -> tuple[bool, str]:
    """Is this turn worth waking the herd for? Returns (wake, reason)."""
    t = turn.strip()
    if not t:
        return False, "empty"
    normalized = re.sub(r"[^\w\s]", "", t.lower()).strip()
    if normalized in _FILLER:
        return False, "filler phrase"
    words = normalized.split()
    if len(words) < 8 and "?" not in t and not re.search(r"[A-Z][a-z]{2,}", t):
        # Short, not a question, no proper noun - almost certainly acknowledgement.
        return False, "short, no question, no proper noun"
    return True, "substantive"


# --- protocol ----------------------------------------------------------------


@dataclass
class Contribution:
    elephant: str
    text: str
    sources: list[str] = field(default_factory=list)
    malformed: bool = False
    truncated: bool = False


def parse_listener(elephant: str, raw: str) -> Contribution | None:
    """Parse a listener reply. Tolerant by necessity.

    Format compliance degrades as questions get broader: a narrow question yields
    clean tags, a broad one can yield untagged prose. Dropping untagged answers
    would silently discard real knowledge, so prose is kept and flagged instead.
    """
    text = (raw or "").strip()
    if not text:
        return None
    if SILENT in text.lower().replace(" ", ""):
        return None

    sources = re.findall(r'path="([^"]+)"', text)
    if not sources:
        sources = re.findall(r"`([^`]+\.(?:md|txt))`", text)

    body = re.search(r"<text>(.*?)</text>", text, re.S)
    if body:
        return Contribution(elephant, body.group(1).strip(), sources)

    # No tags: strip any stray markup and keep the prose, flagged.
    cleaned = re.sub(r"<[^>]+>", "", text).strip()
    return Contribution(elephant, cleaned, sources, malformed=True)


# --- elephant ----------------------------------------------------------------


@dataclass
class Elephant:
    name: str
    corpus: str  # assembled, cacheable blob
    model: str = LISTENER_MODEL
    provider: str = LISTENER_PROVIDER
    system: str | None = None   # conversation elephant answers a different question

    def listen(
        self, turn: str, client: OpenRouterClient, history: str = "", max_tokens: int | None = None
    ) -> tuple[Contribution | None, Response]:
        """One turn. Corpus stays a stable cached prefix; only the turn varies.

        max_tokens must be generous and must scale with corpus size: reasoning is
        on (it is what makes answers correct) and consumes most of the budget before
        any visible output. A truncated elephant looks identical to a silent one -
        it is a silent false negative that no error surfaces. Observed twice: at 800
        tokens nothing was emitted at all, and at a flat 4000 the largest elephant
        (528K tokens) was cut off mid-reasoning while holding the correct answer.
        """
        if max_tokens is None:
            # Bigger corpus -> more to reason over before answering.
            max_tokens = max(4000, min(16000, len(self.corpus) // 120))
        messages = [
            {"role": "system", "content": self.system or LISTENER_SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<corpus>\n{self.corpus}\n</corpus>",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": (f"<conversation>{history}</conversation>\n" if history else "")
                + f"<turn>{turn}</turn>",
            },
        ]
        resp = client.chat(
            model=self.model,
            provider=self.provider,
            messages=messages,
            max_tokens=max_tokens,
            reasoning=True,  # off saves ~$0.0001 and loses correct answers
        )
        contribution = parse_listener(self.name, resp.text)
        if contribution is None and resp.usage.truncated:
            # Not silence - the budget ran out mid-reasoning. Surface it rather than
            # letting it be scored as a deliberate silence.
            contribution = Contribution(
                self.name, "", [], malformed=True, truncated=True
            )
        return contribution, resp


# --- herd --------------------------------------------------------------------


@dataclass
class HerdResult:
    briefing: str | None
    contributions: list[Contribution]
    silent: list[str]
    gated: bool
    gate_reason: str
    cost_usd: float
    latency_s: float
    usages: dict = field(default_factory=dict)


class Herd:
    def __init__(
        self,
        elephants: list[Elephant],
        leader_model: str = "anthropic/claude-sonnet-5",
        leader_provider: str | None = "anthropic",
        session_id: str = "herd",
        context_elephant: Elephant | None = None,
    ) -> None:
        self.elephants = elephants
        # Holds the conversation rather than a document corpus. Consulted before the
        # broadcast so the others know what "it" refers to, then left in the pool for
        # later rounds - because something said days ago and never written down exists
        # only here.
        self.context_elephant = context_elephant
        if context_elephant:
            self.elephants = [*elephants, context_elephant]
        self.leader_model = leader_model
        self.leader_provider = leader_provider
        # One client per elephant: session_id drives sticky routing, and each
        # elephant needs its own cache prefix kept on its own host.
        # Built from self.elephants, which includes the context elephant.
        self.clients = {
            e.name: OpenRouterClient(session_id=f"{session_id}-{e.name}")
            for e in self.elephants
        }
        self.leader_client = OpenRouterClient(session_id=f"{session_id}-leader")

    def hear(self, turn: str, history: str = "", rounds: int = 1,
             patience: int = 3) -> HerdResult:
        """One turn. `rounds` > 1 enables blackboard iteration.

        A single round finds whatever sits whole inside one elephant. Facts chained
        across elephants need one round per hop: the elephant holding "B follows from
        A" can only speak once someone has reported A. Iteration stops as soon as a
        round adds nothing new, so easy turns still cost one round - but each extra
        round is another full cache read per elephant, so this is opt-in, not default.
        """
        import time

        start = time.monotonic()
        wake, reason = should_wake_herd(turn)
        if not wake:
            return HerdResult(None, [], [], True, reason, 0.0, time.monotonic() - start)

        contributions: list[Contribution] = []
        silent: list[str] = []
        usages: dict = {}
        cost = 0.0

        # The conversation elephant goes first and alone. A domain elephant cannot
        # judge whether it holds something relevant until it knows which project the
        # turn is about, so this one sequential call earns its latency.
        broadcast_turn = turn
        if self.context_elephant is not None:
            try:
                ctx, ctx_resp = self.context_elephant.listen(
                    turn, self.clients[self.context_elephant.name], history
                )
                cost += ctx_resp.usage.cost_usd
                usages[self.context_elephant.name] = ctx_resp.usage
                if ctx and ctx.text and not ctx.truncated:
                    contributions.append(ctx)
                    broadcast_turn = (
                        f"{turn}\n\n<context>{ctx.text}</context>"
                    )
                else:
                    silent.append(self.context_elephant.name)
            except Exception as exc:
                silent.append(f"{self.context_elephant.name}(error: {str(exc)[:40]})")

        # Elephants are independent; latency is the slowest one, not the sum.
        with ThreadPoolExecutor(max_workers=len(self.elephants)) as pool:
            futures = {
                pool.submit(e.listen, broadcast_turn, self.clients[e.name], history): e
                for e in self.elephants
                if e is not self.context_elephant   # already spoke
            }
            for fut, elephant in futures.items():
                try:
                    contrib, resp = fut.result()
                except Exception as exc:  # one elephant failing must not kill the turn
                    silent.append(f"{elephant.name}(error: {str(exc)[:60]})")
                    continue
                usages[elephant.name] = resp.usage
                cost += resp.usage.cost_usd
                if contrib and contrib.truncated:
                    silent.append(f"{elephant.name}(TRUNCATED)")
                elif contrib:
                    contributions.append(contrib)
                else:
                    silent.append(elephant.name)

        # Extra rounds: re-broadcast what was found so elephants can extend it.
        quiet = 0
        for _ in range(max(0, rounds - 1)):
            if not contributions:
                break
            board = "\n".join(f"- [{c.elephant}] {c.text}" for c in contributions)
            fresh: list[Contribution] = []
            with ThreadPoolExecutor(max_workers=len(self.elephants)) as pool:
                futs = {
                    pool.submit(
                        e.listen,
                        f"{turn}\n\n<findings_so_far>\n{board}\n</findings_so_far>",
                        self.clients[e.name],
                        history,
                    ): e
                    for e in self.elephants
                }
                for fut, elephant in futs.items():
                    try:
                        contrib, resp = fut.result()
                    except Exception:
                        continue
                    cost += resp.usage.cost_usd
                    if contrib and not contrib.truncated and contrib.text:
                        if all(contrib.text != c.text for c in contributions):
                            fresh.append(contrib)
            if not fresh:
                quiet += 1
                if quiet >= patience:
                    break      # genuinely converged
                continue       # a stalled elephant may still speak next round
            quiet = 0
            contributions.extend(fresh)

        briefing = None
        if contributions:
            briefing, leader_cost = self._synthesize(turn, contributions)
            cost += leader_cost

        return HerdResult(
            briefing, contributions, silent, False, reason, cost, time.monotonic() - start, usages
        )

    def _synthesize(self, turn: str, contributions: list[Contribution]) -> tuple[str | None, float]:
        blocks = "\n\n".join(
            f"[{c.elephant}] sources: {', '.join(c.sources) or 'none cited'}\n{c.text}"
            for c in contributions
        )
        resp = self.leader_client.chat(
            model=self.leader_model,
            provider=self.leader_provider,
            messages=[
                {"role": "system", "content": LEADER_SYSTEM},
                {"role": "user", "content": f"<turn>{turn}</turn>\n\n<contributions>\n{blocks}\n</contributions>"},
            ],
            # Generous: reasoning is on, and a leader cut off mid-thought returns an
            # empty briefing that looks exactly like "nothing was relevant".
            max_tokens=8000,
        )
        if resp.usage.truncated:
            # Never silently report "nothing" when the budget simply ran out.
            return (
                "## From your knowledge base\n"
                f"- (briefing truncated after {resp.usage.completion_tokens} tokens; "
                f"{len(contributions)} specialists contributed)"
            ), resp.usage.cost_usd
        text = resp.text.strip()
        if text.upper().startswith("NOTHING"):
            return None, resp.usage.cost_usd
        return text, resp.usage.cost_usd
