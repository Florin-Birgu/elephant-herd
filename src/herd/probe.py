"""Cache probe - the cheapest possible test of whether Elephant Herd is viable.

The project's entire cost argument is that a resident corpus is re-read from cache
at ~1c per million tokens. If prompt caching does not actually work through
OpenRouter for our chosen model and host, every turn costs ~30x more and the idea
is dead. This probe answers that for a few cents, before any of the system exists.

It sends the same large prefix twice and reports what the provider says about
caching, then optionally waits and sends a third time to observe TTL.

    python -m herd.probe --tokens 200000
    python -m herd.probe --tokens 200000 --ttl-wait 360
"""

from __future__ import annotations

import argparse
import random
import sys
import time

from herd.providers.openrouter import CacheMissError, OpenRouterClient

# First-party hosts, resolved from /api/v1/models/{id}/endpoints. Pinning matters:
# across hosts for the same model, cache-read price spans 10x, quantization spans
# fp4..bf16, and at least one host has no caching at all.
CANDIDATES = [
    ("deepseek/deepseek-v4-flash-0731", "deepseek"),
    ("qwen/qwen3.7-flash", "alibaba"),
]


def synthetic_prefix(target_tokens: int, seed: int = 7) -> str:
    """Filler with a planted fact, so the probe also sanity-checks recall.

    Deliberately not lorem ipsum: repetitive text can be compressed or deduplicated
    by a provider and would flatter the cache numbers. Varied sentences make the
    token count honest.
    """
    rng = random.Random(seed)
    subjects = ["the archivist", "a technician", "the courier", "an auditor", "the curator"]
    verbs = ["catalogued", "misplaced", "annotated", "verified", "returned"]
    objects = ["ledger", "manifest", "sample", "transcript", "invoice"]
    places = ["depot", "annex", "basement", "east wing", "cold store"]

    parts: list[str] = []
    approx = 0
    i = 0
    while approx < target_tokens:
        i += 1
        s = (
            f"Record {i:05d}: {rng.choice(subjects)} {rng.choice(verbs)} the "
            f"{rng.choice(objects)} in the {rng.choice(places)} at "
            f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}.\n"
        )
        parts.append(s)
        approx += len(s) // 4
    # Plant the needle in the middle - the hardest position for long-context recall.
    needle = "\nRecord 00000: the maintenance code for the east wing freight lift is QN-4417.\n"
    parts.insert(len(parts) // 2, needle)
    return "".join(parts)


NEEDLE_ANSWER = "QN-4417"
QUESTION = (
    "What is the maintenance code for the east wing freight lift? "
    "Answer with the code only."
)


def run_probe(model: str, provider: str, tokens: int, ttl_wait: int) -> dict:
    client = OpenRouterClient(session_id=f"probe-{provider}")
    prefix = synthetic_prefix(tokens)
    est_tokens = len(prefix) // 4
    print(f"\n{'=' * 78}\n{model}  @  {provider}")
    print(f"prefix ~{est_tokens:,} tokens ({len(prefix):,} chars)\n{'-' * 78}")

    def call(label: str, require_hit: bool = False) -> object:
        messages = [
            {"role": "system", "content": "You answer strictly from the records provided."},
            {
                "role": "user",
                # cache_control marks the end of the cacheable prefix. DeepSeek
                # caches automatically and ignores it; Anthropic/Qwen require it.
                "content": [
                    {
                        "type": "text",
                        "text": prefix,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": QUESTION},
        ]
        try:
            r = client.chat(
                model=model,
                messages=messages,
                provider=provider,
                max_tokens=32,
                require_cache_hit=require_hit,
            )
        except CacheMissError as exc:
            print(f"  {label:<14} CACHE MISS -> {exc}")
            return None
        except Exception as exc:  # noqa: BLE001 - probe reports, never crashes the run
            print(f"  {label:<14} FAILED: {exc}")
            return None
        found = NEEDLE_ANSWER.lower() in r.text.lower()
        print(f"  {label:<14} {r.usage.summary()}  needle={'FOUND' if found else 'MISSED'}")
        return r

    first = call("cold (write)")
    if first is None:
        return {"model": model, "provider": provider, "ok": False}
    second = call("warm (read)")
    third = None
    if ttl_wait and second is not None:
        print(f"  {'':<14} sleeping {ttl_wait}s to test cache TTL...")
        time.sleep(ttl_wait)
        third = call(f"after {ttl_wait}s")

    result = {
        "model": model,
        "provider": provider,
        "ok": second is not None and second.usage.cache_hit,
        "cold_cost": first.usage.cost_usd,
        "warm_cost": second.usage.cost_usd if second else None,
        "warm_hit_ratio": second.usage.cache_hit_ratio if second else 0.0,
        "ttl_hit": bool(third and third.usage.cache_hit) if third else None,
        "total_spent": client.total_cost_usd,
    }
    if second and first.usage.cost_usd:
        saving = 1 - (second.usage.cost_usd / first.usage.cost_usd)
        print(f"\n  cache saving: {saving:.1%} per re-read")
        per_turn = second.usage.cost_usd * (4_130_000 / max(est_tokens, 1))
        print(f"  extrapolated to a 4.13M-token vault: ${per_turn:.4f} per turn")
        result["extrapolated_per_turn"] = per_turn
    print(f"  probe spend so far: ${client.total_cost_usd:.4f}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, default=200_000, help="prefix size (default 200k)")
    ap.add_argument("--ttl-wait", type=int, default=0, help="seconds to wait for a TTL test")
    ap.add_argument("--model", help="single model id (default: probe all candidates)")
    ap.add_argument("--provider", help="host slug to pin")
    args = ap.parse_args()

    targets = (
        [(args.model, args.provider)] if args.model else CANDIDATES
    )
    results = [run_probe(m, p, args.tokens, args.ttl_wait) for m, p in targets]

    print(f"\n{'=' * 78}\nSUMMARY")
    total = 0.0
    for r in results:
        total += r.get("total_spent", 0.0)
        verdict = "CACHING WORKS" if r["ok"] else "NO CACHE"
        extra = (
            f"  ~${r['extrapolated_per_turn']:.4f}/turn at 4.13M tokens"
            if r.get("extrapolated_per_turn")
            else ""
        )
        print(f"  {r['model']:<38} {verdict}{extra}")
    print(f"\n  total probe spend: ${total:.4f}")
    return 0 if any(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
