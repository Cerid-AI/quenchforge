#!/usr/bin/env python3
"""bench-llama-correctness.py — gate before flipping AMD-Mac chat off CPU.

Runs determinism + semantic probes against a quenchforge chat slot.
Fails loudly if responses drift across calls at temperature=0.

Probes:

1. **deterministic single-prompt** (10 calls, temperature=0).
   Three fixed prompts; each must return the SAME response string
   on all 10 calls. Catches the cross-call race condition that
   produced non-deterministic BERT embeddings before the
   GGML_METAL_CONCURRENCY_DISABLE=1 fix.

2. **semantic sanity** (1 call per question).
   Three factual questions; responses must contain the expected
   keyword. Catches the slot producing fluent-but-wrong output
   (e.g., kernel-state corruption that doesn't trigger garbage
   detection at the token level).

Usage::

    scripts/bench-llama-correctness.py --url http://127.0.0.1:11500
    scripts/bench-llama-correctness.py --n-calls 50  # release-gate run

Exit codes:
    0  All probes passed; safe to consider flipping the CPU-route flag
    1  At least one probe failed; do NOT flip the flag
    2  Daemon unreachable / unexpected protocol error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


DETERMINISTIC_PROMPTS = [
    ("What is 2+2? Answer in one word.", "four"),
    ("What is the capital of France? Answer in one word.", "paris"),
    ("Largest planet in our solar system? Answer in one word.", "jupiter"),
]


def post_chat(url: str, model: str, prompt: str, temperature: float,
              max_tokens: int = 10, timeout: float = 60.0) -> tuple[int, str, str]:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return resp.status, "", f"HTTP {resp.status}"
            body = json.loads(resp.read())
        text = body["choices"][0]["message"]["content"].strip()
        return 200, text, ""
    except urllib.error.URLError as exc:
        return 0, "", f"daemon unreachable at {url}: {exc}"


def probe_deterministic(url: str, model: str, n: int) -> tuple[bool, str]:
    """Each prompt repeated N times at temperature=0 — all responses must match."""
    for prompt, _expected in DETERMINISTIC_PROMPTS:
        responses: list[str] = []
        for i in range(n):
            status, text, err = post_chat(url, model, prompt, temperature=0.0)
            if err:
                return False, f"call {i+1} of {n} failed: {err}"
            responses.append(text)
        first = responses[0]
        for i, r in enumerate(responses[1:], 1):
            if r != first:
                return False, (
                    f"prompt {prompt!r}: call[0]={first!r} but "
                    f"call[{i}]={r!r} — non-deterministic at temperature=0"
                )
    return True, f"all {n} calls per prompt returned identical responses"


def probe_semantic(url: str, model: str) -> tuple[bool, str]:
    """Each prompt answered once; response must contain the expected keyword."""
    for prompt, expected in DETERMINISTIC_PROMPTS:
        status, text, err = post_chat(url, model, prompt, temperature=0.0, max_tokens=30)
        if err:
            return False, f"daemon error on prompt {prompt!r}: {err}"
        if expected.lower() not in text.lower():
            return False, (
                f"prompt {prompt!r}: response {text!r} missing "
                f"expected keyword {expected!r} — semantic regression"
            )
    return True, "all factual prompts contained expected keywords"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default=os.getenv("QUENCHFORGE_URL", "http://127.0.0.1:11434"),
        help="quenchforge gateway URL (default: http://127.0.0.1:11434)",
    )
    parser.add_argument(
        "--model",
        default="llama3.1-8b",
        help="chat model name (default: llama3.1-8b)",
    )
    parser.add_argument(
        "--n-calls",
        type=int,
        default=10,
        help="how many calls per determinism prompt (default: 10)",
    )
    args = parser.parse_args(argv)

    print(f"=== bench-llama-correctness @ {args.url} ({args.model}) ===\n")

    probes: list[tuple[str, callable]] = [
        (f"deterministic single-prompt (n={args.n_calls})",
         lambda: probe_deterministic(args.url, args.model, args.n_calls)),
        ("semantic sanity",
         lambda: probe_semantic(args.url, args.model)),
    ]

    failed = 0
    for name, fn in probes:
        t0 = time.perf_counter()
        try:
            ok, detail = fn()
        except RuntimeError as exc:
            print(f"  [DAEMON-ERROR] {name}: {exc}")
            return 2
        elapsed = time.perf_counter() - t0
        marker = "[PASS]" if ok else "[FAIL]"
        print(f"  {marker} {name}  ({elapsed:.2f}s)\n         {detail}")
        if not ok:
            failed += 1

    print()
    if failed:
        print(f"=== {failed} probe(s) FAILED — do NOT flip the CPU-route flag ===")
        return 1
    print("=== all probes passed ===")
    print("Next step: sustained-load bench via scripts/bench-llama-sustained-load.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
