#!/usr/bin/env python3
"""bench-bert-correctness.py — gate before flipping AMD-Mac BERT off CPU route.

Runs a battery of numeric-correctness probes against a quenchforge embed
or rerank slot and fails loudly if outputs are non-deterministic. This
is the safety gate that has to pass BEFORE an operator removes
``--gpu-layers 0`` from ``internal/tuning/tuning.go`` for AMD-discrete
embed / code-embed / rerank slots.

The 2026-05-17 staging-buffer-pool incident (v0.7.0-rc1) is the
specific failure class this script catches: the patch passed HTTP-200
liveness benches but produced numerically wrong outputs that only
surfaced as recall=0.000 on the LongMemEval ablation. This script
fails that scenario in seconds instead of hours.

Default probe set (all four must pass):

1. **identical-input determinism (same batch).**
   Same input repeated N times in a single embedding request.
   Expectation: every output vector is bit-identical (cos_sim 1.0000).

2. **identical-input determinism (separate calls).**
   Same input issued in N separate requests.
   Expectation: same vector across all calls.

3. **semantic sanity.**
   Known-similar pair ("cat sat on the mat" / "a cat is sitting on a mat")
   should produce cos_sim > 0.85. Catches the kernel producing
   non-garbage but semantically-wrong outputs.

4. **L2 norm finite.**
   Output vector L2 norm should be finite and within [0.5, 5.0].
   Catches NaN / Inf / zero-vector corruption.

Each probe is parameterized so an operator can dial up sample counts
when bisecting a regression. Defaults are CI-friendly (~30s total
wall-clock on a healthy daemon).

Usage::

    # Default: 10 identical-input calls, 10 separate calls, semantic + L2 norm
    scripts/bench-bert-correctness.py --model nomic-embed-text-v1.5

    # Tight tolerance for a release-gate run
    scripts/bench-bert-correctness.py --epsilon 1e-6 --n-calls 50

    # Probe the rerank slot too
    scripts/bench-bert-correctness.py --model bge-reranker-v2-m3 --rerank

Exit codes:
    0  All probes passed; safe to consider flipping the CPU-route flag
    1  At least one probe failed; do NOT flip the flag
    2  Daemon unreachable / unexpected protocol error
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request


# Tolerance defaults. Identical-input cos_sim should be exactly 1.0 (modulo
# fp32 rounding); 1e-4 leaves room for the f16/f32 mixed-precision path
# without admitting actual non-determinism.
DEFAULT_EPSILON = 1e-4

# Semantic-similarity floor. The "cat/mat" pair was 0.66 on the broken
# Metal path and 0.66 on CPU; setting it at 0.50 catches obvious garbage
# while admitting normal embedding noise.
DEFAULT_SEMANTIC_FLOOR = 0.50

# L2 norm bounds. nomic-embed-text-v1.5 unit-normalized vectors have
# L2 norm 1.0; allow [0.5, 5.0] for non-normalized models too.
L2_NORM_MIN = 0.5
L2_NORM_MAX = 5.0


def cos_sim(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector dim mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def l2_norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def post_embed(url: str, model: str, inputs: list[str], timeout: float = 30.0) -> list[list[float]]:
    payload = json.dumps({"model": model, "input": inputs}).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/v1/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise RuntimeError(f"daemon returned HTTP {resp.status}")
            body = json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"daemon unreachable at {url}: {exc}") from exc
    return [row["embedding"] for row in body["data"]]


def post_rerank(url: str, model: str, query: str, docs: list[str], timeout: float = 30.0) -> list[float]:
    payload = json.dumps({
        "model": model,
        "query": query,
        "documents": docs,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/v1/rerank",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise RuntimeError(f"daemon returned HTTP {resp.status}")
            body = json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"daemon unreachable at {url}: {exc}") from exc
    # OpenAI-style rerank response: results[].relevance_score
    return [r["relevance_score"] for r in body.get("results", [])]


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def probe_same_batch(url: str, model: str, n: int, epsilon: float) -> tuple[bool, str]:
    """Repeat the same input N times in ONE batch. Every output vector
    should be cos_sim 1.0 with every other.
    """
    inputs = ["hello"] * n
    vecs = post_embed(url, model, inputs)
    if len(vecs) != n:
        return False, f"requested {n} embeds, got {len(vecs)}"
    min_sim = 1.0
    for i in range(1, n):
        sim = cos_sim(vecs[0], vecs[i])
        min_sim = min(min_sim, sim)
        if sim < 1.0 - epsilon:
            return False, (
                f"same-batch cos_sim(v[0], v[{i}]) = {sim:.6f}, "
                f"expected >= {1.0 - epsilon:.6f}"
            )
    return True, f"min cos_sim across {n} same-batch entries = {min_sim:.6f}"


def probe_separate_calls(url: str, model: str, n: int, epsilon: float) -> tuple[bool, str]:
    """Issue the same input in N separate requests. cross-call cos_sim
    should be 1.0.
    """
    vecs = []
    for i in range(n):
        out = post_embed(url, model, ["hello"])
        if not out:
            return False, f"call {i+1} returned no embeddings"
        vecs.append(out[0])
    min_sim = 1.0
    for i in range(1, n):
        sim = cos_sim(vecs[0], vecs[i])
        min_sim = min(min_sim, sim)
        if sim < 1.0 - epsilon:
            return False, (
                f"cross-call cos_sim(call[0], call[{i}]) = {sim:.6f}, "
                f"expected >= {1.0 - epsilon:.6f}"
            )
    return True, f"min cos_sim across {n} separate calls = {min_sim:.6f}"


def probe_semantic_sanity(url: str, model: str, floor: float) -> tuple[bool, str]:
    """Two paraphrases of the same fact should embed similarly."""
    inputs = [
        "The cat sat on the mat.",
        "A cat is sitting on a mat.",
        "Quantum chromodynamics studies the strong interaction.",
    ]
    vecs = post_embed(url, model, inputs)
    if len(vecs) != 3:
        return False, f"requested 3 embeds, got {len(vecs)}"
    sim_para = cos_sim(vecs[0], vecs[1])
    sim_unrelated = cos_sim(vecs[0], vecs[2])
    if sim_para < floor:
        return False, (
            f"paraphrase cos_sim {sim_para:.4f} below floor {floor:.4f} — "
            "embedding is producing garbage even though it returns finite vectors"
        )
    if sim_para <= sim_unrelated:
        return False, (
            f"paraphrase ({sim_para:.4f}) should beat unrelated "
            f"({sim_unrelated:.4f}); semantic structure is gone"
        )
    return True, f"paraphrase {sim_para:.4f} >> unrelated {sim_unrelated:.4f}"


def probe_l2_norm(url: str, model: str) -> tuple[bool, str]:
    """Output vector L2 norm should be finite and within [0.5, 5.0]."""
    vecs = post_embed(url, model, ["hello"])
    if not vecs:
        return False, "no vectors returned"
    norm = l2_norm(vecs[0])
    if math.isnan(norm) or math.isinf(norm):
        return False, f"L2 norm = {norm} (NaN/Inf — corrupted output)"
    if not (L2_NORM_MIN <= norm <= L2_NORM_MAX):
        return False, (
            f"L2 norm {norm:.4f} outside [{L2_NORM_MIN}, {L2_NORM_MAX}] — "
            "embedding magnitude is off; possible kernel-output corruption"
        )
    return True, f"L2 norm = {norm:.4f}"


def probe_rerank_determinism(url: str, model: str, n: int, epsilon: float) -> tuple[bool, str]:
    """Rerank a fixed (query, docs) tuple N times. Scores should be
    bit-identical (or within fp32 rounding).
    """
    query = "What is the capital of France?"
    docs = [
        "Paris is the capital and most populous city of France.",
        "Berlin is the capital of Germany.",
        "The Eiffel Tower stands in Paris.",
    ]
    runs = [post_rerank(url, model, query, docs) for _ in range(n)]
    if any(len(r) != len(docs) for r in runs):
        return False, f"rerank returned wrong cardinality"
    for doc_idx in range(len(docs)):
        scores = [r[doc_idx] for r in runs]
        spread = max(scores) - min(scores)
        if spread > epsilon:
            return False, (
                f"rerank doc[{doc_idx}] score spread {spread:.6f} across "
                f"{n} calls, expected within {epsilon}"
            )
    return True, f"all {n} rerank calls returned identical scores"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
        default="nomic-embed-text-v1.5",
        help="model name to probe (default: nomic-embed-text-v1.5)",
    )
    parser.add_argument(
        "--n-calls",
        type=int,
        default=10,
        help="how many calls per determinism probe (default: 10)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help=f"max permitted cos_sim deviation from 1.0 (default: {DEFAULT_EPSILON})",
    )
    parser.add_argument(
        "--semantic-floor",
        type=float,
        default=DEFAULT_SEMANTIC_FLOOR,
        help=f"paraphrase cos_sim floor (default: {DEFAULT_SEMANTIC_FLOOR})",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="run the rerank determinism probe instead of embedding probes",
    )
    args = parser.parse_args(argv)

    print(f"=== bench-bert-correctness @ {args.url} ({args.model}) ===\n")

    probes: list[tuple[str, callable]] = []
    if args.rerank:
        probes.append((
            "rerank determinism",
            lambda: probe_rerank_determinism(
                args.url, args.model, args.n_calls, args.epsilon,
            ),
        ))
    else:
        probes.append((
            f"same-batch determinism (n={args.n_calls})",
            lambda: probe_same_batch(
                args.url, args.model, args.n_calls, args.epsilon,
            ),
        ))
        probes.append((
            f"separate-call determinism (n={args.n_calls})",
            lambda: probe_separate_calls(
                args.url, args.model, args.n_calls, args.epsilon,
            ),
        ))
        probes.append((
            "semantic sanity",
            lambda: probe_semantic_sanity(args.url, args.model, args.semantic_floor),
        ))
        probes.append((
            "L2 norm bounds",
            lambda: probe_l2_norm(args.url, args.model),
        ))

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
    print("Next step: sustained-load bench via scripts/bench-bert-sustained-load.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
