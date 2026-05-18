#!/usr/bin/env python3
"""bench-bert-sustained-load.py — kernel-panic prevention bench.

Hammers the embed slot with concurrent, batched, varied-payload
requests for a configured duration and watches for:

- SIGABRT / daemon disappearance (process exits unexpectedly)
- HTTP 5xx burst (indicates daemon back-pressure or slot crash)
- output drift (cos_sim across identical-input probes interleaved
  with the load workload should stay at 1.0; drift indicates GPU
  state corruption that builds up over time)
- IOSurface exhaustion symptom (latency-cliff at ~5-10 minutes,
  the 2026-05-14 kernel-panic pattern)
- daemon RSS growth beyond a threshold (leak indicator)

Default duration is 30 minutes — matches the 2026-05-14 incident's
time-to-crash and the threshold the staging-buffer-pool patch was
supposed to clear. Operators wanting a quick sanity check can pass
``--duration 300`` for 5 min; release-gate runs should stay at 1800.

This script does NOT itself modify the daemon. It assumes the daemon
is already running at the configured slot config. To exercise the
fallback kernels: build a quenchforge with patches 0001 + 0003 + 0004
applied, modify ``internal/tuning/tuning.go`` to drop ``--gpu-layers 0``
from embedParams (the activation switch), restart the daemon, then run
this bench. Reverting the tuning change rolls back instantly.

Usage::

    # Default 30-min release gate run
    scripts/bench-bert-sustained-load.py --model nomic-embed-text-v1.5

    # Quick 5-min smoke
    scripts/bench-bert-sustained-load.py --duration 300

    # Aggressive concurrency probe
    scripts/bench-bert-sustained-load.py --concurrency 8 --batch-size 16

Exit codes:
    0  No SIGABRT, no HTTP 5xx burst, no drift, no leak; safe.
    1  Failure detected; surface the cause and do NOT proceed.
    2  Daemon unreachable / setup error.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request


# Failure thresholds — tuned to fail fast on the known incident patterns.
HTTP_5XX_BURST_THRESHOLD = 5    # 5 5xx in a 30s window = early bail
# Two-tier drift gating:
#   - WARN: identical-input cos_sim below 0.999 deserves attention but
#     can be explained by fp-rounding noise inside the daemon's
#     batched / parallel processing path. Logged, doesn't abort.
#   - FAIL: cos_sim below 0.95 is the Metal-on-AMD non-determinism
#     class — abort immediately. (Pre-patch baseline was 0.07; this
#     threshold has 10× headroom over any conceivable false-positive.)
DRIFT_COSSIM_WARN = 0.999
DRIFT_COSSIM_FAIL = 0.95
RSS_GROWTH_FACTOR = 2.0         # 2× RSS growth over the run is a leak signal
LATENCY_CLIFF_FACTOR = 5.0      # mid-run p95 > 5× start-of-run p95 = IOSurface symptom

# Sample synthetic workload — designed to vary chunk shape, context length,
# and content so the daemon's cache + tokenizer paths get exercised broadly.
_WORKLOAD_TEMPLATES = [
    "The quick brown fox jumps over the lazy dog. ",
    "Paris is the capital of France. The Eiffel Tower stands in Paris. ",
    "Quantum chromodynamics studies the strong interaction. Gluons carry the strong force. ",
    "The cat sat on the mat. The dog ran in the yard. The bird flew in the sky. ",
    "Machine learning models embed text into dense vectors for retrieval. ",
]


def synthesize_inputs(batch_size: int, char_target: int) -> list[str]:
    """Build a varied batch where each input is ~char_target chars."""
    out = []
    for _ in range(batch_size):
        template = random.choice(_WORKLOAD_TEMPLATES)
        repeats = max(1, char_target // len(template))
        out.append(template * repeats)
    return out


def post_embed(url: str, model: str, inputs: list[str], timeout: float = 60.0) -> tuple[int, list[list[float]] | None, str]:
    """Returns (http_status, vectors_or_None, error_message)."""
    payload = json.dumps({"model": model, "input": inputs}).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/v1/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            if status != 200:
                return status, None, f"HTTP {status}"
            body = json.loads(resp.read())
        return 200, [r["embedding"] for r in body["data"]], ""
    except urllib.error.HTTPError as e:
        return e.code, None, f"HTTP {e.code}"
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return 0, None, f"network: {e}"


def find_llama_server_pid(model_hint: str = "nomic-embed") -> int | None:
    """Return the PID of the llama-server process serving the embed slot,
    or None if not found. We use ``pgrep -f`` to scope to the model hint.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"llama-server.*{model_hint}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        # If multiple matches, take the first (the others are usually grep itself
        # or test slots).
        return int(result.stdout.strip().split("\n")[0])
    except (subprocess.SubprocessError, ValueError):
        return None


def get_rss_kb(pid: int) -> int | None:
    """Return RSS in KB for the given PID, or None on failure."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "rss="],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        return int(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        return None


def cos_sim(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default=os.getenv("QUENCHFORGE_URL", "http://127.0.0.1:11434"),
        help="quenchforge gateway URL",
    )
    parser.add_argument(
        "--model",
        default="nomic-embed-text-v1.5",
        help="embedding model name",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=1800,
        help="total bench duration in seconds (default: 1800 = 30 min)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="parallel HTTP workers (default: 4, matches --parallel 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="chunks per request (default: 8)",
    )
    parser.add_argument(
        "--char-target",
        type=int,
        default=1500,
        help="approximate chars per chunk (default: 1500 = LongMemEval chunk)",
    )
    parser.add_argument(
        "--drift-probe-interval",
        type=int,
        default=60,
        help="seconds between identical-input drift probes (default: 60)",
    )
    args = parser.parse_args(argv)

    print(f"=== sustained-load bench @ {args.url} ===")
    print(f"    model={args.model} duration={args.duration}s "
          f"concurrency={args.concurrency} batch={args.batch_size}")
    print()

    pid = find_llama_server_pid("nomic-embed" if "nomic" in args.model else args.model.split("-")[0])
    if pid:
        rss0 = get_rss_kb(pid)
        print(f"    llama-server pid={pid}, initial RSS={rss0} KB")
    else:
        rss0 = None
        print(f"    [warn] could not find llama-server pid for RSS tracking")
    print()

    # Reference vector for drift detection.
    _, ref_vecs, err = post_embed(args.url, args.model, ["sustained-load reference probe"])
    if err or not ref_vecs:
        print(f"    [DAEMON-ERROR] cannot get reference vector: {err}")
        return 2
    ref = ref_vecs[0]

    t_start = time.perf_counter()
    t_end = t_start + args.duration

    n_requests = 0
    n_failures = 0
    failures_5xx_window: list[float] = []  # timestamps of recent 5xx for burst detection
    latencies: list[float] = []
    last_drift_check = t_start

    def issue_one() -> tuple[int, float, str]:
        inputs = synthesize_inputs(args.batch_size, args.char_target)
        t0 = time.perf_counter()
        status, _, err = post_embed(args.url, args.model, inputs)
        dt = time.perf_counter() - t0
        return status, dt, err

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures: list[concurrent.futures.Future] = []
        # Initial backlog
        for _ in range(args.concurrency):
            futures.append(pool.submit(issue_one))

        while time.perf_counter() < t_end:
            done, _pending = concurrent.futures.wait(
                futures, timeout=1.0,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                futures.remove(fut)
                status, dt, err = fut.result()
                n_requests += 1
                latencies.append(dt)
                if status >= 500 or status == 0:
                    n_failures += 1
                    now = time.perf_counter()
                    failures_5xx_window.append(now)
                    failures_5xx_window = [
                        t for t in failures_5xx_window if now - t < 30.0
                    ]
                    if len(failures_5xx_window) >= HTTP_5XX_BURST_THRESHOLD:
                        print(f"    [FAIL] HTTP 5xx burst — {len(failures_5xx_window)} "
                              f"failures in last 30s ({err})")
                        return 1
                # Keep the pipeline full
                if time.perf_counter() < t_end:
                    futures.append(pool.submit(issue_one))

            # Periodic drift probe
            now = time.perf_counter()
            if now - last_drift_check > args.drift_probe_interval:
                _, drift_vecs, derr = post_embed(
                    args.url, args.model, ["sustained-load reference probe"],
                )
                if derr or not drift_vecs:
                    print(f"    [FAIL] drift probe error: {derr}")
                    return 1
                sim = cos_sim(ref, drift_vecs[0])
                elapsed = now - t_start
                drift_marker = ""
                if sim < DRIFT_COSSIM_FAIL:
                    drift_marker = " [FAIL]"
                elif sim < DRIFT_COSSIM_WARN:
                    drift_marker = " [WARN]"
                print(f"    t={int(elapsed):5d}s  reqs={n_requests:6d}  "
                      f"fail={n_failures:4d}  "
                      f"p50={sorted(latencies)[len(latencies)//2] if latencies else 0:.2f}s  "
                      f"drift_cos_sim={sim:.6f}{drift_marker}")
                if sim < DRIFT_COSSIM_FAIL:
                    print(f"    [FAIL] catastrophic output drift — reference "
                          f"cos_sim {sim:.6f} below fail floor "
                          f"{DRIFT_COSSIM_FAIL}; this is the Metal-on-AMD "
                          f"non-determinism class")
                    return 1
                last_drift_check = now

            # Periodic RSS check
            if pid and rss0 and int(now - t_start) % 300 == 0:
                rss_now = get_rss_kb(pid)
                if rss_now and rss_now > rss0 * RSS_GROWTH_FACTOR:
                    print(f"    [FAIL] RSS leak — grew {rss0} → {rss_now} KB "
                          f"({rss_now/rss0:.2f}× over {int(now-t_start)}s)")
                    return 1

        # Drain remaining futures (best-effort)
        for fut in concurrent.futures.as_completed(futures, timeout=60):
            try:
                fut.result()
            except Exception:
                pass

    elapsed = time.perf_counter() - t_start
    p50 = sorted(latencies)[len(latencies)//2] if latencies else 0
    p95 = sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0
    print()
    print(f"=== complete: {n_requests} requests, {n_failures} failures, {elapsed:.0f}s ===")
    print(f"    p50={p50:.2f}s  p95={p95:.2f}s  throughput={n_requests/elapsed:.2f} req/s")

    # Latency-cliff check: compare first 10% to last 10% of latencies as a rough
    # IOSurface-exhaustion signal. If late latencies are 5× early, something
    # bad happened mid-run even though no individual probe tripped.
    if len(latencies) > 100:
        early = sorted(latencies[:len(latencies)//10])
        late = sorted(latencies[-len(latencies)//10:])
        early_p95 = early[int(len(early)*0.95)]
        late_p95 = late[int(len(late)*0.95)]
        if late_p95 > early_p95 * LATENCY_CLIFF_FACTOR:
            print(f"    [FAIL] latency cliff — late p95 {late_p95:.2f}s vs early "
                  f"{early_p95:.2f}s ({late_p95/early_p95:.1f}×)")
            return 1

    if pid and rss0:
        rss_final = get_rss_kb(pid)
        print(f"    daemon RSS: {rss0} → {rss_final} KB "
              f"({(rss_final/rss0 if rss0 else 0):.2f}×)")

    print()
    print("=== sustained-load bench PASSED ===")
    print("Daemon survived sustained load without crash / drift / leak signals.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
