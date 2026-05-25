#!/usr/bin/env python3
"""bench-llama-sustained-load.py — chat-equivalent of bench-bert-sustained-load.

Hammers a chat slot for a configured duration and watches for the failure
modes documented in project_cerid_quenchforge_chat_on_cpu memory:
- SIGABRT / daemon disappearance (PID gone, HTTP 5xx burst)
- response drift on deterministic prompts (temperature=0 should give
  bit-identical text every time; divergence = race-condition pollution)
- latency cliff (mid-run p95 >> start-of-run p95 = compute degradation)
- RSS leak (>2× over the run)

Exit codes:
    0  Stable; safe to consider flipping chat to GPU
    1  Failure detected; do NOT flip
    2  Daemon unreachable / setup error
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request


HTTP_5XX_BURST_THRESHOLD = 5
DRIFT_FAIL_THRESHOLD = 3
RSS_GROWTH_FACTOR = 2.0
LATENCY_CLIFF_FACTOR = 5.0

DETERMINISTIC_PROMPTS = [
    ("What is 2+2? Answer in one word.", None),
    ("What is the capital of France? Answer in one word.", None),
    ("Largest planet in our solar system? Answer in one word.", None),
]

VARYING_PROMPTS = [
    "Write a one-sentence summary of the French Revolution.",
    "Explain photosynthesis in two sentences.",
    "List 3 prime numbers above 10.",
    "Describe how a bicycle works in one short paragraph.",
    "What's the typical weather pattern in a temperate zone?",
    "Name 3 common units of electrical resistance.",
    "What's the difference between TCP and UDP?",
    "Explain the term 'idiomatic Go code'.",
]


def post_chat(url, model, prompt, temperature, max_tokens, timeout=120.0):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            if status != 200:
                return status, "", f"HTTP {status}"
            body = json.loads(resp.read())
        return 200, body["choices"][0]["message"]["content"].strip(), ""
    except urllib.error.HTTPError as e:
        return e.code, "", f"HTTP {e.code}"
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return 0, "", f"network: {e}"


def find_pid(hint):
    try:
        r = subprocess.run(["pgrep", "-f", f"llama-server.*{hint}"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        return int(r.stdout.strip().split("\n")[0])
    except Exception:
        return None


def get_rss_kb(pid):
    try:
        r = subprocess.run(["ps", "-p", str(pid), "-o", "rss="],
                           capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return None
        return int(r.stdout.strip())
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=os.getenv("QUENCHFORGE_URL", "http://127.0.0.1:11434"))
    p.add_argument("--model", default="llama3.1-8b")
    p.add_argument("--duration", type=int, default=1800)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--rss-hint", default="llama3.1-8b",
                   help="substring to match the llama-server cmdline for RSS tracking")
    args = p.parse_args()

    print(f"=== chat sustained-load @ {args.url} ===")
    print(f"    model={args.model} duration={args.duration}s "
          f"concurrency={args.concurrency}")
    print()

    # Reference responses
    refs = {}
    for prompt, _ in DETERMINISTIC_PROMPTS:
        status, resp, err = post_chat(args.url, args.model, prompt,
                                      temperature=0, max_tokens=10)
        if err:
            print(f"    [DAEMON-ERROR] reference probe failed: {err}")
            return 2
        refs[prompt] = resp
        print(f"    ref: {prompt!r} -> {resp!r}")
    print()

    pid = find_pid(args.rss_hint)
    if pid:
        rss0 = get_rss_kb(pid)
        print(f"    llama-server pid={pid} initial RSS={rss0} KB")
    else:
        rss0 = None
        print(f"    [warn] no RSS tracking (pid not found)")
    print()

    t_start = time.perf_counter()
    t_end = t_start + args.duration
    n_req = 0
    n_fail = 0
    n_drift = 0
    latencies = []
    fail_window = []
    last_log = t_start

    def issue_one(prompt, deterministic):
        t0 = time.perf_counter()
        temp = 0.0 if deterministic else 0.7
        max_tok = 10 if deterministic else 80
        status, response, err = post_chat(args.url, args.model, prompt,
                                          temperature=temp, max_tokens=max_tok)
        dt = time.perf_counter() - t0
        return status, dt, err, prompt, response, deterministic

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = []
        for _ in range(args.concurrency):
            futures.append(pool.submit(issue_one,
                                       random.choice(VARYING_PROMPTS), False))

        while time.perf_counter() < t_end:
            done, _ = concurrent.futures.wait(
                futures, timeout=2.0,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                futures.remove(fut)
                status, dt, err, prompt, response, deterministic = fut.result()
                n_req += 1
                latencies.append(dt)

                if status >= 500 or status == 0:
                    n_fail += 1
                    now = time.perf_counter()
                    fail_window.append(now)
                    fail_window = [t for t in fail_window if now - t < 30.0]
                    if len(fail_window) >= HTTP_5XX_BURST_THRESHOLD:
                        print(f"    [FAIL] HTTP 5xx burst — "
                              f"{len(fail_window)} fails / 30s ({err})")
                        return 1

                if deterministic and status == 200:
                    if response != refs[prompt]:
                        n_drift += 1
                        print(f"    [DRIFT] {prompt!r}: "
                              f"ref {refs[prompt]!r} -> got {response!r}")
                        if n_drift >= DRIFT_FAIL_THRESHOLD:
                            print(f"    [FAIL] response drift — {n_drift} "
                                  f"deterministic probes diverged")
                            return 1

                if time.perf_counter() < t_end:
                    if random.random() < 0.2:
                        det_prompt, _ = random.choice(DETERMINISTIC_PROMPTS)
                        futures.append(pool.submit(issue_one, det_prompt, True))
                    else:
                        futures.append(pool.submit(issue_one,
                                                   random.choice(VARYING_PROMPTS),
                                                   False))

            now = time.perf_counter()
            if now - last_log > 60:
                elapsed = int(now - t_start)
                p50 = sorted(latencies)[len(latencies)//2] if latencies else 0
                rss_str = ""
                if pid and rss0:
                    rss_now = get_rss_kb(pid)
                    if rss_now is None:
                        print(f"    [FAIL] llama-server pid {pid} disappeared "
                              "(SIGABRT?)")
                        return 1
                    rss_str = f"  rss={rss_now}KB ({rss_now/rss0:.2f}x)"
                    if rss_now > rss0 * RSS_GROWTH_FACTOR:
                        print(f"    [FAIL] RSS leak — {rss0} -> {rss_now} KB")
                        return 1
                print(f"    t={elapsed:5d}s  reqs={n_req:5d}  "
                      f"fail={n_fail:3d}  drift={n_drift}  p50={p50:.2f}s{rss_str}")
                last_log = now

        for fut in concurrent.futures.as_completed(futures, timeout=180):
            try:
                fut.result()
            except Exception:
                pass

    elapsed = time.perf_counter() - t_start
    if not latencies:
        print("    [FAIL] no requests completed")
        return 1
    p50 = sorted(latencies)[len(latencies)//2]
    p95 = sorted(latencies)[int(len(latencies)*0.95)]
    print()
    print(f"=== complete: {n_req} reqs, {n_fail} failures, "
          f"{n_drift} drifts, {elapsed:.0f}s ===")
    print(f"    p50={p50:.2f}s  p95={p95:.2f}s  "
          f"throughput={n_req/elapsed:.2f} req/s")

    if len(latencies) > 100:
        early = sorted(latencies[:len(latencies)//10])
        late = sorted(latencies[-len(latencies)//10:])
        if early and late:
            early_p95 = early[int(len(early)*0.95)]
            late_p95 = late[int(len(late)*0.95)]
            if early_p95 > 0 and late_p95 > early_p95 * LATENCY_CLIFF_FACTOR:
                print(f"    [FAIL] latency cliff — late p95 {late_p95:.2f}s "
                      f"vs early {early_p95:.2f}s "
                      f"({late_p95/early_p95:.1f}x)")
                return 1

    if pid and rss0:
        rss_final = get_rss_kb(pid)
        if rss_final:
            print(f"    daemon RSS: {rss0} -> {rss_final} KB "
                  f"({rss_final/rss0:.2f}x)")

    if n_drift > 0:
        print(f"    [WARN] {n_drift} drift events but below fail threshold")

    print()
    print("=== chat sustained-load PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
