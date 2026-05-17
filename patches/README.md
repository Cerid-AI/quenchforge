# Patch series

Quenchforge carries **four** patches — one per submodule. All four fix the same root cause (Apple-Silicon-only Metal kernels incorrectly enabled on AMD Mac), in four independently-vendored copies of ggml-metal. Applied at build time by `scripts/apply-patches.sh`; submodule SHAs in `.gitmodules` stay clean.

| File | Submodule path | Target file | Upstream |
|---|---|---|---|
| `llama.cpp/0001-metal-correctness-on-non-apple-silicon.patch` | `llama.cpp/` | `ggml/src/ggml-metal/ggml-metal-device.m` | [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) |
| `whisper.cpp/0001-metal-correctness-on-non-apple-silicon.patch` | `whisper.cpp/` | `ggml/src/ggml-metal/ggml-metal-device.m` | [`ggml-org/whisper.cpp`](https://github.com/ggml-org/whisper.cpp) |
| `sd.cpp/0001-metal-correctness-on-non-apple-silicon.patch` | `sd.cpp/` | `ggml/src/ggml-metal/ggml-metal-device.m` (via nested `ggml-org/ggml` submodule) | [`leejet/stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) |
| `bark.cpp/0001-metal-correctness-on-non-apple-silicon.patch` | `bark.cpp/` | `encodec.cpp/ggml/src/ggml-metal.m` (via two-level nested submodules; older single-file `ggml-metal.m` layout, different API: `support_*` not `has_*`) | [`PABannier/bark.cpp`](https://github.com/PABannier/bark.cpp) → [`PABannier/encodec.cpp`](https://github.com/PABannier/encodec.cpp) |

All four patches address the **same upstream bug** ([ggml-org/llama.cpp#19563](https://github.com/ggml-org/llama.cpp/issues/19563)) — the `|= MTLGPUFamilyMetal3_GGML` line that enables Apple-Silicon-only kernels on AMD Mac. Each consumer of ggml has its own copy of the offending source, so we patch each copy.

## What the patch does

In `ggml/src/ggml-metal/ggml-metal-device.m`:

- `has_simdgroup_reduction` is gated to `MTLGPUFamilyApple7` only. Opt back in via `GGML_METAL_FORCE_SIMDGROUP_REDUCTION=1`.
- `has_bfloat` is gated to `MTLGPUFamilyApple6` only. Opt back in via `GGML_METAL_FORCE_BF16=1`. The existing `GGML_METAL_BF16_DISABLE=1` still works as a hard override.

Trade-off is a slower scalar fallback for reductions. On AMD Vega II + llama3.2:3b we measured ~4 tok/s patched vs garbage tokens unpatched. Correct slow output beats fast garbage; v0.4 will explore rewriting the reduction kernels to use AMD-compatible intrinsics.

## Supervisor-level companion fixes (no second patch)

Two additional Metal correctness issues surface on AMD discrete that
the simdgroup/bfloat gating doesn't reach. Both are addressed by the
Go supervisor passing explicit `llama-server` flags when it detects an
AMD-discrete profile (`Vega Pro`, `W6800X`, `RDNA1`, `RDNA2`) —
**no second llama.cpp patch is required**, which keeps the
"one patch per submodule" rule intact.

### 1. Flash-attention CPU-fallback throttle

With simdgroup reduction disabled, llama-server's `--flash-attn auto`
correctly determines that FA's GPU path is unavailable, but instead of
disabling FA outright it schedules the FA tensor on CPU each decode
step. Result: a GPU↔CPU copy every token, throttling chat to ~3 tok/s
on Vega II despite all 29/29 model layers being resident on MTL0.

Supervisor passes `--flash-attn off` for the chat slot on AMD
profiles. Standard attention is slower per kernel than FA but
stays GPU-resident, yielding a net throughput win.

### 2. Prompt-cache `GGML_ASSERT(buf_dst)` crash

The chat slot's prompt-cache state-save path
(`server_slot::prompt_save` → `llama_context::state_seq_get_data` →
`ggml_metal_buffer_get_tensor`) hits a NULL destination buffer on
Vega II during the second request when LCP similarity > 10%. Slot
aborts with:

```
ggml-metal-device.m:1736: GGML_ASSERT(buf_dst) failed
```

Supervisor passes **two** flags on AMD chat slots:

- `--cache-ram 0` — disables the server-side LCP-similarity slot
  cache, which is the path that calls `prompt_save`. This is the
  load-bearing flag.
- `--no-cache-prompt` — companion: disables per-slot prompt caching
  so the LCP path can't be entered via a different trigger in a
  future llama.cpp release.

`--no-cache-prompt` alone is not sufficient — it controls the
per-slot prompt cache, but the crash is in the server-side
LCP-similarity cache that runs during slot picking
(`get_available_slot`).

Embed and rerank slots don't touch either cache, so they keep the
upstream defaults for the LCP/prompt-cache surface. (They do, however,
have a separate sustained-load failure mode — see section 3 below.)

### 3. Sustained-load graph-compute buffer-corruption

A third Metal-on-AMD failure surfaces on the embed and rerank slots
under sustained batch workloads (eval suites, bulk KB ingest, sustained
MCP retrieval). After ~50-200 successive forward passes the
`ggml_metal_buffer_set_tensor` (or `_get_tensor`) call asserts on a
NULL Metal buffer and the slot SIGABRTs:

```
ggml-metal-device.m:1665+: ggml_metal_buffer_set_tensor →
                            ggml_abort → SIGABRT
```

Reading `~/Develop/quenchforge/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:1665-1717`
the mechanism is clear: `newBufferWithBytesNoCopy` with
`MTLResourceStorageModeShared` requests a CPU-visible Metal buffer
from the AMD discrete driver. On Apple Silicon this is trivial — unified
memory means the `buf->is_shared` fast path uses plain `memcpy` and
never enters the failing code at all (lines 1666-1668, 1720-1722). On
AMD discrete, the driver maintains a finite PCIe staging-buffer pool;
sustained sub-millisecond allocations exhaust it and the API returns
NULL. The neighbouring `GGML_ASSERT(buf_src)` / `GGML_ASSERT(buf_dst)`
calls then abort the process.

The cascade extends past the slot: the `AMDRadeonX5000` kernel mutex
that owned the failing allocation can stall WindowServer for tens of
seconds (the userspace watchdog then resets the WindowServer process
and the operator's UI session is interrupted, even though no kernel
panic occurs).

The existing chat-slot flags from sections 1 and 2 don't help. The
LCP-prompt-save crash and the FA-CPU-fallback throttle are
chat-decode-specific. The buffer-corruption family runs through the
graph-compute path (`process_ubatch → encode → graph_compute →
buffer_{set,get}_tensor`) which fires on every forward pass — chat,
embed, rerank. Chat traffic in production is bursty enough that the
staging-buffer pool drains between requests; eval-style workloads with
gapless POSTs do not give it that recovery window.

Supervisor mitigations (no llama.cpp patch — same "one patch per
submodule" reasoning):

- `--ubatch-size` and `--batch-size` are configurable per-slot via
  `QUENCHFORGE_EMBED_UBATCH_SIZE` (default 0 → inherit MaxContext)
  and `QUENCHFORGE_RERANK_BATCH_SIZE` (default 0 → llama.cpp's
  512-token internal default). Smaller ubatch shrinks per-call Metal
  staging allocations on AMD discrete.
- `GGML_METAL_N_CB` is configurable per-slot via
  `QUENCHFORGE_EMBED_METAL_N_CB` and `QUENCHFORGE_RERANK_METAL_N_CB`.
  Lowering to 1 serialises Metal command-buffer submission so the
  staging-buffer pool drains between commands instead of accumulating
  pressure.
- AMD-discrete embed/rerank slots auto-respawn on SIGABRT (2s / 4s /
  8s exponential backoff, capped at 3 attempts/60s) so the gateway
  recovers without manual `launchctl kickstart`.
- The gateway's rolling-window latency tracker surfaces impending
  family-B exhaustion via `/health` (per-slot status `ok | degraded |
  critical`), so consumers can throttle before the SIGABRT. Opt-in
  `QUENCHFORGE_AUTO_BACKOFF=true` turns `critical` into an automatic
  503+`Retry-After` on the upstream proxy paths.

Per-profile defaults will be tuned on the `[amd-gpu]` self-hosted
runner in a follow-up PR; `quenchforge-bench sustained-embed` is the
harness that finds the smallest stability-preserving overhead. Until
then, defaults preserve current behaviour and operators on the affected
hardware set the env knobs explicitly.

A long-term patch in upstream llama.cpp / ggml-metal that pools the
staging-buffer allocations across calls (instead of
`newBufferWithBytesNoCopy` per call) would address this at the source.
That work is out of scope for the supervisor layer and tracked
separately.

### Why not a second patch?

The "one patch per submodule" rule (`CLAUDE.md` absolute rule #3)
holds: code that produces wrong/missing Metal buffer pointers should
be patched at the source. But the two issues above are fixed at the
correct architectural layer — at the supervisor, by passing
already-supported `llama-server` flags — and a patch would duplicate
behavior llama-server already exposes via its CLI. The original
simdgroup-reduction patch is necessary because that code path has
no runtime opt-out; these two have an opt-out.

Issue refs: tracked at `Cerid-AI/quenchforge#1` (gateway /api/chat
translation, separate concern) and `Cerid-AI/quenchforge#2` (the
prompt-cache crash, now mitigated by `--no-cache-prompt`).

### 3. Sustained-load graph-compute buffer-corruption — patch #2 (v0.7.0)

Closes the third Metal-on-AMD failure class — `GGML_ASSERT(buf_src)`
SIGABRT in `ggml_metal_buffer_set_tensor` and `ggml_metal_buffer_get_tensor`
under sustained embed / chat workloads on AMD discrete.

`v0.6.0` shipped a **supervisor-side** mitigation: AMD-discrete embed,
code-embed, rerank, and (since v0.6.1) chat slots get `AutoRespawn:
true`, so the supervisor brings the slot back within ~30 seconds of
the SIGABRT. The slot is back online, but the caller sees a 502 +
breaker open during the window.

`v0.7.0` ships the **kernel-level** fix as
`llama.cpp/0002-metal-staging-buffer-pool.patch`. The patch replaces
the per-call `newBufferWithBytesNoCopy` allocation — which registers a
new IOMMU page-table entry on AMD discrete and exhausts the driver's
~256-512-slot pool — with a bounded MTLBuffer pool keyed on
power-of-two size classes. One pool buffer = one registration, reused;
worst-case total registrations stays well below the exhaustion
threshold.

Apple Silicon is unaffected: `buf->is_shared` short-circuits to the
`memcpy` fast path before either patched function reaches the pool.

Escape hatch: `GGML_METAL_DISABLE_STAGING_POOL=1` reverts to the
unpatched behaviour for A/B testing during rollout.

**Bench-validated on Mac Pro 2019 + Radeon Pro Vega II 32 GB HBM2,
2026-05-17:** 1597 sustained-embed calls over 3 minutes against
`nomic-embed-text-v1.5`, **zero family-B SIGABRTs**, p50 = 109 ms,
p99 = 147 ms, ratio = 1.34 (well below the critical-ratio = 5
threshold). v0.6.2 unpatched typically hits family-B within ~80
calls under the same workload. The full design, bench acceptance
criteria, and upstream-issue draft live in
[`llama.cpp/drafts/README.md`](llama.cpp/drafts/README.md) (kept as
supplemental documentation for the upstream filing).

## Honesty about whisper.cpp Metal

The patch is necessary on both submodules but **not sufficient on whisper.cpp**. Even with `simdgroup_reduction` and `bfloat` both disabled, whisper-server on Vega II still produces garbage tokens — there's an additional Metal-on-AMD bug in whisper-specific kernels (likely the encoder's convolution or attention paths). The patch silences the obvious failure modes; the deeper issue is still being investigated.

For v0.3, `quenchforge serve` defaults the whisper slot to `--no-gpu` on AMD Mac. CPU on a 32-core Xeon W-3245 runs **12.8× real-time** on tiny.en — fast enough that the latency hit is invisible. Operators can opt in to Metal via `QUENCHFORGE_WHISPER_GPU=true` when they're on hardware where it works (or want to help us debug).

## Live-verified results (2026-05-12, Mac Pro 2019 + Radeon Pro Vega II 32 GB HBM2)

### llama.cpp chat — patched Metal

```
prompt: "In one sentence, what is Quenchforge?"
output: "I couldn't find any information on 'Quenchforge'..."
prompt rate:  72.6 tok/s
predict rate:  4.1 tok/s
```

Coherent, factually-grounded. Validates issue #19563 fix.

### whisper.cpp transcription — patched CPU (Metal still buggy in v0.3)

```
input: whisper.cpp/samples/jfk.wav (11s audio)
output: "And so my fellow Americans ask not what your country can do for you
         ask what you can do for your country."
wall:    0.86s (12.8× real-time)
```

Correct JFK quote. Going through `quenchforge serve` → `/v1/audio/transcriptions` → `whisper-server` → Vega II (CPU mode).

## Provenance

Both patches are **re-derived from the public llama.cpp issue + live debugging on the Mac Pro 2019**. No third-party gist code is copied. Each patch carries an `Upstream-Issue:` git-trailer linking to the canonical conversation. A draft upstream PR will land once the series stabilises further.

## Re-applying after upstream changes

`scripts/rebase-upstream.sh` runs weekly via GitHub Actions and replays the series onto the latest `master` for each submodule with `git am -3`. The three-way merge survives benign refactors because the patches are keyed on stable function names (`ggml_metal_device_init`), not absolute line numbers.

On a real conflict the rebase action stops with rejected hunks visible and opens a PR with the `rebase-conflict` label for human resolution.

## How to apply manually

```sh
git submodule update --init --recursive   # if you cloned without --recursive
bash scripts/apply-patches.sh             # idempotent; --check for dry-run; --reset to start over
bash scripts/build-llama.sh                # patched llama-server (chat / embed / rerank)
bash scripts/build-whisper.sh              # patched whisper-server (audio transcription)
```
