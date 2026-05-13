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
upstream defaults.

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
