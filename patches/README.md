# Patch series

Quenchforge carries **one** patch against `ggml-org/llama.cpp` (pinned at the SHA in `.gitmodules`). The patch is applied at build time by `scripts/apply-patches.sh`, NOT committed back into the submodule.

| File | Upstream issue | Purpose |
|---|---|---|
| `0001-metal-correctness-on-non-apple-silicon.patch` | [ggml-org/llama.cpp#19563](https://github.com/ggml-org/llama.cpp/issues/19563) | Gate `has_simdgroup_reduction` to `MTLGPUFamilyApple7` only and default `has_bfloat` to `MTLGPUFamilyApple6` only. Both kernels miscompile on AMD Metal and produce garbage tokens on Intel Mac + AMD discrete configurations. |

## Why only one (originally planned two)

The original plan called for two patches:

1. **`private-vram-force`** — force `MTLStorageModePrivate` for discrete AMD ([issue #15228](https://github.com/ggml-org/llama.cpp/issues/15228)). **Now obsolete.** Live testing on the current pinned submodule SHA confirms `use shared buffers = false` is already the default behaviour on AMD Vega II: `recommendedMaxWorkingSetSize = 34342.96 MB` is reported correctly and the KV cache lands in VRAM. Upstream has converged on the right path. The patch is dropped from the series.
2. **`disable-simdgroup-mm`** — gate `has_simdgroup_mm` ([issue #19563](https://github.com/ggml-org/llama.cpp/issues/19563)). **Also obsolete in its original form.** Live device init on Vega II reports `simdgroup matrix mul. = false` already — upstream's `[mtl_device supportsFamily:MTLGPUFamilyApple7]` check correctly returns false on AMD.

Quenchforge's actual fix lives in `0001-metal-correctness-on-non-apple-silicon.patch`. It addresses the **same bug** as #19563 (Apple-Silicon-only kernels enabled on non-Apple Metal devices) but the root cause turned out to be `has_simdgroup_reduction` + `has_bfloat`, both of which had a `|=` line that re-enabled them on any Metal3 device including AMD Vega II / W6800X / RDNA1+2 / Intel Iris.

## What the patch does

In `ggml/src/ggml-metal/ggml-metal-device.m`:

- `has_simdgroup_reduction` is gated to `MTLGPUFamilyApple7` only. Operators can opt back in to the previous Metal3-wide behaviour via `GGML_METAL_FORCE_SIMDGROUP_REDUCTION=1`.
- `has_bfloat` is gated to `MTLGPUFamilyApple6` only. Operators can opt back in via `GGML_METAL_FORCE_BF16=1`. The existing `GGML_METAL_BF16_DISABLE=1` still works as a hard override.

The trade-off is a slower scalar fallback for reductions (observed ~4 tok/s on Vega II + llama3.2:3b vs. an estimated ~50-80 tok/s the simdgroup path would deliver if it were correct). Correct slow output beats fast garbage; v0.2 will explore rewriting the reduction kernels to use AMD-compatible intrinsics.

## Live-verified results (2026-05-12, Mac Pro 2019 + Radeon Pro Vega II 32 GB HBM2)

**Without patch (stock llama.cpp @ a9883db):**

```
prompt: "Hello, my name is"
output: "Key key key key key key key key key key key key key key key key[]inedresult menIONweenatrixATRIXATRIXATRIX..."
```

The model loads, the GPU is correctly enumerated as `MTL0 (AMD Radeon Pro Vega II)`, but the simdgroup-reduction kernel produces wrong arithmetic and the sampler emits garbage tokens indefinitely. Confirmed reproducer for issue #19563.

**With patch applied:**

```
prompt: "In one sentence, what is Quenchforge?"
output: "I couldn't find any information on 'Quenchforge'. It's possible that it's a
        lesser-known or emerging technology, or it may be a misspelling or incorrect
        term. If you could provide more context or clarify what Quenchforge refers
        to, I'd be happy to try and help further."

prompt rate:  72.6 tok/s
predict rate:  4.1 tok/s
```

Coherent, factually-grounded text. End-to-end through the `quenchforge serve` gateway working.

## Provenance

The patch is **re-derived from the public issue description** and our own live debugging on the Mac Pro. No third-party gist code is copied. The commit message in the patch credits issue #19563 and documents the GGML_METAL_FORCE_* opt-in env vars.

A draft upstream PR will be filed once this series stabilises so the diff has a permanent reference on the `ggml-org/llama.cpp` issue tracker. Until then, the patch lives only in this repo.

## Re-applying after upstream changes

`scripts/rebase-upstream.sh` runs weekly via GitHub Actions and replays this series onto the latest `ggml-org/llama.cpp` master with `git am -3`. The three-way merge survives benign refactors because the patch is keyed on stable function names (`ggml_metal_device_init`), not absolute line numbers.

On a real conflict the rebase action stops with rejected hunks visible and opens a PR with the `rebase-conflict` label for human resolution.

## How to apply manually

```sh
git submodule update --init --recursive   # if you cloned without --recursive
bash scripts/apply-patches.sh             # idempotent; --check for dry-run; --reset to start over
bash scripts/build-llama.sh                # CMake + Ninja, produces llama-server binary
```
