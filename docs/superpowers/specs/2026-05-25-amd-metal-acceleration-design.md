> ⚠️ **SUPERSEDED (2026-05-25)** — This spec is preserved as a historical research artifact. The wave-width hypothesis (`N_SIMDWIDTH=32` vs AMD's 64-wide waves) identified a real architectural mismatch per Apple MSL Spec §4.4.2, but empirical work later that day isolated the operational bug to `MTLDispatchTypeConcurrent` (a separate code path) plus the family-B staging-buffer-pool exhaustion. The v0.8.0 implementation followed [`2026-05-25-amd-metal-staging-buffer-pool-revival-design.md`](2026-05-25-amd-metal-staging-buffer-pool-revival-design.md) instead — a two-layer fix (patch 0002 staging-buffer pool + `GGML_METAL_CONCURRENCY_DISABLE=1` env-var routing in `tuning.go`) that does NOT require the wave-width kernel rewrite proposed below. The prior-art survey, MSL Spec citations, AMD GCN5 architecture documentation, and llama.cpp Metal backend code references in this spec remain useful reference material.

---

# 2026-05-25 — AMD Vega II Metal acceleration: kernel-level repair design

**Status:** approved 2026-05-25 (technical attack + Phase-0-diagnostic skip).
**Driver:** v0.7.2 ships with all 4 quenchforge slots on CPU (`--gpu-layers 0`). The 2026-05-25 GPU activation experiment proved patches 0003/0004 are **insufficient AND mis-implemented**: nomic-embed cross-call cos_sim **-0.011** (expected ≥ 0.9999) — catastrophic non-determinism rather than mere noise.
**Goal:** Restore GPU inference on Intel Mac + AMD discrete (canonical target: Mac Pro 2019 + Radeon Pro Vega II 32 GB) for the four production models cerid runs (`llama3.1-8b` Q4_K_M chat, `nomic-embed-text-v1.5` embed, `jina-embeddings-v2-base-code` code-embed, `bge-reranker-v2-m3` rerank).

---

## Problem

Quenchforge's headline mission per `CLAUDE.md` — *"first-class local inference for Intel Mac + AMD discrete GPU configurations that are not served by upstream Ollama / llama.cpp on macOS"* — is unmet on the production hardware. All inference runs on the Xeon W's 16 cores; the Vega II's 32 GB VRAM is idle. Measured cost on this Mac Pro: embed throughput limited to 85 req/s (GPU baseline would be ~400), chat throughput limited to 7.4 tok/s (GPU baseline ~40-60).

### Root-cause analysis (the four findings driving this design)

**Finding 1 — `N_SIMDWIDTH = 32` is a latent bug across the patch series.**

`llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:28`:
```c
#define N_SIMDWIDTH 32 // assuming SIMD group size is 32
```

Per Apple's *Metal Shading Language Specification* §4.4.2, the `simdgroup_size` attribute is hardware-dependent:
- Apple GPUs (A11+, M1+): **32 threads**
- Intel iGPU, AMD discrete: **64 threads** (GCN5/RDNA wave width)

Patches 0003/0004 assume `N_SIMDWIDTH == threads_per_simdgroup` and index threadgroup-local buffers as `local_buf[sgitg*NW + tiisg]`. On AMD with 64-wide waves, `tiisg ∈ [0, 63]` but the multiplication uses `NW=32`, so two physical threads write to the same buffer slot. **Race condition on every reduction.** This explains the catastrophic cross-call non-determinism: the "winner" of the race differs run-to-run, producing different output vectors for identical inputs.

This bug exists in upstream llama.cpp too, not just our patches — but upstream's `simd_sum()` fast path happens to give correct (if slow) results on AMD because the Metal compiler may emulate the cross-lane primitives, while our explicit threadgroup-memory tree-reduce hits the race directly.

**Finding 2 — Quantized matmul kernels (the load-bearing chat path) are entirely uncovered.**

`llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` has 74 `simd_sum/simd_max/simd_min` call sites. Patches 0003/0004 cover ~6-8 of them (norm + softmax + fp32/fp16 mat-vec NR0=2). Production-critical uncovered kernels:
- `kernel_mul_mv_q4_K_f32_impl` — every linear layer in `llama3.1-8b` chat
- `kernel_mul_mv_q5_K_f32_impl`, `q6_K`, `q8_0` — broader quant family the model registry stores
- `kernel_mul_mv_q2_K_f32`, `q3_K` — low-quant extremes
- `kernel_rms_norm` (all variants) — Llama-family models use RMSNorm, NOT the LayerNorm that patch 0003 covers

Patch 0003's comment explicitly admits: *"RMSNorm doesn't have an `_fb` template yet — chat correctness held through the v0.6.x cycle without it"* — empirical 2026-05-17 evidence showed this assumption was wrong (257 chat-slot SIGABRTs / 7 days uptime).

**Finding 3 — Patch 0003 made things measurably worse, not better.**

Per patch 0003's own README:
| Probe | Pre-patch (only 0001 active) | Post-patch (0001 + 0003) | CPU reference |
|---|---|---|---|
| nomic identical "hello" same batch | cos_sim 0.07 | cos_sim **0.29** | 1.0000 |
| nomic two separate "hello" calls | cos_sim 0.15 | cos_sim **0.06** | 1.0000 |

Today's 2026-05-25 bench against the v0.7.1 patch stack (0001 + 0003 + 0004) shows nomic cross-call cos_sim **-0.011** — i.e. the additional 0004 patch *did not improve cross-call determinism*, consistent with the wave-width race still corrupting the reduction.

**Finding 4 — Binary gating prevents graduated rollout.**

`has_simdgroup_reduction` is a single boolean. Patch 0001 forces it `false` for AMD, which makes the dispatcher pick `_fb` variants for every kernel that has one — including kernels whose `_fb` is itself broken (per Finding 3). There's no graduated path: validated kernels stay broken alongside the unvalidated ones.

### What this means

The patches don't just under-cover — the ones we have are **mis-implemented** because of the wave-width bug. Adding more `_fb` kernels with the same bug pattern won't help. The first patch in the new series must rewrite the reduction primitive to be wave-width-agnostic; subsequent patches build on that fixed primitive.

---

## Goals

1. **GPU acceleration unlocked** for the four production models on Mac Pro 2019 + Vega II, validated by:
   - `bench-bert-correctness.py` PASSING on `nomic-embed-text-v1.5`, `jina-embeddings-v2-base-code`, `bge-reranker-v2-m3` (cos_sim ≥ 0.9999 same-batch and cross-call)
   - A new `bench-llama-correctness.py` PASSING on `llama3.1-8b` Q4_K_M (perplexity within 0.5% of CPU reference, identical-prompt determinism cos_sim ≥ 0.9999)
   - `bench-bert-sustained-load.py --duration 1800` PASSING (no SIGABRT, no kernel panic, no latency cliff, no RSS leak)
   - A new `bench-llama-sustained-load.py --duration 1800` PASSING for chat
2. **Measurable speedup** validated by side-by-side numbers (`bench-throughput.py`):
   - Embed throughput ≥ 250 req/s (target 3x over CPU 85 req/s; floor is 2x)
   - Rerank latency ≤ 60 ms for 4 docs (target ~2x over CPU 138 ms)
   - Chat throughput ≥ 25 tok/s on `llama3.1-8b` Q4_K_M (target 3x over CPU 7.4 tok/s)
3. **Zero regression on Apple Silicon** validated by the existing test suite + a smoke run on a separate M-series machine (CI runner if available, otherwise spot check).
4. **Patches upstreamable** — each patch contains a clear rationale, a public reference (upstream issue or MSL spec citation), and a reproducer. Goal: land at least the wave-width primitive (patch 0003-rewrite) upstream so the bug stops affecting anyone else.

## Non-goals

- **Linux/CUDA/ROCm/Vulkan support.** Per CLAUDE.md absolute rule 2.
- **RDNA1/2/W6800X validation.** Vega II is the canonical target; other AMD profiles inherit code via the same gating but aren't bench-validated here. Operators on those cards opt in at their own risk.
- **Apple Silicon performance improvements.** The `simd_*` fast path stays untouched on Apple7+; this work only affects the `has_simdgroup_reduction == false` codepath.
- **Image generation / TTS / Whisper acceleration.** Those slots already work or are out of scope.
- **MPSGraph or Apple-Performance-Shaders fallback.** Considered as Approach C, rejected (sacrifices upstream patch contribution and the "useful artifact" mission). Re-evaluate only if the kernel rewrite proves AMD-Metal-on-Apple-driver has bugs beyond wave width that Apple won't fix.

---

## Architecture: four patches in series

Each patch is independently shippable; each comes with a bench that locks in its correctness invariant. The dispatcher gating is graduated — each patch flips its own kernels from "broken upstream" to "fixed fallback" without touching other kernels.

### Patch 0003-rewrite — wave-width-agnostic reduction primitive

**Replaces** patch 0003 entirely (don't keep two versions). The current 0003 stays in git history; the rewrite supersedes it with a corrected implementation.

**Key change:** introduce a runtime-queried simdgroup size via a Metal function constant, replacing the hardcoded `N_SIMDWIDTH = 32`:

```msl
// ggml-metal.metal — at top of file, near the existing N_SIMDWIDTH define
constant uint kRuntimeSimdWidth [[function_constant(FC_SIMD_WIDTH)]];

// Used by every _fb kernel that needs simdgroup geometry. Apple Silicon
// pipelines compile with kRuntimeSimdWidth=32 (matching N_SIMDWIDTH);
// AMD/Intel pipelines compile with kRuntimeSimdWidth=64.
```

```objc
// ggml-metal-device.m — when building pipelines for non-Apple devices
MTLFunctionConstantValues * cv = [[MTLFunctionConstantValues alloc] init];
uint simd_width = (uint)pipelineState.threadExecutionWidth;  // 32 or 64
[cv setConstantValue:&simd_width type:MTLDataTypeUInt atIndex:FC_SIMD_WIDTH];
// pass cv when newComputePipelineStateWithFunction:...
```

**Reduction primitive rewrite — pure linear-thread tree reduction, no simdgroup math:**

```msl
// Helper: reduce `tpg` values held by `tpg` threads via threadgroup
// tree-reduce. Caller passes a local_buf of size >= tpg and the runtime
// tpg = NSG * kRuntimeSimdWidth. Output is in local_buf[0] when tid == 0.
// tpg MUST be a power of two (dispatcher invariant) — single-threaded
// tail handling is unnecessary because dispatchers always issue
// power-of-two thread counts.
static inline void tg_tree_reduce_sum(
        threadgroup float * local_buf,
        float value,
        ushort tid,
        ushort tpg) {
    local_buf[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort stride = tpg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            local_buf[tid] += local_buf[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
// Identical shape for tg_tree_reduce_max — replace `+=` with `= MAX(local_buf[tid], local_buf[tid + stride])`.
// Float-only signatures (no template) keep the call site simple; for half /
// half4 reductions, hand-write a parallel `_h` variant (small code dup, no
// type-erasure complexity).
```

(Code snippets in this spec are illustrative — the implementation will tighten signatures, naming, and the float/half/float4 variants during patch authoring.)

**Why this is correct on both Apple Silicon and AMD:**
- Uses only `[[thread_position_in_threadgroup]]` (1D linear ID); no `sgitg*NW + tiisg` math
- Threadgroup memory write/read is well-defined under `threadgroup_barrier`
- Assumes only that thread count `N` is power-of-2 (enforced by dispatcher)
- No `simd_*` primitives → no Apple-Silicon-specific behavior

**Rewrite all four current `_fb` kernels to use this primitive:**
- `kernel_norm_fuse_fb_impl<T,F>` (norm / norm_mul / norm_mul_add, ×3 fuse depths × {float, float4})
- `kernel_soft_max_fb<T>` (`f16`, `f32`, `f16_4`, `f32_4`)
- `kernel_mul_mv_t_t_fb_impl<T0,T1,NR0>` (NR0=2 only, matching upstream's enabled dispatch)
- `helper_mv_reduce_and_write_fb<NR0>` — now a thin wrapper over `tg_tree_reduce_sum`

**Dispatcher update:** the `_fb` selection logic in `ggml-metal-device.cpp` is unchanged — patch 0001 still forces `has_simdgroup_reduction = false` on AMD, which still routes to `_fb` variants. The variants are now actually correct.

**Bench:** rerun `bench-bert-correctness.py` against `nomic-embed-text-v1.5`. Must PASS (cos_sim ≥ 0.9999 same-batch AND cross-call). This is the single load-bearing gate — if this fails, the wave-width hypothesis is incomplete and we need Approach C escalation.

**Files:**
- `patches/llama.cpp/0003-metal-amd-bert-fallback-kernels.patch` — full rewrite
- `patches/README.md` — Section 3 updated with new mechanism + bench numbers
- `CHANGELOG.md` — v0.8.0-rc2 entry

### Patch 0005 — RMS norm fallback (unblocks Llama chat correctness)

**Why a separate patch:** RMSNorm has different math from LayerNorm (no mean centering, root-mean-square instead of variance) and lives in distinct kernel templates. Mixing the two into 0003-rewrite would muddy the upstream story.

**Coverage:** `kernel_rms_norm_f32`, `kernel_rms_norm_f32_4`, plus the `mul` and `mul_add` fused variants — same `F == 1, 2, 3` fuse depths the existing norm kernels expose.

**Mechanism:** apply `tg_tree_reduce_sum` from patch 0003-rewrite to compute the sum-of-squares; final scaling step is pointwise, no reduction.

```msl
template<typename T, short F>
kernel void kernel_rms_norm_fuse_fb_impl(
        constant ggml_metal_kargs_rms_norm & args,
        device const char * src0,
        // ... same args as upstream
        ) {
    threadgroup float local_buf[1024];  // sized for max tpg=1024
    // ... compute partial sum-of-squares per thread
    tg_tree_reduce_sum<float, 1024>(local_buf, partial_sumsq, tid);  // see patch 0003-rewrite
    // ... apply scale per thread (no reduction needed)
}
```

**Dispatcher:** extend `pipeline_norm` in `ggml-metal-device.cpp` to apply the same `_fb` suffix selection to RMS norm that patch 0003-rewrite uses for layer norm.

**Bench:** new `scripts/bench-llama-correctness.py` (introduced in patch 0007) tests RMS norm via end-to-end Llama probes. Inline mini-test: run `kernel_rms_norm` against a known vector on Vega II, compare to CPU result, assert L2 norm match within 1e-5.

**Files:**
- `patches/llama.cpp/0005-metal-amd-rms-norm-fallback.patch` (new)
- `patches/README.md` — new Section 4

### Patch 0006 — Quantized matmul fallback (Q4_K_M + Q5_K_M + Q8_0)

**Why these three quants:** Q4_K_M is `llama3.1-8b` (production chat); Q5_K_M is the default tier-up; Q8_0 is the high-fidelity option in the model registry. Q1_0 / Q2_K / Q3_K / Q4_0 / Q5_0 / Q6_K can be added in follow-up patches if anyone runs them — none are in cerid's production set.

**Mechanism:** the K-quants (Q4_K_M, Q5_K_M, Q6_K) use 8×32-element super-blocks; each thread processes a partial sum across some super-blocks, then `helper_mv_reduce_and_write` (broken) collapses those partials. Replace the helper call with the `tg_tree_reduce_sum` primitive from patch 0003-rewrite.

```msl
// New _fb variants of the existing quantized kernels.
// Body identical to upstream kernel_mul_mv_q4_K_f32_impl up to the final
// reduce; that reduce becomes a tg_tree_reduce_sum call.
template<short NR0>
void kernel_mul_mv_q4_K_f32_fb_impl(...) {
    float sumf[NR0] = { 0.f };
    // ... compute per-thread partial sums (identical to upstream)
    threadgroup float local_buf[1024 * NR0];  // verify upper bound for actual NR0 values
    for (short row = 0; row < NR0; ++row) {
        tg_tree_reduce_sum<float, /*tpg=*/...>(local_buf + row * tpg, sumf[row], tid);
    }
    // ... write result
}
```

**Dispatcher update:** extend `pipeline_mul_mv` in `ggml-metal-device.cpp` to route Q4_K_M / Q5_K_M / Q8_0 to `_fb` variants when `has_simdgroup_reduction == false`. Other quant types fall through to upstream (still broken on AMD, no regression).

**Bench:** `bench-llama-correctness.py` (introduced in patch 0007) — perplexity within 0.5% of CPU reference on `llama3.1-8b` for a 200-token deterministic prompt.

**Files:**
- `patches/llama.cpp/0006-metal-amd-quantized-matmul-fallback.patch` (new)
- `patches/README.md` — new Section 5
- `CHANGELOG.md` — note quant coverage scope

### Patch 0007 — Bench harness expansion + graduated activation

**Why this is its own "patch":** It's not a llama.cpp kernel patch but a quenchforge-side change. Lives in `scripts/` and `internal/tuning/tuning.go`.

**New scripts:**
- `scripts/bench-llama-correctness.py` — perplexity probe + deterministic-prompt probe; exit codes match `bench-bert-correctness.py` (0 = safe to flip; 1 = do not flip; 2 = daemon unreachable)
- `scripts/bench-llama-sustained-load.py --duration N` — chat-equivalent of the BERT sustained-load bench; watches for SIGABRT, 5xx burst, latency cliff, RSS leak, IOSurface exhaustion
- `scripts/bench-throughput.py` — measure tok/s and req/s for all four slot types; used pre/post activation to quantify speedup

**Tuning policy:** introduce per-slot env-var overrides so each slot can be flipped independently as its patch validates:
```go
// internal/tuning/tuning.go
QUENCHFORGE_AMD_CHAT_GPU_LAYERS  // default 0 (CPU); set to 999 after patch 0006 validates
QUENCHFORGE_AMD_EMBED_GPU_LAYERS  // default 0; set after patch 0003-rewrite validates
QUENCHFORGE_AMD_RERANK_GPU_LAYERS  // default 0; set after patch 0003-rewrite validates
```

**Bench:** the patch itself doesn't need a correctness bench — it's the test harness. Validation is "all 5 bench scripts run and exit cleanly against both CPU and GPU paths."

**Files:**
- `scripts/bench-llama-correctness.py` (new, ~250 LOC matching `bench-bert-correctness.py` shape)
- `scripts/bench-llama-sustained-load.py` (new, ~300 LOC matching `bench-bert-sustained-load.py` shape)
- `scripts/bench-throughput.py` (new, ~150 LOC)
- `internal/tuning/tuning.go` — three new env-var overrides + tests
- `internal/tuning/tuning_test.go` — coverage for new overrides
- `docs/AMD_GPU_ACTIVATION_PROCEDURE.md` (new) — operator-facing procedure documenting bench-then-flip per slot

---

## Order of operations (suggested execution sequence)

1. **Patch 0007's bench harness first.** Need objective validation tooling before any kernel change.
2. **Patch 0003-rewrite.** Validate against `bench-bert-correctness.py` on `nomic-embed-text-v1.5` + `bge-reranker-v2-m3` + `jina-embeddings-v2-base-code`. **GATE: if any fails, escalate to Approach C — wave-width was incomplete.** If all pass, flip embed/rerank slots via `QUENCHFORGE_AMD_EMBED_GPU_LAYERS=999` + `QUENCHFORGE_AMD_RERANK_GPU_LAYERS=999`. Measure throughput uplift via `bench-throughput.py`.
3. **Patch 0005.** Validate via `bench-llama-correctness.py` RMSNorm sub-probe. No slot flip yet — chat needs patch 0006 too.
4. **Patch 0006.** Validate via `bench-llama-correctness.py` end-to-end. If PASS, flip chat slot via `QUENCHFORGE_AMD_CHAT_GPU_LAYERS=999`. Run `bench-llama-sustained-load.py --duration 1800`.
5. **Production observation window** — 7 days of running with all three slots on GPU, watch for kernel panics or quality regressions. If clean, update `internal/tuning/tuning.go` defaults so AMD-discrete profiles ship with GPU on by default. Tag v0.9.0.

## Testing strategy

| Test | Where | When run | Pass criterion |
|---|---|---|---|
| `bench-bert-correctness.py` | local dev + CI | every patch 0003-rewrite / 0005 / 0006 build | cos_sim ≥ 0.9999 on all 4 probes for each of 3 BERT models |
| `bench-llama-correctness.py` | local dev | patches 0005, 0006 | perplexity ≤ 0.5% from CPU; cross-call cos_sim ≥ 0.9999 on deterministic prompt |
| `bench-bert-sustained-load.py --duration 1800` | local dev | post-flip embed/rerank | zero SIGABRT, no 5xx burst, RSS slope < 1 MB/min |
| `bench-llama-sustained-load.py --duration 1800` | local dev | post-flip chat | same + tok/s stable within ±10% across the window |
| `bench-throughput.py` | local dev | pre and post each slot flip | quantifies speedup; logs to `docs/AMD_GPU_BENCH_RESULTS.md` for the record |
| `go test ./...` | local + CI | every commit | unchanged; tuning tests cover new env-vars |
| `apply-patches.sh --check` | CI rebase-upstream | weekly cron | all patches apply cleanly against upstream tip |

## Verification per patch

| Patch | Verification command | Expected outcome |
|---|---|---|
| 0007 bench harness | `python3 scripts/bench-llama-correctness.py --url http://127.0.0.1:11500 --model llama3.1-8b` (against CPU slot) | PASS — establishes the reference |
| 0003-rewrite | `python3 scripts/bench-bert-correctness.py --url http://127.0.0.1:11501 --model nomic-embed-text-v1.5` (after flipping embed to GPU) | PASS on all 4 probes |
| 0005 | `python3 scripts/bench-llama-correctness.py --only=rms_norm` (a sub-probe added in 0007 that hits RMSNorm in isolation) | PASS — RMSNorm output matches CPU |
| 0006 | `python3 scripts/bench-llama-correctness.py` (full Llama probes after flipping chat to GPU) | PASS — perplexity match + determinism |
| Full stack | All 5 bench scripts in sequence | All PASS |

## Rollback

Every patch reverts independently:
- **0007:** revert env-var overrides in `tuning.go`; bench scripts can stay (read-only tools)
- **0003-rewrite:** revert the patch file; dispatcher falls through to upstream (broken on AMD, same as pre-patch-0001 state — patch 0001 stays loaded as the correctness floor)
- **0005:** revert the patch file; RMSNorm dispatches to upstream
- **0006:** revert the patch file; quantized matmul dispatches to upstream
- **Per-slot flip rollback:** set the corresponding `QUENCHFORGE_AMD_*_GPU_LAYERS=0` and `launchctl kickstart`

If 0003-rewrite fails its bench, all subsequent patches are blocked. Pre-validated v0.7.2 CPU route stays as the production fallback indefinitely.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Wave-width hypothesis is incomplete — 0003-rewrite still produces wrong output on AMD | Medium | High (blocks the whole plan) | Escalate to Approach C (MPSGraph wrapping) for the affected kernels; document the discovered second bug class |
| Quant kernel rewrite is more complex than fp32/fp16 (K-quants have super-block structure) | Medium | Medium (delays patch 0006) | Start with Q8_0 (simplest quant) before Q4_K_M; build confidence incrementally |
| `tg_tree_reduce_sum` performs worse than `simd_sum` on Apple Silicon (regression vector) | Low | High if it ships to Apple Silicon | Patches gate strictly on `has_simdgroup_reduction == false`; Apple Silicon never sees the fallback |
| Sustained-load bench triggers a kernel panic on GPU before the bench can report | Medium | High (need another hard reboot) | Run sustained-load AFTER correctness — if correctness is solid the SIGABRT class is less likely; have `launchctl bootout com.cerid.quenchforge` ready before starting; Phase 0 hygiene fixes ensure the system can survive a panic without needing two hard reboots |
| Apple changes Metal compiler in macOS 27 such that the wave-width gating breaks | Low | Medium | Patches use stable Metal attributes (`threadExecutionWidth`, `[[function_constant]]`); revisit only if MSL spec changes |
| Upstream llama.cpp rejects the patch | Low | Low (we maintain locally either way) | File the upstream bug separately; patch 0003-rewrite stands on its own as a public artifact even if not merged |
| Chat-slot Q4_K_M GPU performance is below expectation (e.g., < 15 tok/s instead of ~40) | Medium | Low (still > CPU 7.4) | Profile via Metal Frame Capture; revisit super-block parallelization in patch 0006 |
| Estimated 4-6 week timeline slips to 10+ weeks | Medium | Low (acceptable for the kind of work this is) | Phase 0007 first so partial progress is measurable; each patch is independently shippable |

## Acceptance criteria

Implementation is complete when:
1. All 4 patches in `patches/llama.cpp/` apply cleanly via `scripts/apply-patches.sh --check`.
2. `bench-bert-correctness.py` PASSES on all 3 BERT models against GPU-routed slots on Vega II.
3. `bench-llama-correctness.py` PASSES on `llama3.1-8b` Q4_K_M against GPU-routed chat slot.
4. `bench-bert-sustained-load.py --duration 1800` PASSES on Vega II.
5. `bench-llama-sustained-load.py --duration 1800` PASSES on Vega II.
6. `bench-throughput.py` shows ≥ 2x speedup over CPU baseline for each of the four slots.
7. Production `internal/tuning/tuning.go` defaults for AMD-discrete have `--gpu-layers 0` removed for at least the validated slot types.
8. ≥ 7 days of production uptime under the new GPU defaults with no kernel panic or `quenchforge doctor` non-PASS finding attributable to the patches.
9. `CHANGELOG.md` v0.9.0 entry documents the change + bench numbers; `docs/AMD_GPU_ACTIVATION_PROCEDURE.md` documents the per-slot flip procedure for public users.
10. At least patch 0003-rewrite filed as an upstream llama.cpp PR with the wave-width rationale and the MSL spec citation.

## Open questions deferred to implementation plan

- Exact location for `FC_SIMD_WIDTH` function-constant ID (must not collide with existing `FC_*` IDs in `ggml-metal-impl.h`).
- Whether to gate `tg_tree_reduce_sum` template size on `[[max_total_threads_per_threadgroup]]` (runtime) or a hardcoded 1024 upper bound (simpler).
- Whether `bench-llama-correctness.py` should compute perplexity itself or shell out to existing llama-server `/perplexity` endpoint.
- Whether to file the four patches as four separate upstream PRs or one combined PR; depends on upstream maintainer preference.

These are tactical implementation choices, not architectural; they get decided when the plan turns into code.

## Connection to the v0.7.2 stability work

This spec builds directly on the v0.7.2 stability foundation completed earlier today. The defenses shipped in v0.7.2 (log rotation, pre-bind check, doctor extensions, scheduled-reboot capability) make the bench-and-iterate cycle of this plan safe to execute — without them, a sustained-load bench that triggers a kernel panic would risk the same multi-hour outage that motivated v0.7.2 in the first place. With them, a panic is recoverable in one reboot, the next bench attempt is unblocked, and the system maintains diagnostic visibility throughout.

Both efforts share the same north star: making the AMD Vega II + Intel Mac configuration a first-class inference platform. v0.7.2 ensured the platform doesn't crash itself; this spec ensures the platform actually accelerates the work.
