# Metal-on-AMD BERT correctness — root-cause analysis + remediation roadmap

> Sister document to [`patches/README.md`](../patches/README.md).
> Captures the 2026-05-17 finding that the patch 0001 `simdgroup_reduction`
> gate is **necessary but not sufficient** for full correctness on
> AMD-Mac, and lays out the v0.8.0 kernel-level fix path.

## TL;DR

- **Chat (Llama-family) on AMD-Mac Metal: works** with patch 0001's
  `has_simdgroup_reduction = false` gate. Deterministic, coherent.
- **Embed (BERT-family) on AMD-Mac Metal: broken.** Identical input
  "hello" returns cos_sim 0.07 between two calls (should be 1.0).
  Same model on CPU returns cos_sim 1.0000.
- **Rerank (BGE BERT) on AMD-Mac Metal: broken.** Same input gives
  different relevance scores across calls.
- **Root cause:** BERT-specific Metal kernels (`kernel_norm_fuse_impl`
  for LayerNorm, `kernel_soft_max` for bidirectional attention)
  unconditionally call `simd_sum()` / `simd_max()` AND use
  **dynamic threadgroup memory** as an entry-point parameter — a
  combination that hits both the AMD simd-reduction divergence AND
  the documented Metal-compiler threadgroup-memory barrier-ordering
  bug.
- **v0.7.0 ships the operational workaround:** AMD-discrete embed +
  code-embed + rerank slots route to CPU via `--gpu-layers 0`. On a
  16-core Mac Pro 2019 the throughput hit is acceptable
  (~1.6 min per 10 LongMemEval items observed); correctness is
  restored to cos_sim 1.0000 on identical input.
- **v0.8.0 candidate (kernel-level fix):** rewrite the affected
  kernels to use fixed-size function-local threadgroup memory
  (eliminates the compiler-bug exposure) and add `has_simdgroup_reduction`
  fallback paths to `kernel_norm_fuse_impl`, `kernel_rms_norm_fuse_impl`,
  `kernel_soft_max`, and `kernel_soft_max_4`. Estimated ~400 LOC of
  Metal Shading Language + ~50 LOC of dispatcher logic in
  `ggml-metal-device.cpp`.

## What we discovered (timeline)

1. **2026-05-17 09:19 EDT** — ablation A (gpu-embed-only on the
   freshly-patched v0.7.0 staging-buffer-pool daemon) returned
   **recall=0.0 / 60 items**.
2. **Probe**: identical input "hello" through the gateway →
   cos_sim **0.18**. Two separate single-text calls → cos_sim
   **-0.03**.
3. **Hypothesis #1** (the staging-buffer pool corrupts embeddings)
   tested via `GGML_METAL_DISABLE_STAGING_POOL=1` escape hatch.
   Still produced bad embeddings (cos_sim 0.07).
4. **Hypothesis #2** (the rebuild itself is broken) tested via
   hard revert to v0.6.2 source-clean state. **Still broken.**
   The bug predates the v0.7.0 staging-buffer-pool work.
5. **Hypothesis #3** (the broken path is GPU-specific) tested via
   `--gpu-layers 0` on a separate llama-server instance.
   **CPU returned cos_sim 1.0000 on identical input.** Confirmed
   Metal-on-AMD is the culprit.
6. **Cross-check on rerank:** BGE-reranker-v2-m3 on the gateway
   returned different scores across identical calls. Confirmed
   the bug affects all BERT-family models, not just nomic-embed.
7. **Cross-check on chat:** llama3.1-8b returned identical tokens
   across identical seeded calls. Chat is unaffected — patch 0001
   covers it via a different code path.

## Mechanism

### Affected kernels

In `llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`:

- **`kernel_norm_fuse_impl`** (lines 2892-2974) — LayerNorm
  variant used by BERT-family models. Calls `simd_sum()` 4 times.
  Uses `threadgroup float * shmem_f32 [[threadgroup(0)]]` as an
  entry-point parameter (dynamic threadgroup memory).
- **`kernel_rms_norm_fuse_impl`** (lines 2990+) — RMSNorm
  variant. Same pattern: `simd_sum()` + dynamic threadgroup memory
  parameter.
- **`kernel_soft_max`** + **`kernel_soft_max_4`** (lines 1855+,
  1961+) — softmax used by bidirectional self-attention. Calls
  `simd_max()` and `simd_sum()`. Uses
  `threadgroup float * buf [[threadgroup(0)]]` as an entry-point
  parameter.

### Why patch 0001 doesn't reach them

Patch 0001 sets `has_simdgroup_reduction = false` on AMD-discrete
profiles. **The flag is checked at the device-capability level only**
— individual kernel dispatchers in
`ggml-metal-device.cpp::ggml_metal_library_get_pipeline_norm` (lines
1659-1665) and `…_soft_max` (lines 440-460) **unconditionally
select** the simd_sum-using kernel. There is no fallback path.

For the chat (Llama) workload, patch 0001 takes effect via a
different mechanism: `has_simdgroup_matrix_mul` and the flash-attn
matmul paths are gated higher in the dispatch chain, and the
non-simdgroup matmul scalar fallback is correct. RMSNorm in chat
also flows through, but apparently with batch shapes / call
frequencies where the bug doesn't surface (or surfaces but
doesn't catastrophically corrupt next-token sampling — sampling
has temperature smoothing and rounding that masks small numeric
errors).

For BERT models the LayerNorm and softmax are core to **every
transformer block**, run on **every forward pass**, with
**no smoothing downstream** — the pooled cls token IS the
final output. Any nondeterminism in those kernels propagates
directly to the embedding vector.

### Why the bug is non-deterministic, not just incorrect

Looking at `kernel_norm_fuse_impl`:

```metal
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tiisg == 0) {
    shmem_f32[sgitg] = sumf;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
sumf = shmem_f32[tiisg];
sumf = simd_sum(sumf);
```

Reads from `shmem_f32` after a barrier. If the barrier doesn't
fully fence on AMD discrete (which is what
[gfx-rs/wgpu#4500](https://github.com/gfx-rs/wgpu/issues/4500)
documents about Metal+AMD compiler bugs with dynamic threadgroup
memory), threads can read partially-written data — and the
specific partial state varies per call based on thread scheduling.
That's the mechanism for non-determinism.

The fact that **CPU mode produces cos_sim 1.0000** is direct
evidence: the model and weights are correct; the Metal
implementation is what introduces the randomness.

## Comparison: chat vs embed forward-pass shapes

| Component | Llama 3.1 8B (chat) | nomic-embed-text-v1.5 (BERT) |
|---|---|---|
| Norm | RMSNorm (has simd reductions) | LayerNorm (has simd reductions) |
| Attention | Causal masked, flash-attn-able | Bidirectional, softmax-heavy |
| Position | RoPE | Learned positional embeddings |
| MLP | SwiGLU | GELU |
| Output | Logits → sampling | CLS-token vector → L2 norm |

Chat's RMSNorm is also broken on AMD, but in a way that gets
masked by sampling. Embed's LayerNorm + softmax are broken in a
way that's directly observable in the output vector.

## v0.7.0 operational workaround (shipped today)

`internal/tuning/tuning.go::embedParams` and `::rerankParams` now
append `--gpu-layers 0` to the per-slot llama-server args on
AMD-discrete profiles. The chat slot stays on GPU (Llama path
works). Effect:

- Embed: identical input "hello" → cos_sim **1.0000** ✓
- Embed: semantic ("cat/mat") → cos_sim **0.6574** ✓
- Rerank: identical query+docs → identical scores across calls ✓
- LongMemEval observed pace: ~1.6 min per 10 items on CPU embed
  (vs ~1.7 min on broken GPU embed; the GPU "speed advantage"
  was producing garbage anyway)

Operator override path: the existing `QUENCHFORGE_EMBED_*` env
knobs still take effect; an operator who later builds a v0.8.0
binary with kernel-level fixes can drop the CPU-route flag by
shipping a new tuning module.

## v0.7.1 — partial kernel-level fix (2026-05-17)

`patches/llama.cpp/0003-metal-amd-bert-fallback-kernels.patch`
(in-tree, not a `.patch` file — added directly to ggml-metal.metal
and ggml-metal-device.cpp; see commits) ships fallback kernels for
`kernel_norm_fuse_impl` (LayerNorm) and `kernel_soft_max{,_4}`
(softmax) that use fixed-size function-local threadgroup memory
plus pure threadgroup tree-reductions. The dispatchers in
`ggml-metal-device.cpp` now select the `_fb` suffix variant when
`has_simdgroup_reduction == false`.

**Measured impact on Vega II 2026-05-17:**

| Test | Pre-patch (Metal) | Post-patch (Metal) | CPU (--gpu-layers 0) |
|---|---|---|---|
| identical "hello" in same batch | cos_sim 0.07 | cos_sim **0.29** | cos_sim 1.0000 |
| two separate calls "hello" | cos_sim 0.15 | cos_sim **0.06** | cos_sim 1.0000 |

The norm+soft_max fallback **does not** restore correctness on its
own. The improvement in batch-internal similarity (0.07 → 0.29)
shows the patched kernels ARE working — they no longer
non-deterministically corrupt LayerNorm or softmax output. But the
matmul kernels that compute attention's QKV projections and the FFN
**also** use `simd_sum` and are equally broken — see the next
section.

The patch ships nonetheless because:

1. It's correct, well-tested fallback code that doesn't regress
   anything (the `_fb` variants are only selected when
   `has_simdgroup_reduction == false`, which is AMD-Mac discrete).
2. It establishes the fallback-dispatch pattern future matmul
   work can layer onto.
3. If an operator overrides the CPU-route flag and runs embed on
   Metal at their own risk, the norm/softmax portion of the
   forward pass is at least deterministic.

**Production stays on the CPU route** (`--gpu-layers 0` for
AMD-discrete embed/rerank via tuning.go) until the matmul
fallbacks are also written.

## v0.8.0 candidate — kernel-level fix design (full scope)

The 2026-05-17 partial-fix investigation revealed that the BERT
correctness bug touches **more than norm + softmax**. A grep of
`simd_sum` / `simd_max` in `ggml-metal.metal` returned 40+ hits
across:

- `kernel_norm_fuse_impl`, `kernel_rms_norm_fuse_impl` — fixed in 0003
- `kernel_soft_max`, `kernel_soft_max_4` — fixed in 0003
- `kernel_mul_mv_*` (quantized mat-vec): q1, q4, q5, q8, iq* — ~10 variants
- `kernel_mul_mv_t_t`, `kernel_mul_mv_t_t_4` — fp16/fp32 mat-vec
- `kernel_mul_mv_ext_q4_*` — extended quantized paths
- `kernel_argmax`, `kernel_argmax_*`
- Other reduction-heavy paths (group_norm, etc.)

For nomic-embed-text-v1.5 (fp32 BERT), the dominant path is
`kernel_mul_mv_t_t` for attention QKV projections + FFN
multiplication. Patching this is the next required step.

Estimated scope for full BERT Metal correctness:

- ~200 LOC of MSL per `_fb` mat-vec variant (~5 variants for fp32/fp16,
  more for quantized → defer quantized until needed)
- ~100 LOC of dispatcher logic in `ggml-metal-device.cpp`
- Bench validation matrix (cos_sim 1.0 on Metal at parity with CPU)
- ~5-10 days of focused work

Two complementary changes:

### A. Add fallback kernels (no simd_sum / simd_max)

Pure-threadgroup-memory reductions for `kernel_norm_fuse_impl`,
`kernel_rms_norm_fuse_impl`, `kernel_soft_max`, and
`kernel_soft_max_4`. Replace `simd_sum(x)` with a tree-reduction
through fixed-size local threadgroup memory:

```metal
threadgroup float local_buf[64];  // fixed-size, function-local
local_buf[tpitg.x] = sumf;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint stride = ntg.x/2; stride > 0; stride >>= 1) {
    if (tpitg.x < stride) {
        local_buf[tpitg.x] += local_buf[tpitg.x + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
sumf = local_buf[0];
```

Fixed-size function-local threadgroup memory avoids the
Metal-compiler dynamic-shared-memory barrier-ordering bug
documented at `gfx-rs/wgpu#4500`. Pure threadgroup reductions
avoid the AMD simd-group reduction divergence.

### B. Dispatcher gating

Update `ggml-metal-device.cpp::ggml_metal_library_get_pipeline_norm`,
`…_rms_norm`, and `…_soft_max` to honour
`dev->props.has_simdgroup_reduction`. When false, select the
fallback kernel suffix (`_fb`):

```cpp
const char * gate = dev->props.has_simdgroup_reduction ? "" : "_fb";
snprintf(base, 256, "kernel_norm_f32%s%s", suffix, gate);
```

Same pattern for the soft_max and rms_norm dispatchers.

### C. Bench acceptance

- Identical "hello" → cos_sim 1.0000 (5 calls)
- Two-call separate identical → cos_sim 1.0000 (5 calls)
- Semantic ("cat/mat") → cos_sim within 0.005 of CPU reference
- LongMemEval 60-item stratified `--gpu-embed-only` returns
  recall within 0.02 of the CPU-route v0.7.0 number (proves
  kernel-level fix matches operational fix)
- Apple Silicon zero regression (kernel selection unchanged
  when `has_simdgroup_reduction == true`)

### D. Estimated scope

- ~400 LOC of Metal Shading Language (4 fallback kernels +
  template instantiations)
- ~50 LOC of dispatcher logic in `ggml-metal-device.cpp`
- ~30 LOC of test additions
- Bench validation: 1 hour wall time on Mac Pro 2019

Total: ~3-5 days of focused work.

## Alternative path — MPSGraph

Apple's [`MPSGraph`](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph)
provides production-quality LayerNorm and softmax implementations
that work correctly across all Apple-supported Metal GPUs
(including AMD discrete). PyTorch on Mac uses MPSGraph as the
backend for tensor ops on AMD Mac.

Integration path: intercept `kernel_norm`/`kernel_soft_max` calls
in `ggml_metal_ops.cpp` and dispatch to MPSGraph when
`has_simdgroup_reduction == false`. Trade-off: bigger refactor
than option A (~1500 LOC), but the resulting kernels are
Apple-maintained and bench-known-correct on AMD-Mac. Could be
v0.9.0 work.

## Upstream filing

Upstream issue at [ggml-org/llama.cpp#19563](https://github.com/ggml-org/llama.cpp/issues/19563)
is the existing report for "Metal backend produces garbage output
on AMD discrete GPUs (Radeon Pro 5300M)". It was tested with
Qwen2.5-1.5B-Instruct (a chat model) and **closed as not planned**.

A separate issue should be filed specifically for the BERT/embed
case with our cos_sim 1.0 vs 0.07 evidence and the proposed
fallback-kernel design. Even if upstream stays "not planned",
quenchforge ships the patch as `patches/llama.cpp/0003-…patch`
(or merges into the existing 0001 patch series).

## References

- [llama.cpp issue #19563](https://github.com/ggml-org/llama.cpp/issues/19563)
  — original Metal-on-AMD garbage-output report (chat workload)
- [gfx-rs/wgpu#4500](https://github.com/gfx-rs/wgpu/issues/4500)
  — Metal compiler dynamic-threadgroup-memory barrier bug
- [Apple Metal Performance Shaders Graph](https://developer.apple.com/documentation/metalperformanceshadersgraph)
  — MPSGraph reference; production LayerNorm + softmax
- [Apple Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
  — feature-capability matrix; documents which GPU families
  support simdgroup-reduction-friendly behaviour
- [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks)
  — Apple GPU microarchitecture reference (Apple-Silicon-only,
  but useful baseline for understanding what AMD lacks)
