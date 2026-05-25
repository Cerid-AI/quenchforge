# 2026-05-25 — AMD-discrete GPU mode revival: pool patch + concurrency-disable

**Status:** approved 2026-05-25. Implementation gated by Phase 4 bench validation.

**Supersedes:** `2026-05-25-amd-metal-acceleration-design.md` (the wave-width spec). The earlier spec correctly identified Apple Silicon vs AMD wave-width asymmetry as a real architectural issue but was wrong about it being the operational root cause; today's experiments (validated `GGML_METAL_CONCURRENCY_DISABLE=1` env var) isolated the actual correctness bug elsewhere. The earlier spec is preserved as a historical research artifact — its prior-art survey, Metal Shading Language spec citations, and AMD GCN5 architecture documentation remain useful.

**Driver:** v0.7.2 ships with all four quenchforge slots forced to CPU (`--gpu-layers 0`) on AMD-discrete profiles. The Vega II's 32 GB HBM2 is idle. Embed throughput is bound to ~85 req/s on the Xeon W's 16 cores; chat throughput is bound to ~7.4 tok/s. Today's experiments show both correctness and stability bugs blocking GPU mode are well-understood and have implementations in hand — one shipping (the env var), one parked as a working draft awaiting re-validation (patch 0002).

**Canonical hardware:** Mac Pro 7,1 + Radeon Pro Vega II 32 GB HBM2 (gfx906, 64-wide waves, `MTLGPUFamilyMac2`, `hasUnifiedMemory: false`). macOS 26.5, Metal 4.

---

## Problem

Three distinct bugs block GPU mode on AMD-discrete profiles. Empirically validated today:

### Bug A — `MTLDispatchTypeConcurrent` race on non-UMA

llama.cpp's Metal backend dispatches command buffers concurrently by default. On non-UMA Macs the dispatch synchronization is unreliable, producing non-deterministic output across BERT-family embeddings (observed: cos_sim **−0.011 cross-call** on nomic, vs 1.0000 on CPU). Documented in [llama.cpp issue #19563](https://github.com/ggml-org/llama.cpp/issues/19563) with the workaround env var `GGML_METAL_CONCURRENCY_DISABLE=1`. **Validated fix:** setting the env var produces cos_sim 1.000000 across all 4 production models (nomic, jina, bge, llama3.1-8b chat) — see Phase 1 experiment results in this session's transcript.

### Bug B — wave-width assumption (latent, not operational)

The hardcoded `#define N_SIMDWIDTH 32` in `ggml/src/ggml-metal/ggml-metal.metal:28` is technically incorrect per Apple's MSL Spec §4.4.2: Vega II uses 64-wide simdgroups, confirmed today via direct probe (`MTLComputePipelineState.threadExecutionWidth: 64`). However, this is not the cause of observed correctness failures — those trace fully to Bug A. The wave-width hypothesis (the earlier 4-patch spec) addressed a real but **non-load-bearing** issue. Not in scope for this work.

### Bug C — `newBufferWithBytesNoCopy` exhaustion (the family-B SIGABRT)

`ggml_metal_buffer_set_tensor` and `_get_tensor` at [ggml-metal-device.m:1658](../../../llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:1658) and [:1715](../../../llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:1715) each call `newBufferWithBytesNoCopy` with `MTLResourceStorageModeShared` per invocation. On Apple Silicon `buf->is_shared` short-circuits this path entirely (memcpy fast path). On AMD discrete, `MTLResourceStorageModeShared` goes through IOSurface-backed PCIe-mapped memory, and every call registers a new IOMMU page-table entry. The driver's bounded mapping pool (~256–512 active slots per Apple's documentation) exhausts under sustained sub-millisecond cadence; subsequent calls return nil; `GGML_ASSERT(buf_src)` / `GGML_ASSERT(buf_dst)` SIGABRTs the process.

**Validated today:** Even with Bug A fixed (env var on), sustained-load bench triggered SIGABRT at all three test slots — nomic at ~5 min (98 reqs), jina at ~2.5 min (16 reqs), chat at ~1 min. Adding the documented AMD safety knobs (`ubatch=1024`, `GGML_METAL_N_CB=1`) extended nomic to 585 reqs before crash — still crashes, just slower. The crash class is real and persistent.

**Existing fix:** `patches/llama.cpp/drafts/0002-metal-staging-buffer-pool.patch.broken` — designed, drafted, briefly promoted to live (commit `0b0e7fa`), bench-validated at 1597 sustained-embed calls / 3 min / 0 SIGABRTs (commit message). Parked back to drafts (commit `533c60d`) because Bug A was masking the signal — operators couldn't tell if the patch fixed family-B or if "no crash" meant "garbage vectors but stable." With Bug A now isolated to the env var, the parked patch is unblocked.

---

## Goals

1. **Stable GPU mode on AMD-discrete profiles** for the four production models cerid runs:
   - `nomic-embed-text-v1.5` (embed slot, port 11501)
   - `jina-embeddings-v2-base-code` (code-embed slot, port 11506)
   - `bge-reranker-v2-m3` (rerank slot, port 11502)
   - `llama3.1-8b` Q4_K_M (chat slot, port 11500)
2. **Measurable speedup** validated by side-by-side numbers (today's smoke showed nomic at ~3.25 req/s GPU vs ~0.5 req/s CPU baseline — ~6× floor). Acceptance: any speedup > 1× passes; target ≥ 3× for embed/rerank.
3. **Zero regression on Apple Silicon** — the patch's UMA fast-path short-circuit ensures the pool code never runs on Apple Silicon. Validated by `go test ./...` on `macos-latest` (arm64) CI.
4. **Production stability** — 7 days of running with all three GPU slots, ≤ 10 supervisor AutoRespawn events / week (vs 257/week baseline from May 17–24), zero kernel panics.

## Non-goals

- **Linux/CUDA/ROCm/Vulkan support.** Per CLAUDE.md absolute rule #2.
- **RDNA1/RDNA2/W6800X bench validation.** Vega II is the canonical target; other AMD profiles inherit code via the same gating but aren't bench-validated here.
- **Wave-width kernel rewrite.** Real architectural issue per the superseded spec, but not load-bearing. Defer until evidence shows a correctness or perf gap traceable to wave-width specifically.
- **Apple Silicon performance work.** Patch's UMA fast path keeps Apple Silicon on the existing memcpy route.
- **Image-gen / TTS / Whisper slots.** Out of scope; those slots have their own GPU policies.
- **Upstream PR submission.** Per prior-art survey, every AMD-Metal PR in the last 12 months has been closed unmerged. Carry locally per CLAUDE.md mission ("useful artifact for the community").

---

## Architecture

Two-layer fix. Each layer is independently testable; together they enable GPU mode.

### Kernel layer — patch 0002

Bounded `MTLBuffer` pool replaces per-call `newBufferWithBytesNoCopy` in `ggml_metal_buffer_set_tensor` and `ggml_metal_buffer_get_tensor`. One pool buffer = one IOMMU registration on AMD discrete, reused across calls. Driver's mapping pool never exhausts.

**Design (unchanged from drafts/0002-...patch.broken):**

```c
// New, in ggml-metal-device.m near the top:
#define QF_STAGING_PER_CLASS_CAP 4
#define QF_STAGING_MIN_CLASS_LOG2 12  // 4 KiB
#define QF_STAGING_MAX_CLASS_LOG2 26  // 64 MiB

static NSLock * qf_staging_lock = nil;
static NSMutableDictionary<NSNumber *, NSMutableArray<id<MTLBuffer>> *> *
    qf_staging_pool = nil;

static void qf_staging_init_once(void);
static size_t qf_staging_class(size_t requested);
static bool qf_staging_disabled(void);
static id<MTLBuffer> qf_staging_acquire(id<MTLDevice> device, size_t size);
static void qf_staging_release(id<MTLBuffer> buf);
```

**Properties:**
- 15 power-of-two size classes from 4 KiB to 64 MiB
- Per-class FIFO of free buffers, capped at 4 (worst case: ~512 MiB total pool footprint, log-bounded by class count)
- Single `NSLock` around the dictionary (the staging path is already serialised behind `cmd_buf waitUntilCompleted`; lock contention is not on the critical path)
- `dispatch_once_t` lazy init on first acquire
- Apple Silicon never enters this code path (`buf->is_shared` short-circuits in both functions before any pool access)
- Operator escape hatch: `GGML_METAL_DISABLE_STAGING_POOL=1` reverts to upstream behavior for A/B testing or emergency rollback

**Modified call sites:**

`set_tensor` flow:
```
acquire pool buffer (or alloc on miss)
  → memcpy data into pool buffer
  → blit pool buffer → destination via MTLBlitCommandEncoder
  → waitUntilCompleted (must wait before release; GPU still reading)
  → release pool buffer back to bucket
```

`get_tensor` flow:
```
acquire pool buffer (or alloc on miss)
  → blit source → pool buffer
  → waitUntilCompleted
  → memcpy pool buffer → caller's data
  → release pool buffer
```

Both flows switched from the existing semaphore + `addCompletedHandler` async pattern to synchronous `waitUntilCompleted` so the pool release happens at a well-defined point. Apple's comment on the original code path notes the async/sync perf delta is negligible.

### Supervisor layer — tuning.go + slotEnv

Routes the env var fix to AMD-discrete slots and re-enables GPU layers.

**New field on `SlotTuning`:**
```go
// MetalConcurrencyDisable, when true, sets GGML_METAL_CONCURRENCY_DISABLE=1
// in the slot's env. Required on AMD discrete (non-UMA Metal): upstream's
// MTLDispatchTypeConcurrent path's command-buffer ordering is unreliable on
// non-UMA drivers and causes non-deterministic output across BERT-family
// models and chat-decode races. See llama.cpp issue #19563.
MetalConcurrencyDisable bool
```

**Three branches updated** (`chatParams`, `embedParams`, `rerankParams`), all in the `profileIsAMDDiscrete(profile)` true case:
- `"--gpu-layers", "0"` → `"--gpu-layers", "999"` (re-enable GPU)
- Add `t.MetalConcurrencyDisable = true`
- `AutoRespawn: true` retained as defense in depth for unknown crash classes

**embedParams specifically** also re-enables `ubatch = amdEmbedUbatchDefault` (1024) for the AMD-discrete branch. The current code sets ubatch to `cfg.MaxContext` (8192) because the v0.7.0 CPU-route comment said "the 1024 cap no longer applies on CPU." On GPU it applies again — keeps Metal staging-buffer pressure bounded per CLAUDE.md operational gotcha #2.

**slotEnv extension** in `cmd/quenchforge/main.go`:
```go
env := []string{fmt.Sprintf("GGML_METAL_N_CB=%d", ncb)}
if tn.MetalConcurrencyDisable {
    env = append(env, "GGML_METAL_CONCURRENCY_DISABLE=1")
}
return env
```

**Deliberately NOT added** (per CLAUDE.md anti-overengineering): a `QUENCHFORGE_AMD_FORCE_CPU` or `QUENCHFORGE_AMD_GPU_LAYERS` operator escape hatch. AutoRespawn handles unknown crash classes; `GGML_METAL_DISABLE_STAGING_POOL=1` handles pool-specific issues; binary rollback to v0.7.2 handles everything else.

---

## Patch revision details

The parked draft (`drafts/0002-metal-staging-buffer-pool.patch.broken`) is implementation-complete but needs two fixes before it applies cleanly on the current `0001/0003/0004` series:

1. **Strip duplicate-of-0001 hunks.** Draft lines 167-194 re-emit patch 0001's `has_simdgroup_reduction` / `has_bfloat` gating logic. The draft was generated against an unpatched submodule; on top of 0001 these hunks have no valid `-` context. Action: delete those ~30 patch lines; keep only the pool helpers (lines 67-158) and the `set_tensor`/`get_tensor` body changes (lines 198-311).

2. **Regenerate file indices.** Draft's `index XXX..YYY` lines are from May 17 submodule state. After applying the trimmed patch on a clean `0001/0003/0004`-applied submodule, run `git format-patch HEAD~1 --stdout -1` to regenerate against the actual current tree.

**No code changes to the pool internals.** The design is intact; only the diff metadata needs refreshing.

**Known limitation kept out of scope:** pool is process-global (one `NSMutableDictionary` shared across all `MTLDevice` instances). Vega II Duo (4 GPUs from one physical card) would see cross-device buffer leakage if anyone ran multi-GPU on a Mac Pro. Not the canonical target; flag in `patches/README.md`; per-device pool refactor revisits only if a Duo user reports an issue.

---

## Bench validation protocol

Four-criterion gate before promoting the patch from drafts to live. All must pass on Mac Pro 2019 + Vega II before tagging `v0.8.0-rc2`.

### Criterion 1: Correctness gate

Short probes (~30s/model) against each production slot running on GPU with the full safety profile (env var + ubatch=1024 + GGML_METAL_N_CB=1 + patch 0002 applied):

| Bench | Model | Threshold |
|---|---|---|
| `bench-bert-correctness.py --n-calls 10 --epsilon 1e-4` | nomic-embed-text-v1.5 | cos_sim ≥ 0.9999 same-batch + cross-call |
| `bench-bert-correctness.py --n-calls 10 --epsilon 1e-4` | jina-embeddings-v2-base-code | same |
| `bench-bert-correctness.py --rerank --n-calls 10` | bge-reranker-v2-m3 | identical scores across 10 calls |
| `bench-llama-correctness.py` (new) | llama3.1-8b Q4_K_M | 3 identical responses to fixed prompt at temp=0 |

Already validated today for the env-var alone (without patch 0002). Re-running on patched binary confirms patch doesn't introduce new correctness regressions.

### Criterion 2: Stability gate

30-minute sustained-load benches per slot type:

| Bench | Slot | Pass criteria |
|---|---|---|
| `bench-bert-sustained-load.py --duration 1800 --concurrency 4 --batch-size 4` | nomic embed | 0 SIGABRT, no 5xx burst, drift cos_sim ≥ 0.999, no latency cliff (late p95 ≤ 5× early p95), RSS ≤ 2× initial |
| (same) | jina code-embed | same |
| `bench-llama-sustained-load.py --duration 1800 --concurrency 2` | llama3.1-8b chat | same + 3 identical responses to deterministic probes |

bge-rerank covered by encoder-family overlap with nomic+jina (same BERT encoder kernels).

### Criterion 3: Escape-hatch gate

Confirms the patch is what's doing the work, not an unrelated env change:

```bash
GGML_METAL_DISABLE_STAGING_POOL=1 \
  bench-bert-sustained-load.py --duration 300 --model nomic-embed-text-v1.5
```

Must **FAIL within ~5 minutes** with family-B SIGABRT. If this passes (doesn't crash), something else is providing stability and the patch's contribution is unverified.

### Criterion 4: Apple Silicon zero-regression

`go test ./...` passes on `macos-latest` (arm64) CI runner. Spot check on a separate M-series test machine: throughput delta ≤ 1% vs v0.7.2 baseline.

### Required bench harness work

Three small additions to ship before Phase 4:
- Promote `/tmp/bench-llama-sustained-load.py` (written today, ~290 lines) → `scripts/bench-llama-sustained-load.py`
- New `scripts/bench-llama-correctness.py` (~150 lines, mirrors `bench-bert-correctness.py` shape; uses `/v1/chat/completions` instead of `/v1/embeddings`)
- Verify `bench-bert-sustained-load.py` works at `--batch-size 4` (smaller chunks to fit the 1024 ubatch on GPU); add a flag if defaults don't accommodate

---

## Rollout plan

Six sequential phases, each independently revertible.

### Phase 1: Patch revival (~1 hour)
1. `git mv patches/llama.cpp/drafts/0002-metal-staging-buffer-pool.patch.broken patches/llama.cpp/0002-metal-staging-buffer-pool.patch`
2. Strip duplicate-of-0001 hunks (lines 167-194)
3. `bash scripts/apply-patches.sh --reset && bash scripts/apply-patches.sh`
4. Regenerate indices: `cd llama.cpp && git format-patch HEAD~1 --stdout -1 > ../patches/llama.cpp/0002-metal-staging-buffer-pool.patch`
5. Commit: `feat(patches): 0002-metal-staging-buffer-pool — kernel fix for family-B SIGABRT (v0.8.0)`

### Phase 2: tuning.go + main.go (~1 hour)
1. Add `MetalConcurrencyDisable bool` to `SlotTuning`
2. Flip three AMD-discrete branches: `--gpu-layers 0` → `999`, add `MetalConcurrencyDisable: true`
3. Re-enable `ubatch = amdEmbedUbatchDefault` in `embedParams` AMD branch
4. Extend `slotEnv` to emit `GGML_METAL_CONCURRENCY_DISABLE=1`
5. Update existing tests + add `TestSlotEnv_AMDIncludesConcurrencyDisable`; `go test ./...` passes
6. Commit: `feat(tuning): re-enable AMD-discrete GPU mode (env-var + patch 0002)`

### Phase 3: Bench harness (~30 min)
1. Promote `/tmp/bench-llama-sustained-load.py` → `scripts/bench-llama-sustained-load.py`
2. Write `scripts/bench-llama-correctness.py`
3. Commit: `feat(scripts): bench-llama-{correctness,sustained-load} for chat slot validation`

### Phase 4: Build + bench validation gate (~3 hours)
1. `bash scripts/build-llama.sh && make build && sudo install -m 0755 bin/quenchforge /usr/local/bin/`
2. `launchctl kickstart -k gui/$(id -u)/com.cerid.quenchforge`
3. Run all four bench criteria from §"Bench validation protocol" in sequence
4. If any gate fails → `git revert` Phase 2 commit (Phase 1 patch stays inert), root-cause, iterate
5. If all pass → tag `v0.8.0-rc2`

### Phase 5: Production observation window (7 days)
1. Run `v0.8.0-rc2` on the Mac Pro 7,1 production install
2. Daily check via `quenchforge doctor` + slot-log review
3. **Acceptance:** ≤ 10 AutoRespawn events / week, zero kernel panics, zero `vm_page_wire` events, cerid-AI eval pass rate stable, doctor non-PASS count stable
4. After 7 clean days: tag `v0.8.0`, update `CHANGELOG.md` final entry, ship Homebrew formula update

### Phase 6: Documentation + memory cleanup (~2 hours, overlapping Phase 5)
1. `patches/README.md` — Section 3 updated from "patch #2 (v0.7.0)" to "SHIPPED v0.8.0" with new bench numbers
2. `docs/METAL_AMD_BERT_CORRECTNESS.md` — corrected root-cause analysis (concurrent-dispatch + buffer-pool, not wave-width)
3. Memory updates: `project_cerid_quenchforge_chat_on_cpu.md` rewritten; `feedback_quenchforge_safety.md` operational status update
4. Spec disposition: header added to `2026-05-25-amd-metal-acceleration-design.md` marking it superseded by this spec; preserve as historical research artifact

**Rollback paths:**
- Bench failure in Phase 4: `git revert` Phase 2 commit, keep Phase 1 patch inert
- Regression in Phase 5: `launchctl bootout` + downgrade to v0.7.2 binary
- Pool-specific failure: `GGML_METAL_DISABLE_STAGING_POOL=1` env var (tactical only; family-B returns)

**Total wall-clock estimate:** ~7 hours focused work (Phases 1-4) + 7 days observation (Phase 5) + ~2 hours docs (Phase 6, overlapping).

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Pool patch doesn't actually fix family-B on bench (May 17 bench confounded by Bug A) | Medium | High | Phase 4 bench is the gate; if SIGABRT recurs at 30-min mark, revert Phase 2 and investigate pool sizing or cap=4 assumption |
| Pool footprint exceeds 500 MiB worst-case | Low | Low | drafts/README.md "failure modes": drop cap to 2. Add `max_pool_bytes` logging in v0.8.1 if observed |
| Apple Silicon regression — patch code runs on UMA paths | Very Low | Low | `buf->is_shared` short-circuits before reaching pool; `go test ./...` on macOS arm64 CI catches anything else |
| CPU thread count of 15 causes oversubscription in GPU mode | Low | Low | Threads mostly idle in GPU mode; perf tweak deferred to v0.8.1 if benchmarks show degradation |
| Vega II Duo (4 GPUs from one card) hits the global-pool cross-device bug | Low | Medium (for affected users) | Not the canonical target; flag in patches/README.md; per-device pool refactor revisits if a Duo user reports |
| Apple changes Metal driver semantics in macOS 27 | Low | Medium | Patches use stable Metal APIs; env var checked at runtime by ggml; revisit only if spec/driver behavior changes |
| 7-day observation surfaces a new failure class | Medium | Medium | Rollback to v0.7.2 binary is one `launchctl bootout`; CPU route stays available |
| Patch upstream submission rejected | High | Low | Carry locally indefinitely per CLAUDE.md mission; no production impact |

---

## Acceptance criteria

Implementation is complete when:

1. `patches/llama.cpp/0001`, `0002`, `0003`, `0004` all apply cleanly via `bash scripts/apply-patches.sh --check`
2. `bench-bert-correctness.py` PASSES on all 3 BERT models against GPU-routed slots on Vega II (cos_sim ≥ 0.9999 same-batch + cross-call)
3. `bench-llama-correctness.py` PASSES on `llama3.1-8b` Q4_K_M (3 identical responses to fixed temp-0 prompt)
4. `bench-bert-sustained-load.py --duration 1800` PASSES on nomic + jina (zero family-B SIGABRTs, no 5xx burst, no drift to FAIL threshold, no latency cliff)
5. `bench-llama-sustained-load.py --duration 1800` PASSES on chat (same)
6. Escape-hatch test: `GGML_METAL_DISABLE_STAGING_POOL=1 bench-bert-sustained-load.py --duration 300` FAILS within ~5 min with family-B SIGABRT (confirms patch is doing the work)
7. `go test ./...` passes on both `macos-latest` (arm64) and self-hosted `[amd-gpu]` Vega II runners
8. ≥ 7 days of production uptime under new GPU defaults with no kernel panic, no `vm_page_wire` event, ≤ 10 supervisor AutoRespawn events / week
9. `CHANGELOG.md` v0.8.0 entry documents the change + bench numbers; `patches/README.md` Section 3 reflects SHIPPED state with current bench table
10. Memory updated (`project_cerid_quenchforge_chat_on_cpu.md` rewritten); spec `2026-05-25-amd-metal-acceleration-design.md` marked superseded

---

## Out of scope (deferred)

- Per-device pool refactor for Vega II Duo (waiting on a Duo user report)
- `max_pool_bytes` observability instrumentation (defer to v0.8.1 if Phase 4 bench shows pool footprint > 500 MiB)
- `--threads 4` perf tweak for GPU mode (currently kept at 15; revisit if benchmarks show CPU+GPU contention)
- Operator escape hatch env var (`QUENCHFORGE_AMD_FORCE_CPU`)
- Upstream PR submission (low ROI per prior-art research; close-rate ~0 in 2025–2026)
- Wave-width rewrite (real architectural issue but not load-bearing; revisit if a perf or correctness gap traces specifically to it)

---

## References

**Today's session validation evidence:**
- Hardware probe: `MTLComputePipelineState.threadExecutionWidth = 64` on Vega II
- Correctness validation: cos_sim 1.000000 on all 4 production models with `GGML_METAL_CONCURRENCY_DISABLE=1` (vs −0.011 baseline)
- Stability validation: family-B SIGABRT reproduced at all 3 slots even with env var + AMD safety profile; crash sites confirmed at `ggml-metal-device.m:1658` and `:1715`

**Prior commits:**
- `1d1dc56` — initial draft of 0002 patch
- `0b0e7fa` — promoted to live; bench-validated 1597 sustained-embed calls
- `533c60d` — parked back to drafts/.broken pending Bug A resolution
- `2422ef5` — v0.7.2 stability work (log rotation, pre-bind check, doctor extensions)

**Upstream references:**
- [llama.cpp issue #19563](https://github.com/ggml-org/llama.cpp/issues/19563) — Metal correctness on AMD discrete (Bug A)
- [llama.cpp PR #20615](https://github.com/ggml-org/llama.cpp/pull/20615) — closed: Metal buffer allocation for discrete GPUs (analogous to Bug C fix)
- [Apple MSL Specification v4 §5.2.3.6](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) — `threads_per_simdgroup` semantics
- Apple "Metal Best Practices Guide" § "Manage Memory Allocation Performance" — buffer pool recommendation

**Internal references:**
- `patches/README.md` Section 3 — "Sustained-load graph-compute buffer-corruption" full context
- `patches/llama.cpp/drafts/README.md` — original draft design + bench acceptance criteria + upstream issue draft text
- `CLAUDE.md` absolute rules #2 (macOS-only) and #3 (one patch per submodule, written rationale required for additions)

---

## Connection to v0.7.2 stability foundation

This spec builds on the v0.7.2 stability work shipped earlier today. The defenses in v0.7.2 (log rotation, pre-bind port check, doctor extensions, scheduled-reboot capability) make Phase 4's bench-and-iterate cycle safe to execute. Without them, a sustained-load bench that triggers a kernel panic would risk the multi-hour outage that motivated v0.7.2. With them, a panic is recoverable in one reboot, the next bench attempt is unblocked, and diagnostic visibility is maintained throughout.

Both efforts share the same north star: making the AMD Vega II + Intel Mac configuration a first-class inference platform. v0.7.2 ensured the platform doesn't crash itself; v0.8.0 ensures the platform actually accelerates the work.
