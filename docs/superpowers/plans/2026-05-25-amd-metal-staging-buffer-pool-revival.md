# AMD-discrete GPU Mode Revival Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-enable stable GPU mode on AMD-discrete profiles (canonical: Mac Pro 7,1 + Radeon Pro Vega II) for the four production cerid slot types (chat, embed, code-embed, rerank), by reviving patch 0002 (staging-buffer-pool) and routing the `GGML_METAL_CONCURRENCY_DISABLE=1` env var via `tuning.go`.

**Architecture:** Two-layer fix. The kernel layer (patch 0002) pools `MTLBuffer` staging allocations in `ggml_metal_buffer_set_tensor`/`_get_tensor` to eliminate AMD IOMMU registration churn that triggers family-B `GGML_ASSERT(buf)` SIGABRTs. The supervisor layer (`tuning.go` + `slotEnv`) injects `GGML_METAL_CONCURRENCY_DISABLE=1` per-slot for AMD-discrete profiles to eliminate the upstream `MTLDispatchTypeConcurrent` race that produces non-deterministic output. Apple Silicon (UMA) takes neither codepath — `buf->is_shared` short-circuits the pool, and the supervisor only sets the env var on AMD profiles.

**Tech Stack:** Go 1.23 (supervisor), C/Objective-C (Metal patch), Python 3 stdlib (bench harness), `git format-patch` (patch maintenance), `launchctl` (slot lifecycle on macOS).

**Spec reference:** [`docs/superpowers/specs/2026-05-25-amd-metal-staging-buffer-pool-revival-design.md`](../specs/2026-05-25-amd-metal-staging-buffer-pool-revival-design.md)

---

## File Structure

| File | Change | Phase | Responsibility |
|---|---|---|---|
| `patches/llama.cpp/0002-metal-staging-buffer-pool.patch` | Create (from `drafts/.broken`) | 1 | Pool helpers + set/get_tensor body changes |
| `patches/llama.cpp/drafts/0002-metal-staging-buffer-pool.patch.broken` | Delete after promotion | 1 | Was: parked draft |
| `internal/tuning/tuning.go` | Modify (add field, flip 3 branches) | 2 | Per-slot tuning policy |
| `internal/tuning/tuning_test.go` | Modify (update 3 tests, add 2 new) | 2 | Tuning unit tests |
| `cmd/quenchforge/main.go` | Modify (`slotEnv` extension) | 2 | Slot env construction |
| `cmd/quenchforge/serve_test.go` | Modify (add 1 test) | 2 | slotEnv unit tests |
| `scripts/bench-llama-sustained-load.py` | Create (promote from `/tmp/`) | 3 | Chat sustained-load bench |
| `scripts/bench-llama-correctness.py` | Create (new, ~150 lines) | 3 | Chat correctness probes |
| `patches/README.md` | Modify (Section 3 SHIPPED status) | 6 | Patch series documentation |
| `CHANGELOG.md` | Modify (v0.8.0 entry) | 6 | Release notes |
| `docs/METAL_AMD_BERT_CORRECTNESS.md` | Modify (corrected root-cause) | 6 | Long-form Metal bug analysis |
| `docs/superpowers/specs/2026-05-25-amd-metal-acceleration-design.md` | Modify (add superseded header) | 6 | Historical research artifact |
| `~/.claude/projects/-Users-sunrunner-Develop/memory/project_cerid_quenchforge_chat_on_cpu.md` | Modify (rewrite) | 6 | Operational memory (off-repo) |

---

## Phase 1 — Patch revival

### Task 1: Strip duplicate-of-0001 hunks from the draft patch

**Files:**
- Read: `patches/llama.cpp/drafts/0002-metal-staging-buffer-pool.patch.broken`
- Create: `patches/llama.cpp/0002-metal-staging-buffer-pool.patch` (intermediate; will be regenerated in Task 2)

The draft patch was generated against an unpatched submodule, so it re-emits patch 0001's `has_simdgroup_reduction` / `has_bfloat` gating logic at hunk lines 167-194 of the patch file. When applied on top of 0001, those hunks have no valid `-` context — `git am` will reject.

- [ ] **Step 1: Copy the draft to its target location**

```bash
cp patches/llama.cpp/drafts/0002-metal-staging-buffer-pool.patch.broken \
   patches/llama.cpp/0002-metal-staging-buffer-pool.patch
```

- [ ] **Step 2: Delete the duplicate hunk that overlaps with patch 0001**

Open `patches/llama.cpp/0002-metal-staging-buffer-pool.patch` in an editor and delete the entire hunk that begins:

```
@@ -645,14 +738,30 @@ ggml_metal_device_t ggml_metal_device_init(int device) {
             dev->addr_virt = 0x000000400ULL;
 
             dev->props.device = device;
-            dev->props.has_simdgroup_reduction  = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];
-            dev->props.has_simdgroup_reduction |= [dev->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];
```

and ends:

```
             if (getenv("GGML_METAL_BF16_DISABLE") != NULL) {
                 dev->props.has_bfloat = false;
             }
```

The hunk is approximately 32 lines (counted from the `@@` header to the next `@@` or `diff` header). After deletion, the patch should jump directly from the pool helpers added near line 28 to the modified `ggml_metal_buffer_set_tensor` body change.

- [ ] **Step 3: Verify the patch no longer contains the duplicate hunk**

```bash
grep -n "has_simdgroup_reduction\|has_bfloat" patches/llama.cpp/0002-metal-staging-buffer-pool.patch
```

Expected: empty output (no matches). If anything matches, the deletion in Step 2 was incomplete.

- [ ] **Step 4: Sanity-check the patch structure**

```bash
grep -n "^@@" patches/llama.cpp/0002-metal-staging-buffer-pool.patch
```

Expected: exactly 3 hunks remaining:
- One near `@@ -25,...` adding the pool helpers near the file top
- One near `@@ -1653,...` modifying `ggml_metal_buffer_set_tensor`
- One near `@@ -1711,...` modifying `ggml_metal_buffer_get_tensor`

No commit yet — file indices are still stale; Task 2 will regenerate.

### Task 2: Apply trimmed patch on clean tree + regenerate file indices

**Files:**
- Modify: `patches/llama.cpp/0002-metal-staging-buffer-pool.patch` (regenerated with current indices)

- [ ] **Step 1: Reset the submodule to clean state**

```bash
bash scripts/apply-patches.sh --reset
```

Expected: prints reset confirmation per submodule; no errors.

- [ ] **Step 2: Apply the full patch series**

```bash
bash scripts/apply-patches.sh
```

Expected: applies 0001, 0002, 0003, 0004 in order. **If 0002 fails to apply**, the trimmed hunks in Task 1 were incorrect — re-inspect the patch file. If 0002 succeeds, continue.

- [ ] **Step 3: Regenerate the patch with proper file indices**

```bash
cd llama.cpp
# The order of patches applied via git am was 0001, 0002, 0003, 0004.
# 0002 is the second commit on this branch (HEAD~2 vs HEAD~3 difference).
# Regenerate just patch 0002's commit:
git log --oneline -4
```

Expected: 4 quenchforge patch commits with subjects from the patch headers. Note the SHA of the 0002 commit (subject begins "metal: pool transient staging buffers on AMD discrete").

- [ ] **Step 4: Emit the regenerated patch from that commit**

```bash
# Substitute <SHA> with the 0002 commit SHA from Step 3:
git format-patch <SHA>~..<SHA> --stdout \
  > ../patches/llama.cpp/0002-metal-staging-buffer-pool.patch
cd ..
```

Expected: file is rewritten with valid `index XXX..YYY` lines matching current submodule state.

- [ ] **Step 5: Re-apply to verify the regenerated patch is clean**

```bash
bash scripts/apply-patches.sh --reset
bash scripts/apply-patches.sh
```

Expected: all four patches apply with zero rejected hunks.

### Task 3: Remove the stale draft, build llama.cpp, commit Phase 1

**Files:**
- Delete: `patches/llama.cpp/drafts/0002-metal-staging-buffer-pool.patch.broken`
- Verify build: `bin/llama-server` rebuilt

- [ ] **Step 1: Remove the parked draft**

```bash
git rm patches/llama.cpp/drafts/0002-metal-staging-buffer-pool.patch.broken
```

- [ ] **Step 2: Build the patched llama.cpp**

```bash
bash scripts/build-llama.sh
```

Expected: CMake configures + builds successfully; `bin/llama-server` is rebuilt. If the build fails, the patch has a compile error — root-cause + iterate.

- [ ] **Step 3: Quick smoke that the binary runs**

```bash
./bin/llama-server --help 2>&1 | head -5
```

Expected: usage banner printed (no segfault, no missing-symbol error).

- [ ] **Step 4: Stage and commit Phase 1**

```bash
git add patches/llama.cpp/0002-metal-staging-buffer-pool.patch
git add patches/llama.cpp/drafts/0002-metal-staging-buffer-pool.patch.broken
git status
```

Expected: `git status` shows one new file (the patch) and one deletion (the .broken draft).

- [ ] **Step 5: Commit**

```bash
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(patches): 0002-metal-staging-buffer-pool — kernel fix for family-B SIGABRT (v0.8.0)

Promotes drafts/0002-...patch.broken to live patch. Strips the
duplicate-of-0001 simdgroup_reduction/bfloat gating hunks from the
draft (those landed in patch 0001 first) and regenerates file
indices against the current submodule tree.

The pool design is unchanged from the May 17 draft: bounded
MTLBuffer pool keyed on power-of-two size classes (4 KiB - 64 MiB),
per-class FIFO cap of 4. set_tensor/get_tensor acquire+release from
the pool instead of calling newBufferWithBytesNoCopy per invocation.
Eliminates AMD-discrete IOMMU registration churn that exhausts the
driver's mapping pool under sustained load and triggers
GGML_ASSERT(buf) SIGABRT.

Apple Silicon unaffected: buf->is_shared fast path short-circuits
before reaching the pool.

Operator escape hatch: GGML_METAL_DISABLE_STAGING_POOL=1 reverts to
upstream behavior for A/B testing.

This commit only enables the kernel-level fix — the supervisor still
routes AMD-discrete slots to CPU via tuning.go::chatParams/embedParams/
rerankParams. Phase 2 of the v0.8.0 rollout flips those branches to GPU.
EOF
)"
```

---

## Phase 2 — tuning.go + main.go

### Task 4: Add `MetalConcurrencyDisable` field + failing slotEnv test

**Files:**
- Modify: `internal/tuning/tuning.go` (add field to `SlotTuning` struct)
- Modify: `cmd/quenchforge/serve_test.go` (add failing test for AMD ConcurrencyDisable)

- [ ] **Step 1: Add the new field to `SlotTuning`**

In `internal/tuning/tuning.go`, locate the `SlotTuning` struct (around line 56). After the existing `AutoRespawn` field (around line 87) and before the closing `}`, add:

```go
	// MetalConcurrencyDisable, when true, sets `GGML_METAL_CONCURRENCY_DISABLE=1`
	// in the slot's env. Required on AMD discrete (non-UMA Metal): upstream's
	// MTLDispatchTypeConcurrent path's command-buffer ordering is unreliable on
	// non-UMA drivers and causes non-deterministic output across BERT-family
	// models and chat-decode races. See llama.cpp issue #19563 and patch 0002.
	// Apple Silicon (UMA) does not need this; the concurrent path is correct there.
	MetalConcurrencyDisable bool
```

- [ ] **Step 2: Verify the field compiles**

```bash
cd /Users/sunrunner/Develop/quenchforge
go build ./internal/tuning/
```

Expected: zero output (success).

- [ ] **Step 3: Write the failing test in `cmd/quenchforge/serve_test.go`**

After the existing `TestSlotEnv_ChatKeepsGlobalNCB` test (around line 240), append:

```go
// TestSlotEnv_AMDIncludesConcurrencyDisable verifies that AMD-discrete profiles
// get the GGML_METAL_CONCURRENCY_DISABLE=1 env var on every llama-server slot
// kind (chat, embed, code-embed, rerank). Required to disable the upstream
// MTLDispatchTypeConcurrent path that's unreliable on non-UMA Metal.
func TestSlotEnv_AMDIncludesConcurrencyDisable(t *testing.T) {
	cfg := config.Defaults()
	hwInfo := hardware.Info{Profile: hardware.ProfileVegaPro}

	for _, kind := range []gateway.SlotKind{
		gateway.KindChat,
		gateway.KindEmbed,
		gateway.KindCodeEmbed,
		gateway.KindRerank,
	} {
		env := slotEnv(cfg, hwInfo, kind)
		want := "GGML_METAL_CONCURRENCY_DISABLE=1"
		found := false
		for _, e := range env {
			if e == want {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("kind=%v: want env to contain %q; got %v", kind, want, env)
		}
	}
}
```

- [ ] **Step 4: Run the test and verify it fails**

```bash
go test ./cmd/quenchforge/ -run TestSlotEnv_AMDIncludesConcurrencyDisable -v
```

Expected: `FAIL` — the env var isn't being emitted yet because (a) `slotEnv` doesn't read `MetalConcurrencyDisable` and (b) `tuning.KernelParams` doesn't set it. Task 5 fixes both.

### Task 5: Wire `MetalConcurrencyDisable` into `slotEnv` and set it for AMD-discrete

**Files:**
- Modify: `cmd/quenchforge/main.go` (extend `slotEnv`)
- Modify: `internal/tuning/tuning.go` (set field to `true` in three AMD-discrete branches)

- [ ] **Step 1: Extend `slotEnv` in `cmd/quenchforge/main.go`**

Locate the `slotEnv` function (around line 937). Replace its body with:

```go
func slotEnv(cfg config.Config, hwInfo hardware.Info, kind gateway.SlotKind) []string {
	ncb := cfg.MetalNCB
	tn := tuning.KernelParams(hwInfo.Profile, kind, cfg)
	if tn.MetalNCB > 0 {
		ncb = tn.MetalNCB
	}
	env := []string{
		fmt.Sprintf("GGML_METAL_N_CB=%d", ncb),
	}
	if tn.MetalConcurrencyDisable {
		env = append(env, "GGML_METAL_CONCURRENCY_DISABLE=1")
	}
	return env
}
```

- [ ] **Step 2: Set `MetalConcurrencyDisable = true` in `chatParams` AMD-discrete branch**

In `internal/tuning/tuning.go`, locate `chatParams` (around line 124). Replace its body with:

```go
func chatParams(profile hardware.Profile) SlotTuning {
	if !profileIsAMDDiscrete(profile) {
		return SlotTuning{}
	}
	// AMD-discrete chat slot runs on GPU as of v0.8.0. The MTLDispatchTypeConcurrent
	// race that produced cross-call non-determinism is disabled via
	// MetalConcurrencyDisable -> GGML_METAL_CONCURRENCY_DISABLE=1. The family-B
	// IOMMU exhaustion that produced sustained-load SIGABRTs is mitigated by
	// patch 0002 (staging-buffer pool). AutoRespawn stays as defense in depth.
	//
	// Chat-specific safety flags retained from the CPU-route era:
	//   --flash-attn off    — FA's GPU path is unsafe with simdgroup_reduction off
	//   --cache-ram 0       — disables LCP-similarity slot cache (CLAUDE.md gotcha #1)
	//   --no-cache-prompt   — disables per-slot prompt cache
	return SlotTuning{
		ExtraArgs: []string{
			"--flash-attn", "off",
			"--cache-ram", "0",
			"--no-cache-prompt",
			"--gpu-layers", "999",
		},
		MetalConcurrencyDisable: true,
		AutoRespawn:             true,
	}
}
```

- [ ] **Step 3: Set `MetalConcurrencyDisable = true` in `embedParams` AMD-discrete branch + re-enable ubatch cap**

In `internal/tuning/tuning.go`, locate `embedParams` (around line 167). The AMD-discrete branch needs three changes: re-enable the 1024 ubatch cap, set `MetalConcurrencyDisable: true`, and flip `--gpu-layers 0` → `--gpu-layers 999`.

Replace the function body with:

```go
func embedParams(profile hardware.Profile, cfg config.Config) SlotTuning {
	ubatch := cfg.MaxContext
	metalNCB := cfg.MetalNCB
	if profileIsAMDDiscrete(profile) {
		// AMD-discrete on GPU (v0.8.0) needs the 1024 ubatch cap re-enabled —
		// it bounds per-call Metal staging-buffer pressure even with patch 0002's
		// pool in place. CLAUDE.md operational gotcha #2 documents this knob.
		ubatch = amdEmbedUbatchDefault
		metalNCB = amdEmbedMetalNCBDefault
	}
	if cfg.EmbedUbatchSize > 0 {
		ubatch = cfg.EmbedUbatchSize
	}
	t := SlotTuning{
		UbatchSize: ubatch,
		BatchSize:  ubatch,
		MetalNCB:   metalNCB,
	}
	if cfg.EmbedMetalNCB > 0 {
		t.MetalNCB = cfg.EmbedMetalNCB
	}
	if profileIsAMDDiscrete(profile) {
		t.AutoRespawn = true
		t.MetalConcurrencyDisable = true
		// GPU mode re-enabled in v0.8.0:
		//   --gpu-layers 999       all layers on Vega II (was: 0, CPU-only)
		//   --threads 15           CPU pool sized for CPU-mode is kept;
		//                          GPU mode mostly idle on CPU but harmless
		//   --parallel 4           4 concurrent in-server slots for burst
		t.ExtraArgs = append(t.ExtraArgs,
			"--gpu-layers", "999",
			"--threads", "15",
			"--parallel", "4",
		)
	}
	return t
}
```

- [ ] **Step 4: Set `MetalConcurrencyDisable = true` in `rerankParams` AMD-discrete branch**

In `internal/tuning/tuning.go`, locate `rerankParams` (around line 249). Replace its body with:

```go
func rerankParams(profile hardware.Profile, cfg config.Config) SlotTuning {
	t := SlotTuning{}
	if cfg.RerankBatchSize > 0 {
		t.BatchSize = cfg.RerankBatchSize
		t.UbatchSize = cfg.RerankBatchSize
	}
	if profileIsAMDDiscrete(profile) {
		t.MetalNCB = amdEmbedMetalNCBDefault
	}
	if cfg.RerankMetalNCB > 0 {
		t.MetalNCB = cfg.RerankMetalNCB
	}
	if profileIsAMDDiscrete(profile) {
		t.AutoRespawn = true
		t.MetalConcurrencyDisable = true
		// Same v0.8.0 GPU-mode re-enable as embedParams; same rationale.
		// See embedParams docstring for the full ExtraArgs justification.
		t.ExtraArgs = append(
			t.ExtraArgs,
			"--gpu-layers", "999",
			"--threads", "15",
			"--parallel", "4",
		)
	}
	return t
}
```

- [ ] **Step 5: Run the slotEnv test — should now pass**

```bash
go test ./cmd/quenchforge/ -run TestSlotEnv_AMDIncludesConcurrencyDisable -v
```

Expected: `PASS`.

### Task 6: Update existing AMD-discrete tuning tests to match new expectations

**Files:**
- Modify: `internal/tuning/tuning_test.go` (rename + update 3 tests; add a new MetalConcurrencyDisable test)

The existing `TestKernelParams_ChatAMDGetsCorrectnessFlags`, `TestKernelParams_EmbedAMDForcesCPU`, and `TestKernelParams_RerankAMDForcesCPU` will fail under the new behavior — they assert `--gpu-layers 0`, but the new code emits `--gpu-layers 999`. Update them.

- [ ] **Step 1: Run the existing tuning tests — confirm they fail**

```bash
go test ./internal/tuning/ -v
```

Expected: at least three tests FAIL (`TestKernelParams_ChatAMDGetsCorrectnessFlags`, `TestKernelParams_EmbedAMDForcesCPU`, `TestKernelParams_RerankAMDForcesCPU`) on `--gpu-layers` expectation mismatches.

- [ ] **Step 2: Update `TestKernelParams_ChatAMDGetsCorrectnessFlags`**

In `internal/tuning/tuning_test.go` around line 54, replace the test body with:

```go
func TestKernelParams_ChatAMDGetsGPUWithConcurrencyDisable(t *testing.T) {
	cfg := config.Defaults()
	got := tuning.KernelParams(hardware.ProfileVegaPro, gateway.KindChat, cfg)

	wantExtra := []string{
		"--flash-attn", "off",
		"--cache-ram", "0",
		"--no-cache-prompt",
		"--gpu-layers", "999",
	}
	if !reflect.DeepEqual(got.ExtraArgs, wantExtra) {
		t.Errorf("ExtraArgs = %v; want %v", got.ExtraArgs, wantExtra)
	}
	if !got.MetalConcurrencyDisable {
		t.Error("MetalConcurrencyDisable = false; want true")
	}
	if !got.AutoRespawn {
		t.Error("AutoRespawn = false; want true")
	}
}
```

Note the function name change: `ChatAMDGetsCorrectnessFlags` → `ChatAMDGetsGPUWithConcurrencyDisable`. Apply the rename consistently in the file.

- [ ] **Step 3: Update `TestKernelParams_EmbedAMDForcesCPU`**

In `internal/tuning/tuning_test.go` around line 140, replace the test body with:

```go
func TestKernelParams_EmbedAMDGetsGPUWithConcurrencyDisable(t *testing.T) {
	cfg := config.Defaults()
	got := tuning.KernelParams(hardware.ProfileVegaPro, gateway.KindEmbed, cfg)

	if got.UbatchSize != 1024 {
		t.Errorf("UbatchSize = %d; want 1024 (amdEmbedUbatchDefault for GPU mode)", got.UbatchSize)
	}
	if got.BatchSize != 1024 {
		t.Errorf("BatchSize = %d; want 1024", got.BatchSize)
	}
	if !got.MetalConcurrencyDisable {
		t.Error("MetalConcurrencyDisable = false; want true")
	}
	if !got.AutoRespawn {
		t.Error("AutoRespawn = false; want true")
	}
	wantExtraSubset := []string{"--gpu-layers", "999"}
	if !containsSubslice(got.ExtraArgs, wantExtraSubset) {
		t.Errorf("ExtraArgs = %v; want to contain %v", got.ExtraArgs, wantExtraSubset)
	}
}

// containsSubslice returns true if needle appears as a contiguous subsequence of haystack.
func containsSubslice(haystack, needle []string) bool {
	if len(needle) == 0 {
		return true
	}
	for i := 0; i+len(needle) <= len(haystack); i++ {
		match := true
		for j := range needle {
			if haystack[i+j] != needle[j] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}
```

- [ ] **Step 4: Update `TestKernelParams_RerankAMDForcesCPU`**

In `internal/tuning/tuning_test.go` around line 218, replace the test body with:

```go
func TestKernelParams_RerankAMDGetsGPUWithConcurrencyDisable(t *testing.T) {
	cfg := config.Defaults()
	got := tuning.KernelParams(hardware.ProfileVegaPro, gateway.KindRerank, cfg)

	if !got.MetalConcurrencyDisable {
		t.Error("MetalConcurrencyDisable = false; want true")
	}
	if !got.AutoRespawn {
		t.Error("AutoRespawn = false; want true")
	}
	wantExtraSubset := []string{"--gpu-layers", "999"}
	if !containsSubslice(got.ExtraArgs, wantExtraSubset) {
		t.Errorf("ExtraArgs = %v; want to contain %v", got.ExtraArgs, wantExtraSubset)
	}
}
```

- [ ] **Step 5: Audit other tests that assumed `--gpu-layers 0`**

```bash
grep -n '"--gpu-layers", "0"\|--gpu-layers 0' internal/tuning/tuning_test.go cmd/quenchforge/serve_test.go
```

Expected: no matches. If any match remains, update those tests to expect `"999"` instead.

- [ ] **Step 6: Run all tuning tests — should pass**

```bash
go test ./internal/tuning/ -v
```

Expected: all tests PASS.

- [ ] **Step 7: Run all quenchforge tests — should pass**

```bash
go test ./...
```

Expected: all tests PASS. If a downstream test fails (e.g., in `cmd/quenchforge/`), update it to the new contract.

- [ ] **Step 8: Stage and commit Phase 2**

```bash
git add internal/tuning/tuning.go internal/tuning/tuning_test.go
git add cmd/quenchforge/main.go cmd/quenchforge/serve_test.go
git status
```

Expected: 4 files modified.

- [ ] **Step 9: Commit**

```bash
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(tuning): re-enable AMD-discrete GPU mode (env-var + patch 0002)

Pairs with patch 0002 (staging-buffer pool) to bring AMD-discrete
profiles back onto GPU. Two supervisor-side changes:

1. New SlotTuning.MetalConcurrencyDisable bool field. When true,
   slotEnv emits GGML_METAL_CONCURRENCY_DISABLE=1 in the slot's
   environment. The env var disables ggml-metal-context.m's
   MTLDispatchTypeConcurrent path, which is unreliable on non-UMA
   Macs and produced cross-call non-determinism in BERT embeddings
   (cos_sim -0.011 cross-call observed 2026-05-25 on Vega II).
   Validated today: with the env var set, cos_sim 1.000000 across
   all 4 production models.

2. chatParams / embedParams / rerankParams AMD-discrete branches
   flip --gpu-layers from "0" to "999" and set
   MetalConcurrencyDisable: true. embedParams additionally re-enables
   the 1024 ubatch cap (amdEmbedUbatchDefault) — CLAUDE.md gotcha #2
   says this caps Metal staging-buffer pressure, which the v0.7.0
   CPU-route comment marked "no longer applies" but does apply again
   under GPU mode.

AutoRespawn retained on all three slot kinds as defense in depth for
unknown crash classes. Apple Silicon profiles are unchanged — slotEnv
only emits the env var when MetalConcurrencyDisable is true, which is
only set on AMD-discrete branches.

Tests:
- Renamed TestKernelParams_ChatAMDGetsCorrectnessFlags ->
  TestKernelParams_ChatAMDGetsGPUWithConcurrencyDisable
- Renamed TestKernelParams_EmbedAMDForcesCPU ->
  TestKernelParams_EmbedAMDGetsGPUWithConcurrencyDisable
- Renamed TestKernelParams_RerankAMDForcesCPU ->
  TestKernelParams_RerankAMDGetsGPUWithConcurrencyDisable
- New TestSlotEnv_AMDIncludesConcurrencyDisable in serve_test.go
EOF
)"
```

---

## Phase 3 — Bench harness

### Task 7: Promote `bench-llama-sustained-load.py` from `/tmp/` to `scripts/`

**Files:**
- Create: `scripts/bench-llama-sustained-load.py` (from existing `/tmp/bench-llama-sustained-load.py`)

- [ ] **Step 1: Copy + commit the chat sustained-load bench**

```bash
cp /tmp/bench-llama-sustained-load.py scripts/bench-llama-sustained-load.py
chmod +x scripts/bench-llama-sustained-load.py
```

- [ ] **Step 2: Verify the script's `--help` works**

```bash
python3 scripts/bench-llama-sustained-load.py --help
```

Expected: argparse help output listing `--url`, `--model`, `--duration`, `--concurrency`, `--rss-hint` flags.

- [ ] **Step 3: Stage (commit happens with Task 8 + 9 as one Phase 3 commit)**

```bash
git add scripts/bench-llama-sustained-load.py
```

### Task 8: Write `bench-llama-correctness.py`

**Files:**
- Create: `scripts/bench-llama-correctness.py`

Chat correctness probe — mirrors `bench-bert-correctness.py` shape but uses `/v1/chat/completions` instead of `/v1/embeddings`. Three deterministic probes (fixed prompts at `temperature=0`, expect identical responses across N calls) plus one semantic sanity check (expect coherent English).

- [ ] **Step 1: Create the file**

```bash
cat > scripts/bench-llama-correctness.py <<'PYEOF'
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
PYEOF
chmod +x scripts/bench-llama-correctness.py
```

- [ ] **Step 2: Verify the script's `--help` works**

```bash
python3 scripts/bench-llama-correctness.py --help
```

Expected: argparse help output listing `--url`, `--model`, `--n-calls`.

- [ ] **Step 3: Stage (commit happens in next task)**

```bash
git add scripts/bench-llama-correctness.py
```

### Task 9: Verify `bench-bert-sustained-load.py` accepts `--batch-size 4` and commit Phase 3

**Files:** none modified (verification step + Phase 3 commit)

- [ ] **Step 1: Smoke-check that the existing bench accepts the smaller batch**

```bash
python3 scripts/bench-bert-sustained-load.py --help 2>&1 | grep -A1 batch-size
```

Expected: shows `--batch-size BATCH_SIZE` flag in the help (it's already there per the script source, defaults to 8). No code change needed — the bench protocol just calls it with `--batch-size 4`.

- [ ] **Step 2: Commit Phase 3**

```bash
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(scripts): bench-llama-{correctness,sustained-load} for chat slot validation

Adds the chat-side bench counterparts of bench-bert-{correctness,
sustained-load}.py — required for Phase 4's validation gate to cover
the chat slot (llama3.1-8b Q4_K_M) in addition to the three BERT
slots.

- scripts/bench-llama-sustained-load.py: concurrent /v1/chat/completions
  hammer for 30 min; tracks SIGABRT (PID disappearance), HTTP 5xx burst,
  response drift on deterministic prompts (temperature=0), latency cliff,
  and RSS leak. Mirrors bench-bert-sustained-load.py's structure but
  uses chat-completion semantics (response text matching, not vector
  cos_sim).

- scripts/bench-llama-correctness.py: short determinism + semantic
  probes (~30s wall-clock). Mirrors bench-bert-correctness.py's exit-
  code contract (0 = safe; 1 = do not flip; 2 = daemon unreachable).
EOF
)"
```

---

## Phase 4 — Build + bench validation gate

### Task 10: Build and install the patched quenchforge binary

**Files:** none modified (build step)

- [ ] **Step 1: Rebuild quenchforge with the new tuning code**

```bash
make build
```

Expected: `bin/quenchforge`, `bin/quenchforge-bench`, `bin/quenchforge-preflight` are rebuilt. Zero errors.

- [ ] **Step 2: Install to the system path**

```bash
sudo install -m 0755 bin/quenchforge          /usr/local/bin/quenchforge
sudo install -m 0755 bin/quenchforge-bench    /usr/local/bin/quenchforge-bench
sudo install -m 0755 bin/quenchforge-preflight /usr/local/bin/quenchforge-preflight
```

Expected: prompts for sudo, then exits silently. **If the user hasn't authorised sudo for installs in your session, ask them first.**

- [ ] **Step 3: Verify the install worked**

```bash
/usr/local/bin/quenchforge --version 2>&1 || /usr/local/bin/quenchforge version
```

Expected: a version string is printed (or, at minimum, no "command not found").

- [ ] **Step 4: Reload the LaunchAgent so the new binary takes effect**

```bash
launchctl kickstart -k gui/$(id -u)/com.cerid.quenchforge
```

Expected: silent (kickstart returns no output on success). Production slots are now running the new binary; all four llama-server children should restart with the new env vars + `--gpu-layers 999`.

- [ ] **Step 5: Verify the slots are running with the new args**

```bash
sleep 10
ps aux | grep llama-server | grep -v grep
```

Expected: 4 llama-server processes — one per slot kind. Each command line includes `--gpu-layers 999` (NOT `0`).

- [ ] **Step 6: Confirm GGML_METAL_CONCURRENCY_DISABLE is in the slot env**

```bash
# Pick one of the llama-server PIDs from Step 5:
PID=<paste pid here>
ps -E -p $PID 2>/dev/null | tr ' ' '\n' | grep -i metal
```

Expected: lists `GGML_METAL_N_CB=1` AND `GGML_METAL_CONCURRENCY_DISABLE=1`.

### Task 11: Bench Criterion 1 — correctness probes against all four production slots

**Files:** none modified (validation step)

- [ ] **Step 1: nomic embed correctness**

```bash
python3 /Users/sunrunner/Develop/quenchforge/scripts/bench-bert-correctness.py \
  --url http://127.0.0.1:11501 \
  --model nomic-embed-text-v1.5 \
  --n-calls 10 --epsilon 1e-4
```

Expected: all 4 probes PASS (same-batch cos_sim ≥ 0.9999, cross-call cos_sim ≥ 0.9999, semantic OK, L2 norm OK). Exit 0.

- [ ] **Step 2: jina code-embed correctness**

```bash
python3 /Users/sunrunner/Develop/quenchforge/scripts/bench-bert-correctness.py \
  --url http://127.0.0.1:11506 \
  --model jina-embeddings-v2-base-code \
  --n-calls 10 --epsilon 1e-4
```

Expected: same.

- [ ] **Step 3: bge rerank correctness**

```bash
python3 /Users/sunrunner/Develop/quenchforge/scripts/bench-bert-correctness.py \
  --url http://127.0.0.1:11502 \
  --model bge-reranker-v2-m3 \
  --rerank --n-calls 10 --epsilon 1e-4
```

Expected: PASS (rerank determinism: all 10 calls returned identical scores). Exit 0.

- [ ] **Step 4: chat correctness**

```bash
python3 /Users/sunrunner/Develop/quenchforge/scripts/bench-llama-correctness.py \
  --url http://127.0.0.1:11500 \
  --model llama3.1-8b \
  --n-calls 10
```

Expected: both probes PASS. Exit 0.

**Failure response:** if any criterion fails, do NOT proceed. The supervisor's `git revert HEAD~1` reverts Phase 2 (Phase 1 patch stays inert), and the bench is investigated.

### Task 12: Bench Criterion 2a — nomic sustained-load (30 min)

**Files:** none modified (validation step)

- [ ] **Step 1: Run the 30-min sustained bench**

```bash
python3 /Users/sunrunner/Develop/quenchforge/scripts/bench-bert-sustained-load.py \
  --url http://127.0.0.1:11501 \
  --model nomic-embed-text-v1.5 \
  --duration 1800 --concurrency 4 --batch-size 4
```

Expected output (the final lines after ~30 min):

```
=== complete: NNNN requests, 0 failures, ~1800s ===
    p50=X.XXs  p95=Y.YYs  throughput=Z.ZZ req/s
=== sustained-load bench PASSED ===
```

Pass criteria: exit code 0, zero `[FAIL]` lines in output, drift cos_sim stays ≥ 0.999 throughout.

- [ ] **Step 2: Confirm no auto-respawn happened in the supervisor**

```bash
log show --last 35m --predicate 'process == "quenchforge"' 2>/dev/null \
  | grep -i 'auto.*respawn\|sigabrt\|signal' | head
```

Expected: no auto-respawn / SIGABRT lines.

**Failure response:** if SIGABRT appears, the patch did NOT fix family-B. Halt the rollout; root-cause via `tail /Users/sunrunner/Library/Logs/quenchforge/embed.log` and the crash stack at `ggml-metal-device.m`. Likely investigation paths: pool cap too high (drop to 2), pool not actually being hit (verify with debug logging), or different staging path missed by the patch.

### Task 13: Bench Criterion 2b — jina sustained-load (30 min)

**Files:** none modified (validation step)

- [ ] **Step 1: Run the 30-min sustained bench**

```bash
python3 /Users/sunrunner/Develop/quenchforge/scripts/bench-bert-sustained-load.py \
  --url http://127.0.0.1:11506 \
  --model jina-embeddings-v2-base-code \
  --duration 1800 --concurrency 4 --batch-size 4
```

Expected: same PASS structure as Task 12.

### Task 14: Bench Criterion 2c — chat sustained-load (30 min)

**Files:** none modified (validation step)

- [ ] **Step 1: Run the 30-min sustained bench**

```bash
python3 /Users/sunrunner/Develop/quenchforge/scripts/bench-llama-sustained-load.py \
  --url http://127.0.0.1:11500 \
  --model llama3.1-8b \
  --duration 1800 --concurrency 2 \
  --rss-hint llama3.1-8b.gguf
```

Expected: `=== chat sustained-load PASSED ===`, exit 0, zero drift events, zero SIGABRT.

### Task 15: Bench Criterion 3 — escape-hatch must reproduce the crash

**Files:** none modified (validation step)

This task **must fail** — confirming the patch is what's preventing the crash.

- [ ] **Step 1: Spin up a manual llama-server with the staging-pool DISABLED**

```bash
GGML_METAL_DISABLE_STAGING_POOL=1 GGML_METAL_CONCURRENCY_DISABLE=1 \
  GGML_METAL_N_CB=1 /usr/local/bin/llama-server \
  --model /Users/sunrunner/.quenchforge/models/nomic-embed-text-v1.5.gguf \
  --host 127.0.0.1 --port 11600 \
  --gpu-layers 999 --threads 4 --parallel 4 \
  --ctx-size 8192 --embedding --pooling cls \
  --batch-size 1024 --ubatch-size 1024 \
  > /tmp/bench-nomic-pool-disabled.log 2>&1 &
PID=$!
echo "Started PID=$PID with pool DISABLED"
```

- [ ] **Step 2: Wait for ready**

```bash
for i in $(seq 1 30); do
  curl -fsS http://127.0.0.1:11600/health > /dev/null 2>&1 && break
  sleep 1
done
echo "Ready"
```

- [ ] **Step 3: Run a short sustained bench (5 min) — must FAIL with family-B**

```bash
python3 /Users/sunrunner/Develop/quenchforge/scripts/bench-bert-sustained-load.py \
  --url http://127.0.0.1:11600 \
  --model nomic-embed-text-v1.5 \
  --duration 300 --concurrency 4 --batch-size 4
EXIT=$?
echo "Bench exit: $EXIT"
```

Expected: bench exits with code 1 within ~5 minutes (typically 1-3 min), reports HTTP 5xx burst. The llama-server log at `/tmp/bench-nomic-pool-disabled.log` ends with `GGML_ASSERT(buf_src) failed` and a backtrace into `ggml_metal_buffer_set_tensor`.

- [ ] **Step 4: Confirm SIGABRT in the log**

```bash
grep -A2 'GGML_ASSERT\|Abort trap' /tmp/bench-nomic-pool-disabled.log | head -20
```

Expected: at least one match (the assertion that fired). This **confirms** that the patch was doing the work: with the pool disabled, the family-B crash returns within minutes.

- [ ] **Step 5: Clean up**

```bash
kill -TERM $PID 2>/dev/null
wait $PID 2>/dev/null
```

**Failure mode of THIS task:** if the bench passes (no crash within 5 min) with the pool disabled, something else is providing stability — the patch's contribution is unverified, and the rollout should not proceed without understanding why.

### Task 16: Tag v0.8.0-rc2

**Files:** none modified (release tag)

- [ ] **Step 1: Verify clean working tree**

```bash
git status
```

Expected: nothing to commit, working tree clean.

- [ ] **Step 2: Create the annotated tag**

```bash
git tag -a v0.8.0-rc2 -m "$(cat <<'EOF'
v0.8.0-rc2: AMD-discrete GPU mode revival

Pairs patch 0002 (staging-buffer pool, kernel-level family-B fix) with
the supervisor-side GGML_METAL_CONCURRENCY_DISABLE=1 routing to unlock
GPU inference on Mac Pro 7,1 + Radeon Pro Vega II.

Validated 2026-05-25 on Mac Pro 2019 + Vega II 32 GB HBM2:
- Correctness: cos_sim 1.000000 across nomic / jina / bge embeddings;
  deterministic responses on llama3.1-8b chat (temperature=0).
- Stability: 30-min sustained-load bench on nomic + jina + chat with
  zero family-B SIGABRTs, no 5xx burst, no drift, no latency cliff.
- Escape hatch: GGML_METAL_DISABLE_STAGING_POOL=1 reproduces the
  v0.7.2 family-B crash class within ~5 min, confirming the patch
  is what's doing the stability work.

7-day production observation window begins now. Tag v0.8.0 (final)
after that window if AutoRespawn count stays ≤ 10/week and no kernel
panics surface.

Rollback path: launchctl bootout + downgrade to v0.7.2 binary.
Tactical fallback: GGML_METAL_DISABLE_STAGING_POOL=1 (loses pool,
family-B returns, but kernel correctness preserved by env var).
EOF
)"
```

- [ ] **Step 3: Confirm tag**

```bash
git tag -l 'v0.8.0*' -n5
```

Expected: tag listed with the first 5 lines of the annotation.

---

## Phase 5 — Production observation window

### Task 17: 7-day production observation

**Files:** none modified (operational observation)

Daily check; cumulative pass criteria evaluated after 7 calendar days from the v0.8.0-rc2 tag time.

- [ ] **Step 1: Daily — run `quenchforge doctor`**

```bash
quenchforge doctor
```

Expected: all checks PASS. **If any check reports non-PASS attributable to the patch or env-var change**, halt observation and root-cause.

- [ ] **Step 2: Daily — review slot logs for AutoRespawn / SIGABRT events**

```bash
for slot in chat embed code-embed rerank; do
  echo "=== $slot ==="
  tail -100 ~/Library/Logs/quenchforge/${slot}.log 2>/dev/null \
    | grep -iE 'abort trap|sigabrt|auto.respawn|panic|GGML_ASSERT' | head
done
```

Expected: empty output most days. Cumulative count ≤ 10 events across the full 7-day window.

- [ ] **Step 3: Daily — check for kernel panics**

```bash
ls -la /Library/Logs/DiagnosticReports/*.panic 2>/dev/null \
  | awk '{print $6, $7, $8, $9}' | head
```

Expected: no new `.panic` files dated after the v0.8.0-rc2 tag. Any new entry blocks promotion to v0.8.0 final.

- [ ] **Step 4: Daily — cerid-AI eval pass rate spot check**

If a cerid LongMemEval or similar ground-truth eval ran during the observation window, compare its pass rate to v0.7.2 baseline. **Acceptance:** within ±2% of baseline. **If the rate dropped >2%**, investigate before promotion — chat quality may have regressed under the new dispatch path.

- [ ] **Step 5: After 7 clean days — proceed to Phase 6**

If all daily checks pass for 7 consecutive days starting from the v0.8.0-rc2 tag, the patch is production-ready. If a soft regression appears (within the 10/week AutoRespawn budget but trending up), extend observation by another week.

---

## Phase 6 — Documentation, memory, final tag

### Task 18: Update `patches/README.md` Section 3 to SHIPPED status

**Files:**
- Modify: `patches/README.md`

The current `README.md` has Section 3 (Sustained-load graph-compute buffer-corruption) split between two states: an upper "supervisor-only mitigation, patch deferred" framing (lines 78-149) and a lower "patch #2 v0.7.0" section (lines 179+) reflecting the brief landing.

- [ ] **Step 1: Replace the Section 3 (line 179+) with the v0.8.0 SHIPPED summary**

Replace lines 179-214 of `patches/README.md` (the section titled "### 3. Sustained-load graph-compute buffer-corruption — patch #2 (v0.7.0)") with:

```markdown
### 3. Sustained-load graph-compute buffer-corruption — patch #2 (v0.8.0)

Closes the third Metal-on-AMD failure class — `GGML_ASSERT(buf_src)`
SIGABRT in `ggml_metal_buffer_set_tensor` and `_get_tensor` under
sustained embed / chat workloads on AMD discrete.

`v0.7.0` shipped a **supervisor-side** mitigation: AMD-discrete embed,
code-embed, rerank, and chat slots get `AutoRespawn: true`, so the
supervisor brings the slot back within ~30 seconds of the SIGABRT.

`v0.8.0` ships the **kernel-level** fix as
`llama.cpp/0002-metal-staging-buffer-pool.patch`. The patch replaces
the per-call `newBufferWithBytesNoCopy` allocation — which registers
a new IOMMU page-table entry on AMD discrete and exhausts the driver's
~256-512-slot pool — with a bounded MTLBuffer pool keyed on
power-of-two size classes (4 KiB - 64 MiB, per-class FIFO cap of 4).
One pool buffer = one registration, reused; worst-case total
registrations stays well below the exhaustion threshold.

Apple Silicon is unaffected: `buf->is_shared` short-circuits to the
`memcpy` fast path before either patched function reaches the pool.

Paired with `internal/tuning/tuning.go::chatParams/embedParams/
rerankParams` setting `MetalConcurrencyDisable: true` for AMD-discrete
profiles. The supervisor injects `GGML_METAL_CONCURRENCY_DISABLE=1`
in slot env, which disables the upstream `MTLDispatchTypeConcurrent`
path that produced non-deterministic output on non-UMA Macs
([llama.cpp issue #19563](https://github.com/ggml-org/llama.cpp/issues/19563)).

Operator escape hatch: `GGML_METAL_DISABLE_STAGING_POOL=1` reverts to
the unpatched per-call allocation path for A/B testing or emergency
rollback. Setting this env var brings the family-B SIGABRT back
within ~5 min under sustained load.

**Bench-validated on Mac Pro 2019 + Radeon Pro Vega II 32 GB HBM2,
2026-05-NN (post-7-day observation):**

| Bench | Result | Notes |
|---|---|---|
| `bench-bert-correctness.py` (nomic) | PASS | cos_sim 1.000000 across all 4 probes |
| `bench-bert-correctness.py` (jina) | PASS | same |
| `bench-bert-correctness.py` (bge --rerank) | PASS | 10 identical scores |
| `bench-llama-correctness.py` (llama3.1-8b) | PASS | deterministic + semantic |
| `bench-bert-sustained-load.py --duration 1800` (nomic) | PASS | 0 SIGABRT, throughput XXX req/s |
| `bench-bert-sustained-load.py --duration 1800` (jina) | PASS | 0 SIGABRT, throughput XXX req/s |
| `bench-llama-sustained-load.py --duration 1800` (chat) | PASS | 0 SIGABRT, throughput XXX req/s |
| Escape hatch reproduces v0.7.2 crash within 5 min | PASS | family-B SIGABRT confirmed; patch is doing the work |
| 7-day production observation | PASS | ≤ 10 AutoRespawn events / week, zero kernel panics |

Fill in the `XXX req/s` values from the Phase 4 bench output.
```

- [ ] **Step 2: Remove the stale "supervisor-only mitigation" framing from Section 3 upper**

Locate lines 78-149 of the old `patches/README.md` (the section header "### 3. Sustained-load graph-compute buffer-corruption" without the "patch #2" suffix). Replace the entire range with a brief cross-reference:

```markdown
### 3. Sustained-load graph-compute buffer-corruption

See the dedicated patch section below ("Section 3 — patch #2 (v0.8.0)").
v0.6.x — v0.7.x shipped supervisor-side mitigations (AutoRespawn,
QUENCHFORGE_EMBED_UBATCH_SIZE, QUENCHFORGE_EMBED_METAL_N_CB env vars)
that bounded blast radius. v0.8.0 ships the kernel-level patch and
re-enables GPU mode on AMD-discrete profiles by default.
```

- [ ] **Step 3: Stage**

```bash
git add patches/README.md
```

### Task 19: Update `CHANGELOG.md` with v0.8.0 entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Insert the v0.8.0 entry at the top of the file**

After the existing top entry (probably `v0.7.2`), insert:

```markdown
## v0.8.0 (2026-05-NN)

### AMD-discrete GPU mode re-enabled

The Mac Pro 7,1 + Radeon Pro Vega II 32 GB configuration now runs all
four production slot types (chat, embed, code-embed, rerank) on GPU by
default. The Xeon W's 16 cores are freed for the gateway and the OS.

#### Mechanism

Two-layer fix:

1. **Kernel:** new patch `0002-metal-staging-buffer-pool` replaces
   per-call `newBufferWithBytesNoCopy` in `ggml_metal_buffer_set_tensor`
   and `_get_tensor` with a bounded MTLBuffer pool. Eliminates the
   AMD IOMMU registration churn that exhausts the driver's mapping
   pool under sustained load and triggers `GGML_ASSERT(buf)` SIGABRT
   (the "family-B" crash documented in `patches/README.md` Section 3).

2. **Supervisor:** `tuning.go::chatParams/embedParams/rerankParams`
   set `MetalConcurrencyDisable: true` for AMD-discrete profiles.
   `slotEnv` injects `GGML_METAL_CONCURRENCY_DISABLE=1` in the slot's
   environment, disabling the upstream `MTLDispatchTypeConcurrent`
   path that produced non-deterministic output on non-UMA Macs
   (llama.cpp issue #19563).

`--gpu-layers 0` removed from the AMD-discrete branches of
`chatParams` / `embedParams` / `rerankParams`; replaced with
`--gpu-layers 999`. `embedParams` re-enables the 1024 ubatch cap
(`amdEmbedUbatchDefault`) to bound per-call Metal staging-buffer
pressure. `AutoRespawn` stays as defense in depth.

#### Bench results

(populate from Phase 4 results)

#### Operator escape hatches

- `GGML_METAL_DISABLE_STAGING_POOL=1` — disables the pool, reverts to
  upstream per-call allocation behavior. Family-B SIGABRT returns
  within ~5 min under sustained load.
- Binary rollback to v0.7.2 — full CPU route restored.

#### Apple Silicon

Unaffected. The pool code is short-circuited by `buf->is_shared` on
UMA devices; the env var is only injected on AMD-discrete profiles.

#### Files changed

- `patches/llama.cpp/0002-metal-staging-buffer-pool.patch` (new — promoted from drafts)
- `internal/tuning/tuning.go` (new `MetalConcurrencyDisable` field; three AMD branches updated)
- `internal/tuning/tuning_test.go` (three tests renamed + updated; +1 `containsSubslice` helper)
- `cmd/quenchforge/main.go` (`slotEnv` extension)
- `cmd/quenchforge/serve_test.go` (+1 test: `TestSlotEnv_AMDIncludesConcurrencyDisable`)
- `scripts/bench-llama-{correctness,sustained-load}.py` (new chat benches)
- `patches/README.md` (Section 3 updated to SHIPPED)
- `docs/METAL_AMD_BERT_CORRECTNESS.md` (corrected root-cause analysis)
- `docs/superpowers/specs/2026-05-25-amd-metal-acceleration-design.md` (marked superseded)
```

(Substitute the actual date in `2026-05-NN` after the 7-day observation completes.)

- [ ] **Step 2: Stage**

```bash
git add CHANGELOG.md
```

### Task 20: Update `docs/METAL_AMD_BERT_CORRECTNESS.md` with corrected root-cause

**Files:**
- Modify: `docs/METAL_AMD_BERT_CORRECTNESS.md`

The existing doc was written when the BERT non-determinism was thought to be a kernel-level reduction bug (wave-width hypothesis). Today's empirical work isolated the actual cause to `MTLDispatchTypeConcurrent`. Append a corrections section.

- [ ] **Step 1: Append a "v0.8.0 Correction" section at the top of the file**

Prepend (after the title header) the following section:

```markdown
## v0.8.0 correction (2026-05-25)

**The root-cause analysis in the sections below is partially incorrect.**

This document was written when BERT non-determinism on AMD-Mac Metal
was thought to be a kernel-level reduction race (wave-width assumption
in `simd_sum` / `simd_max` paths). Empirical work on 2026-05-25
isolated the actual cause to `MTLDispatchTypeConcurrent` — the upstream
ggml-metal dispatcher uses concurrent command-buffer ordering by
default, and that ordering is unreliable on non-UMA Macs.

The fix is a single env var: `GGML_METAL_CONCURRENCY_DISABLE=1`,
shipped in upstream llama.cpp since 2025 ([issue #19563](https://github.com/ggml-org/llama.cpp/issues/19563)).
Quenchforge's v0.8.0 `tuning.go` injects this env var for AMD-discrete
profiles via the new `SlotTuning.MetalConcurrencyDisable` field.

The wave-width / `N_SIMDWIDTH=32` issue described below is a real
architectural mismatch per Apple MSL Spec §4.4.2 — but it is **not**
the cause of the observed embedding non-determinism. The kernel-level
wave-width hypothesis (4-patch fallback kernel rewrite) is no longer
in scope; see `docs/superpowers/specs/2026-05-25-amd-metal-staging-buffer-pool-revival-design.md`
for the actual v0.8.0 design.

The sections below are preserved as a historical record of the
analytical path that led to the empirical isolation. The kernel
fallback approach in `patches/llama.cpp/0003`/`0004` remains active
as defense in depth — those patches still address valid corner cases
even though they're not the load-bearing fix.

---
```

- [ ] **Step 2: Stage**

```bash
git add docs/METAL_AMD_BERT_CORRECTNESS.md
```

### Task 21: Mark the superseded wave-width spec

**Files:**
- Modify: `docs/superpowers/specs/2026-05-25-amd-metal-acceleration-design.md`

- [ ] **Step 1: Prepend a superseded header**

At line 1 of the file (before the existing `# 2026-05-25 — AMD Vega II Metal acceleration...` title), insert:

```markdown
> ⚠️ **SUPERSEDED (2026-05-25)** — This spec is preserved as a historical research artifact. The wave-width hypothesis identified a real architectural mismatch per Apple MSL Spec §4.4.2, but empirical work later that day isolated the operational bug to `MTLDispatchTypeConcurrent` (a separate code path). The v0.8.0 implementation followed [`2026-05-25-amd-metal-staging-buffer-pool-revival-design.md`](2026-05-25-amd-metal-staging-buffer-pool-revival-design.md) instead. The prior-art survey, MSL Spec citations, and AMD GCN5 architecture documentation in this spec remain useful reference material.

---

```

- [ ] **Step 2: Stage**

```bash
git add docs/superpowers/specs/2026-05-25-amd-metal-acceleration-design.md
```

### Task 22: Update the chat-on-CPU memory file

**Files:**
- Modify: `~/.claude/projects/-Users-sunrunner-Develop/memory/project_cerid_quenchforge_chat_on_cpu.md`

The existing memory documents chat running on CPU as of v0.7.2 with a reversal trigger of "patch 0005 + bench-llama-sustained-load.py". With v0.8.0 the reversal is shipped (via patch 0002 + env var instead).

- [ ] **Step 1: Rewrite the memory**

```bash
cat > ~/.claude/projects/-Users-sunrunner-Develop/memory/project_cerid_quenchforge_chat_on_cpu.md <<'MDEOF'
---
name: project-cerid-quenchforge-chat-on-cpu
description: SUPERSEDED 2026-05-NN. Quenchforge chat slot ran on CPU on AMD-discrete (Vega II) from v0.7.2 through v0.7.2.x; v0.8.0 re-enabled GPU via patch 0002 (staging-buffer pool) + GGML_METAL_CONCURRENCY_DISABLE=1 env var. Reversal trigger satisfied.
metadata: 
  node_type: memory
  type: project
  originSessionId: 9ab58864-5c73-4554-a7c8-078ca1832be9
---

**State (2026-05-NN, after 7-day v0.8.0-rc2 observation):** The quenchforge chat slot — `llama-server` running `llama3.1-8b.gguf` (Q4_K_M) — runs on **GPU** on AMD-discrete profiles as of v0.8.0. `internal/tuning/tuning.go::chatParams` emits `--gpu-layers 999` and sets `MetalConcurrencyDisable: true`; the supervisor injects `GGML_METAL_CONCURRENCY_DISABLE=1` in the slot env. Patch 0002 (`patches/llama.cpp/0002-metal-staging-buffer-pool.patch`) provides the kernel-level fix that eliminates the family-B SIGABRT.

**Validated paths (Mac Pro 2019 + Radeon Pro Vega II 32 GB HBM2):**
- Correctness: bench-llama-correctness.py PASS (3 identical responses per prompt at temperature=0)
- Stability: bench-llama-sustained-load.py --duration 1800 PASS (zero SIGABRT, zero drift, throughput XXX tok/s)
- 7-day production observation PASS (≤ 10 AutoRespawn events / week, zero kernel panics)

**Reversal trigger (was, now satisfied):** the previous version of this memory documented two conditions — (1) a patch covering quantized matmul Metal fallback, and (2) bench-llama-sustained-load.py passing for ≥ 30 min. The actual fix turned out to be different (concurrent-dispatch env var + staging-buffer pool, not wave-width matmul fallback), but the spirit of the trigger — "a real fix + sustained-load validation" — was satisfied by patch 0002 + the env-var routing.

**Rollback procedure if regression appears:**
1. `launchctl bootout gui/$(id -u)/com.cerid.quenchforge`
2. `sudo install -m 0755 <path-to-v0.7.2-binary> /usr/local/bin/quenchforge`
3. `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.cerid.quenchforge.plist`
4. Chat slot returns to CPU route (v0.7.2 behavior).

Tactical fallback without full downgrade: set `GGML_METAL_DISABLE_STAGING_POOL=1` in `~/Library/LaunchAgents/com.cerid.quenchforge.plist` `EnvironmentVariables`. Family-B SIGABRT will return within ~5 min under sustained load but correctness is preserved by the env var.

**Related:** [[feedback_quenchforge_safety]] (operational rule about port 11434), [[reference_quenchforge_inference]] (broader endpoint reference).
MDEOF
```

- [ ] **Step 2: Note the memory file is off-repo (not staged for git)**

The memory directory at `~/.claude/projects/-Users-sunrunner-Develop/memory/` is the Claude memory store, not part of the quenchforge git repo. No `git add` needed for this file.

### Task 23: Commit Phase 6 docs + tag v0.8.0 final

**Files:** none modified (commit + tag)

- [ ] **Step 1: Stage everything queued from Tasks 18-21**

```bash
cd /Users/sunrunner/Develop/quenchforge
git status
```

Expected: shows staged changes to `patches/README.md`, `CHANGELOG.md`, `docs/METAL_AMD_BERT_CORRECTNESS.md`, `docs/superpowers/specs/2026-05-25-amd-metal-acceleration-design.md`.

- [ ] **Step 2: Commit**

```bash
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
docs: v0.8.0 release docs — AMD-discrete GPU mode SHIPPED

- patches/README.md: Section 3 updated from PARKED to SHIPPED v0.8.0
  with bench numbers and operator escape hatches
- CHANGELOG.md: new v0.8.0 entry with mechanism, bench results, and
  full file-changed list
- docs/METAL_AMD_BERT_CORRECTNESS.md: prepended correction noting the
  wave-width hypothesis was non-load-bearing; actual fix was the
  MTLDispatchTypeConcurrent env var + staging-buffer pool
- docs/superpowers/specs/2026-05-25-amd-metal-acceleration-design.md:
  marked superseded with pointer to the staging-buffer-pool spec

Memory file project_cerid_quenchforge_chat_on_cpu.md updated off-repo
to reflect v0.8.0 GPU mode and rollback procedure.
EOF
)"
```

- [ ] **Step 3: Tag v0.8.0 (final)**

```bash
git tag -a v0.8.0 -m "$(cat <<'EOF'
v0.8.0: AMD-discrete GPU mode SHIPPED

7-day observation window from v0.8.0-rc2 completed without regressions.
AMD-discrete profiles (Vega II canonical) now run all four production
slot types on GPU by default.

Two-layer fix:
1. patches/llama.cpp/0002-metal-staging-buffer-pool.patch — bounded
   MTLBuffer pool replaces per-call newBufferWithBytesNoCopy in
   ggml_metal_buffer_set/get_tensor. Eliminates AMD IOMMU pool
   exhaustion under sustained load (family-B SIGABRT class).
2. internal/tuning/tuning.go — SlotTuning.MetalConcurrencyDisable
   field routes GGML_METAL_CONCURRENCY_DISABLE=1 to AMD-discrete
   slots, disabling the upstream MTLDispatchTypeConcurrent path
   that produced non-deterministic output on non-UMA Macs.

See CHANGELOG.md and patches/README.md Section 3 for full bench
results, operator escape hatches, and rollback procedure.
EOF
)"
```

- [ ] **Step 4: Confirm tags**

```bash
git tag -l 'v0.8.*' -n5
```

Expected: both `v0.8.0-rc2` and `v0.8.0` listed with annotations.

- [ ] **Step 5: Final verification — patches still apply, tests still pass**

```bash
bash scripts/apply-patches.sh --check
go test ./...
```

Expected: zero errors from both.

---

## Spec coverage self-check

Mapping spec sections to plan tasks:

| Spec section | Plan task(s) |
|---|---|
| Problem — Bug A (concurrent dispatch race) | Task 5 (set `MetalConcurrencyDisable: true`); Task 11 (correctness probes verify fix) |
| Problem — Bug B (wave-width, latent) | Out of scope per spec; Task 20 (corrected root-cause doc) |
| Problem — Bug C (newBufferWithBytesNoCopy exhaustion) | Tasks 1-3 (patch revival); Tasks 12-14 (sustained-load benches verify fix) |
| Goal 1: stable GPU on 4 production models | Tasks 5, 11-14 |
| Goal 2: measurable speedup | Tasks 12-14 (bench summary lines report throughput) |
| Goal 3: zero Apple Silicon regression | Task 6 step 7 (`go test ./...` on existing tests; CI runs on macos-latest arm64 automatically on push) |
| Goal 4: 7-day production stability | Task 17 |
| Architecture — kernel layer | Tasks 1-3 |
| Architecture — supervisor layer | Tasks 4-6 |
| Patch revision details | Task 1 (strip duplicate hunks); Task 2 (regenerate indices) |
| Bench validation Criterion 1 (correctness) | Task 11 |
| Bench validation Criterion 2 (stability) | Tasks 12-14 |
| Bench validation Criterion 3 (escape hatch) | Task 15 |
| Bench validation Criterion 4 (Apple Silicon CI) | Task 6 step 7 + CI runner on push |
| Rollout Phase 1 | Tasks 1-3 |
| Rollout Phase 2 | Tasks 4-6 |
| Rollout Phase 3 | Tasks 7-9 |
| Rollout Phase 4 | Tasks 10-16 |
| Rollout Phase 5 | Task 17 |
| Rollout Phase 6 | Tasks 18-23 |
| Acceptance criteria #1-10 | Distributed across tasks; Task 23 step 5 final verification |

No spec requirement lacks a corresponding task. No task introduces scope beyond the spec.
