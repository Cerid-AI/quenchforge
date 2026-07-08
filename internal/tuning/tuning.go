// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

// Package tuning maps a (hardware profile, slot kind, runtime config)
// triple to the llama-server tuning knobs that profile needs for stable
// and well-throughput inference.
//
// Why a dedicated package: the supervisor's `buildSlotArgs` previously
// hard-coded per-kind and per-profile branches inline (chat-AMD safety
// flags, embed batch overrides). That worked for the two known crash
// families (FA throttle, prompt-cache GGML_ASSERT) but missed a third —
// the graph-compute buffer-corruption crash on sustained embed/rerank
// load on AMD discrete (see `patches/README.md`). Extending the inline
// branches further makes them harder to reason about and test. The
// tuning package is the single owner of "what flags does this slot
// kind on this hardware profile want", with table-driven tests.
//
// The function is intentionally pure (no I/O, no globals): it consumes
// a profile, the detected GPU VRAM (GB), a slot kind, and a config
// snapshot, and returns a `SlotTuning` describing the additional
// llama-server flags and env vars the supervisor should layer on top of
// the base argv. VRAM drives the adaptive context/ubatch sizing (see
// amdSizing) so smaller AMD cards fit without operator hand-tuning.
//
// Honors operator overrides: env-driven config fields (cfg.EmbedUbatchSize,
// cfg.EmbedMetalNCB, cfg.RerankBatchSize, cfg.RerankMetalNCB) win over
// profile-derived defaults so an operator with a different hardware
// configuration can always reach a known-good state.
package tuning

import (
	"github.com/cerid-ai/quenchforge/internal/config"
	"github.com/cerid-ai/quenchforge/internal/gateway"
	"github.com/cerid-ai/quenchforge/internal/hardware"
	"github.com/cerid-ai/quenchforge/internal/placement"
)

// AMD-discrete embed-slot defaults, baked from the Vega II bench at
// v0.6.0 + v0.6.1 release validation. See `embedParams` docstring for
// the empirical basis.
const (
	// amdEmbedUbatchDefault caps per-call Metal staging-buffer
	// pressure. ubatch=1024 sustained 0.5 req/s for ~70 min in the
	// cerid LongMemEval canonical run with a single auto-respawned
	// family-B crash at ~23 min uptime. ubatch=8192 (v0.5.x default)
	// crashed within ~80 calls / ~2 min on the same hardware.
	amdEmbedUbatchDefault = 1024

	// amdEmbedMetalNCBDefault serialises Metal command-buffer
	// submission so the staging-buffer pool drains between commands.
	// =1 was the smallest-overhead stability-preserving value in the
	// bench sweep; =2 (the global default) produced 5x more SIGABRTs.
	amdEmbedMetalNCBDefault = 1
)

// SlotTuning describes the per-slot llama-server flags and env vars
// the supervisor layers on top of the base argv built by
// `buildSlotArgs`. Zero values are valid and mean "no override".
type SlotTuning struct {
	// UbatchSize, when non-zero, is added as `--ubatch-size N`. For
	// embed and code-embed slots, this is the physical batch llama.cpp
	// uploads to the Metal staging buffer per forward pass. Smaller =
	// less per-call staging pressure (relevant for AMD discrete);
	// larger = fewer kernel launches per request.
	UbatchSize int

	// BatchSize, when non-zero, is added as `--batch-size N`. Must be
	// >= UbatchSize. For embed slots, equal to UbatchSize so single
	// large inputs fit a single batch; for rerank, this is the
	// physical batch budget for (query, doc) pairs.
	BatchSize int

	// MetalNCB, when non-zero, sets `GGML_METAL_N_CB` in the slot's
	// env. Caps concurrent in-flight Metal command buffers. Lowering
	// to 1 forces serialisation and lets the AMD discrete driver's
	// PCIe-staging-buffer pool drain between commands — primary
	// mitigation for the family-B sustained-load crash.
	MetalNCB int

	// ExtraArgs are profile-specific flags appended verbatim. Used for
	// the chat-slot AMD safety flags (--flash-attn off, --cache-ram 0,
	// --no-cache-prompt) that protect against the LCP-prompt-save and
	// FA-CPU-fallback crashes already documented in patches/README.md.
	ExtraArgs []string

	// AutoRespawn signals whether the supervisor should restart this
	// slot on non-zero exit. AMD discrete embed/rerank slots are the
	// only ones that benefit today — sustained-load Metal corruption
	// is non-deterministic; restarting clears the broken state.
	AutoRespawn bool

	// MetalConcurrencyDisable, when true, sets `GGML_METAL_CONCURRENCY_DISABLE=1`
	// in the slot's env. Required on AMD discrete (non-UMA Metal): upstream's
	// MTLDispatchTypeConcurrent path's command-buffer ordering is unreliable on
	// non-UMA drivers and causes non-deterministic output across BERT-family
	// models and chat-decode races. See llama.cpp issue #19563 and patch 0002.
	// Apple Silicon (UMA) does not need this; the concurrent path is correct there.
	MetalConcurrencyDisable bool

	// ContextSize, when non-zero, is a VRAM-tier-derived ceiling on the
	// slot's --ctx-size. buildSlotArgs applies it as min(cfg.MaxContext,
	// ContextSize), so it only ever LOWERS the configured context: small
	// AMD cards (<= 11 GB) get a KV cache that fits without manual tuning,
	// while >= 12 GB cards and non-AMD profiles leave this 0 (no cap) and
	// keep cfg.MaxContext verbatim. See amdSizing.
	ContextSize int
}

// KernelParams returns the tuning the supervisor should apply for the
// given (profile, kind) pair, honouring any operator overrides in cfg.
//
// The contract is additive: returned values are layered on top of the
// base llama-server args built by buildSlotArgs. Empty / zero fields
// signal "use the upstream default".
//
// Apple Silicon and unknown profiles get zero tuning across all slot
// kinds: their Metal stack is unified-memory and never enters the
// `newBufferWithBytesNoCopy` staging-buffer path (see
// `~/Develop/quenchforge/llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m:1665-1717`
// — the `buf->is_shared` fast path uses plain `memcpy`). Adding flags
// on those profiles would regress throughput without any safety win.
func KernelParams(profile hardware.Profile, vramGB int, kind gateway.SlotKind, cfg config.Config) SlotTuning {
	// Device placement comes first: the GPU safety tuning below is only
	// meaningful when the slot actually runs on the GPU. On a host where the
	// placement policy routes this kind to the CPU (e.g. chat on AMD-discrete,
	// where the Metal path is ~7x slower than CPU), return the minimal CPU
	// tuning instead.
	if PolicyFor(profile, cfg).Device(string(kind)) == placement.CPU {
		return cpuTuning(kind, cfg)
	}
	return gpuKernelParams(profile, vramGB, kind, cfg)
}

// KernelParamsForDevice returns the tuning for a slot whose device is decided
// by the caller, bypassing the placement policy. The supervisor uses this for
// "auto"-placed kinds where it launches BOTH a GPU and a CPU instance of the
// same (profile, kind): each instance must get the tuning for the device it
// actually runs on, not the single device the policy would pick. dev==CPU
// yields the minimal CPU tuning; dev==GPU yields the same per-kind GPU params
// KernelParams produces for a GPU-placed slot.
func KernelParamsForDevice(profile hardware.Profile, vramGB int, kind gateway.SlotKind, cfg config.Config, dev placement.Device) SlotTuning {
	if dev == placement.CPU {
		return cpuTuning(kind, cfg)
	}
	return gpuKernelParams(profile, vramGB, kind, cfg)
}

// gpuKernelParams is the per-kind GPU tuning switch shared by KernelParams and
// KernelParamsForDevice. It assumes the slot runs on the GPU; the CPU decision
// is made by the callers above.
func gpuKernelParams(profile hardware.Profile, vramGB int, kind gateway.SlotKind, cfg config.Config) SlotTuning {
	switch kind {
	case gateway.KindChat:
		return chatParams(profile, vramGB)
	case gateway.KindEmbed, gateway.KindCodeEmbed:
		return embedParams(profile, vramGB, cfg)
	case gateway.KindRerank:
		return rerankParams(profile, vramGB, cfg)
	}
	// Whisper / imagegen and any future kinds fall through unchanged.
	return SlotTuning{}
}

// PolicyFor builds the device-placement policy for a host profile plus any
// operator overrides in cfg. Exposed so the gateway can consult the same
// policy for per-request "auto" routing decisions.
func PolicyFor(profile hardware.Profile, cfg config.Config) placement.Policy {
	return placement.NewPolicy(profileIsAMDDiscrete(profile), map[string]string{
		placement.KindChat:      cfg.PlaceChat,
		placement.KindEmbed:     cfg.PlaceEmbed,
		placement.KindCodeEmbed: cfg.PlaceCodeEmbed,
		placement.KindRerank:    cfg.PlaceRerank,
	})
}

// cpuTuning is the CPU-route tuning: all layers on CPU and none of the
// GPU-only Metal env or cache-disabling flags, so the CPU prompt cache stays
// on (the measured fast path: chat ~3.8s vs ~27s GPU for 32 tokens on Vega II).
// The --embedding / --reranking slot flags are added by the slotSpec, not here.
//
// Embedding-family kinds (embed / code-embed / rerank) size the physical
// batch to the context: llama-server's embedding and rerank pooling paths
// reject any single sequence longer than n_ubatch ("input too large to
// process", HTTP 500), and the CPU backend has none of the Metal
// staging-buffer pressure that motivates the AMD-GPU ubatch caps — so the
// non-AMD embedParams principle applies verbatim: any input that fits the
// context fits a single batch. Before v0.9.1 this returned only
// `--gpu-layers 0`, silently reverting CPU-placed slots to llama-server's
// 512-token default batch — which both reintroduced the v0.5.0
// 512-token-limit bug on the placement CPU twins and made the documented
// QUENCHFORGE_RERANK_BATCH_SIZE / QUENCHFORGE_EMBED_UBATCH_SIZE overrides
// dead knobs on CPU-placed slots (2026-07-08 cerid eval incident).
func cpuTuning(kind gateway.SlotKind, cfg config.Config) SlotTuning {
	t := SlotTuning{ExtraArgs: []string{"--gpu-layers", "0"}}
	switch kind {
	case gateway.KindEmbed, gateway.KindCodeEmbed:
		b := cfg.MaxContext
		if cfg.EmbedUbatchSize > 0 {
			b = cfg.EmbedUbatchSize
		}
		t.UbatchSize, t.BatchSize = b, b
	case gateway.KindRerank:
		b := cfg.MaxContext
		if cfg.RerankBatchSize > 0 {
			b = cfg.RerankBatchSize
		}
		t.UbatchSize, t.BatchSize = b, b
	}
	return t
}

// chatParams returns the chat-slot tuning. AMD-discrete profiles get
// the existing three safety flags AND AutoRespawn — sustained chat
// inference (cerid LongMemEval extraction, agentic tool-use loops)
// produces family-B `GGML_ASSERT(buf_src)` crashes the same as embed
// under sustained load. v0.6.0 missed wiring AutoRespawn here on the
// theory that chat is naturally bursty; cerid eval workloads broke
// that assumption (chat.log entry at 2026-05-16T23:14 — task 143
// hit GGML_ASSERT at `set_tensor` after ~30 successful chat calls).
func chatParams(profile hardware.Profile, vramGB int) SlotTuning {
	if !profileIsAMDDiscrete(profile) {
		return SlotTuning{}
	}
	ctxCap, _ := amdSizing(vramGB)
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
		ContextSize:             ctxCap,
		MetalConcurrencyDisable: true,
		AutoRespawn:             true,
	}
}

// embedParams returns embed (and code-embed) slot tuning.
//
// Behaviour:
//   - Apple Silicon / non-AMD profiles: ubatch and batch default to
//     cfg.MaxContext (preserves the v0.5.0 contextplus 512-token-limit
//     fix — any input that fits the context fits a single batch).
//   - AMD-discrete profiles get a Vega-II-tested-stable default of 1024.
//     Empirical evidence (cerid LongMemEval canonical run 2026-05-17,
//     plus the v0.6.0 release-validation bench): ubatch=1024 with
//     MetalNCB=1 sustains 0.5 req/sec indefinitely on Vega II with
//     zero crashes over an hour; ubatch=8192 crashed within ~80 calls
//     via the family-B `ggml_metal_buffer_set_tensor` SIGABRT. Other
//     AMD profiles (W6800X, RDNA1/2) inherit Vega II's value until
//     benched independently.
//   - Operator overrides (cfg.EmbedUbatchSize, cfg.EmbedMetalNCB) win
//     over the profile-derived defaults whenever they are non-zero.
//   - AMD discrete profiles additionally enable AutoRespawn — the
//     supervisor brings the slot back on a Metal SIGABRT instead of
//     leaving it dead until manual restart.
func embedParams(profile hardware.Profile, vramGB int, cfg config.Config) SlotTuning {
	ubatch := cfg.MaxContext
	metalNCB := cfg.MetalNCB
	ctxCap := 0
	if profileIsAMDDiscrete(profile) {
		// AMD-discrete on GPU (v0.8.0) needs the ubatch cap re-enabled —
		// it bounds per-call Metal staging-buffer pressure even with patch 0002's
		// pool in place. As of v0.8.0 the cap is VRAM-tier-adaptive (1024 on
		// >= 12 GB cards, 512 on 8 GB, 256 on 4 GB) so smaller cards don't OOM
		// without an operator setting QUENCHFORGE_EMBED_UBATCH_SIZE by hand.
		// CLAUDE.md operational gotcha #2 documents this knob.
		ctxCap, ubatch = amdSizing(vramGB)
		metalNCB = amdEmbedMetalNCBDefault
	}
	if cfg.EmbedUbatchSize > 0 {
		ubatch = cfg.EmbedUbatchSize
	}
	t := SlotTuning{
		UbatchSize:  ubatch,
		BatchSize:   ubatch,
		MetalNCB:    metalNCB,
		ContextSize: ctxCap,
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

// rerankParams returns rerank (GPU-placed) slot tuning.
//
// llama-server's rerank pooling path rejects any (query, doc) pair longer
// than n_ubatch ("input too large to process", HTTP 500), and its 512-token
// default is too small for the 600–2k-token chunks modern rerankers
// (bge-reranker-v2-m3 and kin) are routinely fed. v0.9.0 and earlier shipped
// no default on the theory that the right value is workload-specific; the
// 2026-07-08 cerid eval incident showed the 512 default is a deterministic
// 500 generator, which is strictly worse than any reasonable default. So:
// non-AMD profiles size the batch to the context (same principle as
// embedParams — any pair that fits the context fits a batch); AMD-discrete
// keeps the bench-validated amdSizing ubatch as a Metal staging-pressure
// cap. QUENCHFORGE_RERANK_BATCH_SIZE still overrides both.
//
// AutoRespawn fires on AMD discrete same as embed. AMD profiles also
// get the conservative MetalNCB=1 default same as embed.
func rerankParams(profile hardware.Profile, vramGB int, cfg config.Config) SlotTuning {
	t := SlotTuning{}
	batch := cfg.MaxContext
	if profileIsAMDDiscrete(profile) {
		ctxCap, ubatch := amdSizing(vramGB)
		t.ContextSize = ctxCap
		t.MetalNCB = amdEmbedMetalNCBDefault
		batch = ubatch
	}
	if cfg.RerankBatchSize > 0 {
		batch = cfg.RerankBatchSize
	}
	t.BatchSize = batch
	t.UbatchSize = batch
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

// amdSizing returns the VRAM-tier-adaptive context ceiling and embed
// ubatch for an AMD-discrete profile. A contextCap of 0 means "no cap —
// honour cfg.MaxContext as-is".
//
// The high tier (>= 12 GB: Vega II/Duo, W6800X, W6900X, Vega 56/64,
// 5600M) keeps the Vega-II-validated values: no context cap, ubatch
// 1024. Smaller cards scale both down so the KV cache + Metal staging
// buffers fit without an operator hand-tuning QUENCHFORGE_MAX_CONTEXT /
// QUENCHFORGE_EMBED_UBATCH_SIZE:
//
//	VRAM        context cap   embed ubatch   example cards
//	>= 12 GB    none (0)      1024           Vega II/Duo, W6800X, W6900X, Vega 56/64, 5600M
//	7-11 GB     4096          512            RX 5700 / 5700 XT, W5700X
//	<= 6 GB     2048          256            4 GB MacBook Pro dGPUs (5300M/5500M), Polaris 560X
//
// vramGB <= 0 means detection could not read VRAM; treat it as the high
// tier so a probe miss never throttles the validated Vega II path. The
// caps only ever LOWER cfg.MaxContext (buildSlotArgs takes the min), so a
// high-VRAM operator who raised QUENCHFORGE_MAX_CONTEXT is unaffected.
func amdSizing(vramGB int) (contextCap, ubatch int) {
	switch {
	case vramGB <= 0 || vramGB >= 12:
		return 0, amdEmbedUbatchDefault
	case vramGB >= 7:
		return 4096, 512
	default:
		return 2048, 256
	}
}

// profileIsAMDDiscrete inlines hardware.Info.IsAMDDiscrete logic
// against a raw Profile (we don't have a full Info here). Kept in sync
// with internal/hardware/hardware.go::IsAMDDiscrete by the test in
// tuning_test.go that exhaustively iterates all hardware.Profile values.
func profileIsAMDDiscrete(p hardware.Profile) bool {
	switch p {
	case hardware.ProfileVegaPro,
		hardware.ProfileW6800X,
		hardware.ProfileRDNA1,
		hardware.ProfileRDNA2:
		return true
	}
	return false
}
