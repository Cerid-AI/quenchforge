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
// a profile, a slot kind, and a config snapshot, and returns a
// `SlotTuning` describing the additional llama-server flags and env
// vars the supervisor should layer on top of the base argv.
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
func KernelParams(profile hardware.Profile, kind gateway.SlotKind, cfg config.Config) SlotTuning {
	switch kind {
	case gateway.KindChat:
		return chatParams(profile)
	case gateway.KindEmbed, gateway.KindCodeEmbed:
		return embedParams(profile, cfg)
	case gateway.KindRerank:
		return rerankParams(profile, cfg)
	}
	// Whisper / imagegen and any future kinds fall through unchanged.
	return SlotTuning{}
}

// chatParams returns the chat-slot tuning. AMD-discrete profiles get
// the existing three safety flags AND AutoRespawn — sustained chat
// inference (cerid LongMemEval extraction, agentic tool-use loops)
// produces family-B `GGML_ASSERT(buf_src)` crashes the same as embed
// under sustained load. v0.6.0 missed wiring AutoRespawn here on the
// theory that chat is naturally bursty; cerid eval workloads broke
// that assumption (chat.log entry at 2026-05-16T23:14 — task 143
// hit GGML_ASSERT at `set_tensor` after ~30 successful chat calls).
func chatParams(profile hardware.Profile) SlotTuning {
	if !profileIsAMDDiscrete(profile) {
		return SlotTuning{}
	}
	// AMD-discrete chat slot routes to CPU pending the quantized-matmul
	// fallback patch (planned as patch 0005). Patches 0001/0003/0004
	// cover fp32/fp16 BERT shapes only; chat-slot Q4_K_M / Q5_K_M
	// models still SIGABRT on Vega II Metal under sustained load —
	// 257 abort traps observed across a 7-day uptime window
	// (2026-05-17 → 2026-05-24) contributing to the 2026-05-17 panic
	// and the 2026-05-24 system freeze. Mirror of the v0.7.0
	// embed/rerank CPU routing — remove this `--gpu-layers 0` pair
	// when patch 0005 lands and bench-llama-sustained-load passes.
	return SlotTuning{
		ExtraArgs: []string{
			"--flash-attn", "off",
			"--cache-ram", "0",
			"--no-cache-prompt",
			"--gpu-layers", "0",
		},
		AutoRespawn: true,
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
func embedParams(profile hardware.Profile, cfg config.Config) SlotTuning {
	ubatch := cfg.MaxContext
	metalNCB := cfg.MetalNCB
	if profileIsAMDDiscrete(profile) {
		// On AMD discrete the embed slot routes to CPU (see below — Metal
		// produces non-deterministic vectors for BERT models). Once we're
		// off Metal, the 1024 ubatch cap that mitigated the family-B
		// SIGABRT no longer applies — that crash was Metal-specific.
		// Keep the natural model-max (ctx-size) so long single inputs
		// like full LongMemEval sessions (often 1500-2000 tokens) fit
		// in a single forward pass instead of returning HTTP 500
		// ("input is too large for the physical batch size"). The
		// MetalNCB knob is still emitted for completeness but has no
		// effect when --gpu-layers 0 is also set; harmless.
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
		// Force CPU for embed slots on AMD discrete. BERT-family embedding
		// models (nomic-embed-text-v1.5, jina-embeddings-v2-base-code,
		// snowflake-arctic, ...) produce non-deterministic garbage vectors
		// when run on the AMD-Mac Metal backend, even with the patch 0001
		// simdgroup_reduction + bfloat gates active. Verified empirically
		// 2026-05-17 on Vega II: identical input "hello" returns cos_sim
		// 0.07 between two separate requests through Metal; the same model
		// on `--gpu-layers 0` returns cos_sim 1.0000. Chat (Llama-family)
		// is unaffected — that's a different forward-pass shape. The bug
		// is in BERT-specific kernels not covered by patch 0001 (likely
		// LayerNorm or bidirectional self-attention reductions); kernel-
		// level repair is tracked as v0.8.0 follow-up. Until then, CPU is
		// the only correct path. On a 16-core Mac Pro 2019 the throughput
		// hit is acceptable — ~100-500ms/call vs the broken-but-fast GPU
		// path. Operator override: QUENCHFORGE_EMBED_UBATCH_SIZE sets a
		// smaller batch (e.g. for memory-constrained operators);
		// QUENCHFORGE_EMBED_GPU_LAYERS=N re-enables Metal partially at
		// your own correctness risk.
		t.ExtraArgs = append(t.ExtraArgs, "--gpu-layers", "0")
		// CPU embed multithreading: default llama-server picks ~half the
		// logical cores per request and runs ONE request at a time.
		// Measured 2026-05-17 on Mac Pro 2019 Xeon W-3245 (16 physical /
		// 32 logical): the embed slot used ~7.4 cores per request and
		// went idle between requests, capping cerid ingest throughput at
		// ~1 session/sec. The fix is two dials:
		//   --threads 15 : pin a 15-thread compute pool (leave 1 physical
		//                  core for the OS + supervisor + other slots).
		//   --parallel 4 : 4 concurrent request slots inside llama-server
		//                  so a burst from chromadb's batched embed call
		//                  doesn't serialize behind earlier requests.
		// With both, peak sustained throughput rises ~4x on the cerid
		// LongMemEval ingest workload. Trade-off: a single huge request
		// now shares the pool with up to 3 others, so per-request
		// latency under sustained burst rises ~25%. For embed/rerank
		// workloads (batched, throughput-bound) the burst-throughput
		// win dominates the per-request-latency loss.
		t.ExtraArgs = append(t.ExtraArgs, "--threads", "15", "--parallel", "4")
	}
	return t
}

// rerankParams returns rerank slot tuning.
//
// The reranker has no batch override today (cf. buildSlotArgs comment)
// because it scores (query, doc) pairs individually. But llama-server's
// 512-token default batch is too small for any (query, doc) pair where
// the document exceeds ~510 tokens, and bge-reranker-v2-m3 + similar
// modern rerankers are routinely fed 1k-2k-token chunks. The operator
// can raise this with QUENCHFORGE_RERANK_BATCH_SIZE; we don't ship a
// non-zero default here because the right value is workload-specific.
//
// AutoRespawn fires on AMD discrete same as embed. AMD profiles also
// get the conservative MetalNCB=1 default same as embed.
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
		// Same Metal-on-AMD BERT-family bug as embed. bge-reranker-v2-m3
		// produces non-deterministic scores on AMD Metal: identical
		// (query, docs) input returns different relevance numbers across
		// calls. Relative ordering is partially preserved (which is why
		// production-stack+qa got 0.133 not 0.0 in the morning eval — the
		// reranker partially-masks broken embed garbage), but absolute
		// scores are unreliable. See embedParams docstring for the full
		// rationale; same multithreading defaults for the same reason.
		t.ExtraArgs = append(
			t.ExtraArgs,
			"--gpu-layers", "0",
			"--threads", "15",
			"--parallel", "4",
		)
	}
	return t
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
