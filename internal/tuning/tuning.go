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
	return SlotTuning{
		ExtraArgs: []string{
			"--flash-attn", "off",
			"--cache-ram", "0",
			"--no-cache-prompt",
		},
		AutoRespawn: true,
	}
}

// embedParams returns embed (and code-embed) slot tuning.
//
// Behaviour:
//   - All profiles: ubatch and batch default to cfg.MaxContext (preserves
//     the v0.5.0 contextplus 512-token-limit fix — any input that fits
//     the context fits a single batch).
//   - Operator overrides (cfg.EmbedUbatchSize, cfg.EmbedMetalNCB) win
//     over the profile-derived defaults whenever they are non-zero.
//   - AMD discrete profiles additionally enable AutoRespawn — the
//     supervisor brings the slot back on a Metal SIGABRT instead of
//     leaving it dead until manual restart.
//
// We deliberately do NOT lower the AMD defaults in this PR (per the
// approved plan). PR 2 benches Vega II to find the smallest stable
// overhead and flips the defaults there.
func embedParams(profile hardware.Profile, cfg config.Config) SlotTuning {
	ubatch := cfg.MaxContext
	if cfg.EmbedUbatchSize > 0 {
		ubatch = cfg.EmbedUbatchSize
	}
	t := SlotTuning{
		UbatchSize: ubatch,
		BatchSize:  ubatch,
	}
	if cfg.EmbedMetalNCB > 0 {
		t.MetalNCB = cfg.EmbedMetalNCB
	}
	if profileIsAMDDiscrete(profile) {
		t.AutoRespawn = true
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
// AutoRespawn fires on AMD discrete same as embed.
func rerankParams(profile hardware.Profile, cfg config.Config) SlotTuning {
	t := SlotTuning{}
	if cfg.RerankBatchSize > 0 {
		t.BatchSize = cfg.RerankBatchSize
		t.UbatchSize = cfg.RerankBatchSize
	}
	if cfg.RerankMetalNCB > 0 {
		t.MetalNCB = cfg.RerankMetalNCB
	}
	if profileIsAMDDiscrete(profile) {
		t.AutoRespawn = true
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
