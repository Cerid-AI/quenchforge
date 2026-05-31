// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package tuning

import (
	"slices"
	"testing"

	"github.com/cerid-ai/quenchforge/internal/config"
	"github.com/cerid-ai/quenchforge/internal/gateway"
	"github.com/cerid-ai/quenchforge/internal/hardware"
)

// allProfiles is the exhaustive list of hardware.Profile values the
// tuning module's behaviour must be defined over. When a new profile is
// added in internal/hardware, this slice must grow and KernelParams's
// AMD-detection switch must be re-audited.
var allProfiles = []hardware.Profile{
	hardware.ProfileVegaPro,
	hardware.ProfileW6800X,
	hardware.ProfileRDNA1,
	hardware.ProfileRDNA2,
	hardware.ProfileAppleSilicon,
	hardware.ProfileIGPU,
	hardware.ProfileCPU,
	hardware.ProfileUnknown,
}

// amdProfiles must match hardware.Info.IsAMDDiscrete (whitebox). The
// constancy test below enforces it.
var amdProfiles = []hardware.Profile{
	hardware.ProfileVegaPro,
	hardware.ProfileW6800X,
	hardware.ProfileRDNA1,
	hardware.ProfileRDNA2,
}

// vramHigh is a >= 12 GB headline VRAM (e.g. Vega II's 32 GB). At this
// tier amdSizing imposes no context cap and keeps ubatch 1024, so the
// pre-v0.8.0 assertions below remain valid verbatim. Low-VRAM behaviour
// has its own dedicated tests.
const vramHigh = 32

func TestProfileIsAMDDiscrete_MatchesHardwarePackage(t *testing.T) {
	// Cross-check our inline AMD predicate against hardware.Info's
	// IsAMDDiscrete. If hardware adds or removes a profile from the
	// AMD list and we miss the update, this test catches it.
	for _, p := range allProfiles {
		info := hardware.Info{Profile: p}
		got := profileIsAMDDiscrete(p)
		want := info.IsAMDDiscrete()
		if got != want {
			t.Errorf("profile %s: tuning says AMD=%v, hardware says AMD=%v",
				p, got, want)
		}
	}
}

func TestKernelParams_ChatAMDGetsGPUWithConcurrencyDisable(t *testing.T) {
	cfg := config.Config{MaxContext: 8192}
	for _, p := range amdProfiles {
		t.Run(string(p), func(t *testing.T) {
			tn := KernelParams(p, vramHigh, gateway.KindChat, cfg)
			wantExtra := []string{
				"--flash-attn", "off",
				"--cache-ram", "0",
				"--no-cache-prompt",
				"--gpu-layers", "999",
			}
			if !slices.Equal(tn.ExtraArgs, wantExtra) {
				t.Errorf("chat AMD %s ExtraArgs = %v, want %v",
					p, tn.ExtraArgs, wantExtra)
			}
			// Chat doesn't get embed-style ubatch / batch overrides,
			// but DOES get AutoRespawn on AMD discrete — sustained
			// chat workloads (cerid LongMemEval extraction, agent
			// loops) hit family-B SIGABRT same as embed. v0.6.0
			// shipped without this and the chat slot stayed dead;
			// v0.6.1 fixed it.
			if tn.UbatchSize != 0 || tn.BatchSize != 0 || tn.MetalNCB != 0 {
				t.Errorf("chat AMD %s unexpected non-zero tuning: %+v", p, tn)
			}
			if !tn.MetalConcurrencyDisable {
				t.Errorf("chat AMD %s should have MetalConcurrencyDisable=true", p)
			}
			if !tn.AutoRespawn {
				t.Errorf("chat AMD %s should request AutoRespawn", p)
			}
		})
	}
}

func TestKernelParams_ChatNonAMDIsEmpty(t *testing.T) {
	// Apple Silicon, CPU, iGPU, unknown — none should get the AMD
	// chat safety flags.
	cfg := config.Config{MaxContext: 8192}
	for _, p := range allProfiles {
		if profileIsAMDDiscrete(p) {
			continue
		}
		t.Run(string(p), func(t *testing.T) {
			tn := KernelParams(p, vramHigh, gateway.KindChat, cfg)
			if !slices.Equal(tn.ExtraArgs, nil) && len(tn.ExtraArgs) != 0 {
				t.Errorf("chat %s should emit no ExtraArgs, got %v",
					p, tn.ExtraArgs)
			}
		})
	}
}

func TestKernelParams_EmbedDefaultsByProfile(t *testing.T) {
	// Non-AMD profiles use MaxContext for ubatch/batch on embed slots —
	// long single inputs need the full natural model context as a single
	// forward pass, otherwise llama-server returns HTTP 500 "input is
	// too large for the physical batch size".
	// AMD-discrete profiles cap at amdEmbedUbatchDefault (1024) — GPU
	// mode re-enabled in v0.8.0 re-exposes the Metal staging-buffer
	// pressure that this cap bounds. See embedParams docstring.
	cfg := config.Config{MaxContext: 8192}
	for _, p := range allProfiles {
		for _, k := range []gateway.SlotKind{gateway.KindEmbed, gateway.KindCodeEmbed} {
			t.Run(string(p)+"/"+string(k), func(t *testing.T) {
				tn := KernelParams(p, vramHigh, k, cfg)
				wantUbatch := 8192
				wantNCB := 0
				if profileIsAMDDiscrete(p) {
					wantUbatch = amdEmbedUbatchDefault
					wantNCB = amdEmbedMetalNCBDefault
				}
				if tn.UbatchSize != wantUbatch {
					t.Errorf("%s %s UbatchSize = %d, want %d",
						p, k, tn.UbatchSize, wantUbatch)
				}
				if tn.BatchSize != wantUbatch {
					t.Errorf("%s %s BatchSize = %d, want %d",
						p, k, tn.BatchSize, wantUbatch)
				}
				if tn.MetalNCB != wantNCB {
					t.Errorf("%s %s MetalNCB = %d, want %d",
						p, k, tn.MetalNCB, wantNCB)
				}
			})
		}
	}
}

func TestKernelParams_EmbedAMDGetsGPUWithConcurrencyDisable(t *testing.T) {
	// AMD discrete profiles must include `--gpu-layers 999` (GPU mode,
	// re-enabled v0.8.0) and MetalConcurrencyDisable=true. Apple Silicon
	// and unknown profiles must NOT get either — the MTLDispatchTypeConcurrent
	// race is AMD-discrete-only and the GPU-layers flag is unnecessary on UMA.
	cfg := config.Config{MaxContext: 8192}
	for _, p := range allProfiles {
		for _, k := range []gateway.SlotKind{gateway.KindEmbed, gateway.KindCodeEmbed} {
			t.Run(string(p)+"/"+string(k), func(t *testing.T) {
				tn := KernelParams(p, vramHigh, k, cfg)
				hasGPUFlag := containsSubslice(tn.ExtraArgs, []string{"--gpu-layers", "999"})
				if profileIsAMDDiscrete(p) {
					if !hasGPUFlag {
						t.Errorf("%s %s missing --gpu-layers 999; ExtraArgs=%v",
							p, k, tn.ExtraArgs)
					}
					if !tn.MetalConcurrencyDisable {
						t.Errorf("%s %s should have MetalConcurrencyDisable=true", p, k)
					}
				} else {
					if hasGPUFlag {
						t.Errorf("%s %s should NOT set --gpu-layers 999; ExtraArgs=%v",
							p, k, tn.ExtraArgs)
					}
					if tn.MetalConcurrencyDisable {
						t.Errorf("%s %s should NOT have MetalConcurrencyDisable=true", p, k)
					}
				}
			})
		}
	}
}

func TestKernelParams_EmbedAMDMultithreading(t *testing.T) {
	// AMD discrete CPU-routed embed must also surface --threads 15
	// --parallel 4 so the 16-physical-core Mac Pro 2019 doesn't bottle-
	// neck on the default ~7-cores-per-request-one-at-a-time pattern.
	// See embedParams docstring. Non-AMD profiles get neither flag.
	cfg := config.Config{MaxContext: 8192}
	for _, p := range allProfiles {
		for _, k := range []gateway.SlotKind{gateway.KindEmbed, gateway.KindCodeEmbed, gateway.KindRerank} {
			t.Run(string(p)+"/"+string(k), func(t *testing.T) {
				tn := KernelParams(p, vramHigh, k, cfg)
				hasThreads := containsArgPair(tn.ExtraArgs, "--threads", "15")
				hasParallel := containsArgPair(tn.ExtraArgs, "--parallel", "4")
				if profileIsAMDDiscrete(p) {
					if !hasThreads {
						t.Errorf("%s %s missing --threads 15; ExtraArgs=%v",
							p, k, tn.ExtraArgs)
					}
					if !hasParallel {
						t.Errorf("%s %s missing --parallel 4; ExtraArgs=%v",
							p, k, tn.ExtraArgs)
					}
				} else {
					if hasThreads {
						t.Errorf("%s %s should NOT set --threads; ExtraArgs=%v",
							p, k, tn.ExtraArgs)
					}
					if hasParallel {
						t.Errorf("%s %s should NOT set --parallel; ExtraArgs=%v",
							p, k, tn.ExtraArgs)
					}
				}
			})
		}
	}
}

func containsArgPair(args []string, flag, value string) bool {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == flag && args[i+1] == value {
			return true
		}
	}
	return false
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

func TestKernelParams_RerankAMDGetsGPUWithConcurrencyDisable(t *testing.T) {
	// Same BERT-family Metal concurrency fix applies to bge-reranker-v2-m3.
	// AMD discrete must have --gpu-layers 999 and MetalConcurrencyDisable=true;
	// non-AMD profiles must not.
	cfg := config.Config{MaxContext: 8192}
	for _, p := range allProfiles {
		t.Run(string(p), func(t *testing.T) {
			tn := KernelParams(p, vramHigh, gateway.KindRerank, cfg)
			hasGPUFlag := containsSubslice(tn.ExtraArgs, []string{"--gpu-layers", "999"})
			if profileIsAMDDiscrete(p) {
				if !hasGPUFlag {
					t.Errorf("%s rerank missing --gpu-layers 999; ExtraArgs=%v",
						p, tn.ExtraArgs)
				}
				if !tn.MetalConcurrencyDisable {
					t.Errorf("%s rerank should have MetalConcurrencyDisable=true", p)
				}
			} else {
				if hasGPUFlag {
					t.Errorf("%s rerank should NOT set --gpu-layers 999; ExtraArgs=%v",
						p, tn.ExtraArgs)
				}
				if tn.MetalConcurrencyDisable {
					t.Errorf("%s rerank should NOT have MetalConcurrencyDisable=true", p)
				}
			}
		})
	}
}

func TestKernelParams_RerankAMDGetsMetalNCBDefault(t *testing.T) {
	// AMD rerank slots also inherit MetalNCB=1; non-AMD keeps 0
	// (inherit global).
	cfg := config.Config{MaxContext: 8192}
	for _, p := range allProfiles {
		t.Run(string(p), func(t *testing.T) {
			tn := KernelParams(p, vramHigh, gateway.KindRerank, cfg)
			wantNCB := 0
			if profileIsAMDDiscrete(p) {
				wantNCB = amdEmbedMetalNCBDefault
			}
			if tn.MetalNCB != wantNCB {
				t.Errorf("%s rerank MetalNCB = %d, want %d",
					p, tn.MetalNCB, wantNCB)
			}
		})
	}
}

func TestKernelParams_EmbedHonoursUbatchOverride(t *testing.T) {
	// Operator-set QUENCHFORGE_EMBED_UBATCH_SIZE wins.
	cfg := config.Config{MaxContext: 8192, EmbedUbatchSize: 1024}
	tn := KernelParams(hardware.ProfileVegaPro, vramHigh, gateway.KindEmbed, cfg)
	if tn.UbatchSize != 1024 {
		t.Errorf("UbatchSize = %d, want 1024 (env override)", tn.UbatchSize)
	}
	if tn.BatchSize != 1024 {
		t.Errorf("BatchSize = %d, want 1024 (mirrors UbatchSize)", tn.BatchSize)
	}
}

func TestKernelParams_EmbedHonoursMetalNCBOverride(t *testing.T) {
	cfg := config.Config{MaxContext: 8192, EmbedMetalNCB: 1}
	tn := KernelParams(hardware.ProfileVegaPro, vramHigh, gateway.KindEmbed, cfg)
	if tn.MetalNCB != 1 {
		t.Errorf("MetalNCB = %d, want 1 (env override)", tn.MetalNCB)
	}
}

func TestKernelParams_EmbedAMDGetsAutoRespawn(t *testing.T) {
	cfg := config.Config{MaxContext: 8192}
	for _, p := range amdProfiles {
		for _, k := range []gateway.SlotKind{gateway.KindEmbed, gateway.KindCodeEmbed} {
			t.Run(string(p)+"/"+string(k), func(t *testing.T) {
				tn := KernelParams(p, vramHigh, k, cfg)
				if !tn.AutoRespawn {
					t.Errorf("%s %s should request AutoRespawn on AMD", p, k)
				}
			})
		}
	}
}

func TestKernelParams_EmbedNonAMDNoAutoRespawn(t *testing.T) {
	// Apple Silicon doesn't hit family-B; auto-respawn is dead weight
	// (and risks masking unrelated bugs).
	cfg := config.Config{MaxContext: 8192}
	for _, p := range allProfiles {
		if profileIsAMDDiscrete(p) {
			continue
		}
		t.Run(string(p), func(t *testing.T) {
			tn := KernelParams(p, vramHigh, gateway.KindEmbed, cfg)
			if tn.AutoRespawn {
				t.Errorf("%s should NOT request AutoRespawn", p)
			}
		})
	}
}

func TestKernelParams_RerankNoBatchOverrideByDefault(t *testing.T) {
	// Without QUENCHFORGE_RERANK_BATCH_SIZE the rerank slot keeps
	// llama-server's 512 default (matches current behaviour). Operators
	// who need larger pairs set the env var.
	cfg := config.Config{MaxContext: 8192}
	for _, p := range allProfiles {
		t.Run(string(p), func(t *testing.T) {
			tn := KernelParams(p, vramHigh, gateway.KindRerank, cfg)
			if tn.BatchSize != 0 {
				t.Errorf("%s rerank BatchSize = %d, want 0 (no override)",
					p, tn.BatchSize)
			}
		})
	}
}

func TestKernelParams_RerankHonoursBatchOverride(t *testing.T) {
	cfg := config.Config{MaxContext: 8192, RerankBatchSize: 2048}
	tn := KernelParams(hardware.ProfileVegaPro, vramHigh, gateway.KindRerank, cfg)
	if tn.BatchSize != 2048 {
		t.Errorf("BatchSize = %d, want 2048", tn.BatchSize)
	}
	if tn.UbatchSize != 2048 {
		t.Errorf("UbatchSize = %d, want 2048 (mirrors BatchSize)", tn.UbatchSize)
	}
}

func TestKernelParams_RerankAMDGetsAutoRespawn(t *testing.T) {
	cfg := config.Config{MaxContext: 8192}
	for _, p := range amdProfiles {
		t.Run(string(p), func(t *testing.T) {
			tn := KernelParams(p, vramHigh, gateway.KindRerank, cfg)
			if !tn.AutoRespawn {
				t.Errorf("%s rerank should request AutoRespawn on AMD", p)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// VRAM-tier-adaptive sizing (v0.8.0)
// ---------------------------------------------------------------------------

func TestAmdSizing_Tiers(t *testing.T) {
	// Whitebox: the (contextCap, ubatch) curve over VRAM. <=0 and >=12
	// are the high tier (no cap, validated 1024) so a probe miss or a
	// big card never throttles. 8 GB scales to 4096/512; 4 GB to 2048/256.
	cases := []struct {
		vram       int
		wantCtx    int
		wantUbatch int
	}{
		{vram: 0, wantCtx: 0, wantUbatch: amdEmbedUbatchDefault},  // probe miss -> high
		{vram: -1, wantCtx: 0, wantUbatch: amdEmbedUbatchDefault}, // negative -> high
		{vram: 32, wantCtx: 0, wantUbatch: amdEmbedUbatchDefault}, // Vega II
		{vram: 16, wantCtx: 0, wantUbatch: amdEmbedUbatchDefault}, // W6800X-class
		{vram: 12, wantCtx: 0, wantUbatch: amdEmbedUbatchDefault}, // tier boundary (incl)
		{vram: 11, wantCtx: 4096, wantUbatch: 512},                // just below high
		{vram: 8, wantCtx: 4096, wantUbatch: 512},                 // RX 5700
		{vram: 7, wantCtx: 4096, wantUbatch: 512},                 // low boundary (incl)
		{vram: 6, wantCtx: 2048, wantUbatch: 256},                 // tiny boundary
		{vram: 4, wantCtx: 2048, wantUbatch: 256},                 // 4 GB MBP dGPU
	}
	for _, c := range cases {
		ctx, ub := amdSizing(c.vram)
		if ctx != c.wantCtx || ub != c.wantUbatch {
			t.Errorf("amdSizing(%d) = (ctx %d, ubatch %d), want (ctx %d, ubatch %d)",
				c.vram, ctx, ub, c.wantCtx, c.wantUbatch)
		}
	}
}

func TestKernelParams_EmbedLowVRAMScalesDown(t *testing.T) {
	// An 8 GB AMD card must get the reduced embed ubatch (512) and a
	// context ceiling (4096) without any operator env var.
	cfg := config.Config{MaxContext: 8192}
	for _, p := range amdProfiles {
		for _, k := range []gateway.SlotKind{gateway.KindEmbed, gateway.KindCodeEmbed} {
			t.Run(string(p)+"/"+string(k), func(t *testing.T) {
				tn := KernelParams(p, 8, k, cfg)
				if tn.UbatchSize != 512 || tn.BatchSize != 512 {
					t.Errorf("%s %s ubatch/batch = %d/%d, want 512/512",
						p, k, tn.UbatchSize, tn.BatchSize)
				}
				if tn.ContextSize != 4096 {
					t.Errorf("%s %s ContextSize = %d, want 4096", p, k, tn.ContextSize)
				}
			})
		}
	}
}

func TestKernelParams_ContextCapAppliesToAllAMDSlots(t *testing.T) {
	// A 4 GB card caps context to 2048 on every AMD slot kind (chat,
	// embed, code-embed, rerank) — the KV cache is the dominant VRAM
	// consumer and must shrink uniformly.
	cfg := config.Config{MaxContext: 8192}
	for _, k := range []gateway.SlotKind{
		gateway.KindChat, gateway.KindEmbed, gateway.KindCodeEmbed, gateway.KindRerank,
	} {
		t.Run(string(k), func(t *testing.T) {
			tn := KernelParams(hardware.ProfileVegaPro, 4, k, cfg)
			if tn.ContextSize != 2048 {
				t.Errorf("%s ContextSize = %d, want 2048", k, tn.ContextSize)
			}
		})
	}
}

func TestKernelParams_HighVRAMAndNonAMDHaveNoContextCap(t *testing.T) {
	// >= 12 GB AMD and every non-AMD profile must leave ContextSize 0 so
	// buildSlotArgs keeps cfg.MaxContext verbatim (zero regression).
	cfg := config.Config{MaxContext: 8192}
	for _, p := range allProfiles {
		for _, k := range []gateway.SlotKind{
			gateway.KindChat, gateway.KindEmbed, gateway.KindRerank,
		} {
			t.Run(string(p)+"/"+string(k), func(t *testing.T) {
				tn := KernelParams(p, vramHigh, k, cfg)
				if tn.ContextSize != 0 {
					t.Errorf("%s %s ContextSize = %d, want 0 (no cap)",
						p, k, tn.ContextSize)
				}
			})
		}
	}
}

func TestKernelParams_UbatchOverrideBeatsTierButCapStands(t *testing.T) {
	// An explicit QUENCHFORGE_EMBED_UBATCH_SIZE wins over the tier ubatch,
	// but the VRAM context cap is independent and still applies — the two
	// knobs protect different resources.
	cfg := config.Config{MaxContext: 8192, EmbedUbatchSize: 2048}
	tn := KernelParams(hardware.ProfileVegaPro, 4, gateway.KindEmbed, cfg)
	if tn.UbatchSize != 2048 {
		t.Errorf("UbatchSize = %d, want 2048 (operator override wins)", tn.UbatchSize)
	}
	if tn.ContextSize != 2048 {
		t.Errorf("ContextSize = %d, want 2048 (cap independent of ubatch override)", tn.ContextSize)
	}
}

func TestKernelParams_UnknownKindsAreEmpty(t *testing.T) {
	// Whisper / imagegen / future kinds: tuning module shouldn't emit
	// anything until explicitly added. Prevents accidental flag
	// injection on slot types we haven't reasoned about.
	cfg := config.Config{MaxContext: 8192}
	for _, k := range []gateway.SlotKind{gateway.KindWhisper, gateway.KindImageGen} {
		t.Run(string(k), func(t *testing.T) {
			tn := KernelParams(hardware.ProfileVegaPro, vramHigh, k, cfg)
			if tn.UbatchSize != 0 || tn.BatchSize != 0 || tn.MetalNCB != 0 ||
				len(tn.ExtraArgs) != 0 || tn.AutoRespawn {
				t.Errorf("%s should emit empty SlotTuning, got %+v", k, tn)
			}
		})
	}
}
