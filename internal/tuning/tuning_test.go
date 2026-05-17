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

func TestKernelParams_ChatAMDGetsCorrectnessFlags(t *testing.T) {
	cfg := config.Config{MaxContext: 8192}
	for _, p := range amdProfiles {
		t.Run(string(p), func(t *testing.T) {
			tn := KernelParams(p, gateway.KindChat, cfg)
			wantExtra := []string{
				"--flash-attn", "off",
				"--cache-ram", "0",
				"--no-cache-prompt",
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
			tn := KernelParams(p, gateway.KindChat, cfg)
			if !slices.Equal(tn.ExtraArgs, nil) && len(tn.ExtraArgs) != 0 {
				t.Errorf("chat %s should emit no ExtraArgs, got %v",
					p, tn.ExtraArgs)
			}
		})
	}
}

func TestKernelParams_EmbedDefaultsMatchMaxContext(t *testing.T) {
	// All profiles: embed/code-embed ubatch and batch default to
	// cfg.MaxContext so any input that fits the context fits a single
	// batch (preserves the v0.5.0 contextplus fix).
	cfg := config.Config{MaxContext: 8192}
	for _, p := range allProfiles {
		for _, k := range []gateway.SlotKind{gateway.KindEmbed, gateway.KindCodeEmbed} {
			t.Run(string(p)+"/"+string(k), func(t *testing.T) {
				tn := KernelParams(p, k, cfg)
				if tn.UbatchSize != 8192 {
					t.Errorf("%s %s UbatchSize = %d, want 8192",
						p, k, tn.UbatchSize)
				}
				if tn.BatchSize != 8192 {
					t.Errorf("%s %s BatchSize = %d, want 8192",
						p, k, tn.BatchSize)
				}
			})
		}
	}
}

func TestKernelParams_EmbedHonoursUbatchOverride(t *testing.T) {
	// Operator-set QUENCHFORGE_EMBED_UBATCH_SIZE wins.
	cfg := config.Config{MaxContext: 8192, EmbedUbatchSize: 1024}
	tn := KernelParams(hardware.ProfileVegaPro, gateway.KindEmbed, cfg)
	if tn.UbatchSize != 1024 {
		t.Errorf("UbatchSize = %d, want 1024 (env override)", tn.UbatchSize)
	}
	if tn.BatchSize != 1024 {
		t.Errorf("BatchSize = %d, want 1024 (mirrors UbatchSize)", tn.BatchSize)
	}
}

func TestKernelParams_EmbedHonoursMetalNCBOverride(t *testing.T) {
	cfg := config.Config{MaxContext: 8192, EmbedMetalNCB: 1}
	tn := KernelParams(hardware.ProfileVegaPro, gateway.KindEmbed, cfg)
	if tn.MetalNCB != 1 {
		t.Errorf("MetalNCB = %d, want 1 (env override)", tn.MetalNCB)
	}
}

func TestKernelParams_EmbedAMDGetsAutoRespawn(t *testing.T) {
	cfg := config.Config{MaxContext: 8192}
	for _, p := range amdProfiles {
		for _, k := range []gateway.SlotKind{gateway.KindEmbed, gateway.KindCodeEmbed} {
			t.Run(string(p)+"/"+string(k), func(t *testing.T) {
				tn := KernelParams(p, k, cfg)
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
			tn := KernelParams(p, gateway.KindEmbed, cfg)
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
			tn := KernelParams(p, gateway.KindRerank, cfg)
			if tn.BatchSize != 0 {
				t.Errorf("%s rerank BatchSize = %d, want 0 (no override)",
					p, tn.BatchSize)
			}
		})
	}
}

func TestKernelParams_RerankHonoursBatchOverride(t *testing.T) {
	cfg := config.Config{MaxContext: 8192, RerankBatchSize: 2048}
	tn := KernelParams(hardware.ProfileVegaPro, gateway.KindRerank, cfg)
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
			tn := KernelParams(p, gateway.KindRerank, cfg)
			if !tn.AutoRespawn {
				t.Errorf("%s rerank should request AutoRespawn on AMD", p)
			}
		})
	}
}

func TestKernelParams_UnknownKindsAreEmpty(t *testing.T) {
	// Whisper / imagegen / future kinds: tuning module shouldn't emit
	// anything until explicitly added. Prevents accidental flag
	// injection on slot types we haven't reasoned about.
	cfg := config.Config{MaxContext: 8192}
	for _, k := range []gateway.SlotKind{gateway.KindWhisper, gateway.KindImageGen} {
		t.Run(string(k), func(t *testing.T) {
			tn := KernelParams(hardware.ProfileVegaPro, k, cfg)
			if tn.UbatchSize != 0 || tn.BatchSize != 0 || tn.MetalNCB != 0 ||
				len(tn.ExtraArgs) != 0 || tn.AutoRespawn {
				t.Errorf("%s should emit empty SlotTuning, got %+v", k, tn)
			}
		})
	}
}
