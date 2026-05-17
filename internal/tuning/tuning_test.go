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

func TestKernelParams_EmbedDefaultsByProfile(t *testing.T) {
	// All profiles use MaxContext for ubatch/batch on embed slots —
	// long single inputs (full LongMemEval sessions, multi-paragraph
	// documents) need the full natural model context as a single
	// forward pass, otherwise llama-server returns HTTP 500 "input is
	// too large for the physical batch size". The 1024 ubatch
	// mitigation that v0.6.2 used was Metal-family-B-specific; with
	// AMD embed slots now routed to CPU via --gpu-layers 0, that
	// crash mode is structurally impossible. MetalNCB stays at the
	// AMD default for completeness but is a no-op on the CPU path.
	cfg := config.Config{MaxContext: 8192}
	for _, p := range allProfiles {
		for _, k := range []gateway.SlotKind{gateway.KindEmbed, gateway.KindCodeEmbed} {
			t.Run(string(p)+"/"+string(k), func(t *testing.T) {
				tn := KernelParams(p, k, cfg)
				wantUbatch := 8192
				wantNCB := 0
				if profileIsAMDDiscrete(p) {
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

func TestKernelParams_EmbedAMDForcesCPU(t *testing.T) {
	// AMD discrete profiles must include `--gpu-layers 0` to route embed
	// off the broken Metal-on-AMD path for BERT-family models. See the
	// embedParams docstring for the empirical evidence (Vega II 2026-05-17:
	// identical "hello" gives cos_sim 0.07 on Metal vs 1.0 on CPU). Apple
	// Silicon and unknown profiles must NOT get this flag — the bug is
	// structurally AMD-discrete-only and CPU on UMA hardware would be a
	// large unnecessary throughput hit.
	cfg := config.Config{MaxContext: 8192}
	for _, p := range allProfiles {
		for _, k := range []gateway.SlotKind{gateway.KindEmbed, gateway.KindCodeEmbed} {
			t.Run(string(p)+"/"+string(k), func(t *testing.T) {
				tn := KernelParams(p, k, cfg)
				hasFlag := false
				for i := 0; i+1 < len(tn.ExtraArgs); i++ {
					if tn.ExtraArgs[i] == "--gpu-layers" && tn.ExtraArgs[i+1] == "0" {
						hasFlag = true
						break
					}
				}
				if profileIsAMDDiscrete(p) && !hasFlag {
					t.Errorf("%s %s missing --gpu-layers 0; ExtraArgs=%v",
						p, k, tn.ExtraArgs)
				}
				if !profileIsAMDDiscrete(p) && hasFlag {
					t.Errorf("%s %s should NOT force CPU; ExtraArgs=%v",
						p, k, tn.ExtraArgs)
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
				tn := KernelParams(p, k, cfg)
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

func TestKernelParams_RerankAMDForcesCPU(t *testing.T) {
	// Same BERT-family bug applies to bge-reranker-v2-m3. AMD discrete
	// must route rerank to CPU; non-AMD profiles must not.
	cfg := config.Config{MaxContext: 8192}
	for _, p := range allProfiles {
		t.Run(string(p), func(t *testing.T) {
			tn := KernelParams(p, gateway.KindRerank, cfg)
			hasFlag := false
			for i := 0; i+1 < len(tn.ExtraArgs); i++ {
				if tn.ExtraArgs[i] == "--gpu-layers" && tn.ExtraArgs[i+1] == "0" {
					hasFlag = true
					break
				}
			}
			if profileIsAMDDiscrete(p) && !hasFlag {
				t.Errorf("%s rerank missing --gpu-layers 0; ExtraArgs=%v",
					p, tn.ExtraArgs)
			}
			if !profileIsAMDDiscrete(p) && hasFlag {
				t.Errorf("%s rerank should NOT force CPU; ExtraArgs=%v",
					p, tn.ExtraArgs)
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
			tn := KernelParams(p, gateway.KindRerank, cfg)
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
