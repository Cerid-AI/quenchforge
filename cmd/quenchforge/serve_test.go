// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"strings"
	"testing"

	"github.com/cerid-ai/quenchforge/internal/config"
	"github.com/cerid-ai/quenchforge/internal/gateway"
	"github.com/cerid-ai/quenchforge/internal/hardware"
)

// containsArgPair returns true when `flag` appears at args[i] and `value`
// at args[i+1] for some i.
func containsArgPair(args []string, flag, value string) bool {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == flag && args[i+1] == value {
			return true
		}
	}
	return false
}

// containsArg returns true when `flag` appears anywhere in args.
func containsArg(args []string, flag string) bool {
	for _, a := range args {
		if a == flag {
			return true
		}
	}
	return false
}

func TestBuildSlotArgs_BaseShape(t *testing.T) {
	// Base shape is profile-independent: --model, --host, --port,
	// --ctx-size plus the spec.ExtraArgs trailer.
	cfg := config.Config{MaxContext: 4096}
	info := hardware.Info{Profile: hardware.ProfileAppleSilicon}
	spec := slotSpec{
		Kind:      gateway.KindEmbed,
		Name:      "embed",
		Port:      11501,
		ExtraArgs: []string{"--embedding", "--pooling", "cls"},
	}
	args := buildSlotArgs(cfg, info, spec, "/tmp/model.gguf")

	if !containsArgPair(args, "--model", "/tmp/model.gguf") {
		t.Errorf("missing --model /tmp/model.gguf in %v", args)
	}
	if !containsArgPair(args, "--host", "127.0.0.1") {
		t.Errorf("missing --host 127.0.0.1 in %v", args)
	}
	if !containsArgPair(args, "--port", "11501") {
		t.Errorf("missing --port 11501 in %v", args)
	}
	if !containsArgPair(args, "--ctx-size", "4096") {
		t.Errorf("missing --ctx-size 4096 in %v", args)
	}
	// ExtraArgs must land contiguously.
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "--embedding --pooling cls") {
		t.Errorf("ExtraArgs not preserved contiguously in %q", joined)
	}
}

func TestBuildSlotArgs_AMDChatGetsCorrectnessFlags(t *testing.T) {
	// All four AMD-discrete profiles must add the three correctness
	// flags to the chat slot:
	//   --flash-attn off     keeps attention GPU-resident (no FA tensor
	//                        CPU fallback per decode step)
	//   --cache-ram 0        disables the server-side LCP-similarity
	//                        slot cache (the prompt_save → buf_dst=NULL
	//                        GGML_ASSERT path)
	//   --no-cache-prompt    disables per-slot prompt caching as a
	//                        belt-and-suspenders companion
	for _, profile := range []hardware.Profile{
		hardware.ProfileVegaPro,
		hardware.ProfileW6800X,
		hardware.ProfileRDNA1,
		hardware.ProfileRDNA2,
	} {
		t.Run(string(profile), func(t *testing.T) {
			cfg := config.Config{MaxContext: 8192}
			info := hardware.Info{Profile: profile}
			spec := slotSpec{
				Kind: gateway.KindChat,
				Name: "chat",
				Port: 11500,
			}
			args := buildSlotArgs(cfg, info, spec, "/tmp/chat.gguf")

			if !containsArgPair(args, "--flash-attn", "off") {
				t.Errorf("chat slot on %s missing --flash-attn off: %v",
					profile, args)
			}
			if !containsArgPair(args, "--cache-ram", "0") {
				t.Errorf("chat slot on %s missing --cache-ram 0: %v",
					profile, args)
			}
			if !containsArg(args, "--no-cache-prompt") {
				t.Errorf("chat slot on %s missing --no-cache-prompt: %v",
					profile, args)
			}
		})
	}
}

func TestBuildSlotArgs_AMDEmbedRerankUnchanged(t *testing.T) {
	// Embed and rerank slots don't autoregressively decode and don't
	// touch the prompt-cache state-save path; they must NOT get the
	// chat-slot correctness flags even on AMD discrete.
	cfg := config.Config{MaxContext: 8192}
	info := hardware.Info{Profile: hardware.ProfileVegaPro}

	for _, kind := range []gateway.SlotKind{gateway.KindEmbed, gateway.KindRerank} {
		t.Run(string(kind), func(t *testing.T) {
			spec := slotSpec{Kind: kind, Name: string(kind), Port: 11501}
			args := buildSlotArgs(cfg, info, spec, "/tmp/model.gguf")

			if containsArgPair(args, "--flash-attn", "off") {
				t.Errorf("%s slot on AMD must not pass --flash-attn off: %v",
					kind, args)
			}
			if containsArgPair(args, "--cache-ram", "0") {
				t.Errorf("%s slot on AMD must not pass --cache-ram 0: %v",
					kind, args)
			}
			if containsArg(args, "--no-cache-prompt") {
				t.Errorf("%s slot on AMD must not pass --no-cache-prompt: %v",
					kind, args)
			}
		})
	}
}

func TestBuildSlotArgs_AppleSiliconChatUnchanged(t *testing.T) {
	// Apple Silicon and the unknown fallback must keep the upstream
	// defaults — flash-attn=auto, cache-ram=8192, cache-prompt enabled.
	// Adding the AMD flags here would regress Apple Silicon throughput.
	cfg := config.Config{MaxContext: 8192}
	for _, profile := range []hardware.Profile{
		hardware.ProfileAppleSilicon,
		hardware.ProfileIGPU,
		hardware.ProfileCPU,
		hardware.ProfileUnknown,
	} {
		t.Run(string(profile), func(t *testing.T) {
			info := hardware.Info{Profile: profile}
			spec := slotSpec{Kind: gateway.KindChat, Name: "chat", Port: 11500}
			args := buildSlotArgs(cfg, info, spec, "/tmp/chat.gguf")

			if containsArgPair(args, "--flash-attn", "off") {
				t.Errorf("chat slot on %s must not pass --flash-attn off: %v",
					profile, args)
			}
			if containsArgPair(args, "--cache-ram", "0") {
				t.Errorf("chat slot on %s must not pass --cache-ram 0: %v",
					profile, args)
			}
			if containsArg(args, "--no-cache-prompt") {
				t.Errorf("chat slot on %s must not pass --no-cache-prompt: %v",
					profile, args)
			}
		})
	}
}
