// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"regexp"
	"runtime"
	"strings"
	"testing"

	"github.com/cerid-ai/quenchforge/internal/config"
	"github.com/cerid-ai/quenchforge/internal/gateway"
	"github.com/cerid-ai/quenchforge/internal/hardware"
)

// skipIfNotDarwin short-circuits doctor extension tests on non-Darwin
// platforms. cmdDoctor emits its "UNSUPPORTED (quenchforge is macOS-only)"
// banner and returns early on Linux/Windows per CLAUDE.md's macOS-only
// rule, so the Phase 3 sections (Ollama LaunchAgent, Disk space, Slot
// log sizes, Port 11434) only render on darwin. Tests covering those
// sections must skip elsewhere.
func skipIfNotDarwin(t *testing.T) {
	t.Helper()
	if runtime.GOOS != "darwin" {
		t.Skipf("cmdDoctor extensions are macOS-only (GOOS=%s)", runtime.GOOS)
	}
}

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
			cfg := config.Config{MaxContext: 8192, PlaceChat: "gpu"} // exercise GPU chat path (default is now CPU)
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

func TestBuildSlotArgs_AMDEmbedRerankDontGetChatFlags(t *testing.T) {
	// Embed and rerank slots don't autoregressively decode and don't
	// touch the LCP-prompt-save path; the chat-slot safety flags
	// (--flash-attn off, --cache-ram 0, --no-cache-prompt) are
	// chat-specific. They must NOT appear on AMD embed/rerank.
	//
	// Note: the AMD embed/rerank slots DO get their own protections via
	// the tuning module (auto-respawn + the env-tunable ubatch/N_CB
	// knobs documented in `patches/README.md` section 3). Those are
	// orthogonal to the chat-slot flag set this test asserts against.
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

func TestBuildSlotArgs_DelegatesToTuningModule(t *testing.T) {
	// Regression guard: buildSlotArgs must surface every flag the
	// tuning module asks for. Failing this test means a new
	// SlotTuning field was added without the corresponding wire-up
	// in main.go (or vice-versa).
	cfg := config.Config{
		MaxContext:      8192,
		EmbedUbatchSize: 1024, // exercises the embed override
	}
	info := hardware.Info{Profile: hardware.ProfileVegaPro}
	spec := slotSpec{Kind: gateway.KindEmbed, Name: "embed", Port: 11501}
	args := buildSlotArgs(cfg, info, spec, "/tmp/embed.gguf")

	// The env-override value (1024) must show up, not MaxContext (8192).
	if !containsArgPair(args, "--ubatch-size", "1024") {
		t.Errorf("--ubatch-size 1024 from env override not in args: %v", args)
	}
	if !containsArgPair(args, "--batch-size", "1024") {
		t.Errorf("--batch-size 1024 from env override not in args: %v", args)
	}
	// MaxContext-based default must NOT appear when the override is set.
	if containsArgPair(args, "--ubatch-size", "8192") {
		t.Errorf("env override should have replaced MaxContext default: %v", args)
	}
}

func TestSlotEnv_AppleInheritsGlobal(t *testing.T) {
	// Apple Silicon doesn't get the AMD-specific tuning, so the embed
	// slot's GGML_METAL_N_CB falls through to the global default.
	cfg := config.Config{MetalNCB: 2}
	info := hardware.Info{Profile: hardware.ProfileAppleSilicon}
	env := slotEnv(cfg, info, gateway.KindEmbed)
	want := "GGML_METAL_N_CB=2"
	found := false
	for _, e := range env {
		if e == want {
			found = true
		}
	}
	if !found {
		t.Errorf("expected Apple Silicon embed env to contain %q, got %v",
			want, env)
	}
}

func TestSlotEnv_AMDEmbedGetsBakedDefault(t *testing.T) {
	// AMD discrete embed slot inherits the v0.6.2 bench-validated
	// conservative default (MetalNCB=1) baked into tuning.go — even
	// when the global cfg.MetalNCB is the larger v0.5.x default.
	cfg := config.Config{MetalNCB: 2}
	info := hardware.Info{Profile: hardware.ProfileVegaPro}
	env := slotEnv(cfg, info, gateway.KindEmbed)
	want := "GGML_METAL_N_CB=1"
	found := false
	for _, e := range env {
		if e == want {
			found = true
		}
	}
	if !found {
		t.Errorf("AMD embed env should bake MetalNCB=1, got %v", env)
	}
}

func TestSlotEnv_EmbedMetalNCBOverride(t *testing.T) {
	cfg := config.Config{MetalNCB: 2, EmbedMetalNCB: 1}
	info := hardware.Info{Profile: hardware.ProfileVegaPro}
	env := slotEnv(cfg, info, gateway.KindEmbed)
	want := "GGML_METAL_N_CB=1"
	found := false
	for _, e := range env {
		if e == want {
			found = true
		}
	}
	if !found {
		t.Errorf("EmbedMetalNCB override must produce %q, got %v", want, env)
	}
}

func TestSlotEnv_ChatKeepsGlobalNCB(t *testing.T) {
	// Embed-specific override must not bleed into chat.
	cfg := config.Config{MetalNCB: 2, EmbedMetalNCB: 1}
	info := hardware.Info{Profile: hardware.ProfileVegaPro}
	env := slotEnv(cfg, info, gateway.KindChat)
	want := "GGML_METAL_N_CB=2"
	found := false
	for _, e := range env {
		if e == want {
			found = true
		}
	}
	if !found {
		t.Errorf("Chat slot should use global MetalNCB %q, got %v", want, env)
	}
}

// TestSlotEnv_AMDIncludesConcurrencyDisable verifies that AMD-discrete profiles
// get the GGML_METAL_CONCURRENCY_DISABLE=1 env var on every llama-server slot
// kind (chat, embed, code-embed, rerank). Required to disable the upstream
// MTLDispatchTypeConcurrent path that's unreliable on non-UMA Metal.
func TestSlotEnv_AMDIncludesConcurrencyDisable(t *testing.T) {
	// chat + rerank default to CPU now; force both to GPU to test their Metal env.
	cfg := config.Config{PlaceChat: "gpu", PlaceRerank: "gpu"}
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

func TestBuildSlotArgs_EmbedKindsBatchSizing(t *testing.T) {
	// Non-AMD profiles size their batch to MaxContext (the model's ctx-size).
	// AMD-discrete profiles cap at 1024 (amdEmbedUbatchDefault) — GPU mode
	// re-enabled in v0.8.0 re-exposes Metal staging-buffer pressure that
	// this cap bounds. Apple Silicon was never family-B exposed.
	cfg := config.Config{MaxContext: 8192}
	infoAMD := hardware.Info{Profile: hardware.ProfileVegaPro}
	infoApple := hardware.Info{Profile: hardware.ProfileAppleSilicon}

	for _, kind := range []gateway.SlotKind{gateway.KindEmbed, gateway.KindCodeEmbed} {
		t.Run("amd-"+string(kind), func(t *testing.T) {
			spec := slotSpec{Kind: kind, Name: string(kind), Port: 11501}
			args := buildSlotArgs(cfg, infoAMD, spec, "/tmp/model.gguf")

			if !containsArgPair(args, "--batch-size", "1024") {
				t.Errorf("AMD %s slot expected --batch-size 1024 "+
					"(GPU mode + staging-buffer cap, v0.8.0): %v", kind, args)
			}
			if !containsArgPair(args, "--ubatch-size", "1024") {
				t.Errorf("AMD %s slot expected --ubatch-size 1024 "+
					"(GPU mode + staging-buffer cap, v0.8.0): %v", kind, args)
			}
		})
		t.Run("apple-"+string(kind), func(t *testing.T) {
			spec := slotSpec{Kind: kind, Name: string(kind), Port: 11501}
			args := buildSlotArgs(cfg, infoApple, spec, "/tmp/model.gguf")

			if !containsArgPair(args, "--batch-size", "8192") {
				t.Errorf("Apple %s slot missing --batch-size 8192: %v", kind, args)
			}
			if !containsArgPair(args, "--ubatch-size", "8192") {
				t.Errorf("Apple %s slot missing --ubatch-size 8192: %v", kind, args)
			}
		})
	}
}

func TestBuildSlotArgs_EmbedKindsBatchOverride(t *testing.T) {
	// Operators can still pin a smaller ubatch via EmbedUbatchSize for
	// environments where the natural MaxContext is too large (e.g. memory-
	// constrained host). Pins both --batch-size and --ubatch-size to the
	// same value.
	cfg := config.Config{MaxContext: 8192, EmbedUbatchSize: 1024}
	infoAMD := hardware.Info{Profile: hardware.ProfileVegaPro}

	for _, kind := range []gateway.SlotKind{gateway.KindEmbed, gateway.KindCodeEmbed} {
		t.Run("amd-"+string(kind)+"-override", func(t *testing.T) {
			spec := slotSpec{Kind: kind, Name: string(kind), Port: 11501}
			args := buildSlotArgs(cfg, infoAMD, spec, "/tmp/model.gguf")

			if !containsArgPair(args, "--batch-size", "1024") {
				t.Errorf("AMD %s slot with EmbedUbatchSize=1024 missing --batch-size 1024: %v", kind, args)
			}
			if !containsArgPair(args, "--ubatch-size", "1024") {
				t.Errorf("AMD %s slot with EmbedUbatchSize=1024 missing --ubatch-size 1024: %v", kind, args)
			}
		})
	}
}

func TestBuildSlotArgs_LowVRAMAMDCapsContextAndUbatch(t *testing.T) {
	// v0.8.0 adaptive sizing: an 8 GB AMD card (e.g. RX 5700) must get a
	// capped --ctx-size 4096 (down from MaxContext 8192) and --ubatch-size
	// 512 on embed/chat without any operator env var — the gap that used
	// to force manual QUENCHFORGE_MAX_CONTEXT / _EMBED_UBATCH_SIZE tuning.
	cfg := config.Config{MaxContext: 8192, PlaceChat: "gpu"} // chat default is now CPU; force GPU to test the context cap
	info := hardware.Info{Profile: hardware.ProfileRDNA1, GPUVRAMGB: 8}

	embedArgs := buildSlotArgs(cfg, info, slotSpec{Kind: gateway.KindEmbed, Name: "embed", Port: 11501}, "/tmp/e.gguf")
	if !containsArgPair(embedArgs, "--ctx-size", "4096") {
		t.Errorf("8 GB AMD embed missing capped --ctx-size 4096: %v", embedArgs)
	}
	if !containsArgPair(embedArgs, "--ubatch-size", "512") {
		t.Errorf("8 GB AMD embed missing scaled --ubatch-size 512: %v", embedArgs)
	}

	chatArgs := buildSlotArgs(cfg, info, slotSpec{Kind: gateway.KindChat, Name: "chat", Port: 11500}, "/tmp/c.gguf")
	if !containsArgPair(chatArgs, "--ctx-size", "4096") {
		t.Errorf("8 GB AMD chat missing capped --ctx-size 4096: %v", chatArgs)
	}

	// Regression: a 32 GB card keeps the full configured context.
	hi := hardware.Info{Profile: hardware.ProfileVegaPro, GPUVRAMGB: 32}
	hiArgs := buildSlotArgs(cfg, hi, slotSpec{Kind: gateway.KindChat, Name: "chat", Port: 11500}, "/tmp/c.gguf")
	if !containsArgPair(hiArgs, "--ctx-size", "8192") {
		t.Errorf("32 GB AMD chat should keep --ctx-size 8192 (no cap): %v", hiArgs)
	}
}

func TestBuildSlotArgs_ChatSkipsBatchOverride(t *testing.T) {
	// The chat slot doesn't need the embed batch override — it decodes
	// autoregressively and its prompt path splits across ubatches; adding it
	// would waste VRAM. (Rerank USED to be in this list; since v0.9.1 it
	// carries a context-sized batch because llama-server's rerank pooling
	// path 500s any (query, doc) pair longer than n_ubatch — see
	// tuning.rerankParams and the 2026-07-08 incident regression tests.)
	cfg := config.Config{MaxContext: 8192}
	info := hardware.Info{Profile: hardware.ProfileVegaPro}

	spec := slotSpec{Kind: gateway.KindChat, Name: "chat", Port: 11500}
	args := buildSlotArgs(cfg, info, spec, "/tmp/model.gguf")
	if containsArgPair(args, "--batch-size", "8192") {
		t.Errorf("chat slot must not pass --batch-size 8192: %v", args)
	}
}

func TestBuildSlotArgs_RerankCarriesContextSizedBatch(t *testing.T) {
	// Regression (2026-07-08): the CPU-placed rerank slot spawned with no
	// batch flags, reverting to llama-server's 512 default — every
	// >512-token (query, doc) pair returned HTTP 500 "input too large".
	cfg := config.Config{MaxContext: 8192}
	info := hardware.Info{Profile: hardware.ProfileVegaPro}

	spec := slotSpec{Kind: gateway.KindRerank, Name: "rerank", Port: 11502}
	args := buildSlotArgs(cfg, info, spec, "/tmp/model.gguf")
	if !containsArgPair(args, "--batch-size", "8192") || !containsArgPair(args, "--ubatch-size", "8192") {
		t.Errorf("CPU-placed rerank slot must carry context-sized batch args: %v", args)
	}
}

// ---------------------------------------------------------------------------
// doctor — Phase 3 extension tests.
// Each test asserts cmdDoctor's stdout contains the new section header.
// The header strings are part of the bug-report-triage paste contract;
// renaming them is a breaking change for the .github/ISSUE_TEMPLATE/bug.yml
// downstream consumers (see CLAUDE.md absolute rule 4).
// ---------------------------------------------------------------------------

func TestDoctor_IncludesOllamaLaunchAgentCheck(t *testing.T) {
	skipIfNotDarwin(t)
	var stdout, stderr bytes.Buffer
	if err := cmdDoctor(nil, &stdout, &stderr); err != nil {
		t.Fatalf("cmdDoctor: %v", err)
	}
	out := stdout.String()
	if !strings.Contains(out, "Ollama LaunchAgent") {
		t.Errorf("doctor output missing 'Ollama LaunchAgent' section.\nGot:\n%s", out)
	}
	// The Ollama LaunchAgent line must emit a recognisable status string
	// (one of the four checkOllamaLaunchAgent return shapes). A buggy
	// helper that always returned "" would still pass the header check
	// above — this assertion catches that.
	matched, err := regexp.MatchString(
		`(?m)Ollama LaunchAgent\n-+\n\s+status:.*(not installed|disabled|loaded \(PID|loaded but stopped)`,
		out,
	)
	if err != nil {
		t.Fatalf("regex: %v", err)
	}
	if !matched {
		t.Errorf("Ollama LaunchAgent section missing recognisable status verdict.\nGot:\n%s", out)
	}
}

func TestDoctor_IncludesDiskFreeCheck(t *testing.T) {
	skipIfNotDarwin(t)
	var stdout, stderr bytes.Buffer
	if err := cmdDoctor(nil, &stdout, &stderr); err != nil {
		t.Fatalf("cmdDoctor: %v", err)
	}
	out := stdout.String()
	if !strings.Contains(out, "Disk space") {
		t.Errorf("doctor output missing 'Disk space' section")
	}
	matched, err := regexp.MatchString(
		`(?m)Disk space\n-+\n\s+/System/Volumes/Data:.*\b(PASS|WARN|CRITICAL)\b`,
		out,
	)
	if err != nil {
		t.Fatalf("regex: %v", err)
	}
	if !matched {
		t.Errorf("Disk space section missing classification (PASS/WARN/CRITICAL).\nGot:\n%s", out)
	}
}

func TestDoctor_IncludesLogSizeCheck(t *testing.T) {
	skipIfNotDarwin(t)
	var stdout, stderr bytes.Buffer
	if err := cmdDoctor(nil, &stdout, &stderr); err != nil {
		t.Fatalf("cmdDoctor: %v", err)
	}
	out := stdout.String()
	if !strings.Contains(out, "Slot log sizes") {
		t.Errorf("doctor output missing 'Slot log sizes' section")
	}
	// On a CI box with no slot logs yet, the section emits the
	// "(no slot logs yet)" sentinel. On a developer machine with logs,
	// each row carries a PASS/WARN/CRITICAL verdict. Either shape proves
	// the helper did real work.
	matched, err := regexp.MatchString(
		`(?m)Slot log sizes\n-+\n\s+(\(no slot logs yet\)|.*\b(PASS|WARN|CRITICAL)\b)`,
		out,
	)
	if err != nil {
		t.Fatalf("regex: %v", err)
	}
	if !matched {
		t.Errorf("Slot log sizes section missing classification or empty-state sentinel.\nGot:\n%s", out)
	}
}

func TestDoctor_IncludesPortCheck(t *testing.T) {
	skipIfNotDarwin(t)
	var stdout, stderr bytes.Buffer
	if err := cmdDoctor(nil, &stdout, &stderr); err != nil {
		t.Fatalf("cmdDoctor: %v", err)
	}
	out := stdout.String()
	if !strings.Contains(out, "Port 11434") {
		t.Errorf("doctor output missing 'Port 11434' section")
	}
	// Port check verdicts don't map to the PASS/WARN/CRITICAL string set;
	// they're worded ("free", "held by", "in use"). Assert one of those
	// phrases shows up so we know the section did real work.
	matched, err := regexp.MatchString(
		`(?m)Port 11434\n-+\n\s+.*\b(free|held by|in use|could not probe)\b`,
		out,
	)
	if err != nil {
		t.Fatalf("regex: %v", err)
	}
	if !matched {
		t.Errorf("Port 11434 section missing verdict (free|held by|in use).\nGot:\n%s", out)
	}
}

func TestDoctor_ExplainModeAddsRemediation(t *testing.T) {
	skipIfNotDarwin(t)
	var stdout, stderr bytes.Buffer
	if err := cmdDoctor([]string{"--explain"}, &stdout, &stderr); err != nil {
		t.Fatalf("cmdDoctor --explain: %v", err)
	}
	out := stdout.String()
	if !strings.Contains(out, "Common remediations") {
		t.Errorf("--explain output missing 'Common remediations' section.\nGot:\n%s", out)
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
