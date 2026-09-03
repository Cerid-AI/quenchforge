// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cerid-ai/quenchforge/internal/config"
	"github.com/cerid-ai/quenchforge/internal/hardware"
	"github.com/cerid-ai/quenchforge/internal/tuning"
)

// makeFakeGGUF creates a sparse file of the given size in modelsDir.
// Returns the name (without .gguf) the operator would set in
// QUENCHFORGE_*_MODEL.
func makeFakeGGUF(t *testing.T, modelsDir, name string, sizeBytes int64) {
	t.Helper()
	if err := os.MkdirAll(modelsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(modelsDir, name+".gguf")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	// Use sparse-file truncate — fast, no real disk usage.
	if err := f.Truncate(sizeBytes); err != nil {
		t.Fatal(err)
	}
}

func TestVRAMCheck_NoMetal_NoOp(t *testing.T) {
	tmp := t.TempDir()
	cfg := config.Config{ModelsDir: tmp, DefaultModel: "huge"}
	makeFakeGGUF(t, tmp, "huge", 100<<30) // 100 GB — would never fit

	// HasMetal=false — check must skip
	info := hardware.Info{Profile: hardware.ProfileCPU, HasMetal: false, GPUVRAMGB: 0}

	var buf bytes.Buffer
	if err := checkVRAMBudget(cfg, info, tuning.PolicyFor(info.Profile, cfg), &buf); err != nil {
		t.Errorf("non-Metal host should be a no-op, got: %v", err)
	}
}

func TestVRAMCheck_VRAMSizeUnknown_WarnAndContinue(t *testing.T) {
	tmp := t.TempDir()
	cfg := config.Config{ModelsDir: tmp, DefaultModel: "x"}
	makeFakeGGUF(t, tmp, "x", 1<<30)

	// HasMetal=true but VRAM not detected
	info := hardware.Info{Profile: hardware.ProfileIGPU, HasMetal: true, GPUVRAMGB: 0}

	var buf bytes.Buffer
	if err := checkVRAMBudget(cfg, info, tuning.PolicyFor(info.Profile, cfg), &buf); err != nil {
		t.Errorf("unknown VRAM should warn-and-continue, got: %v", err)
	}
	if !strings.Contains(buf.String(), "VRAM size unknown") {
		t.Errorf("expected unknown-VRAM warning in output, got: %q", buf.String())
	}
}

func TestVRAMCheck_NoSlotsConfigured_NoOp(t *testing.T) {
	tmp := t.TempDir()
	// DefaultModel empty AND no embed/rerank model set
	cfg := config.Config{ModelsDir: tmp, DefaultModel: ""}
	info := hardware.Info{Profile: hardware.ProfileVegaPro, HasMetal: true, GPUVRAMGB: 32, GPU: "AMD Vega II"}

	var buf bytes.Buffer
	if err := checkVRAMBudget(cfg, info, tuning.PolicyFor(info.Profile, cfg), &buf); err != nil {
		t.Errorf("no slots configured should be a no-op, got: %v", err)
	}
}

func TestVRAMCheck_FitsComfortably(t *testing.T) {
	tmp := t.TempDir()
	makeFakeGGUF(t, tmp, "chat-model", 4<<30)
	makeFakeGGUF(t, tmp, "embed-model", 200<<20)
	cfg := config.Config{
		ModelsDir:    tmp,
		DefaultModel: "chat-model",
		EmbedModel:   "embed-model",
	}
	info := hardware.Info{
		Profile: hardware.ProfileVegaPro, HasMetal: true,
		GPUVRAMGB: 32, GPU: "AMD Radeon Pro Vega II",
	}

	var buf bytes.Buffer
	if err := checkVRAMBudget(cfg, info, tuning.PolicyFor(info.Profile, cfg), &buf); err != nil {
		t.Errorf("4GB + 200MB should fit in 32GB, got: %v", err)
	}
	if !strings.Contains(buf.String(), "VRAM check OK") {
		t.Errorf("expected OK message in output, got: %q", buf.String())
	}
}

func TestVRAMCheck_Oversubscribed_HelpfulError(t *testing.T) {
	tmp := t.TempDir()
	makeFakeGGUF(t, tmp, "huge-chat", 28<<30)
	makeFakeGGUF(t, tmp, "huge-embed", 8<<30)
	cfg := config.Config{
		ModelsDir:    tmp,
		DefaultModel: "huge-chat",
		EmbedModel:   "huge-embed",
		PlaceChat:    "gpu", // both slots on the GPU, so both count
	}
	// 16 GB GPU — 28+8 + overhead won't fit
	info := hardware.Info{
		Profile: hardware.ProfileVegaPro, HasMetal: true,
		GPUVRAMGB: 16, GPU: "AMD Radeon Pro Vega II",
	}

	var buf bytes.Buffer
	err := checkVRAMBudget(cfg, info, tuning.PolicyFor(info.Profile, cfg), &buf)
	if err == nil {
		t.Fatalf("expected oversubscription error")
	}
	// The error message must be helpful — break down per-slot AND
	// suggest fixes.
	msg := err.Error()
	for _, want := range []string{
		"exceed available VRAM",
		"huge-chat", // per-slot breakdown
		"huge-embed",
		"unset",              // suggested fix
		"smaller model",      // suggested fix
		"VRAM_CHECK_DISABLE", // escape hatch
	} {
		if !strings.Contains(msg, want) {
			t.Errorf("error missing %q. Full message:\n%s", want, msg)
		}
	}
}

func TestVRAMCheck_MissingModel_Skipped(t *testing.T) {
	// A model name set but the file doesn't exist on disk. checkVRAMBudget
	// should skip rather than error — the per-slot startup path has
	// better error context for missing-model.
	tmp := t.TempDir()
	cfg := config.Config{
		ModelsDir:    tmp,
		DefaultModel: "model-that-doesnt-exist",
	}
	info := hardware.Info{
		Profile: hardware.ProfileVegaPro, HasMetal: true,
		GPUVRAMGB: 32, GPU: "AMD Vega II",
	}

	var buf bytes.Buffer
	if err := checkVRAMBudget(cfg, info, tuning.PolicyFor(info.Profile, cfg), &buf); err != nil {
		t.Errorf("missing-model should be skipped, not errored: %v", err)
	}
}

func TestVRAMCheckDisabled_EnvVar(t *testing.T) {
	t.Setenv("QUENCHFORGE_VRAM_CHECK_DISABLE", "1")
	if !vramCheckDisabled() {
		t.Errorf("QUENCHFORGE_VRAM_CHECK_DISABLE=1 should disable the check")
	}
	t.Setenv("QUENCHFORGE_VRAM_CHECK_DISABLE", "")
	if vramCheckDisabled() {
		t.Errorf("empty env var should leave the check enabled")
	}
}

func TestHumanBytes(t *testing.T) {
	cases := []struct {
		n    int64
		want string
	}{
		{500, "0.5 KB"},
		{1500, "1.5 KB"},
		{int64(1.5 * (1 << 20)), "1.5 MB"},
		{int64(2.5 * (1 << 30)), "2.50 GB"},
	}
	for _, c := range cases {
		got := humanBytes(c.n)
		if got != c.want {
			t.Errorf("humanBytes(%d) = %q, want %q", c.n, got, c.want)
		}
	}
}

func TestVRAMCheck_SkipsCPUPlacedSlots(t *testing.T) {
	// AMD-discrete defaults route chat and rerank to the CPU (placement.go),
	// so their weights never land in VRAM. Counting them against the GPU
	// budget refuses a configuration that would run fine.
	tmp := t.TempDir()
	makeFakeGGUF(t, tmp, "chat-model", 12<<30)
	makeFakeGGUF(t, tmp, "rerank-model", 2<<30)
	makeFakeGGUF(t, tmp, "embed-model", 1<<30)
	cfg := config.Config{
		ModelsDir:    tmp,
		DefaultModel: "chat-model",
		RerankModel:  "rerank-model",
		EmbedModel:   "embed-model",
	}
	info := hardware.Info{
		Profile: hardware.ProfileVegaPro, HasMetal: true,
		GPUVRAMGB: 8, GPU: "AMD Radeon Pro 580X",
	}

	var buf bytes.Buffer
	if err := checkVRAMBudget(cfg, info, tuning.PolicyFor(info.Profile, cfg), &buf); err != nil {
		t.Errorf("chat and rerank are CPU-placed on AMD-discrete; only the 1 GB embed "+
			"slot competes for the 8 GB card. Got:\n%v", err)
	}
}

func TestVRAMCheck_CountsCodeEmbedSlot(t *testing.T) {
	tmp := t.TempDir()
	makeFakeGGUF(t, tmp, "code-embed-model", 20<<30)
	cfg := config.Config{
		ModelsDir:      tmp,
		CodeEmbedModel: "code-embed-model",
	}
	info := hardware.Info{
		Profile: hardware.ProfileVegaPro, HasMetal: true,
		GPUVRAMGB: 8, GPU: "AMD Radeon Pro 580X",
	}

	var buf bytes.Buffer
	err := checkVRAMBudget(cfg, info, tuning.PolicyFor(info.Profile, cfg), &buf)
	if err == nil {
		t.Fatalf("a 20 GB GPU-placed code-embed model does not fit an 8 GB card, "+
			"but the budget reported OK: %q", buf.String())
	}
	if !strings.Contains(err.Error(), "code-embed") {
		t.Errorf("error must name the code-embed slot:\n%v", err)
	}
}

func TestVRAMCheck_MissingModelIsReported(t *testing.T) {
	tmp := t.TempDir()
	cfg := config.Config{ModelsDir: tmp, EmbedModel: "absent-embed-model"}
	info := hardware.Info{
		Profile: hardware.ProfileVegaPro, HasMetal: true,
		GPUVRAMGB: 32, GPU: "AMD Vega II",
	}

	var buf bytes.Buffer
	if err := checkVRAMBudget(cfg, info, tuning.PolicyFor(info.Profile, cfg), &buf); err != nil {
		t.Fatalf("a missing model is not a budget failure: %v", err)
	}
	out := buf.String()
	if !strings.Contains(out, "absent-embed-model") {
		t.Errorf("dropping a slot from the budget without saying so lets the "+
			"pre-flight print an OK it cannot back. Got:\n%s", out)
	}
	if strings.Contains(out, "VRAM check OK") {
		t.Errorf("the budget is incomplete; it must not claim OK. Got:\n%s", out)
	}
}
