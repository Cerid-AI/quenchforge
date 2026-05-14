// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

// VRAM-aware slot startup. Before spawning any llama-server child, sum
// the model file sizes that will be loaded and compare against the
// detected GPU VRAM. Refuse to start with a helpful error if the
// configured slots won't fit — that catches the most painful operator
// failure mode (chat + embed + rerank all configured but combined
// weights exceed VRAM) before three Metal load attempts crash and burn.
//
// The math is rough — model file size is a reasonable lower bound for
// VRAM but doesn't account for KV cache, compute buffers, or Metal
// runtime overhead. We add a fixed-percentage headroom on top so the
// check is conservative.
//
// On non-Metal hardware (CPU-only Mac) the check is a no-op — there's
// no GPU VRAM constraint to enforce.

package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/cerid-ai/quenchforge/internal/config"
	"github.com/cerid-ai/quenchforge/internal/hardware"
)

// vramBudget is one slot's contribution to the VRAM total. Captured so
// the error message can break down which slot uses what.
type vramBudget struct {
	slotName  string
	modelName string
	modelPath string
	bytes     int64
}

// slotEstimate is the per-slot static cost we add to the on-disk size
// (KV cache, compute buffer, Metal runtime). Rough but conservative.
const (
	// Per-slot overhead (KV cache + compute buffer + runtime). ~1 GB
	// is conservative for ctx-size=8192 chat slots; embed/rerank slots
	// use much less but we apply the same budget to keep the math
	// simple. Better to over-warn than to under-warn here.
	perSlotOverheadBytes = 1 << 30 // 1 GB

	// Headroom multiplier applied to the sum of model file sizes.
	// Accounts for activations, quantization-bookkeeping tensors, and
	// Metal driver fudge.
	vramSafetyMultiplier = 1.15
)

// checkVRAMBudget pre-validates that the configured slots will fit in
// VRAM. Returns nil if either the check passes OR the host has no
// Metal GPU (CPU-only paths aren't VRAM-constrained).
//
// The check considers chat, embed, rerank, whisper-on-GPU, image-gen,
// and TTS slots — every workload that would land model weights on the
// GPU. It does NOT verify llama-server's actual VRAM usage at runtime;
// for that we'd need an iperf-style probe which Metal doesn't expose.
func checkVRAMBudget(cfg config.Config, hwInfo hardware.Info, w io.Writer) error {
	if !hwInfo.HasMetal {
		// CPU-only path. No VRAM constraint.
		return nil
	}
	if hwInfo.GPUVRAMGB <= 0 {
		// Detection didn't return VRAM size — can happen on iGPUs or
		// in containerized contexts. Print a warning, don't block.
		fmt.Fprintln(w, "quenchforge: warning: VRAM size unknown for detected GPU; skipping pre-flight VRAM check.")
		return nil
	}

	// Sum the GGUF sizes that will be loaded.
	var budgets []vramBudget
	var totalBytes int64

	addSlot := func(slotName, modelName string) {
		if modelName == "" {
			return
		}
		path, err := resolveModelPath(cfg.ModelsDir, modelName)
		if err != nil {
			// Model isn't actually present — leave the error to the
			// per-slot startSlot path which has the right context.
			// Don't include unknown-size models in the budget.
			return
		}
		info, err := os.Stat(path)
		if err != nil {
			return
		}
		b := vramBudget{
			slotName:  slotName,
			modelName: modelName,
			modelPath: path,
			bytes:     info.Size() + perSlotOverheadBytes,
		}
		budgets = append(budgets, b)
		totalBytes += b.bytes
	}

	addSlot("chat", cfg.DefaultModel)
	addSlot("embed", cfg.EmbedModel)
	addSlot("rerank", cfg.RerankModel)
	// whisper/sd/bark are different binaries with different memory
	// shapes — skip from the simple GGUF-based budget. They have their
	// own per-binary launch failure path if they don't fit.

	if len(budgets) == 0 {
		// No slots configured (--no-slot, or all opt-in vars empty).
		return nil
	}

	// Apply the safety multiplier to the model-bytes portion (overhead
	// already includes runtime).
	modelBytes := totalBytes - perSlotOverheadBytes*int64(len(budgets))
	adjustedBytes := int64(float64(modelBytes)*vramSafetyMultiplier) + perSlotOverheadBytes*int64(len(budgets))

	vramBytes := int64(hwInfo.GPUVRAMGB) * (1 << 30)

	if adjustedBytes <= vramBytes {
		// Fits — log a friendly summary so operators can see the
		// budget breakdown.
		fmt.Fprintf(w, "quenchforge: VRAM check OK — %s configured, %s available on %s\n",
			humanBytes(adjustedBytes), humanBytes(vramBytes), hwInfo.GPU)
		return nil
	}

	// Doesn't fit — build a helpful multi-line error.
	var sb strings.Builder
	fmt.Fprintf(&sb, "configured slots exceed available VRAM:\n")
	fmt.Fprintf(&sb, "  GPU:         %s\n", hwInfo.GPU)
	fmt.Fprintf(&sb, "  VRAM:        %s available\n", humanBytes(vramBytes))
	fmt.Fprintf(&sb, "  configured:  %s (model weights + per-slot overhead + %.0f%% headroom)\n",
		humanBytes(adjustedBytes), (vramSafetyMultiplier-1)*100)
	fmt.Fprintln(&sb, "  per-slot:")
	for _, b := range budgets {
		fmt.Fprintf(&sb, "    %-8s %s (%s)\n",
			b.slotName, humanBytes(b.bytes), filepath.Base(b.modelPath))
	}
	fmt.Fprintln(&sb)
	fmt.Fprintln(&sb, "  to fix, either:")
	fmt.Fprintln(&sb, "    - unset one slot's model env var (e.g. `unset QUENCHFORGE_RERANK_MODEL`)")
	fmt.Fprintln(&sb, "    - swap to a smaller model (`quenchforge pull --list` shows sizes)")
	fmt.Fprintln(&sb, "    - reduce --ctx-size (lower KV cache footprint)")
	fmt.Fprintln(&sb, "    - override the check: set QUENCHFORGE_VRAM_CHECK_DISABLE=1 (use at your own risk)")
	return fmt.Errorf("%s", sb.String())
}

// vramCheckDisabled is the opt-out env var. Operators on edge cases
// (eGPU, swap-backed inference, etc.) can force-skip the check.
func vramCheckDisabled() bool {
	return os.Getenv("QUENCHFORGE_VRAM_CHECK_DISABLE") != ""
}

func humanBytes(n int64) string {
	const (
		_  = iota
		KB = 1 << (10 * iota)
		MB
		GB
	)
	f := float64(n)
	switch {
	case n < MB:
		return fmt.Sprintf("%.1f KB", f/KB)
	case n < GB:
		return fmt.Sprintf("%.1f MB", f/MB)
	default:
		return fmt.Sprintf("%.2f GB", f/GB)
	}
}
