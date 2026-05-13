// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

// Package hardware discovers the host's CPU/GPU/RAM and maps the result to
// a named profile that downstream tuning logic keys off of.
//
// The detection is intentionally split per-platform: real probes live in
// detect_darwin.go (IOKit + Metal via CGo) and the non-Darwin file holds a
// stub that returns a "cpu" profile so `go test ./...` works on CI runners
// that don't have macOS.
package hardware

import (
	"fmt"
	"runtime"
)

// Profile is a stable identifier for a hardware bucket. New profiles are
// added when a new GPU class proves it needs different kernel parameters.
type Profile string

const (
	// ProfileVegaPro covers the Mac Pro 2019's Radeon Pro Vega II / Vega II Duo
	// (16 / 32 GB HBM2) — the primary MVP target.
	ProfileVegaPro Profile = "vega-pro"

	// ProfileW6800X covers W6800X and W6800X Duo MPX modules.
	ProfileW6800X Profile = "w6800x"

	// ProfileRDNA1 covers RDNA1 mobile / discrete on Intel Mac (5500M, 5700).
	ProfileRDNA1 Profile = "rdna1"

	// ProfileRDNA2 covers RDNA2 mobile / discrete on Intel Mac (6700M).
	ProfileRDNA2 Profile = "rdna2"

	// ProfileAppleSilicon covers Apple Silicon (M-series). Not a primary target
	// but documented as a non-degraded secondary path.
	ProfileAppleSilicon Profile = "apple-silicon"

	// ProfileIGPU covers Intel Mac integrated GPUs (Iris Plus, Intel HD). Metal
	// works but performance is so far below discrete + CPU that we surface this
	// as a degraded path with a warning.
	ProfileIGPU Profile = "igpu"

	// ProfileCPU is the fallback when no acceleration is available — also what
	// the Linux stub returns on CI.
	ProfileCPU Profile = "cpu"

	// ProfileUnknown is returned only when detection itself failed (CGo error,
	// missing IOKit framework, etc.). Callers should refuse to start.
	ProfileUnknown Profile = "unknown"
)

// String implements fmt.Stringer.
func (p Profile) String() string { return string(p) }

// Info is the hardware snapshot the rest of the binary consumes. All fields
// are best-effort: zero values are valid (mean "not detected") and downstream
// code must not panic on them.
type Info struct {
	Profile Profile

	// OSVersion is the user-facing macOS version string, e.g. "macOS 14.5".
	// Empty on non-Darwin platforms.
	OSVersion string

	// CPU is the marketing name for the CPU, e.g. "Intel Xeon W-3245M" or
	// "Apple M2 Max". Detection is best-effort.
	CPU string

	// CPUCores is logical (HT-counted) core count.
	CPUCores int

	// TotalRAMGB is host RAM rounded to whole GB.
	TotalRAMGB int

	// GPU is the marketing name of the active discrete GPU. Multi-GPU hosts
	// report only the highest-VRAM device here; the full list lives in
	// Devices.
	GPU string

	// GPUVRAMGB is total VRAM in GB across all detected discrete devices.
	// On Apple Silicon this is unified memory.
	GPUVRAMGB int

	// Devices is the per-device breakdown. On Mac Pro 2019 with a Vega II Duo
	// this is two entries (Infinity Fabric is invisible to Metal). On a
	// single-GPU host this is one entry.
	Devices []Device

	// HasMetal is true when at least one Metal-capable device is present and
	// not low-power (i.e., not the iGPU when a discrete is also present).
	HasMetal bool
}

// Device describes a single Metal-capable adapter.
type Device struct {
	// Name is the marketing name (e.g., "AMD Radeon Pro Vega II").
	Name string

	// VRAMGB is recommendedMaxWorkingSetSize for discrete devices, or unified
	// memory for Apple Silicon — both rounded to whole GB.
	VRAMGB int

	// LowPower is true for integrated/iGPU devices. The supervisor uses this
	// to skip Intel iGPUs when a discrete is present (MTLCreateSystemDefaultDevice
	// silently picks the iGPU on dual-GPU Macs — see llama.cpp#2407).
	LowPower bool

	// AppleSilicon is true when MTLDevice.architecture.name reports the Apple
	// GPU family. Patch 0002 gates simdgroup_mm on this.
	AppleSilicon bool
}

// Detect returns a fresh hardware snapshot. The implementation is in
// detect_darwin.go on macOS and detect_other.go everywhere else.
//
// The function is deliberately synchronous (no context) — it runs once at
// startup and is fast enough not to need cancellation.
func Detect() (Info, error) {
	return detect()
}

// String renders an Info as a single-line human-readable summary suitable
// for `quenchforge doctor` output and bug-report pastes.
func (i Info) String() string {
	return fmt.Sprintf(
		"profile=%s os=%q cpu=%q cores=%d ram=%dGB gpu=%q vram=%dGB metal=%v devices=%d",
		i.Profile, i.OSVersion, i.CPU, i.CPUCores, i.TotalRAMGB, i.GPU, i.GPUVRAMGB, i.HasMetal, len(i.Devices),
	)
}

// IsAMDDiscrete reports whether the host is running an AMD discrete GPU
// that needs the Metal-correctness workarounds — i.e., one of the four
// AMD profiles (Vega Pro, W6800X, RDNA1, RDNA2). False on Apple Silicon,
// integrated GPUs, and CPU-only paths.
//
// Used by the supervisor to gate chat-slot launch flags:
//   - `--flash-attn off` (auto-detect tries to keep FA on GPU but ferries
//     the FA tensor to CPU each decode, throttling tok/s)
//   - `--no-cache-prompt` (prompt-cache state-save asserts on
//     ggml_metal_buffer_get_tensor with NULL buf_dst on Vega II)
//
// Both fixes match the spirit of the simdgroup-reduction patch — gating
// Metal code paths that produce wrong/missing buffer pointers on AMD —
// but live at the supervisor level so we honour the "one patch" rule.
func (i Info) IsAMDDiscrete() bool {
	switch i.Profile {
	case ProfileVegaPro, ProfileW6800X, ProfileRDNA1, ProfileRDNA2:
		return true
	default:
		return false
	}
}

// runtimePlatform is a thin shim so tests can spoof GOOS — used by the stub
// implementation to choose between "cpu" and "unknown" when neither macOS
// nor a known Linux GPU is present.
var runtimePlatform = func() string { return runtime.GOOS }
