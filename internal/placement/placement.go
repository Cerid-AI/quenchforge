// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

// Package placement is quenchforge's device-placement control plane: it
// decides whether a given inference workload runs on the GPU or the CPU.
//
// Why this exists: on some hardware the GPU is not the right device for every
// workload. On the Mac Pro 2019 + Radeon Pro Vega II, measured 2026-06-07:
//
//   - chat (autoregressive, latency-sensitive, single-request): CPU ~7x
//     faster than the AMD Metal path (CPU ~3.8s vs GPU ~27-29s for 32 tokens;
//     the GPU figure is corroborated by bench-llama-sustained-load p50=29.4s).
//   - embed / code-embed / rerank (BERT, batched/concurrent throughput): GPU
//     ~1.7-2.5x faster under sustained batched load (quenchforge v0.8.0 bench),
//     but CPU wins single-request latency.
//
// So placement is workload-aware and hardware-adaptive:
//   - "cpu"  : always CPU (latency-bound kinds on a weak-GPU host).
//   - "gpu"  : always GPU (throughput-bound kinds; any kind on a strong GPU).
//   - "auto" : dual-placed — route per request by batch shape (single -> CPU,
//     batched -> GPU) via RouteRequest. The gateway picks the instance.
//
// The admission/duty-cycle governor (internal/scheduler + gateway) applies to
// whatever lands on the GPU; placement decides what that is. Together they are
// the resource-control framework: placement (where) + governor (how much).
//
// Decoupled by design: kinds are plain strings so neither gateway nor tuning
// creates an import cycle through this package.
package placement

import "strings"

// Device is the compute device a workload runs on.
type Device int

const (
	GPU Device = iota
	CPU
)

func (d Device) String() string {
	if d == CPU {
		return "cpu"
	}
	return "gpu"
}

// GPULayers is the llama-server `--gpu-layers` value for this device: all
// layers on GPU, none on CPU.
func (d Device) GPULayers() string {
	if d == CPU {
		return "0"
	}
	return "999"
}

// Placement modes (per kind).
const (
	ModeGPU  = "gpu"
	ModeCPU  = "cpu"
	ModeAuto = "auto"
)

// Canonical slot-kind keys (mirror gateway.SlotKind string values; kept as
// strings here to avoid an import cycle).
const (
	KindChat      = "chat"
	KindEmbed     = "embed"
	KindCodeEmbed = "code-embed"
	KindRerank    = "rerank"
)

// Policy maps each slot kind to a placement mode.
type Policy struct {
	mode map[string]string
}

// NewPolicy builds the default policy for the host, then applies operator
// overrides. amdDiscrete selects the hardware-adaptive defaults:
//
//   - AMD-discrete (Vega II / W6800X / RDNA1/2): chat -> CPU (latency, GPU
//     slow); embed/code-embed -> GPU (batched throughput win); rerank -> CPU
//     (query-time, single-request, measured faster on CPU than the AMD Metal
//     path).
//   - everything else (Apple Silicon UMA / unknown): all GPU — the Metal
//     path is fast and the contention class doesn't apply.
//
// overrides maps a kind to a mode ("gpu"/"cpu"/"auto"); invalid or empty
// values are ignored so a partial/typo'd override never silently disables a
// slot. Unknown kinds default to GPU.
func NewPolicy(amdDiscrete bool, overrides map[string]string) Policy {
	m := map[string]string{}
	if amdDiscrete {
		m[KindChat] = ModeCPU
		m[KindEmbed] = ModeGPU
		m[KindCodeEmbed] = ModeGPU
		m[KindRerank] = ModeCPU
	} else {
		m[KindChat] = ModeGPU
		m[KindEmbed] = ModeGPU
		m[KindCodeEmbed] = ModeGPU
		m[KindRerank] = ModeGPU
	}
	for k, v := range overrides {
		if mode := normalizeMode(v); mode != "" {
			m[strings.TrimSpace(k)] = mode
		}
	}
	return Policy{mode: m}
}

func normalizeMode(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case ModeGPU:
		return ModeGPU
	case ModeCPU:
		return ModeCPU
	case ModeAuto:
		return ModeAuto
	}
	return ""
}

// Mode returns the configured mode for a kind ("gpu" if unset/unknown).
func (p Policy) Mode(kind string) string {
	if m, ok := p.mode[kind]; ok {
		return m
	}
	return ModeGPU
}

// Device returns the placement device for a kind. For "auto" kinds this is the
// default (single-request) instance device — CPU — because the latency path is
// the common interactive case; the gateway calls RouteRequest to send batched
// requests to the GPU instance instead.
func (p Policy) Device(kind string) Device {
	switch p.Mode(kind) {
	case ModeCPU, ModeAuto:
		return CPU
	default:
		return GPU
	}
}

// RouteRequest decides the device for a single request. For fixed gpu/cpu
// kinds it returns that device. For "auto" kinds it routes by workload shape:
// a batch larger than threshold (bulk/throughput) -> GPU; a single/small
// request (latency) -> CPU. threshold < 1 is treated as 1.
func (p Policy) RouteRequest(kind string, batchN, threshold int) Device {
	if p.Mode(kind) != ModeAuto {
		return p.Device(kind)
	}
	if threshold < 1 {
		threshold = 1
	}
	if batchN > threshold {
		return GPU
	}
	return CPU
}
