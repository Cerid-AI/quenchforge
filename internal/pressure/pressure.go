// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

// Package pressure adapts quenchforge's GPU admission concurrency to host
// pressure signals so sustained inference never starves the macOS display
// compositor (WindowServer).
//
// On a single-GPU Mac, llama.cpp Metal compute and WindowServer share one
// device. Uncapped, sustained concurrent inference (eval suites, bulk KB
// ingest, back-to-back RAG) can monopolize the Metal command queue long
// enough that WindowServer misses its kernel watchdog check-ins — at which
// point macOS panics and reboots the machine. Interactive use survives
// because it's bursty (idle gaps let the compositor in); sustained batch
// load removes those gaps.
//
// The governor's lever is the scheduler's admission ceiling. Its primary
// signal is whether a display is actively being driven: when a screen is on
// it reserves GPU headroom (a lower ceiling) so the compositor always gets
// time; when the host is headless or the display is asleep it restores full
// throughput. Memory pressure is a secondary backoff. This generalizes
// across user configurations — a headless inference server sees no
// throttling, while a workstation driving a display is protected.
package pressure

// macOS memory-pressure levels, mirroring kern.memorystatus_vm_pressure_level.
const (
	MemNormal   = 1
	MemWarn     = 2
	MemCritical = 4
)

// Reading is a point-in-time snapshot of host pressure.
type Reading struct {
	// DisplayActive is true when a display is being driven at full power
	// (a screen is on and compositing). False when headless or asleep — the
	// compositor isn't competing for the GPU, so inference can run flat out.
	DisplayActive bool
	// MemPressure is the macOS memory-pressure level (1 normal / 2 warn /
	// 4 critical). 1 when unknown.
	MemPressure int
}

// Limits configures the governor's target concurrency in each regime.
type Limits struct {
	// Max is the admission ceiling when the GPU is ours alone (headless or
	// display asleep) — full throughput.
	Max int
	// DisplayActive is the reduced ceiling while a screen is being driven,
	// reserving GPU headroom for WindowServer. Clamped to [1, Max].
	DisplayActive int
}

// Target maps a Reading to the scheduler concurrency the governor should set:
//
//   - headless / display asleep      → Max (full throughput)
//   - display active, memory normal  → DisplayActive (reserve compositor headroom)
//   - display active, memory warn    → halve DisplayActive (min 1)
//   - memory critical (any display)  → 1 (shed almost everything)
//
// The result is always clamped to [1, Max].
func (l Limits) Target(r Reading) int {
	max := l.Max
	if max < 1 {
		max = 1
	}
	da := l.DisplayActive
	if da < 1 {
		da = 1
	}
	if da > max {
		da = max
	}

	if r.MemPressure >= MemCritical {
		return 1
	}
	if !r.DisplayActive {
		return max
	}
	t := da
	if r.MemPressure >= MemWarn {
		t = (da + 1) / 2 // halve, round up
		if t < 1 {
			t = 1
		}
	}
	return t
}

// Sensor reads host pressure. Read never blocks beyond its internal probe
// timeout and never returns an error — on any probe failure it returns the
// safe default for the platform (headless + normal memory → full throughput;
// see sense_darwin.go / sense_other.go).
type Sensor interface {
	Read() Reading
}
