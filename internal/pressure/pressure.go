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

// Limits configures the governor's target plan in each regime.
type Limits struct {
	// Max is the admission ceiling when the GPU is ours alone (headless or
	// display asleep) — full throughput.
	Max int
	// DisplayActive is the admission ceiling while a screen is being driven.
	// Serialized (1) by default so the duty-cycle gaps are clean. Clamped to
	// [1, Max].
	DisplayActive int
	// DisplayActiveDuty is the target GPU busy fraction (0<d<=1) while a
	// screen is being driven. Below 1, the gateway inserts proportional GPU
	// idle gaps so the compositor gets time slices. The single most important
	// knob — concurrency capping alone does NOT prevent compositor starvation
	// (sustained gapless GPU work at any concurrency does), the idle gaps do.
	DisplayActiveDuty float64
}

// Plan is the governor's output: how many concurrent GPU workloads to admit
// and what GPU busy fraction to hold to.
type Plan struct {
	Concurrency int
	Duty        float64 // 1.0 = no idle-gap limit
}

// For maps a Reading to the plan the governor should apply:
//
//   - headless / display asleep      → {Max, 1.0}   full throughput, no gaps
//   - display active, memory normal  → {DisplayActive, DisplayActiveDuty}
//   - display active, memory warn    → {1, DisplayActiveDuty}  serialize
//   - display active, memory critical→ {1, tighter duty}       shed hard
//   - headless, memory critical      → {1, 1.0}     relieve memory, no gap need
func (l Limits) For(r Reading) Plan {
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
	duty := l.DisplayActiveDuty
	if duty <= 0 || duty > 1 {
		duty = 0.5
	}

	if !r.DisplayActive {
		if r.MemPressure >= MemCritical {
			return Plan{Concurrency: 1, Duty: 1.0}
		}
		return Plan{Concurrency: max, Duty: 1.0}
	}
	// Display active: the compositor is competing for the GPU.
	if r.MemPressure >= MemCritical {
		return Plan{Concurrency: 1, Duty: tighten(duty)}
	}
	if r.MemPressure >= MemWarn {
		return Plan{Concurrency: 1, Duty: duty}
	}
	return Plan{Concurrency: da, Duty: duty}
}

// tighten lowers the duty cycle under memory-critical pressure, with a floor
// so inference still makes some progress.
func tighten(d float64) float64 {
	t := d * 0.6
	if t < 0.25 {
		t = 0.25
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
