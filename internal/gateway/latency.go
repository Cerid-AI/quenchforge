// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

// Per-slot rolling latency + error-rate tracker. Surfaces "this slot is
// about to die" before the SIGABRT in the family-B sustained-load Metal
// crash on AMD discrete (see `patches/README.md` section 3). The
// gateway records every upstream call; /health exposes the per-slot
// status so external consumers (cerid-ai, monitors) can throttle
// pre-emptively.
//
// Statistical shape: ring buffer of the last N samples per kind, with
// timestamps so the snapshot drops anything older than the configured
// window. Sample-count cap defends against unbounded memory growth
// under sustained traffic; window-trimming defends against stale
// p50/p99 values when traffic drops.

package gateway

import (
	"sort"
	"sync"
	"time"
)

const (
	// latencyWindow is the rolling window for the p50 / p99 calculation.
	// 60s is long enough to absorb a single slow query, short enough to
	// surface a developing problem before the slot dies.
	latencyWindow = 60 * time.Second

	// latencySampleCap bounds the per-kind ring-buffer size. ~1000 in a
	// 60-second window is 16 req/s sustained — plenty for any realistic
	// quenchforge slot, and small enough that the on-demand percentile
	// sort stays sub-millisecond.
	latencySampleCap = 1000

	// statusDegradedRatio: p99 must exceed this multiple of p50 for the
	// slot to be classified "degraded". 2x catches a slot that has begun
	// to suffer Metal staging-buffer pressure but is still serving.
	statusDegradedRatio = 2.0

	// statusCriticalRatio: p99 above this multiple of p50 is the "impending
	// crash" signal. Empirically (from the May-16 family-B crash) the
	// last 10-15 requests before SIGABRT see p99 / p50 climb past 5x.
	statusCriticalRatio = 5.0

	// statusCriticalErrorRate: if more than this fraction of recent calls
	// errored, also classify critical regardless of latency. Catches
	// "slot already crashing" without waiting for the latency tail.
	statusCriticalErrorRate = 0.05

	// statusMinSamples below which we report "ok" regardless — a slot
	// with two samples isn't a statistically-meaningful signal.
	statusMinSamples = 20
)

// SlotStatus is the public-facing health classification per slot kind.
type SlotStatus string

const (
	// StatusOK means latency is within bounds and error rate is low.
	StatusOK SlotStatus = "ok"

	// StatusDegraded means p99 latency has climbed past 2x p50 — operators
	// should investigate but the slot is still serving.
	StatusDegraded SlotStatus = "degraded"

	// StatusCritical means p99 latency or error rate signals impending
	// crash. Consumers should back off; when QUENCHFORGE_AUTO_BACKOFF=true
	// the gateway returns 503+Retry-After.
	StatusCritical SlotStatus = "critical"
)

// latencySample is a single recorded upstream call outcome. Stored in
// the ring buffer; the timestamp lets us prune entries older than
// latencyWindow on every snapshot.
type latencySample struct {
	at      time.Time
	dur     time.Duration
	isError bool
}

// latencyTracker is the per-kind ring-buffer rolling tracker. Safe for
// concurrent use; the lock is held only across the buffer mutation /
// snapshot computation, both of which are O(N) with N ≤ latencySampleCap.
type latencyTracker struct {
	mu      sync.Mutex
	samples map[SlotKind][]latencySample
}

// newLatencyTracker returns an initialised tracker.
func newLatencyTracker() *latencyTracker {
	return &latencyTracker{
		samples: make(map[SlotKind][]latencySample),
	}
}

// Record appends one observation. dur is the wall-time of the upstream
// call; isError flags any non-2xx response or proxy transport error.
// Callers that can't measure duration accurately (e.g. the request was
// rejected before reaching the proxy) should not call Record at all.
func (t *latencyTracker) Record(kind SlotKind, dur time.Duration, isError bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	buf := t.samples[kind]
	if len(buf) >= latencySampleCap {
		// Drop the oldest sample. Cheap because we trim on the right by
		// growth and on the left by the window check in snapshot().
		buf = buf[1:]
	}
	t.samples[kind] = append(buf, latencySample{
		at:      time.Now(),
		dur:     dur,
		isError: isError,
	})
}

// LatencySnapshot is one slot's rolling-window summary. Status is
// derived from p50/p99 and the error rate at snapshot time.
type LatencySnapshot struct {
	Kind       SlotKind   `json:"kind"`
	Samples    int        `json:"samples"`
	P50Ms      float64    `json:"p50_ms"`
	P99Ms      float64    `json:"p99_ms"`
	ErrorRate  float64    `json:"error_rate"`
	Status     SlotStatus `json:"status"`
	WindowSecs int        `json:"window_secs"`
}

// Snapshot computes the per-kind p50/p99/error-rate from samples within
// the rolling window. Returns one entry per kind that has any sample
// activity in the window. Side-effect: prunes expired samples (so the
// in-memory footprint reflects current activity, not lifetime traffic).
func (t *latencyTracker) Snapshot() map[SlotKind]LatencySnapshot {
	t.mu.Lock()
	defer t.mu.Unlock()
	cutoff := time.Now().Add(-latencyWindow)
	out := make(map[SlotKind]LatencySnapshot)
	for kind, buf := range t.samples {
		// Find the first in-window index. Samples are time-ordered
		// (Record appends in real-time), so a simple linear scan is fine.
		first := 0
		for first < len(buf) && buf[first].at.Before(cutoff) {
			first++
		}
		if first > 0 {
			buf = buf[first:]
			t.samples[kind] = buf
		}
		if len(buf) == 0 {
			continue
		}
		out[kind] = computeSnapshot(kind, buf)
	}
	return out
}

// SnapshotKind returns the snapshot for a single kind, or a zero-Samples
// snapshot when no samples are in-window. Same pruning side-effect as
// Snapshot for that kind.
func (t *latencyTracker) SnapshotKind(kind SlotKind) LatencySnapshot {
	t.mu.Lock()
	defer t.mu.Unlock()
	cutoff := time.Now().Add(-latencyWindow)
	buf := t.samples[kind]
	first := 0
	for first < len(buf) && buf[first].at.Before(cutoff) {
		first++
	}
	if first > 0 {
		buf = buf[first:]
		t.samples[kind] = buf
	}
	if len(buf) == 0 {
		return LatencySnapshot{
			Kind:       kind,
			Status:     StatusOK,
			WindowSecs: int(latencyWindow / time.Second),
		}
	}
	return computeSnapshot(kind, buf)
}

// computeSnapshot is the pure inner-loop. Caller holds the lock.
func computeSnapshot(kind SlotKind, buf []latencySample) LatencySnapshot {
	n := len(buf)
	durs := make([]float64, n)
	errs := 0
	for i, s := range buf {
		durs[i] = float64(s.dur.Nanoseconds()) / float64(time.Millisecond)
		if s.isError {
			errs++
		}
	}
	sort.Float64s(durs)
	p50 := percentile(durs, 0.50)
	p99 := percentile(durs, 0.99)
	errorRate := float64(errs) / float64(n)
	status := classify(n, p50, p99, errorRate)
	return LatencySnapshot{
		Kind:       kind,
		Samples:    n,
		P50Ms:      round2(p50),
		P99Ms:      round2(p99),
		ErrorRate:  round2(errorRate),
		Status:     status,
		WindowSecs: int(latencyWindow / time.Second),
	}
}

// classify implements the threshold logic. Pulled out so the test suite
// can drive the table of (samples, p50, p99, errorRate) → status without
// having to fabricate latency-sample buffers.
func classify(samples int, p50, p99, errorRate float64) SlotStatus {
	if samples < statusMinSamples {
		return StatusOK
	}
	if errorRate > statusCriticalErrorRate {
		return StatusCritical
	}
	if p50 <= 0 {
		return StatusOK
	}
	ratio := p99 / p50
	if ratio >= statusCriticalRatio {
		return StatusCritical
	}
	if ratio >= statusDegradedRatio {
		return StatusDegraded
	}
	return StatusOK
}

// percentile returns the q-th percentile (q ∈ [0,1]) from a sorted
// slice using the nearest-rank method. Returns 0 for an empty slice.
func percentile(sorted []float64, q float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if q <= 0 {
		return sorted[0]
	}
	if q >= 1 {
		return sorted[len(sorted)-1]
	}
	rank := q * float64(len(sorted)-1)
	lo := int(rank)
	hi := lo + 1
	if hi >= len(sorted) {
		return sorted[lo]
	}
	frac := rank - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}

func round2(f float64) float64 {
	// Avoid emitting 12-decimal float noise into the JSON payload.
	return float64(int64(f*100+0.5)) / 100
}
