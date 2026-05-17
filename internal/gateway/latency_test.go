// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package gateway

import (
	"testing"
	"time"
)

func TestLatencyTracker_EmptySnapshotReturnsOK(t *testing.T) {
	tr := newLatencyTracker()
	snap := tr.SnapshotKind(KindEmbed)
	if snap.Samples != 0 {
		t.Errorf("expected 0 samples on a fresh tracker, got %d", snap.Samples)
	}
	if snap.Status != StatusOK {
		t.Errorf("expected status ok on a fresh tracker, got %s", snap.Status)
	}
}

func TestLatencyTracker_BelowMinSamplesStaysOK(t *testing.T) {
	// Even with a huge spike, < statusMinSamples observations is "not
	// enough signal" — we report ok to avoid flapping on transient
	// single-request glitches.
	tr := newLatencyTracker()
	for range 5 {
		tr.Record(KindEmbed, 10*time.Millisecond, false)
	}
	tr.Record(KindEmbed, 5*time.Second, false)
	snap := tr.SnapshotKind(KindEmbed)
	if snap.Status != StatusOK {
		t.Errorf("expected ok below %d samples, got %s",
			statusMinSamples, snap.Status)
	}
}

func TestLatencyTracker_HealthyRunReportsOK(t *testing.T) {
	tr := newLatencyTracker()
	for range 50 {
		tr.Record(KindEmbed, 20*time.Millisecond, false)
	}
	snap := tr.SnapshotKind(KindEmbed)
	if snap.Status != StatusOK {
		t.Errorf("expected ok under steady latency, got %s", snap.Status)
	}
	if snap.Samples != 50 {
		t.Errorf("expected 50 samples, got %d", snap.Samples)
	}
}

func TestLatencyTracker_TailSpikeReportsDegraded(t *testing.T) {
	// 49 fast + 1 slow → p99 interpolates between the last fast sample
	// and the slow one (nearest-rank with 50 entries hits index 48.51).
	// Use 80ms slow value so p99 lands ≈ 50ms (2.5x p50 = degraded
	// threshold) but well below critical (100ms = 5x).
	tr := newLatencyTracker()
	for range 49 {
		tr.Record(KindEmbed, 20*time.Millisecond, false)
	}
	tr.Record(KindEmbed, 80*time.Millisecond, false)
	snap := tr.SnapshotKind(KindEmbed)
	if snap.Status != StatusDegraded {
		t.Errorf("expected degraded on tail-spike, got %s "+
			"(p50=%.1fms p99=%.1fms)",
			snap.Status, snap.P50Ms, snap.P99Ms)
	}
}

func TestLatencyTracker_DeepTailReportsCritical(t *testing.T) {
	// 49 fast + 1 catastrophically slow → p99 / p50 > 5x → critical.
	tr := newLatencyTracker()
	for range 49 {
		tr.Record(KindEmbed, 20*time.Millisecond, false)
	}
	tr.Record(KindEmbed, 500*time.Millisecond, false) // 25x p50
	snap := tr.SnapshotKind(KindEmbed)
	if snap.Status != StatusCritical {
		t.Errorf("expected critical on 25x tail spike, got %s "+
			"(p50=%.1fms p99=%.1fms)",
			snap.Status, snap.P50Ms, snap.P99Ms)
	}
}

func TestLatencyTracker_HighErrorRateReportsCritical(t *testing.T) {
	// Latency is fine, but 10% of calls erroring is the impending-crash
	// signal — error rate above 5% → critical regardless of latency.
	tr := newLatencyTracker()
	for i := range 50 {
		isErr := i%10 == 0 // 10% error rate
		tr.Record(KindEmbed, 20*time.Millisecond, isErr)
	}
	snap := tr.SnapshotKind(KindEmbed)
	if snap.Status != StatusCritical {
		t.Errorf("expected critical on 10%% error rate, got %s "+
			"(error_rate=%.2f)",
			snap.Status, snap.ErrorRate)
	}
}

func TestLatencyTracker_PerKindIsolation(t *testing.T) {
	// Embed degraded must not bleed into rerank status.
	tr := newLatencyTracker()
	for range 49 {
		tr.Record(KindEmbed, 20*time.Millisecond, false)
	}
	tr.Record(KindEmbed, 500*time.Millisecond, false)
	for range 50 {
		tr.Record(KindRerank, 30*time.Millisecond, false)
	}
	embed := tr.SnapshotKind(KindEmbed)
	rerank := tr.SnapshotKind(KindRerank)
	if embed.Status != StatusCritical {
		t.Errorf("embed should be critical, got %s", embed.Status)
	}
	if rerank.Status != StatusOK {
		t.Errorf("rerank should be ok, got %s", rerank.Status)
	}
}

func TestLatencyTracker_WindowTrimming(t *testing.T) {
	// Samples older than the window must not influence the snapshot.
	// We can't easily move the clock without dependency injection,
	// so instead simulate by directly mutating the slice.
	tr := newLatencyTracker()
	old := time.Now().Add(-2 * latencyWindow)
	tr.mu.Lock()
	for range 60 {
		tr.samples[KindEmbed] = append(tr.samples[KindEmbed], latencySample{
			at:  old,
			dur: 5 * time.Second, // would scream critical if not trimmed
		})
	}
	tr.mu.Unlock()

	// Now record a normal batch in-window.
	for range 30 {
		tr.Record(KindEmbed, 20*time.Millisecond, false)
	}
	snap := tr.SnapshotKind(KindEmbed)
	if snap.Samples != 30 {
		t.Errorf("expected 30 in-window samples after trim, got %d",
			snap.Samples)
	}
	if snap.Status != StatusOK {
		t.Errorf("expected ok after window-trim, got %s "+
			"(p50=%.1fms p99=%.1fms)",
			snap.Status, snap.P50Ms, snap.P99Ms)
	}
}

func TestLatencyTracker_RingBufferCapBounded(t *testing.T) {
	// Sustained traffic above the cap should not grow memory unboundedly.
	tr := newLatencyTracker()
	for range latencySampleCap + 200 {
		tr.Record(KindEmbed, 20*time.Millisecond, false)
	}
	tr.mu.Lock()
	n := len(tr.samples[KindEmbed])
	tr.mu.Unlock()
	if n > latencySampleCap {
		t.Errorf("ring buffer grew past cap: got %d, want <= %d",
			n, latencySampleCap)
	}
}

func TestClassify_TableDriven(t *testing.T) {
	// Decouple threshold semantics from latency-sample plumbing so the
	// thresholds are individually verifiable.
	cases := []struct {
		name   string
		n      int
		p50    float64
		p99    float64
		errs   float64
		want   SlotStatus
	}{
		{"below-min-samples", 5, 10, 1000, 0, StatusOK},
		{"healthy-ratio", 50, 10, 18, 0, StatusOK},
		{"degraded-2x", 50, 10, 20, 0, StatusDegraded},
		{"degraded-4x", 50, 10, 40, 0, StatusDegraded},
		{"critical-5x", 50, 10, 50, 0, StatusCritical},
		{"critical-10x", 50, 10, 100, 0, StatusCritical},
		{"critical-by-error-rate", 50, 10, 15, 0.10, StatusCritical},
		{"zero-p50-stays-ok", 50, 0, 0, 0, StatusOK},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := classify(c.n, c.p50, c.p99, c.errs)
			if got != c.want {
				t.Errorf("classify(n=%d, p50=%.1f, p99=%.1f, err=%.2f) = %s, want %s",
					c.n, c.p50, c.p99, c.errs, got, c.want)
			}
		})
	}
}

func TestPercentile(t *testing.T) {
	// Sanity-check the nearest-rank percentile against hand-picked values.
	sorted := []float64{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
	if got := percentile(sorted, 0.5); got < 50 || got > 60 {
		t.Errorf("p50 = %.1f, expected ~50-60", got)
	}
	if got := percentile(sorted, 0.99); got < 99 || got > 100 {
		t.Errorf("p99 = %.1f, expected ~99-100", got)
	}
	if got := percentile([]float64{}, 0.5); got != 0 {
		t.Errorf("empty slice should return 0, got %.1f", got)
	}
}
