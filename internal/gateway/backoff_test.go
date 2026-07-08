// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package gateway

import (
	"testing"
	"time"

	"github.com/cerid-ai/quenchforge/internal/config"
)

// These tests pin the v0.9.1 auto-backoff contract, written against the
// 2026-07-08 cerid eval incident: shedding fires on a critical ERROR RATE
// only — the family-B crash signature — never on the p99/p50 latency ratio,
// which is workload-shape sensitive (one slow GPU batch against millisecond
// CPU singles read as ratio≈1700 with zero failures and 503'd all embed
// traffic for the rest of the rolling window).

func recordN(g *Gateway, kind SlotKind, n int, dur time.Duration, isError bool) {
	for i := 0; i < n; i++ {
		g.latency.Record(kind, dur, isError)
	}
}

func TestShouldBackoff_ErrorRateCriticalSheds(t *testing.T) {
	g := New(config.Config{AutoBackoffEnabled: true})
	// 30 samples, 4 errors (13% > 5% threshold) — the crash signature.
	recordN(g, KindEmbed, 26, 10*time.Millisecond, false)
	recordN(g, KindEmbed, 4, 10*time.Millisecond, true)
	if !g.shouldBackoff(KindEmbed) {
		t.Error("critical error rate must shed when AutoBackoffEnabled")
	}
}

func TestShouldBackoff_LatencyRatioDoesNotShed(t *testing.T) {
	// Incident regression: 20 fast singles + 1 slow batch = ratio >> 5
	// (classify reports critical) but zero errors — must NOT shed.
	g := New(config.Config{AutoBackoffEnabled: true})
	recordN(g, KindEmbed, 20, 12*time.Millisecond, false)
	recordN(g, KindEmbed, 1, 26*time.Second, false)

	if snap := g.latency.SnapshotKind(KindEmbed); snap.Status != StatusCritical {
		t.Fatalf("fixture must reproduce the ratio-critical snapshot, got %s", snap.Status)
	}
	if g.shouldBackoff(KindEmbed) {
		t.Error("latency-ratio critical with zero errors must NOT shed (observability-only)")
	}
}

func TestShouldBackoff_DisabledByDefault(t *testing.T) {
	g := New(config.Config{}) // AutoBackoffEnabled false
	recordN(g, KindEmbed, 30, 10*time.Millisecond, true) // 100% errors
	if g.shouldBackoff(KindEmbed) {
		t.Error("backoff must stay off without QUENCHFORGE_AUTO_BACKOFF")
	}
}

func TestShouldBackoff_InstanceIsolation(t *testing.T) {
	// A failing GPU instance must not shed traffic bound for the healthy CPU
	// twin: the two record under distinct tracker keys.
	g := New(config.Config{AutoBackoffEnabled: true})
	recordN(g, KindEmbed, 30, 10*time.Millisecond, true) // GPU instance failing hard
	recordN(g, cpuTrackKind(KindEmbed), 30, 8*time.Millisecond, false)

	if !g.shouldBackoff(KindEmbed) {
		t.Error("failing GPU instance should shed")
	}
	if g.shouldBackoff(cpuTrackKind(KindEmbed)) {
		t.Error("healthy CPU twin must NOT shed because the GPU instance is failing")
	}
}

func TestShouldBackoff_BelowMinSamplesNeverSheds(t *testing.T) {
	g := New(config.Config{AutoBackoffEnabled: true})
	recordN(g, KindEmbed, statusMinSamples-1, 10*time.Millisecond, true)
	if g.shouldBackoff(KindEmbed) {
		t.Error("below statusMinSamples the tracker has no signal — must not shed")
	}
}
