// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package gateway

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/cerid-ai/quenchforge/internal/config"
	"github.com/cerid-ai/quenchforge/internal/placement"
	"github.com/cerid-ai/quenchforge/internal/scheduler"
)

const (
	gpuEmbedURL = "http://127.0.0.1:11501"
	cpuEmbedURL = "http://127.0.0.1:11511"
)

func TestCountEmbedInputs(t *testing.T) {
	cases := []struct {
		name   string
		input  interface{}
		prompt string
		want   int
	}{
		{"nil falls back to prompt", nil, "hi", 1},
		{"nil no prompt -> 1", nil, "", 1},
		{"empty string falls back to prompt", "", "hi", 1},
		{"empty string no prompt -> 1", "", "", 1},
		{"single string", "one", "", 1},
		{"string slice", []string{"a", "b", "c"}, "", 3},
		{"interface slice", []interface{}{"a", "b"}, "", 2},
		{"empty interface slice -> 1", []interface{}{}, "", 1},
		{"empty string slice falls back to prompt", []string{}, "hi", 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := countEmbedInputs(c.input, c.prompt); got != c.want {
				t.Errorf("countEmbedInputs(%v, %q) = %d, want %d", c.input, c.prompt, got, c.want)
			}
		})
	}
}

// routeEmbedFixture builds a gateway with embed upstreams registered and the
// given placement mode installed for the embed kind.
func routeEmbedFixture(t *testing.T, mode string, withCPU bool) *Gateway {
	t.Helper()
	g := New(config.Config{})
	if err := g.SetUpstream(KindEmbed, gpuEmbedURL); err != nil {
		t.Fatalf("SetUpstream: %v", err)
	}
	if withCPU {
		if err := g.SetCPUUpstream(KindEmbed, cpuEmbedURL); err != nil {
			t.Fatalf("SetCPUUpstream: %v", err)
		}
	}
	g.SetPlacement(placement.NewPolicy(false, map[string]string{placement.KindEmbed: mode}), 1)
	return g
}

func TestRouteEmbed_GPUMode(t *testing.T) {
	g := routeEmbedFixture(t, placement.ModeGPU, false)
	// Both single and batch go to the GPU upstream, governed.
	for _, batchN := range []int{1, 32} {
		entry, onGPU, track, ok := g.routeEmbed(KindEmbed, batchN)
		if !ok || !onGPU {
			t.Fatalf("batchN=%d: ok=%v onGPU=%v, want true/true", batchN, ok, onGPU)
		}
		if entry.url.String() != gpuEmbedURL {
			t.Errorf("batchN=%d: routed to %s, want %s", batchN, entry.url, gpuEmbedURL)
		}
		if track != KindEmbed {
			t.Errorf("batchN=%d: track=%s, want %s", batchN, track, KindEmbed)
		}
	}
}

func TestRouteEmbed_CPUMode(t *testing.T) {
	g := routeEmbedFixture(t, placement.ModeCPU, false)
	// cpu mode always uses the (single) registered upstream, ungoverned. The
	// tracker key stays the kind — a single homogeneous instance.
	entry, onGPU, track, ok := g.routeEmbed(KindEmbed, 64)
	if !ok || onGPU {
		t.Fatalf("ok=%v onGPU=%v, want true/false", ok, onGPU)
	}
	if entry.url.String() != gpuEmbedURL {
		t.Errorf("cpu mode routed to %s, want the primary upstream %s", entry.url, gpuEmbedURL)
	}
	if track != KindEmbed {
		t.Errorf("track=%s, want %s (single instance keeps the kind key)", track, KindEmbed)
	}
}

func TestRouteEmbed_AutoRoutesByBatch(t *testing.T) {
	g := routeEmbedFixture(t, placement.ModeAuto, true)

	// Single (<= threshold 1) -> CPU instance, ungoverned, tracked under the
	// "-cpu" instance key so its millisecond latencies never mix with the GPU
	// instance's multi-second batches (the 2026-07-08 false-critical bug).
	entry, onGPU, track, ok := g.routeEmbed(KindEmbed, 1)
	if !ok || onGPU {
		t.Fatalf("single: ok=%v onGPU=%v, want true/false", ok, onGPU)
	}
	if entry.url.String() != cpuEmbedURL {
		t.Errorf("single auto routed to %s, want CPU %s", entry.url, cpuEmbedURL)
	}
	if track != cpuTrackKind(KindEmbed) {
		t.Errorf("single auto track=%s, want %s", track, cpuTrackKind(KindEmbed))
	}

	// Batch (> threshold) -> GPU instance, governed, tracked under the kind.
	entry, onGPU, track, ok = g.routeEmbed(KindEmbed, 8)
	if !ok || !onGPU {
		t.Fatalf("batch: ok=%v onGPU=%v, want true/true", ok, onGPU)
	}
	if entry.url.String() != gpuEmbedURL {
		t.Errorf("batch auto routed to %s, want GPU %s", entry.url, gpuEmbedURL)
	}
	if track != KindEmbed {
		t.Errorf("batch auto track=%s, want %s", track, KindEmbed)
	}
}

func TestRouteEmbed_AutoFallsBackToGPUWhenNoCPU(t *testing.T) {
	// No CPU upstream registered: a single request that would route CPU must
	// fall back to the GPU instance (degrade to working, not 503).
	g := routeEmbedFixture(t, placement.ModeAuto, false)
	entry, onGPU, track, ok := g.routeEmbed(KindEmbed, 1)
	if !ok || !onGPU {
		t.Fatalf("ok=%v onGPU=%v, want true/true (GPU fallback)", ok, onGPU)
	}
	if entry.url.String() != gpuEmbedURL {
		t.Errorf("auto fallback routed to %s, want GPU %s", entry.url, gpuEmbedURL)
	}
	if track != KindEmbed {
		t.Errorf("fallback track=%s, want %s (it IS the GPU instance)", track, KindEmbed)
	}
}

func TestRouteEmbed_NoUpstreamIsNotOK(t *testing.T) {
	g := New(config.Config{})
	g.SetPlacement(placement.NewPolicy(false, nil), 1)
	if _, _, _, ok := g.routeEmbed(KindEmbed, 1); ok {
		t.Error("routeEmbed ok=true with no upstream registered")
	}
}

func TestRouteEmbed_ZeroPolicyDefaultsToGPUUpstream(t *testing.T) {
	// A gateway that never had SetPlacement called must behave like the
	// pre-placement single-upstream path: GPU upstream, governed.
	g := New(config.Config{})
	if err := g.SetUpstream(KindEmbed, gpuEmbedURL); err != nil {
		t.Fatalf("SetUpstream: %v", err)
	}
	entry, onGPU, track, ok := g.routeEmbed(KindEmbed, 1)
	if !ok || !onGPU || entry.url.String() != gpuEmbedURL {
		t.Fatalf("zero policy: entry=%v onGPU=%v ok=%v, want gpu upstream/true/true",
			entry.url, onGPU, ok)
	}
	if track != KindEmbed {
		t.Errorf("zero-policy track=%s, want %s", track, KindEmbed)
	}
}

// TestGatedCPUKindSkipsAdmission proves a CPU-placed kind runs ungoverned: even
// with the scheduler fully saturated (and an un-cancellable request context, so
// a GPU-placed kind would block forever), the handler runs immediately.
func TestGatedCPUKindSkipsAdmission(t *testing.T) {
	g := New(config.Config{})
	s := scheduler.New(1)
	g.SetScheduler(s)
	hold, _ := s.Acquire(context.Background(), scheduler.PriorityChat) // saturate
	defer hold()
	g.SetPlacement(placement.NewPolicy(false, map[string]string{placement.KindChat: placement.ModeCPU}), 1)

	ran := false
	h := g.gated(KindChat, func(w http.ResponseWriter, _ *http.Request) {
		ran = true
		w.WriteHeader(http.StatusOK)
	})
	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest(http.MethodPost, "/api/chat", nil))
	if !ran {
		t.Fatal("CPU-placed handler should run despite a saturated scheduler")
	}
	if rec.Code != http.StatusOK {
		t.Fatalf("code = %d, want 200", rec.Code)
	}
	if s.Active() != 1 {
		t.Fatalf("CPU kind must not consume a scheduler slot: active=%d, want 1 (the held one)", s.Active())
	}
}

// TestGatedGPUKindStillGoverned is the companion: a GPU-placed kind under the
// same saturated scheduler is backpressured (503) once its context expires —
// confirming the refactor preserved gated's GPU admission semantics.
func TestGatedGPUKindStillGoverned(t *testing.T) {
	g := New(config.Config{})
	s := scheduler.New(1)
	g.SetScheduler(s)
	hold, _ := s.Acquire(context.Background(), scheduler.PriorityChat)
	defer hold()
	g.SetPlacement(placement.NewPolicy(false, nil), 1) // all GPU

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already expired -> Acquire returns immediately with err
	ran := false
	h := g.gated(KindChat, func(http.ResponseWriter, *http.Request) { ran = true })
	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest(http.MethodPost, "/api/chat", nil).WithContext(ctx))
	if ran {
		t.Error("GPU-placed handler ran despite saturation + expired context")
	}
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("code = %d, want 503", rec.Code)
	}
}
