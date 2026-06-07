// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package gateway

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/cerid-ai/quenchforge/internal/config"
	"github.com/cerid-ai/quenchforge/internal/scheduler"
)

func TestGatedPassthroughWhenNoScheduler(t *testing.T) {
	g := New(config.Config{})
	ran := false
	h := g.gated(KindChat, func(w http.ResponseWriter, _ *http.Request) {
		ran = true
		w.WriteHeader(http.StatusOK)
	})
	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest(http.MethodPost, "/api/chat", nil))
	if !ran || rec.Code != http.StatusOK {
		t.Fatalf("passthrough failed: ran=%v code=%d", ran, rec.Code)
	}
}

func TestGatedAdmitsWhenSlotFree(t *testing.T) {
	g := New(config.Config{})
	g.SetScheduler(scheduler.New(2))
	ran := false
	h := g.gated(KindEmbed, func(w http.ResponseWriter, _ *http.Request) {
		ran = true
		w.WriteHeader(http.StatusOK)
	})
	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest(http.MethodPost, "/api/embeddings", nil))
	if !ran || rec.Code != http.StatusOK {
		t.Fatalf("admit failed: ran=%v code=%d", ran, rec.Code)
	}
}

func TestGatedBackpressures503WhenSaturated(t *testing.T) {
	g := New(config.Config{})
	s := scheduler.New(1)
	g.SetScheduler(s)

	// Occupy the only slot directly.
	hold, _ := s.Acquire(context.Background(), scheduler.PriorityChat)
	defer hold()

	// A gated request whose context times out while waiting must 503.
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()
	ran := false
	h := g.gated(KindChat, func(http.ResponseWriter, *http.Request) { ran = true })
	rec := httptest.NewRecorder()
	h(rec, httptest.NewRequest(http.MethodPost, "/api/chat", nil).WithContext(ctx))

	if ran {
		t.Error("handler ran despite saturation")
	}
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("code = %d, want 503", rec.Code)
	}
	if rec.Header().Get("Retry-After") == "" {
		t.Error("missing Retry-After header on backpressure 503")
	}
}

func TestGatedReleasesSlotAfterHandler(t *testing.T) {
	g := New(config.Config{})
	s := scheduler.New(1)
	g.SetScheduler(s)
	h := g.gated(KindChat, func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	for i := 0; i < 3; i++ {
		h(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/api/chat", nil))
	}
	if s.Active() != 0 {
		t.Fatalf("slot leaked: active=%d want 0", s.Active())
	}
}
