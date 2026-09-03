// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"testing"
	"time"
)

// getHealth fetches /health and decodes it into the loose map shape a
// consumer (cerid-ai, a monitor) would parse.
func getHealth(t *testing.T, addr string) map[string]any {
	t.Helper()
	resp, err := http.Get("http://" + addr + "/health")
	if err != nil {
		t.Fatalf("GET /health: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("/health status = %d, want 200", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("decode /health: %v body=%s", err, body)
	}
	return got
}

func healthSlot(t *testing.T, health map[string]any, kind SlotKind) map[string]any {
	t.Helper()
	slots, ok := health["slots"].(map[string]any)
	if !ok {
		t.Fatalf("/health has no slots object: %v", health)
	}
	entry, ok := slots[string(kind)].(map[string]any)
	if !ok {
		t.Fatalf("/health slots has no %q entry: %v", kind, slots)
	}
	return entry
}

// A route mounted for a kind the operator configured, but with no upstream
// registered, is the exact shape of the rerank outage: /v1/rerank 503s on
// every call while /health answers "ok".
func TestHealthReportsConfiguredSlotWithNoUpstreamAsDegraded(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	cfg.RerankModel = "bge-reranker-v2-m3"
	newRunningGateway(t, cfg)

	health := getHealth(t, cfg.ListenAddr)
	if health["status"] != string(StatusDegraded) {
		t.Errorf("overall status = %v, want degraded (rerank is configured but has no upstream)", health["status"])
	}
	rerank := healthSlot(t, health, KindRerank)
	if rerank["status"] != string(StatusUnconfigured) {
		t.Errorf("rerank status = %v, want %q", rerank["status"], StatusUnconfigured)
	}
	if rerank["configured"] != false {
		t.Errorf("rerank configured = %v, want false", rerank["configured"])
	}
}

// A kind the operator never asked for (no model env var) is reported as
// disabled and must NOT degrade the overall status — otherwise a
// chat-only deployment reads "degraded" forever and consumers learn to
// ignore the field.
func TestHealthReportsUnrequestedSlotAsDisabledWithoutDegrading(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	g := newRunningGateway(t, cfg)
	upstream, _ := newCapturingUpstream(t, http.StatusOK, `{}`, "application/json")
	if err := g.SetUpstream(KindChat, upstream.URL); err != nil {
		t.Fatalf("SetUpstream: %v", err)
	}

	health := getHealth(t, cfg.ListenAddr)
	if health["status"] != string(StatusOK) {
		t.Errorf("overall status = %v, want ok (only chat is requested and it is up)", health["status"])
	}
	tts := healthSlot(t, health, KindTTS)
	if tts["status"] != string(StatusDisabled) {
		t.Errorf("tts status = %v, want %q", tts["status"], StatusDisabled)
	}
	chat := healthSlot(t, health, KindChat)
	if chat["configured"] != true {
		t.Errorf("chat configured = %v, want true", chat["configured"])
	}
	if chat["upstream"] != upstream.URL {
		t.Errorf("chat upstream = %v, want %s", chat["upstream"], upstream.URL)
	}
}

// The existing latency keys are a published contract (cerid reads this
// endpoint); the readiness fields are additive.
func TestHealthKeepsLatencySchemaForServingSlot(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	cfg.EmbedModel = "nomic-embed-text-v1.5"
	g := newRunningGateway(t, cfg)
	upstream, _ := newCapturingUpstream(t, http.StatusOK, `{}`, "application/json")
	if err := g.SetUpstream(KindEmbed, upstream.URL); err != nil {
		t.Fatalf("SetUpstream: %v", err)
	}
	g.latency.Record(KindEmbed, 10*time.Millisecond, false)

	health := getHealth(t, cfg.ListenAddr)
	if _, ok := health["auto_backoff_enabled"]; !ok {
		t.Errorf("/health lost auto_backoff_enabled key: %v", health)
	}
	embed := healthSlot(t, health, KindEmbed)
	for _, key := range []string{"kind", "samples", "p50_ms", "p99_ms", "error_rate", "status", "window_secs"} {
		if _, ok := embed[key]; !ok {
			t.Errorf("/health embed slot lost key %q: %v", key, embed)
		}
	}
	if embed["samples"] != float64(1) {
		t.Errorf("embed samples = %v, want 1", embed["samples"])
	}
	if embed["configured"] != true {
		t.Errorf("embed configured = %v, want true", embed["configured"])
	}
}
