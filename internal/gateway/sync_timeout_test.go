// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package gateway

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func postJSON(t *testing.T, url, body string) (int, string) {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(raw)
}

// A wedged embed slot must fail the caller instead of hanging it forever.
func TestEmbedUpstreamTimesOut(t *testing.T) {
	prev := syncUpstreamTimeout
	syncUpstreamTimeout = 50 * time.Millisecond
	t.Cleanup(func() { syncUpstreamTimeout = prev })

	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	cfg.EmbedModel = "nomic-embed-text-v1.5"
	g := newRunningGateway(t, cfg)
	wedged := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second) // slot stalled in a Metal call
		writeJSON(w, http.StatusOK, map[string]any{"data": []any{}})
	}))
	t.Cleanup(wedged.Close)
	if err := g.SetUpstream(KindEmbed, wedged.URL); err != nil {
		t.Fatalf("SetUpstream: %v", err)
	}

	done := make(chan struct{})
	var status int
	var body string
	go func() {
		status, body = postJSON(t, "http://"+cfg.ListenAddr+"/api/embeddings",
			`{"model":"nomic-embed-text-v1.5","prompt":"hello"}`)
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("embed request did not return within 1s — no upstream timeout")
	}
	if status != http.StatusGatewayTimeout {
		t.Errorf("status = %d body=%s, want 504", status, body)
	}
}
