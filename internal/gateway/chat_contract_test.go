// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package gateway

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/cerid-ai/quenchforge/internal/config"
)

// newChatGateway starts a gateway whose chat slot is backed by h.
func newChatGateway(t *testing.T, cfg config.Config, h http.Handler) (*Gateway, string) {
	t.Helper()
	upstream := httptest.NewServer(h)
	t.Cleanup(upstream.Close)
	cfg.ListenAddr = pickListenAddr(t)
	g := newRunningGateway(t, cfg)
	if err := g.SetUpstream(KindChat, upstream.URL); err != nil {
		t.Fatalf("SetUpstream: %v", err)
	}
	return g, cfg.ListenAddr
}

// The Ollama chat surface is the primary path; without a latency sample it
// is invisible to /health and to auto-backoff.
func TestChatRecordsLatencySample(t *testing.T) {
	cfg := newTestConfig(t)
	chatResp := `{"id":"c1","model":"` + cfg.DefaultModel + `","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{}}`
	g, addr := newChatGateway(t, cfg, chatUpstreamHandler(cfg.DefaultModel, chatResp, "application/json"))

	body := `{"model":"` + cfg.DefaultModel + `","stream":false,"messages":[{"role":"user","content":"hi"}]}`
	if status, respBody := postChat(t, addr, body); status != http.StatusOK {
		t.Fatalf("status = %d body=%s", status, respBody)
	}
	if got := g.latency.SnapshotKind(KindChat).Samples; got != 1 {
		t.Errorf("chat latency samples = %d, want 1", got)
	}
}

// Auto-backoff must be able to shed chat load, same as the proxy routes.
func TestChatShedsUnderAutoBackoff(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.AutoBackoffEnabled = true
	chatResp := `{"id":"c1","model":"` + cfg.DefaultModel + `","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{}}`
	g, addr := newChatGateway(t, cfg, chatUpstreamHandler(cfg.DefaultModel, chatResp, "application/json"))
	for i := 0; i < statusMinSamples+5; i++ {
		g.latency.Record(KindChat, 10*time.Millisecond, true)
	}

	body := `{"model":"` + cfg.DefaultModel + `","stream":false,"messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post("http://"+addr+"/api/chat", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503 while the chat slot is shedding", resp.StatusCode)
	}
	if resp.Header.Get("Retry-After") == "" {
		t.Errorf("missing Retry-After header on a shed chat request")
	}
}

// A stream that ends before the upstream signalled completion is a severed
// answer; rendering it as done_reason "stop" makes a truncation
// indistinguishable from a finished reply.
func TestTruncatedChatStreamIsNotReportedAsStop(t *testing.T) {
	cfg := newTestConfig(t)
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			writeJSON(w, http.StatusOK, map[string]any{
				"object": "list", "data": []map[string]any{{"id": cfg.DefaultModel}},
			})
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`data: {"id":"c1","model":"` + cfg.DefaultModel + `","choices":[{"index":0,"delta":{"role":"assistant","content":"par"}}]}` + "\n\n"))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		// Upstream dies mid-generation: no finish_reason, no [DONE].
	})
	g, addr := newChatGateway(t, cfg, h)

	body := `{"model":"` + cfg.DefaultModel + `","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post("http://"+addr+"/api/chat", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	var lines []map[string]any
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		var rec map[string]any
		if err := json.Unmarshal(scanner.Bytes(), &rec); err != nil {
			t.Fatalf("non-JSON NDJSON line %q", scanner.Text())
		}
		lines = append(lines, rec)
	}
	if len(lines) == 0 {
		t.Fatal("no NDJSON lines")
	}
	last := lines[len(lines)-1]
	if last["done"] != true {
		t.Fatalf("last line is not terminal: %v", last)
	}
	if last["done_reason"] == "stop" {
		t.Errorf("done_reason = stop on a truncated stream: %v", last)
	}
	if last["done_reason"] != "error" {
		t.Errorf("done_reason = %v, want error", last["done_reason"])
	}
	if _, ok := last["error"]; !ok {
		t.Errorf("truncated stream carries no error detail: %v", last)
	}
	if snap := g.latency.SnapshotKind(KindChat); snap.Samples != 1 || snap.ErrorRate == 0 {
		t.Errorf("truncation not recorded as an error sample: %+v", snap)
	}
}

// A complete stream keeps reporting done_reason "stop".
func TestCompleteChatStreamStillReportsStop(t *testing.T) {
	cfg := newTestConfig(t)
	sse := strings.Join([]string{
		`data: {"id":"c1","model":"` + cfg.DefaultModel + `","choices":[{"index":0,"delta":{"role":"assistant","content":"foo"}}]}`,
		``,
		`data: {"id":"c1","model":"` + cfg.DefaultModel + `","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	_, addr := newChatGateway(t, cfg, chatUpstreamHandler(cfg.DefaultModel, sse, "text/event-stream"))

	body := `{"model":"` + cfg.DefaultModel + `","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post("http://"+addr+"/api/chat", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	var last map[string]any
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		var rec map[string]any
		_ = json.Unmarshal(scanner.Bytes(), &rec)
		last = rec
	}
	if last["done_reason"] != "stop" {
		t.Errorf("done_reason = %v, want stop", last["done_reason"])
	}
}

// A slot that died leaves a registered upstream behind: the gateway keeps
// proxying at a corpse and /health never notices.
func TestDeadChatUpstreamIsDeregistered(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	g := newRunningGateway(t, cfg)
	dead := pickListenAddr(t) // nothing is listening here
	if err := g.SetUpstream(KindChat, "http://"+dead); err != nil {
		t.Fatalf("SetUpstream: %v", err)
	}

	body := `{"model":"` + cfg.DefaultModel + `","stream":false,"messages":[{"role":"user","content":"hi"}]}`
	if status, _ := postChat(t, cfg.ListenAddr, body); status != http.StatusBadGateway {
		t.Fatalf("first call status = %d, want 502", status)
	}

	health := getHealth(t, cfg.ListenAddr)
	chat := healthSlot(t, health, KindChat)
	if chat["status"] != string(StatusUnreachable) {
		t.Errorf("chat status = %v, want %q", chat["status"], StatusUnreachable)
	}
	if chat["configured"] != false {
		t.Errorf("chat configured = %v, want false after the upstream proved dead", chat["configured"])
	}
	if health["status"] == string(StatusOK) {
		t.Errorf("overall status = ok while the chat upstream is dead")
	}

	status, respBody := postChat(t, cfg.ListenAddr, body)
	if status != http.StatusServiceUnavailable {
		t.Errorf("second call status = %d, want 503 (deregistered, not proxied at a corpse)", status)
	}
	if !strings.Contains(respBody, "unreachable") {
		t.Errorf("second call body = %s, want an unreachable explanation", respBody)
	}
}
