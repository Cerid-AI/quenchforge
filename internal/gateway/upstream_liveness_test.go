// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package gateway

import (
	"encoding/json"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"
)

// serveOnAddr starts a server on an exact address so a test can kill an
// upstream and bring a replacement up on the same port — the respawn shape
// the gateway has to recover from.
func serveOnAddr(t *testing.T, addr string, h http.Handler) *http.Server {
	t.Helper()
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		t.Fatalf("listen %s: %v", addr, err)
	}
	srv := &http.Server{Handler: h, ReadHeaderTimeout: time.Second}
	go func() { _ = srv.Serve(ln) }()
	t.Cleanup(func() { _ = srv.Close() })
	return srv
}

// chatUpstreamHandler fakes the llama-server surface the gateway talks to:
// GET /v1/models advertises the loaded model, POST /v1/chat/completions
// answers.
func chatUpstreamHandler(model, chatBody, contentType string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			writeJSON(w, http.StatusOK, map[string]any{
				"object": "list",
				"data":   []map[string]any{{"id": model, "object": "model"}},
			})
			return
		}
		w.Header().Set("Content-Type", contentType)
		_, _ = w.Write([]byte(chatBody))
	}
}

func postChat(t *testing.T, addr, body string) (int, string) {
	t.Helper()
	resp, err := http.Post("http://"+addr+"/api/chat", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST /api/chat: %v", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(raw)
}

// Same for the reverse-proxy routes: a dead rerank slot must deregister
// rather than answer 502 forever.
func TestDeadProxyUpstreamIsDeregistered(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	cfg.RerankModel = "bge-reranker-v2-m3"
	g := newRunningGateway(t, cfg)
	dead := pickListenAddr(t)
	if err := g.SetUpstream(KindRerank, "http://"+dead); err != nil {
		t.Fatalf("SetUpstream: %v", err)
	}

	resp, err := http.Post("http://"+cfg.ListenAddr+"/v1/rerank", "application/json",
		strings.NewReader(`{"query":"cat","documents":["a cat sat"]}`))
	if err != nil {
		t.Fatalf("POST /v1/rerank: %v", err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("first call status = %d, want 502", resp.StatusCode)
	}

	rerank := healthSlot(t, getHealth(t, cfg.ListenAddr), KindRerank)
	if rerank["status"] != string(StatusUnreachable) {
		t.Errorf("rerank status = %v, want %q", rerank["status"], StatusUnreachable)
	}
}

// Deregistration must not be a one-way door: a slot that respawns on the
// same port has to be picked up again without restarting quenchforge.
func TestUnreachableUpstreamIsRetriedAfterCooloff(t *testing.T) {
	prev := upstreamRetryCooloff
	upstreamRetryCooloff = 50 * time.Millisecond
	t.Cleanup(func() { upstreamRetryCooloff = prev })

	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	g := newRunningGateway(t, cfg)
	slotAddr := pickListenAddr(t)
	if err := g.SetUpstream(KindChat, "http://"+slotAddr); err != nil {
		t.Fatalf("SetUpstream: %v", err)
	}

	body := `{"model":"` + cfg.DefaultModel + `","stream":false,"messages":[{"role":"user","content":"hi"}]}`
	if status, _ := postChat(t, cfg.ListenAddr, body); status != http.StatusBadGateway {
		t.Fatalf("pre-respawn status = %d, want 502", status)
	}

	// Slot respawns on the same port.
	chatResp := `{"id":"c1","model":"` + cfg.DefaultModel + `","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{}}`
	serveOnAddr(t, slotAddr, chatUpstreamHandler(cfg.DefaultModel, chatResp, "application/json"))
	time.Sleep(2 * upstreamRetryCooloff)

	status, respBody := postChat(t, cfg.ListenAddr, body)
	if status != http.StatusOK {
		t.Fatalf("post-respawn status = %d body=%s, want 200", status, respBody)
	}
	var out map[string]any
	if err := json.Unmarshal([]byte(respBody), &out); err != nil {
		t.Fatalf("decode: %v body=%s", err, respBody)
	}
	if out["done"] != true {
		t.Errorf("post-respawn response = %v, want a completed chat reply", out)
	}
}
