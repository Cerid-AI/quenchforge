// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package gateway

import (
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The chat slot serves exactly one model. Forwarding a mismatched pin
// verbatim means the caller is answered by a different model than it asked
// for and cannot tell.
func TestChatRejectsModelMismatch(t *testing.T) {
	cfg := newTestConfig(t)
	served := "qwen2.5-7b-instruct-q4_k_m.gguf" // what llama-server reports
	chatResp := `{"id":"c1","model":"` + served + `","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{}}`
	_, addr := newChatGateway(t, cfg, chatUpstreamHandler(served, chatResp, "application/json"))

	status, body := postChat(t, addr,
		`{"model":"llama3.1-8b","stream":false,"messages":[{"role":"user","content":"hi"}]}`)
	if status != http.StatusBadRequest {
		t.Fatalf("status = %d body=%s, want 400 for a mismatched model pin", status, body)
	}
	if !strings.Contains(body, served) {
		t.Errorf("400 body = %s, want it to name the served model %q", body, served)
	}
}

// The same slot must still answer the name it actually loaded, including
// the Ollama-style spelling of the configured model.
func TestChatAcceptsRequestedModelThatMatchesSlot(t *testing.T) {
	cfg := newTestConfig(t)
	served := "qwen2.5-7b-instruct-q4_k_m.gguf"
	chatResp := `{"id":"c1","model":"` + served + `","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{}}`
	_, addr := newChatGateway(t, cfg, chatUpstreamHandler(served, chatResp, "application/json"))

	for _, model := range []string{"", "qwen2.5:7b-instruct-q4_k_m", "qwen2.5:7b", served} {
		status, body := postChat(t, addr,
			`{"model":"`+model+`","stream":false,"messages":[{"role":"user","content":"hi"}]}`)
		if status != http.StatusOK {
			t.Errorf("model %q: status = %d body=%s, want 200", model, status, body)
		}
	}
}

// /api/tags enumerates every cached GGUF, but only the model a slot has
// loaded can actually answer a request.
func TestTagsMarksWhichModelsAreLoaded(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	if err := os.MkdirAll(cfg.ModelsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"qwen2.5-7b-instruct-q4_k_m.gguf", "llama-3.2-3b.gguf"} {
		if err := os.WriteFile(filepath.Join(cfg.ModelsDir, name), []byte("not a real model"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	g := newRunningGateway(t, cfg)
	served := "qwen2.5-7b-instruct-q4_k_m.gguf"
	upstream, _ := newCapturingUpstream(t, http.StatusOK,
		`{"object":"list","data":[{"id":"`+served+`","object":"model"}]}`, "application/json")
	if err := g.SetUpstream(KindChat, upstream.URL); err != nil {
		t.Fatalf("SetUpstream: %v", err)
	}

	tags := getJSON(t, "http://"+cfg.ListenAddr+"/api/tags")
	models, _ := tags["models"].([]any)
	if len(models) != 2 {
		t.Fatalf("tags models = %v, want both cached GGUFs listed", models)
	}
	loaded := map[string]any{}
	for _, m := range models {
		e := m.(map[string]any)
		loaded[e["name"].(string)] = e["loaded"]
	}
	if loaded["qwen2.5-7b-instruct-q4_k_m"] != true {
		t.Errorf("qwen tag loaded = %v, want true (the chat slot has it loaded)", loaded["qwen2.5-7b-instruct-q4_k_m"])
	}
	if loaded["llama-3.2-3b"] != false {
		t.Errorf("llama tag loaded = %v, want false (cached but no slot serves it)", loaded["llama-3.2-3b"])
	}
}
