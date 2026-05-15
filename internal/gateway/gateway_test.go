// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cerid-ai/quenchforge/internal/config"
)

func newTestConfig(t *testing.T) config.Config {
	t.Helper()
	tmp := t.TempDir()
	return config.Config{
		ListenAddr:   "127.0.0.1:0",
		ModelsDir:    filepath.Join(tmp, "models"),
		LogDir:       filepath.Join(tmp, "logs"),
		PIDDir:       filepath.Join(tmp, "pids"),
		DefaultModel: "qwen2.5:7b-instruct-q4_k_m",
		MaxContext:   8192,
		MetalNCB:     2,
	}
}

// pickListenAddr binds to ":0" to grab a free port, closes the listener,
// and returns the address. Race-prone (the port could be re-taken) but
// good enough for tests on a quiet machine.
func pickListenAddr(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("pickListenAddr: %v", err)
	}
	addr := ln.Addr().String()
	_ = ln.Close()
	return addr
}

func newRunningGateway(t *testing.T, cfg config.Config) *Gateway {
	t.Helper()
	g := New(cfg)
	if err := g.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(func() { _ = g.Stop(time.Second) })
	// Wait for the port to accept connections — Start returns once the
	// listener is bound but Serve may not have looped yet.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", cfg.ListenAddr, 50*time.Millisecond)
		if err == nil {
			_ = conn.Close()
			return g
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("gateway did not become ready on %s", cfg.ListenAddr)
	return nil
}

func TestRootRespondsWithServiceInfo(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	g := newRunningGateway(t, cfg)
	g.SetVersion("9.9.9")

	resp, err := http.Get("http://" + cfg.ListenAddr + "/")
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("decode /: %v body=%q", err, body)
	}
	if got["service"] != "quenchforge" {
		t.Errorf("service = %v, want quenchforge", got["service"])
	}
	if got["version"] != "9.9.9" {
		t.Errorf("version = %v, want 9.9.9", got["version"])
	}
}

func TestHealthReturns200(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	newRunningGateway(t, cfg)
	resp, err := http.Get("http://" + cfg.ListenAddr + "/health")
	if err != nil {
		t.Fatalf("GET /health: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("/health: status=%d", resp.StatusCode)
	}
}

func TestTagsListsGGUFsInModelsDir(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	// Drop two fake gguf files in modelsDir before starting.
	if err := os.MkdirAll(cfg.ModelsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"qwen2.5-7b.gguf", "llama-3.2-3b-q4.gguf"} {
		if err := os.WriteFile(filepath.Join(cfg.ModelsDir, name), []byte("not a real model"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	newRunningGateway(t, cfg)

	resp, err := http.Get("http://" + cfg.ListenAddr + "/api/tags")
	if err != nil {
		t.Fatalf("GET /api/tags: %v", err)
	}
	defer resp.Body.Close()
	var got struct {
		Models []struct {
			Name string `json:"name"`
			Size int64  `json:"size"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatalf("decode tags: %v", err)
	}
	if len(got.Models) != 2 {
		t.Fatalf("models length = %d, want 2 (got %+v)", len(got.Models), got)
	}
	names := map[string]bool{}
	for _, m := range got.Models {
		names[m.Name] = true
		if m.Size == 0 {
			t.Errorf("model %q size=0", m.Name)
		}
	}
	if !names["qwen2.5-7b"] || !names["llama-3.2-3b-q4"] {
		t.Errorf("expected names not present: %v", names)
	}
}

func TestTagsEmptyWhenModelsDirMissing(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	// Don't create cfg.ModelsDir — handler should return {"models":[]}.
	newRunningGateway(t, cfg)

	resp, err := http.Get("http://" + cfg.ListenAddr + "/api/tags")
	if err != nil {
		t.Fatalf("GET /api/tags: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), `"models":`) {
		t.Errorf("body %q lacks models field", body)
	}
}

func TestChatProxyReturns503WithoutUpstream(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	newRunningGateway(t, cfg)

	resp, err := http.Post("http://"+cfg.ListenAddr+"/api/chat",
		"application/json", strings.NewReader(`{"model":"x","messages":[]}`))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "no chat slot configured") {
		t.Errorf("body %q does not mention chat slot", body)
	}
}

func TestEmbedProxyReturns503WithoutUpstream(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	newRunningGateway(t, cfg)

	resp, err := http.Post("http://"+cfg.ListenAddr+"/api/embeddings",
		"application/json", strings.NewReader(`{"model":"x","input":"hi"}`))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "no embed slot configured") {
		t.Errorf("body %q does not mention embed slot", body)
	}
}

func TestChatProxiesToUpstreamWhenSet(t *testing.T) {
	// /api/chat is now translated server-side: the gateway rewrites the
	// path to /v1/chat/completions and converts the Ollama-shape body
	// into an OpenAI-shape body before forwarding. The non-streaming
	// upstream response is then translated back into Ollama JSON shape
	// for the caller.
	var gotPath string
	var gotBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","model":"x","choices":[{"index":0,"message":{"role":"assistant","content":"hi back"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`))
	}))
	defer upstream.Close()

	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	g := newRunningGateway(t, cfg)
	if err := g.SetUpstream(KindChat, upstream.URL); err != nil {
		t.Fatalf("SetUpstream: %v", err)
	}

	resp, err := http.Post("http://"+cfg.ListenAddr+"/api/chat",
		"application/json", strings.NewReader(`{"model":"x","stream":false,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	// Path translation
	if gotPath != "/v1/chat/completions" {
		t.Errorf("upstream got path = %q, want /v1/chat/completions", gotPath)
	}
	// Body translation: Ollama wire {messages, options} → OpenAI wire
	// {messages, max_tokens, ...}. We can confirm the messages survived
	// the translation and the body is now OpenAI-shaped (no `options` field).
	if !strings.Contains(string(gotBody), `"content":"hi"`) {
		t.Errorf("upstream got body = %q, want it to carry the user message", gotBody)
	}
	if strings.Contains(string(gotBody), `"options"`) {
		t.Errorf("upstream got body = %q, must NOT carry Ollama-wire options key", gotBody)
	}
	// Response translation: upstream returned OpenAI {choices: [{message}]},
	// caller must see Ollama {message, done, done_reason}.
	respBody, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(respBody), `"message":`) ||
		!strings.Contains(string(respBody), `"done":true`) {
		t.Errorf("response body = %q, must be Ollama-shape with message+done", respBody)
	}
	if strings.Contains(string(respBody), `"choices":`) {
		t.Errorf("response body = %q, must NOT pass through OpenAI choices array", respBody)
	}
}

// TestSlotKindRouting confirms each Ollama-wire / OpenAI-wire route
// reaches the correct slot upstream. Bodies are translator-valid for
// the wire surface they hit:
//   - Ollama chat: {model, messages: [{role,content}]}
//   - Ollama embed: {model, prompt} or {model, input}
//   - OpenAI chat: {model, messages}
//   - OpenAI embed: {model, input}
func TestSlotKindRouting(t *testing.T) {
	var chatHits, embedHits int
	chatUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		chatHits++
		// Return a minimal OpenAI-shape body so the translator can
		// fold it back into Ollama wire when needed.
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","model":"x","choices":[{"index":0,"message":{"role":"assistant","content":""},"finish_reason":"stop"}],"usage":{}}`))
	}))
	defer chatUpstream.Close()
	embedUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		embedHits++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1]}],"model":"x","usage":{}}`))
	}))
	defer embedUpstream.Close()

	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	g := newRunningGateway(t, cfg)
	if err := g.SetUpstream(KindChat, chatUpstream.URL); err != nil {
		t.Fatalf("set chat: %v", err)
	}
	if err := g.SetUpstream(KindEmbed, embedUpstream.URL); err != nil {
		t.Fatalf("set embed: %v", err)
	}

	cases := []struct {
		method, path, body string
		wantChat           int
		wantEmbed          int
	}{
		{"POST", "/api/chat", `{"model":"x","stream":false,"messages":[{"role":"user","content":"hi"}]}`, 1, 0},
		{"POST", "/api/generate", `{"model":"x","stream":false,"prompt":"hi"}`, 1, 0},
		{"POST", "/v1/chat/completions", `{"model":"x","messages":[{"role":"user","content":"hi"}]}`, 1, 0},
		{"POST", "/api/embeddings", `{"model":"x","prompt":"hi"}`, 0, 1},
		{"POST", "/api/embed", `{"model":"x","input":"hi"}`, 0, 1},
		{"POST", "/v1/embeddings", `{"model":"x","input":"hi"}`, 0, 1},
	}
	for _, tc := range cases {
		chatHits, embedHits = 0, 0
		req, _ := http.NewRequest(tc.method, "http://"+cfg.ListenAddr+tc.path,
			strings.NewReader(tc.body))
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Errorf("%s %s: %v", tc.method, tc.path, err)
			continue
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
		if chatHits != tc.wantChat || embedHits != tc.wantEmbed {
			t.Errorf("%s %s: chat=%d embed=%d (want chat=%d embed=%d)",
				tc.method, tc.path, chatHits, embedHits, tc.wantChat, tc.wantEmbed)
		}
	}
}

// TestEmbedDispatchByModelName confirms the model-name routing rule:
// requests whose body's `model` field matches Config.CodeEmbedModel land
// on the KindCodeEmbed upstream; anything else lands on KindEmbed. Covers
// both the Ollama-translated path (/api/embeddings, /api/embed) and the
// OpenAI-native passthrough (/v1/embeddings).
func TestEmbedDispatchByModelName(t *testing.T) {
	var embedHits, codeEmbedHits int
	embedUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		embedHits++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1]}],"model":"text-emb","usage":{}}`))
	}))
	defer embedUpstream.Close()
	codeEmbedUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		codeEmbedHits++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.2]}],"model":"code-emb","usage":{}}`))
	}))
	defer codeEmbedUpstream.Close()

	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	cfg.CodeEmbedModel = "code-emb" // arm dispatch
	g := newRunningGateway(t, cfg)
	if err := g.SetUpstream(KindEmbed, embedUpstream.URL); err != nil {
		t.Fatalf("set embed: %v", err)
	}
	if err := g.SetUpstream(KindCodeEmbed, codeEmbedUpstream.URL); err != nil {
		t.Fatalf("set code-embed: %v", err)
	}

	cases := []struct {
		path, body         string
		wantEmbed, wantCode int
	}{
		{"/api/embeddings", `{"model":"text-emb","prompt":"hi"}`, 1, 0},
		{"/api/embeddings", `{"model":"code-emb","prompt":"def f(): pass"}`, 0, 1},
		{"/api/embed", `{"model":"text-emb","input":"hi"}`, 1, 0},
		{"/api/embed", `{"model":"code-emb","input":"def f(): pass"}`, 0, 1},
		{"/v1/embeddings", `{"model":"text-emb","input":"hi"}`, 1, 0},
		{"/v1/embeddings", `{"model":"code-emb","input":"def f(): pass"}`, 0, 1},
		// Unknown model name falls through to the regular embed slot
		// instead of 503-ing — keeps the legacy single-slot UX intact.
		{"/v1/embeddings", `{"model":"unknown","input":"hi"}`, 1, 0},
	}
	for _, tc := range cases {
		embedHits, codeEmbedHits = 0, 0
		resp, err := http.Post("http://"+cfg.ListenAddr+tc.path,
			"application/json", strings.NewReader(tc.body))
		if err != nil {
			t.Errorf("%s body=%q: %v", tc.path, tc.body, err)
			continue
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
		if embedHits != tc.wantEmbed || codeEmbedHits != tc.wantCode {
			t.Errorf("%s body=%q: embed=%d code=%d (want embed=%d code=%d)",
				tc.path, tc.body, embedHits, codeEmbedHits, tc.wantEmbed, tc.wantCode)
		}
	}
}

// TestPullReturns501 — Quenchforge's MVP doesn't pull from a registry.
func TestPullReturns501(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	newRunningGateway(t, cfg)
	resp, err := http.Post("http://"+cfg.ListenAddr+"/api/pull",
		"application/json", strings.NewReader(`{"name":"llama3:latest"}`))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotImplemented {
		t.Errorf("status = %d, want 501", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "migrate-from-ollama") {
		t.Errorf("body %q does not point at migrate-from-ollama", body)
	}
}

// TestRootReportsKnownSlots — / surfaces all known slot kinds with
// their configured status.
func TestRootReportsKnownSlots(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	g := newRunningGateway(t, cfg)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer upstream.Close()
	if err := g.SetUpstream(KindChat, upstream.URL); err != nil {
		t.Fatalf("set chat: %v", err)
	}

	resp, err := http.Get("http://" + cfg.ListenAddr + "/")
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	for _, kind := range []string{"chat", "embed", "rerank"} {
		if !strings.Contains(string(body), `"`+kind+`"`) {
			t.Errorf("/ body missing %q in slots: %s", kind, body)
		}
	}
	// Chat should report configured=true; embed/rerank should report false.
	if !strings.Contains(string(body), `"chat":{"configured":true`) {
		t.Errorf("/ body should show chat as configured=true: %s", body)
	}
}

func TestPortConflictDetection(t *testing.T) {
	addr := pickListenAddr(t)
	// Hold the port with a stand-in listener.
	hold, err := net.Listen("tcp", addr)
	if err != nil {
		t.Fatalf("hold: %v", err)
	}
	defer hold.Close()

	cfg := newTestConfig(t)
	cfg.ListenAddr = addr
	g := New(cfg)
	err = g.Start(context.Background())
	if err == nil {
		_ = g.Stop(time.Second)
		t.Fatal("Start: expected ErrAddrInUse, got nil")
	}
	if !errors.Is(err, ErrAddrInUse) {
		t.Errorf("Start: error = %v, want ErrAddrInUse", err)
	}
}

func TestEnumerateModelsSkipsNonGGUF(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"a.gguf", "b.GGUF", "readme.md", "metadata.json"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	models, err := EnumerateModels(dir)
	if err != nil {
		t.Fatalf("EnumerateModels: %v", err)
	}
	if len(models) != 2 {
		t.Errorf("got %d models, want 2 (got %+v)", len(models), models)
	}
}

// TestEnumerateModelsHandlesNestedDirs confirms walk semantics (a model can
// live in modelsDir/qwen2.5/7b-q4.gguf etc).
func TestEnumerateModelsHandlesNestedDirs(t *testing.T) {
	dir := t.TempDir()
	nested := filepath.Join(dir, "qwen2.5")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(nested, "7b-q4.gguf"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	models, err := EnumerateModels(dir)
	if err != nil {
		t.Fatalf("EnumerateModels: %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("got %d, want 1", len(models))
	}
	want := filepath.Join("qwen2.5", "7b-q4")
	if models[0].Name != want {
		t.Errorf("nested model name = %q, want %q", models[0].Name, want)
	}
}

// TestSetUpstreamRejectsBadURL keeps the error path tested.
func TestSetUpstreamRejectsBadURL(t *testing.T) {
	g := New(newTestConfig(t))
	if err := g.SetUpstream(KindChat, "://not a url"); err == nil {
		t.Error("SetUpstream: nil error on bad URL")
	}
	if err := g.SetUpstream(KindChat, ""); err != nil {
		t.Errorf("SetUpstream(chat, ''): %v, want nil clear", err)
	}
	if err := g.SetUpstream(KindEmbed, ""); err != nil {
		t.Errorf("SetUpstream(embed, ''): %v, want nil clear", err)
	}
}

// Smoke: the test binary doesn't need to import fmt for the test to compile,
// but the rest of the package does. Keep an unused-detector at bay.
var _ = fmt.Sprint
