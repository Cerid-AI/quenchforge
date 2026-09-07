// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package gateway

import (
	"bytes"
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

// TestTagsReportsLoadedForServedModel — a cached .gguf whose trimmed name
// matches a configured slot with a registered upstream reports loaded=true;
// a cached file with no matching slot reports loaded=false; every entry
// carries the field.
func TestTagsReportsLoadedForServedModel(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	cfg.DefaultModel = "qwen2.5-7b.gguf"
	if err := os.MkdirAll(cfg.ModelsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"qwen2.5-7b.gguf", "llama-3.2-3b-q4.gguf"} {
		if err := os.WriteFile(filepath.Join(cfg.ModelsDir, name), []byte("not a real model"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	g := newRunningGateway(t, cfg)
	chatUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer chatUpstream.Close()
	if err := g.SetUpstream(KindChat, chatUpstream.URL); err != nil {
		t.Fatalf("set chat: %v", err)
	}

	resp, err := http.Get("http://" + cfg.ListenAddr + "/api/tags")
	if err != nil {
		t.Fatalf("GET /api/tags: %v", err)
	}
	defer resp.Body.Close()
	var got struct {
		Models []map[string]any `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatalf("decode tags: %v", err)
	}
	if len(got.Models) != 2 {
		t.Fatalf("models length = %d, want 2 (got %+v)", len(got.Models), got)
	}
	for _, m := range got.Models {
		loaded, present := m["loaded"]
		if !present {
			t.Fatalf("model %v missing loaded field", m)
		}
		want := m["name"] == "qwen2.5-7b"
		if loaded != want {
			t.Errorf("model %v loaded = %v, want %v", m["name"], loaded, want)
		}
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
		path, body          string
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

// TestChatDispatchByModelName confirms the model-name routing rule for
// chat: requests whose body's `model` field names Config.BackgroundModel
// (as GGUF basename, optionally with .gguf or an Ollama-style :tag suffix)
// land on the KindBackground upstream; anything else lands on KindChat.
// Unlike TestEmbedDispatchByModelName's fallback-on-miss, an explicit
// background-model request gets a 503 — not a silent redirect to chat —
// when no background upstream is registered. Covers both the
// Ollama-translated path (/api/chat) and the OpenAI-native passthrough
// (/v1/chat/completions).
func TestChatDispatchByModelName(t *testing.T) {
	const openAIChatBody = `{"id":"x","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{}}`
	// A tiny SSE stream shaped like llama-server's streaming output. Used to
	// confirm /v1/chat/completions forwards a streaming upstream response to
	// the client byte-for-byte (no buffering/re-encoding).
	const sseBody = "data: {\"id\":\"x\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n"

	var (
		chatHits, backgroundHits         int
		lastChatBody, lastBackgroundBody []byte
	)
	// Each stub records the exact bytes it received and, for a
	// `"stream":true` request, echoes back an SSE body instead of a plain
	// JSON completion — lets the same stub cover both the non-streaming
	// hit-counting cases and the streaming-passthrough case below.
	stub := func(hits *int, lastBody *[]byte) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			*hits++
			b, _ := io.ReadAll(r.Body)
			*lastBody = b
			if bytes.Contains(b, []byte(`"stream":true`)) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, sseBody)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(openAIChatBody))
		}
	}
	chatUpstream := httptest.NewServer(stub(&chatHits, &lastChatBody))
	defer chatUpstream.Close()
	backgroundUpstream := httptest.NewServer(stub(&backgroundHits, &lastBackgroundBody))
	defer backgroundUpstream.Close()

	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	cfg.BackgroundModel = "bg-model" // arm dispatch
	g := newRunningGateway(t, cfg)
	if err := g.SetUpstream(KindChat, chatUpstream.URL); err != nil {
		t.Fatalf("set chat: %v", err)
	}
	if err := g.SetUpstream(KindBackground, backgroundUpstream.URL); err != nil {
		t.Fatalf("set background: %v", err)
	}

	reqBody := func(model string) string {
		return fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hi"}],"stream":false}`, model)
	}

	cases := []struct {
		path                     string
		body                     string
		wantChat, wantBackground int
		// wantEcho, when true, additionally asserts the upstream received
		// exactly `body` — only holds for /v1/chat/completions, which is a
		// raw byte passthrough; /api/chat re-encodes into the OpenAI shape.
		wantEcho bool
	}{
		{"/api/chat", reqBody("main-chat"), 1, 0, false},
		{"/api/chat", reqBody("bg-model"), 0, 1, false},
		// Name-normalization variants: .gguf suffix and an Ollama-style
		// :tag suffix must both still match.
		{"/api/chat", reqBody("bg-model.gguf"), 0, 1, false},
		{"/api/chat", reqBody("bg-model:latest"), 0, 1, false},
		{"/v1/chat/completions", reqBody("main-chat"), 1, 0, true},
		{"/v1/chat/completions", reqBody("bg-model"), 0, 1, true},
	}
	for _, tc := range cases {
		chatHits, backgroundHits = 0, 0
		lastChatBody, lastBackgroundBody = nil, nil
		resp, err := http.Post("http://"+cfg.ListenAddr+tc.path,
			"application/json", strings.NewReader(tc.body))
		if err != nil {
			t.Errorf("%s body=%q: %v", tc.path, tc.body, err)
			continue
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
		if chatHits != tc.wantChat || backgroundHits != tc.wantBackground {
			t.Errorf("%s body=%q: chat=%d background=%d (want chat=%d background=%d)",
				tc.path, tc.body, chatHits, backgroundHits, tc.wantChat, tc.wantBackground)
		}
		if tc.wantEcho {
			got := lastChatBody
			if tc.wantBackground == 1 {
				got = lastBackgroundBody
			}
			if string(got) != tc.body {
				t.Errorf("%s body=%q: upstream received %q, want exact echo",
					tc.path, tc.body, got)
			}
		}
	}

	// A >1 MB body on the OpenAI-native passthrough must reach the
	// background upstream byte-for-byte — the peek-and-reattach path in
	// handleOpenAIChat must not truncate or otherwise mutate it.
	chatHits, backgroundHits = 0, 0
	lastChatBody, lastBackgroundBody = nil, nil
	bigBody := fmt.Sprintf(`{"model":"bg-model","messages":[{"role":"user","content":%q}],"stream":false}`,
		strings.Repeat("A", 1<<20+4096)) // > 1 MB
	resp, err := http.Post("http://"+cfg.ListenAddr+"/v1/chat/completions",
		"application/json", strings.NewReader(bigBody))
	if err != nil {
		t.Fatalf("large body POST: %v", err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
	if backgroundHits != 1 {
		t.Fatalf("large body: background hits = %d, want 1", backgroundHits)
	}
	if string(lastBackgroundBody) != bigBody {
		t.Errorf("large body (%d bytes): upstream received %d bytes, want exact echo",
			len(bigBody), len(lastBackgroundBody))
	}

	// A `"stream":true` request's SSE response must reach the client
	// unchanged — no buffering, no re-encoding.
	streamBody := `{"model":"bg-model","messages":[{"role":"user","content":"hi"}],"stream":true}`
	resp, err = http.Post("http://"+cfg.ListenAddr+"/v1/chat/completions",
		"application/json", strings.NewReader(streamBody))
	if err != nil {
		t.Fatalf("streaming POST: %v", err)
	}
	got, err := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if err != nil {
		t.Fatalf("read streaming response: %v", err)
	}
	if string(got) != sseBody {
		t.Errorf("streaming response = %q, want %q (unchanged passthrough)", got, sseBody)
	}

	// No background upstream registered: an explicit background-model
	// request must 503, not silently fall back to the chat slot.
	if err := g.SetUpstream(KindBackground, ""); err != nil {
		t.Fatalf("clear background: %v", err)
	}
	chatHits, backgroundHits = 0, 0
	for _, path := range []string{"/api/chat", "/v1/chat/completions"} {
		resp, err := http.Post("http://"+cfg.ListenAddr+path,
			"application/json", strings.NewReader(reqBody("bg-model")))
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		if resp.StatusCode != http.StatusServiceUnavailable {
			t.Errorf("%s background w/o upstream: status = %d, want 503", path, resp.StatusCode)
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
	}
	if chatHits != 0 || backgroundHits != 0 {
		t.Errorf("background w/o upstream should not touch either upstream: chat=%d background=%d",
			chatHits, backgroundHits)
	}
}

// TestModelMatchesBackground exercises the model-name matching rules
// resolveChatKind relies on (see modelMatchesBackground): exact match,
// both sides sans a ".gguf" suffix, and a request's Ollama-style ":tag"
// suffix stripped as a fallback — but only when the configured background
// model itself has no colon (bg is never canonicalized/stripped).
func TestModelMatchesBackground(t *testing.T) {
	cases := []struct {
		name, model, bg string
		want            bool
	}{
		{"exact bare match", "qwen2.5-3b-instruct-q4_k_m", "qwen2.5-3b-instruct-q4_k_m", true},
		{"request .gguf, bg bare", "qwen2.5-3b-instruct-q4_k_m.gguf", "qwen2.5-3b-instruct-q4_k_m", true},
		{"request :tag, bg bare", "qwen2.5-3b-instruct-q4_k_m:latest", "qwen2.5-3b-instruct-q4_k_m", true},
		{"request bare, bg .gguf", "qwen2.5-3b-instruct-q4_k_m", "qwen2.5-3b-instruct-q4_k_m.gguf", true},
		{"request .gguf, bg .gguf (exact)", "qwen2.5-3b-instruct-q4_k_m.gguf", "qwen2.5-3b-instruct-q4_k_m.gguf", true},
		{"request :tag, bg .gguf", "qwen2.5-3b-instruct-q4_k_m:latest", "qwen2.5-3b-instruct-q4_k_m.gguf", true},
		{"colon-named bg matches exact colon request", "qwen2.5:3b", "qwen2.5:3b", true},
		{"colon-named bg: stripped request must NOT match", "qwen2.5", "qwen2.5:3b", false},
		{"unrelated name", "llama3", "qwen2.5-3b-instruct-q4_k_m", false},
		{"empty model", "", "qwen2.5-3b-instruct-q4_k_m", false},
		{"empty background", "qwen2.5-3b-instruct-q4_k_m", "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := modelMatchesBackground(tc.model, tc.bg); got != tc.want {
				t.Errorf("modelMatchesBackground(%q, %q) = %v, want %v", tc.model, tc.bg, got, tc.want)
			}
		})
	}
}

// TestChatOpenAIPassthroughUnboundedWhenBackgroundUnset confirms
// handleOpenAIChat is a byte-for-byte, no-read passthrough to the
// pre-dispatch behavior when QUENCHFORGE_BACKGROUND_MODEL is unset: a
// body far larger than the 8 MB Ollama-translation cap (and larger than
// maxChatPeekBodyBytes would even need to be) must still reach the chat
// upstream unchanged, because this path never buffers it at all.
func TestChatOpenAIPassthroughUnboundedWhenBackgroundUnset(t *testing.T) {
	var received int
	chatUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n, _ := io.Copy(io.Discard, r.Body)
		received = int(n)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","model":"m","choices":[],"usage":{}}`))
	}))
	defer chatUpstream.Close()

	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	// BackgroundModel intentionally left unset.
	g := newRunningGateway(t, cfg)
	if err := g.SetUpstream(KindChat, chatUpstream.URL); err != nil {
		t.Fatalf("set chat: %v", err)
	}

	bigBody := fmt.Sprintf(`{"model":"chat","messages":[{"role":"user","content":%q}]}`,
		strings.Repeat("A", 9*1024*1024)) // > 8 MB (maxRequestBodyBytes)
	resp, err := http.Post("http://"+cfg.ListenAddr+"/v1/chat/completions",
		"application/json", strings.NewReader(bigBody))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
	if received != len(bigBody) {
		t.Errorf("chat upstream received %d bytes, want %d (unbounded passthrough)", received, len(bigBody))
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

// TestRootReportsSlotModels — a configured slot's entry names its model
// (with any .gguf suffix trimmed); an unconfigured slot reports none.
func TestRootReportsSlotModels(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	cfg.DefaultModel = "qwen2.5-7b-instruct-q4_k_m.gguf"
	cfg.BackgroundModel = "qwen2.5-3b-instruct-q4_k_m"
	g := newRunningGateway(t, cfg)

	chatUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer chatUpstream.Close()
	backgroundUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer backgroundUpstream.Close()
	if err := g.SetUpstream(KindChat, chatUpstream.URL); err != nil {
		t.Fatalf("set chat: %v", err)
	}
	if err := g.SetUpstream(KindBackground, backgroundUpstream.URL); err != nil {
		t.Fatalf("set background: %v", err)
	}

	resp, err := http.Get("http://" + cfg.ListenAddr + "/")
	if err != nil {
		t.Fatalf("GET /: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	var got struct {
		Slots map[string]struct {
			Configured bool   `json:"configured"`
			Model      string `json:"model"`
		} `json:"slots"`
	}
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("decode /: %v body=%q", err, body)
	}
	if got.Slots["chat"].Model != "qwen2.5-7b-instruct-q4_k_m" {
		t.Errorf("chat model = %q, want .gguf suffix trimmed", got.Slots["chat"].Model)
	}
	if got.Slots["background"].Model != cfg.BackgroundModel {
		t.Errorf("background model = %q, want %q", got.Slots["background"].Model, cfg.BackgroundModel)
	}
	if got.Slots["embed"].Model != "" {
		t.Errorf("embed model = %q, want empty (unconfigured)", got.Slots["embed"].Model)
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
