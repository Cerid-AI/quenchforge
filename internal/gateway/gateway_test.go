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
	if !strings.Contains(string(body), "upstream not configured") {
		t.Errorf("body %q does not mention upstream", body)
	}
}

func TestChatProxiesToUpstreamWhenSet(t *testing.T) {
	// Stand up a fake upstream that records what it received.
	var gotPath string
	var gotBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"done":true,"model":"x"}` + "\n"))
	}))
	defer upstream.Close()

	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	g := newRunningGateway(t, cfg)
	if err := g.SetUpstream(upstream.URL); err != nil {
		t.Fatalf("SetUpstream: %v", err)
	}

	resp, err := http.Post("http://"+cfg.ListenAddr+"/api/chat",
		"application/json", strings.NewReader(`{"model":"x","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if gotPath != "/api/chat" {
		t.Errorf("upstream got path = %q, want /api/chat", gotPath)
	}
	if !strings.Contains(string(gotBody), "hi") {
		t.Errorf("upstream got body = %q, want 'hi'", gotBody)
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
	if err := g.SetUpstream("://not a url"); err == nil {
		t.Error("SetUpstream: nil error on bad URL")
	}
	if err := g.SetUpstream(""); err != nil {
		t.Errorf("SetUpstream(''): %v, want nil clear", err)
	}
}

// Smoke: the test binary doesn't need to import fmt for the test to compile,
// but the rest of the package does. Keep an unused-detector at bay.
var _ = fmt.Sprint
