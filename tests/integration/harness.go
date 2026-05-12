// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

//go:build amd_gpu || apple_silicon

package integration

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// llamaBin returns the path to the patched llama-server binary the test
// will exercise. Resolution order:
//
//  1. $QUENCHFORGE_LLAMA_BIN if non-empty.
//  2. ./llama.cpp/build-<goarch>/bin/llama-server relative to the repo
//     root (the standard `scripts/build-llama.sh` output).
//  3. ./llama.cpp/build/bin/llama-server (fallback).
//
// Fails the test if none exist or aren't executable.
func llamaBin(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("QUENCHFORGE_LLAMA_BIN"); p != "" {
		if isExec(p) {
			return p
		}
		t.Fatalf("QUENCHFORGE_LLAMA_BIN=%q is not an executable file", p)
	}
	root := repoRoot(t)
	candidates := []string{
		filepath.Join(root, "llama.cpp", "build-x86_64", "bin", "llama-server"),
		filepath.Join(root, "llama.cpp", "build-arm64", "bin", "llama-server"),
		filepath.Join(root, "llama.cpp", "build-universal", "bin", "llama-server"),
		filepath.Join(root, "llama.cpp", "build", "bin", "llama-server"),
	}
	for _, p := range candidates {
		if isExec(p) {
			return p
		}
	}
	t.Fatalf("no llama-server found. Build it with: bash scripts/build-llama.sh\n"+
		"  tried: %v", candidates)
	return ""
}

// testModel returns a GGUF model path to load. Resolution order:
//
//  1. $TEST_MODEL_PATH if non-empty.
//  2. The Ollama llama3.2:3b blob (the model the patch was validated
//     against). Resolved by reading the manifest.
//  3. Skip the test with a helpful message — we don't want CI to fail
//     just because the contributor hasn't downloaded a model yet.
func testModel(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("TEST_MODEL_PATH"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
		t.Fatalf("TEST_MODEL_PATH=%q does not exist", p)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skipf("cannot determine home dir: %v", err)
	}
	// Ollama llama3.2:3b blob hash — stable across the Ollama registry.
	blob := filepath.Join(home, ".ollama", "models", "blobs",
		"sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff")
	if _, err := os.Stat(blob); err == nil {
		return blob
	}
	t.Skipf("no test model available — set TEST_MODEL_PATH or `ollama pull llama3.2:3b`")
	return ""
}

// startLlamaServer spawns the patched binary on a random localhost port
// and returns the base URL once /health responds. The cleanup function
// runs SIGKILL with a 5s grace.
//
// The harness is intentionally minimal — we don't use the quenchforge
// supervisor here because we want to exercise the binary in isolation,
// failing fast if the patch itself broke. The supervisor integration
// is covered by a separate test.
func startLlamaServer(t *testing.T, modelPath string) (baseURL string, cleanup func()) {
	t.Helper()

	port := freePort(t)
	bin := llamaBin(t)

	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(ctx, bin,
		"--model", modelPath,
		"--host", "127.0.0.1",
		"--port", fmt.Sprintf("%d", port),
		"--ctx-size", "2048",
	)
	// Surface stderr to the test log on failure but don't spam stdout.
	logBuf := &lineCapture{name: t.Name()}
	cmd.Stdout = logBuf
	cmd.Stderr = logBuf
	if err := cmd.Start(); err != nil {
		cancel()
		t.Fatalf("start llama-server: %v", err)
	}

	cleanup = func() {
		cancel()
		_ = cmd.Wait()
		// Flush captured output to the test log only on test failure.
		if t.Failed() {
			t.Logf("llama-server output (last %d lines):\n%s", logBuf.count, logBuf.tail())
		}
	}

	baseURL = fmt.Sprintf("http://127.0.0.1:%d", port)
	if !waitForHealth(t, baseURL, 90*time.Second) {
		cleanup()
		t.Fatalf("llama-server at %s never reported /health within 90s", baseURL)
	}
	return baseURL, cleanup
}

// waitForHealth polls /health until 200 or until the deadline expires.
// llama-server's /health returns 503 with `{"error":...,"type":"unavailable_error"}`
// while the model is still loading; we treat that as "not ready yet".
func waitForHealth(t *testing.T, baseURL string, timeout time.Duration) bool {
	t.Helper()
	client := &http.Client{Timeout: 2 * time.Second}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		resp, err := client.Get(baseURL + "/health")
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == 200 {
				return true
			}
		}
		time.Sleep(1 * time.Second)
	}
	return false
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func isExec(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir() && st.Mode()&0o111 != 0
}

func repoRoot(t *testing.T) string {
	t.Helper()
	// We're in tests/integration; the repo root is two levels up.
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	return filepath.Join(wd, "..", "..")
}

func freePort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("free port: %v", err)
	}
	addr := ln.Addr().(*net.TCPAddr)
	port := addr.Port
	if err := ln.Close(); err != nil {
		t.Logf("close listener: %v", err)
	}
	return port
}

// lineCapture is a tiny io.Writer that retains the last 64 KB of output
// for failure-time diagnostics. We don't need a ring buffer — these tests
// run for seconds at most.
type lineCapture struct {
	name  string
	buf   []byte
	count int
}

func (l *lineCapture) Write(p []byte) (int, error) {
	const max = 64 * 1024
	l.buf = append(l.buf, p...)
	if len(l.buf) > max {
		l.buf = l.buf[len(l.buf)-max:]
	}
	// Cheap line count for the failure log header.
	for _, b := range p {
		if b == '\n' {
			l.count++
		}
	}
	return len(p), nil
}

func (l *lineCapture) tail() string {
	if len(l.buf) == 0 {
		return "(empty)"
	}
	return string(l.buf)
}
