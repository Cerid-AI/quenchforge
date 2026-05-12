// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

//go:build amd_gpu || apple_silicon

package integration

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestWhisperTranscribesJFKSample is the structural regression guard for
// the whisper.cpp slot. Exercises the FULL path:
//
//	whisper-server (patched whisper.cpp, --no-gpu on AMD Mac by default)
//	   ↑ HTTP POST /inference (multipart form-data)
//	   ↑ same shape OpenAI uses for /v1/audio/transcriptions
//
// Asserts the model produces text containing key fragments of the JFK
// quote, since the audio file is unambiguous and any decent whisper
// model gets it right. The point is to catch a future rebase that
// silently breaks audio decoding or the whisper-server HTTP surface.
//
// Falls back to whisper-server's own /inference endpoint rather than
// going through the gateway — keeps this test focused on the binary
// itself, not the proxy layer.
func TestWhisperTranscribesJFKSample(t *testing.T) {
	bin := whisperBin(t)
	model := whisperModel(t)
	audio := whisperSample(t)

	port := freePort(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cmd := exec.CommandContext(ctx, bin,
		"--model", model,
		"--host", "127.0.0.1",
		"--port", fmt.Sprintf("%d", port),
		"--threads", "4",
		"--no-gpu", // whisper.cpp's Metal path is still buggy on AMD Mac
		//             beyond what patches/whisper.cpp/0001 fixes
	)
	logBuf := &lineCapture{name: t.Name()}
	cmd.Stdout = logBuf
	cmd.Stderr = logBuf
	if err := cmd.Start(); err != nil {
		t.Fatalf("start whisper-server: %v", err)
	}
	defer func() {
		cancel()
		_ = cmd.Wait()
		if t.Failed() {
			t.Logf("whisper-server output (last %d lines):\n%s", logBuf.count, logBuf.tail())
		}
	}()

	// Whisper loads the model from disk synchronously on first request,
	// but the HTTP listener binds immediately. Poll for connectivity.
	baseURL := fmt.Sprintf("http://127.0.0.1:%d", port)
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		if conn, err := tryConnect(baseURL); err == nil {
			conn()
			break
		}
		time.Sleep(500 * time.Millisecond)
	}

	text, err := postWhisperInference(baseURL, audio, 90*time.Second)
	if err != nil {
		t.Fatalf("whisper /inference call failed: %v", err)
	}
	t.Logf("transcription: %q", text)

	lower := strings.ToLower(text)
	// JFK's "ask not what your country" line. We assert on substrings,
	// not exact match, because whisper-tiny.en may add/drop punctuation
	// or capitalize differently. The point: it transcribed something
	// recognizably JFK, not garbage.
	for _, want := range []string{
		"ask not what your country",
		"can do for you",
	} {
		if !strings.Contains(lower, want) {
			t.Errorf("transcription missing expected substring %q\n  got: %q", want, text)
		}
	}

	// Garbage-signature check: if the patch ever regresses and we get
	// the garbage-token output we observed on Metal pre-patch,
	// uniqueness will be ~0 and the test catches it.
	uniqueRatio := uniqueWordRatio(text)
	if uniqueRatio < 0.30 {
		t.Errorf("unique-word ratio %.2f below 0.30 floor — suspect garbage output\n  got: %q",
			uniqueRatio, text)
	}
	t.Logf("unique-word ratio: %.2f", uniqueRatio)
}

// ---------------------------------------------------------------------------
// helpers specific to the whisper test
// ---------------------------------------------------------------------------

// whisperBin returns the whisper-server path, falling back through
// scripts/build-whisper.sh output paths. Skips the test when missing.
func whisperBin(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("QUENCHFORGE_WHISPER_BIN"); p != "" {
		if isExec(p) {
			return p
		}
		t.Fatalf("QUENCHFORGE_WHISPER_BIN=%q is not an executable", p)
	}
	root := repoRoot(t)
	for _, p := range []string{
		filepath.Join(root, "whisper.cpp", "build-x86_64", "bin", "whisper-server"),
		filepath.Join(root, "whisper.cpp", "build-arm64", "bin", "whisper-server"),
		filepath.Join(root, "whisper.cpp", "build-universal", "bin", "whisper-server"),
		filepath.Join(root, "whisper.cpp", "build", "bin", "whisper-server"),
	} {
		if isExec(p) {
			return p
		}
	}
	t.Skip("whisper-server not built — run: bash scripts/build-whisper.sh")
	return ""
}

// whisperModel returns a ggml whisper model path. Looks for the tiny.en
// model the build pipeline downloads by default.
func whisperModel(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("WHISPER_MODEL_PATH"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
		t.Fatalf("WHISPER_MODEL_PATH=%q does not exist", p)
	}
	candidate := filepath.Join(repoRoot(t), "whisper.cpp", "models", "ggml-tiny.en.bin")
	if _, err := os.Stat(candidate); err == nil {
		return candidate
	}
	t.Skip("no whisper model — run: bash whisper.cpp/models/download-ggml-model.sh tiny.en")
	return ""
}

// whisperSample returns whisper.cpp's bundled JFK audio sample.
func whisperSample(t *testing.T) string {
	t.Helper()
	p := filepath.Join(repoRoot(t), "whisper.cpp", "samples", "jfk.wav")
	if _, err := os.Stat(p); err != nil {
		t.Skipf("no jfk.wav sample at %s: %v", p, err)
	}
	return p
}

// tryConnect dials baseURL/health (or just the port if /health 404s) and
// returns a closer. Used to detect "listener bound" without paying the
// full /health latency cost.
func tryConnect(baseURL string) (func(), error) {
	resp, err := http.Get(baseURL + "/")
	if err != nil {
		return nil, err
	}
	return func() { _ = resp.Body.Close() }, nil
}

// postWhisperInference uploads audio to /inference using the same
// multipart/form-data shape whisper-server expects (and which OpenAI's
// /v1/audio/transcriptions also uses). Returns the JSON {"text":...} value.
func postWhisperInference(baseURL, audioPath string, timeout time.Duration) (string, error) {
	body := &bytes.Buffer{}
	mw := multipart.NewWriter(body)

	// file field
	f, err := os.Open(audioPath)
	if err != nil {
		return "", err
	}
	defer f.Close()
	fw, err := mw.CreateFormFile("file", filepath.Base(audioPath))
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(fw, f); err != nil {
		return "", err
	}
	_ = mw.WriteField("response_format", "json")
	_ = mw.Close()

	req, err := http.NewRequest(http.MethodPost, baseURL+"/inference", body)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())

	client := &http.Client{Timeout: timeout}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("status %d: %s", resp.StatusCode, respBody)
	}
	// Match the JSON shape: {"text":"..."}.
	// Avoid pulling in encoding/json type machinery for a single field.
	const key = `"text":"`
	i := bytes.Index(respBody, []byte(key))
	if i < 0 {
		return "", fmt.Errorf("response missing text field: %s", respBody)
	}
	rest := respBody[i+len(key):]
	end := bytes.IndexByte(rest, '"')
	for end > 0 && rest[end-1] == '\\' {
		// skip escaped quotes
		next := bytes.IndexByte(rest[end+1:], '"')
		if next < 0 {
			break
		}
		end = end + 1 + next
	}
	if end < 0 {
		return "", fmt.Errorf("unterminated text value: %s", respBody)
	}
	// Best-effort unescape of \n into literal newlines for the assertion.
	out := strings.ReplaceAll(string(rest[:end]), `\n`, "\n")
	return out, nil
}
