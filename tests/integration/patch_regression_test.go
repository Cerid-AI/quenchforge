// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

//go:build amd_gpu || apple_silicon

package integration

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

// TestPatchProducesCoherentOutput is the structural regression guard for
// patches/0001-metal-correctness-on-non-apple-silicon.patch.
//
// Pre-patch, this same prompt against this same model on AMD Vega II
// produced an infinite stream of "Key key key key key... ATRIX ATRIX
// ATRIX...". We assert post-patch output:
//
//   - is non-empty
//   - is not the garbage-token signature (any of the known patterns)
//   - contains at least one whitespace character (real sentences have spaces)
//   - has a reasonable unique-token ratio (garbage is highly repetitive)
//
// The point isn't to validate model quality; it's to catch the day a
// rebase silently re-enables the buggy Metal kernels and the model starts
// emitting garbage again. If you change this test, the new assertions
// must still distinguish coherent text from the documented garbage
// signature in patches/README.md.
func TestPatchProducesCoherentOutput(t *testing.T) {
	model := testModel(t)
	baseURL, cleanup := startLlamaServer(t, model)
	defer cleanup()

	// Use a prompt + temperature that historically reproduced the bug.
	// temperature=0 makes the output deterministic so the regression test
	// is reproducible across runs.
	req := chatRequest{
		Model:       "test",
		Temperature: 0.0,
		MaxTokens:   80,
		Messages: []chatMessage{
			{Role: "user", Content: "In one sentence, what is the capital of France?"},
		},
	}
	resp, err := chatCompletion(baseURL, req, 90*time.Second)
	if err != nil {
		t.Fatalf("chat call failed: %v", err)
	}

	content := strings.TrimSpace(resp.Choices[0].Message.Content)
	t.Logf("model output: %q", content)
	t.Logf("usage: prompt=%d completion=%d total=%d",
		resp.Usage.PromptTokens, resp.Usage.CompletionTokens, resp.Usage.TotalTokens)
	if tm := resp.Timings; tm != nil {
		// No thresholds — performance varies wildly across hardware.
		// Logging the actuals so the test report doubles as a perf history.
		if tm.PromptPerTokenMs > 0 {
			t.Logf("prompt rate:  %.1f tok/s", 1000.0/tm.PromptPerTokenMs)
		}
		if tm.PredictedPerTokenMs > 0 {
			t.Logf("predict rate: %.1f tok/s", 1000.0/tm.PredictedPerTokenMs)
		}
	}

	if content == "" {
		t.Fatal("model returned empty content — slot didn't generate anything")
	}

	// --- Garbage-signature checks ---

	// 1. The "Key key key" pattern observed pre-patch on Vega II.
	if hasRepeatedWord(content, "key", 5) {
		t.Fatalf("output contains 5+ repeats of 'key' — patch likely regressed\n"+
			"  output: %q", content)
	}
	// 2. The "compat compat" or "ATRIX ATRIX" patterns observed in HTTP responses.
	for _, garbage := range []string{"compat", "ATRIX", "cast cast cast"} {
		if hasRepeatedWord(content, garbage, 5) {
			t.Fatalf("output contains 5+ repeats of %q — patch likely regressed\n"+
				"  output: %q", garbage, content)
		}
	}

	// --- Generic coherence checks ---

	// 3. Coherent text has whitespace.
	if !strings.ContainsAny(content, " \t\n") {
		t.Errorf("output has no whitespace — unusual for natural language\n"+
			"  output: %q", content)
	}

	// 4. Unique-token ratio: garbage outputs reuse the same word over and
	// over. Coherent text has at least 30% unique words.
	uniqueRatio := uniqueWordRatio(content)
	if uniqueRatio < 0.30 {
		t.Errorf("unique-word ratio %.2f below 0.30 floor — suspect garbage output\n"+
			"  output: %q", uniqueRatio, content)
	}
	t.Logf("unique-word ratio: %.2f", uniqueRatio)

	// 5. The model should mention "Paris" given the prompt. This is the
	// only model-quality assertion (and only because Paris being the
	// capital of France is in every model's training data; a model that
	// gets this wrong is broken regardless of patches).
	if !strings.Contains(strings.ToLower(content), "paris") {
		t.Logf("WARNING: output doesn't mention 'Paris' — this is a soft signal " +
			"that something is off, but doesn't fail the test")
	}
}

// ---------------------------------------------------------------------------
// JSON request/response types matching llama-server's OpenAI surface
// ---------------------------------------------------------------------------

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature float64       `json:"temperature"`
}

type chatResponse struct {
	Choices []struct {
		Message chatMessage `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
	Timings *struct {
		PromptPerTokenMs    float64 `json:"prompt_per_token_ms"`
		PredictedPerTokenMs float64 `json:"predicted_per_token_ms"`
	} `json:"timings,omitempty"`
}

func chatCompletion(baseURL string, req chatRequest, timeout time.Duration) (*chatResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequest(http.MethodPost,
		baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: timeout}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status %d: %s", resp.StatusCode, respBytes)
	}
	var out chatResponse
	if err := json.Unmarshal(respBytes, &out); err != nil {
		return nil, fmt.Errorf("decode: %w (body: %s)", err, respBytes)
	}
	if len(out.Choices) == 0 {
		return nil, fmt.Errorf("response has no choices: %s", respBytes)
	}
	return &out, nil
}

// ---------------------------------------------------------------------------
// content analysis helpers
// ---------------------------------------------------------------------------

func hasRepeatedWord(text, word string, minRuns int) bool {
	// case-sensitive token match — both 'key' and 'KEY' patterns are checked
	// by passing each case separately.
	lower := strings.ToLower(text)
	target := strings.ToLower(word)
	tokens := strings.FieldsFunc(lower, isTokenSep)
	run := 0
	for _, tok := range tokens {
		if tok == target {
			run++
			if run >= minRuns {
				return true
			}
		} else {
			run = 0
		}
	}
	return false
}

func uniqueWordRatio(text string) float64 {
	tokens := strings.FieldsFunc(strings.ToLower(text), isTokenSep)
	if len(tokens) < 2 {
		return 1.0
	}
	seen := make(map[string]struct{}, len(tokens))
	for _, t := range tokens {
		seen[t] = struct{}{}
	}
	return float64(len(seen)) / float64(len(tokens))
}

func isTokenSep(r rune) bool {
	switch {
	case r == ' ', r == '\t', r == '\n', r == '\r':
		return true
	case r == '.', r == ',', r == '!', r == '?', r == ';', r == ':':
		return true
	case r == '(', r == ')', r == '[', r == ']', r == '{', r == '}':
		return true
	case r == '"', r == '\'', r == '`':
		return true
	}
	return false
}
