// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package gateway

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// upstreamFunc returns an httptest server that records the path + body
// it received and replies with the caller-supplied response body /
// status. Used by every translation test.
type capture struct {
	method      string
	path        string
	body        []byte
	contentType string
}

func newCapturingUpstream(t *testing.T, status int, response string, contentType string) (*httptest.Server, *capture) {
	t.Helper()
	cap := &capture{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.method = r.Method
		cap.path = r.URL.Path
		cap.contentType = r.Header.Get("Content-Type")
		cap.body, _ = io.ReadAll(r.Body)
		if response == "" {
			w.WriteHeader(status)
			return
		}
		w.Header().Set("Content-Type", contentType)
		w.WriteHeader(status)
		_, _ = w.Write([]byte(response))
	}))
	t.Cleanup(srv.Close)
	return srv, cap
}

func newRunningGatewayWithUpstreams(t *testing.T, chat, embed string) (*Gateway, string) {
	t.Helper()
	cfg := newTestConfig(t)
	cfg.ListenAddr = pickListenAddr(t)
	g := newRunningGateway(t, cfg)
	if chat != "" {
		if err := g.SetUpstream(KindChat, chat); err != nil {
			t.Fatalf("set chat upstream: %v", err)
		}
	}
	if embed != "" {
		if err := g.SetUpstream(KindEmbed, embed); err != nil {
			t.Fatalf("set embed upstream: %v", err)
		}
	}
	return g, cfg.ListenAddr
}

// ---------------------------------------------------------------------------
// Chat — non-streaming
// ---------------------------------------------------------------------------

func TestOllamaChatNonStreaming_TranslatesRequestAndResponse(t *testing.T) {
	openAIResp := `{
		"id":"chatcmpl-1",
		"model":"qwen2.5:7b",
		"choices":[{"index":0,"message":{"role":"assistant","content":"pong"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}
	}`
	upstream, cap := newCapturingUpstream(t, http.StatusOK, openAIResp, "application/json")
	_, addr := newRunningGatewayWithUpstreams(t, upstream.URL, "")

	body := `{
		"model":"qwen2.5:7b",
		"stream":false,
		"messages":[{"role":"user","content":"ping"}],
		"options":{"temperature":0.7,"num_predict":42,"top_p":0.95,"top_k":40,"seed":7,"stop":["<eot>"]},
		"format":"json"
	}`
	resp, err := http.Post("http://"+addr+"/api/chat", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}

	// Upstream sees the OpenAI-wire body and path.
	if cap.path != "/v1/chat/completions" {
		t.Errorf("upstream path = %q, want /v1/chat/completions", cap.path)
	}
	var openaiReq map[string]interface{}
	if err := json.Unmarshal(cap.body, &openaiReq); err != nil {
		t.Fatalf("upstream body not JSON: %v body=%s", err, cap.body)
	}
	if openaiReq["model"] != "qwen2.5:7b" {
		t.Errorf("upstream model = %v", openaiReq["model"])
	}
	if openaiReq["stream"] != false {
		t.Errorf("upstream stream = %v, want false", openaiReq["stream"])
	}
	// options.* must be flattened to OpenAI top-level
	if v, _ := openaiReq["temperature"].(float64); v != 0.7 {
		t.Errorf("upstream temperature = %v, want 0.7", openaiReq["temperature"])
	}
	if v, _ := openaiReq["max_tokens"].(float64); int(v) != 42 {
		t.Errorf("upstream max_tokens = %v, want 42", openaiReq["max_tokens"])
	}
	if v, _ := openaiReq["top_p"].(float64); v != 0.95 {
		t.Errorf("upstream top_p = %v, want 0.95", openaiReq["top_p"])
	}
	if v, _ := openaiReq["top_k"].(float64); int(v) != 40 {
		t.Errorf("upstream top_k = %v, want 40", openaiReq["top_k"])
	}
	if v, _ := openaiReq["seed"].(float64); int(v) != 7 {
		t.Errorf("upstream seed = %v, want 7", openaiReq["seed"])
	}
	stop, _ := openaiReq["stop"].([]interface{})
	if len(stop) != 1 || stop[0] != "<eot>" {
		t.Errorf("upstream stop = %v, want [<eot>]", openaiReq["stop"])
	}
	rf, _ := openaiReq["response_format"].(map[string]interface{})
	if rf == nil || rf["type"] != "json_object" {
		t.Errorf("upstream response_format = %v, want json_object", openaiReq["response_format"])
	}
	if _, hasOptions := openaiReq["options"]; hasOptions {
		t.Errorf("upstream body must not carry Ollama-wire `options` key")
	}

	// Response shape — Ollama wire.
	respBody, _ := io.ReadAll(resp.Body)
	var ollamaResp map[string]interface{}
	if err := json.Unmarshal(respBody, &ollamaResp); err != nil {
		t.Fatalf("response not JSON: %v body=%s", err, respBody)
	}
	msg, _ := ollamaResp["message"].(map[string]interface{})
	if msg == nil || msg["content"] != "pong" {
		t.Errorf("ollama response message = %v, want {content: pong}", ollamaResp["message"])
	}
	if ollamaResp["done"] != true {
		t.Errorf("ollama response done = %v, want true", ollamaResp["done"])
	}
	if ollamaResp["done_reason"] != "stop" {
		t.Errorf("ollama response done_reason = %v, want stop", ollamaResp["done_reason"])
	}
	if _, hasChoices := ollamaResp["choices"]; hasChoices {
		t.Errorf("ollama response must not carry OpenAI `choices` key")
	}
}

// ---------------------------------------------------------------------------
// Chat — streaming
// ---------------------------------------------------------------------------

func TestOllamaChatStreaming_SSEtoNDJSON(t *testing.T) {
	// Upstream emits a 3-event OpenAI SSE stream plus [DONE] sentinel.
	sse := strings.Join([]string{
		`data: {"id":"c1","model":"qwen2.5:7b","choices":[{"index":0,"delta":{"role":"assistant","content":"foo"}}]}`,
		``, // SSE event blank-line separator
		`data: {"id":"c1","model":"qwen2.5:7b","choices":[{"index":0,"delta":{"content":" bar"}}]}`,
		``,
		`data: {"id":"c1","model":"qwen2.5:7b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	upstream, _ := newCapturingUpstream(t, http.StatusOK, sse, "text/event-stream")
	_, addr := newRunningGatewayWithUpstreams(t, upstream.URL, "")

	body := `{"model":"qwen2.5:7b","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post("http://"+addr+"/api/chat", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/x-ndjson") {
		t.Errorf("Content-Type = %q, want application/x-ndjson", ct)
	}

	scanner := bufio.NewScanner(resp.Body)
	var lines []map[string]interface{}
	for scanner.Scan() {
		var rec map[string]interface{}
		if err := json.Unmarshal(scanner.Bytes(), &rec); err != nil {
			t.Fatalf("non-JSON NDJSON line %q: %v", scanner.Text(), err)
		}
		lines = append(lines, rec)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scanner: %v", err)
	}
	if len(lines) < 3 {
		t.Fatalf("got %d NDJSON lines, want >= 3 (2 chunks + final): %v",
			len(lines), lines)
	}

	// First line must be a non-final chunk with content "foo".
	first := lines[0]
	if first["done"] != false {
		t.Errorf("first line done = %v, want false", first["done"])
	}
	if msg, _ := first["message"].(map[string]interface{}); msg["content"] != "foo" {
		t.Errorf("first line content = %v, want foo", msg["content"])
	}

	// Last line must be done:true with done_reason set.
	last := lines[len(lines)-1]
	if last["done"] != true {
		t.Errorf("last line done = %v, want true", last["done"])
	}
	if last["done_reason"] != "stop" {
		t.Errorf("last line done_reason = %v, want stop", last["done_reason"])
	}
}

// ---------------------------------------------------------------------------
// Generate — translates prompt → single user message
// ---------------------------------------------------------------------------

func TestOllamaGenerate_TranslatesPromptToMessage(t *testing.T) {
	openAIResp := `{"id":"c1","model":"x","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{}}`
	upstream, cap := newCapturingUpstream(t, http.StatusOK, openAIResp, "application/json")
	_, addr := newRunningGatewayWithUpstreams(t, upstream.URL, "")

	body := `{"model":"x","stream":false,"system":"You are a unit test.","prompt":"say ok"}`
	resp, err := http.Post("http://"+addr+"/api/generate", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	var openaiReq map[string]interface{}
	if err := json.Unmarshal(cap.body, &openaiReq); err != nil {
		t.Fatalf("upstream body not JSON: %v body=%s", err, cap.body)
	}
	msgs, _ := openaiReq["messages"].([]interface{})
	if len(msgs) != 2 {
		t.Fatalf("upstream messages len = %d, want 2 (system + user): %v", len(msgs), msgs)
	}
	first, _ := msgs[0].(map[string]interface{})
	second, _ := msgs[1].(map[string]interface{})
	if first["role"] != "system" || first["content"] != "You are a unit test." {
		t.Errorf("first message = %v, want system/'You are...'", first)
	}
	if second["role"] != "user" || second["content"] != "say ok" {
		t.Errorf("second message = %v, want user/'say ok'", second)
	}
}

// ---------------------------------------------------------------------------
// Embeddings — both prompt and input variants
// ---------------------------------------------------------------------------

func TestOllamaEmbeddings_LegacyPromptShape(t *testing.T) {
	openAIResp := `{
		"object":"list",
		"model":"nomic-embed-text-v1.5",
		"data":[{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]}],
		"usage":{"prompt_tokens":2,"total_tokens":2}
	}`
	upstream, cap := newCapturingUpstream(t, http.StatusOK, openAIResp, "application/json")
	_, addr := newRunningGatewayWithUpstreams(t, "", upstream.URL)

	// Legacy /api/embeddings shape — `prompt` is a single string.
	resp, err := http.Post("http://"+addr+"/api/embeddings", "application/json",
		strings.NewReader(`{"model":"nomic-embed-text-v1.5","prompt":"hello world"}`))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}

	if cap.path != "/v1/embeddings" {
		t.Errorf("upstream path = %q, want /v1/embeddings", cap.path)
	}
	var upBody map[string]interface{}
	if err := json.Unmarshal(cap.body, &upBody); err != nil {
		t.Fatalf("upstream body not JSON: %v", err)
	}
	if upBody["model"] != "nomic-embed-text-v1.5" {
		t.Errorf("upstream model = %v", upBody["model"])
	}
	if upBody["input"] != "hello world" {
		t.Errorf("upstream input = %v, want 'hello world'", upBody["input"])
	}

	// Response carries BOTH `embedding` (legacy) AND `embeddings` (newer)
	// so old callers don't break and new callers can opt into batches.
	respBody, _ := io.ReadAll(resp.Body)
	var out map[string]interface{}
	if err := json.Unmarshal(respBody, &out); err != nil {
		t.Fatalf("response not JSON: %v", err)
	}
	emb, _ := out["embedding"].([]interface{})
	if len(emb) != 3 {
		t.Errorf("legacy embedding key length = %d, want 3", len(emb))
	}
	embs, _ := out["embeddings"].([]interface{})
	if len(embs) != 1 {
		t.Errorf("newer embeddings key length = %d, want 1", len(embs))
	}
}

func TestOllamaEmbeddings_BatchInputShape(t *testing.T) {
	openAIResp := `{
		"object":"list",
		"model":"nomic-embed-text-v1.5",
		"data":[
			{"object":"embedding","index":0,"embedding":[0.1]},
			{"object":"embedding","index":1,"embedding":[0.2]}
		],
		"usage":{}
	}`
	upstream, cap := newCapturingUpstream(t, http.StatusOK, openAIResp, "application/json")
	_, addr := newRunningGatewayWithUpstreams(t, "", upstream.URL)

	resp, err := http.Post("http://"+addr+"/api/embed", "application/json",
		strings.NewReader(`{"model":"nomic-embed-text-v1.5","input":["one","two"]}`))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d", resp.StatusCode)
	}

	var upBody map[string]interface{}
	_ = json.Unmarshal(cap.body, &upBody)
	inputs, _ := upBody["input"].([]interface{})
	if len(inputs) != 2 || inputs[0] != "one" || inputs[1] != "two" {
		t.Errorf("upstream input = %v, want [one, two]", upBody["input"])
	}

	respBody, _ := io.ReadAll(resp.Body)
	var out map[string]interface{}
	_ = json.Unmarshal(respBody, &out)
	// Batch input must NOT set the legacy singular `embedding` key.
	if _, hasLegacy := out["embedding"]; hasLegacy {
		t.Errorf("batch response must not carry singular `embedding` key: %v", out)
	}
	embs, _ := out["embeddings"].([]interface{})
	if len(embs) != 2 {
		t.Errorf("batch embeddings length = %d, want 2", len(embs))
	}
}

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

func TestOllamaChat_UpstreamErrorMapsToOllamaErrorShape(t *testing.T) {
	openAIErr := `{"error":{"message":"model not found","type":"invalid_request_error","code":404}}`
	upstream, _ := newCapturingUpstream(t, http.StatusNotFound, openAIErr, "application/json")
	_, addr := newRunningGatewayWithUpstreams(t, upstream.URL, "")

	resp, err := http.Post("http://"+addr+"/api/chat", "application/json",
		strings.NewReader(`{"model":"missing","stream":false,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("status = %d, want 404", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	var env map[string]interface{}
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatalf("response not JSON: %v body=%s", err, body)
	}
	if env["error"] != "model not found" {
		t.Errorf("error key = %v, want 'model not found'", env["error"])
	}
}

func TestOllamaChat_NoMessagesIs400(t *testing.T) {
	upstream, _ := newCapturingUpstream(t, http.StatusOK, `{}`, "application/json")
	_, addr := newRunningGatewayWithUpstreams(t, upstream.URL, "")

	resp, err := http.Post("http://"+addr+"/api/chat", "application/json",
		strings.NewReader(`{"model":"x","messages":[]}`))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestOllamaEmbeddings_EmptyInputIs400(t *testing.T) {
	upstream, _ := newCapturingUpstream(t, http.StatusOK, `{}`, "application/json")
	_, addr := newRunningGatewayWithUpstreams(t, "", upstream.URL)

	resp, err := http.Post("http://"+addr+"/api/embeddings", "application/json",
		strings.NewReader(`{"model":"x"}`))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}
