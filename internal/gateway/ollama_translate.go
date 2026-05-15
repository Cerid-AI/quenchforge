// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package gateway

// Ollama-wire ↔ OpenAI-wire body translation.
//
// llama-server speaks only the OpenAI surface (/v1/chat/completions,
// /v1/embeddings). Ollama clients hit /api/chat, /api/generate,
// /api/embeddings, /api/embed expecting Ollama-shaped requests and
// responses. The gateway is the only thing between the two — so it has
// to translate, not just rewrite paths.
//
// Spec references:
//   - Ollama API:  https://github.com/ollama/ollama/blob/main/docs/api.md
//   - OpenAI chat: https://platform.openai.com/docs/api-reference/chat
//   - OpenAI embed: https://platform.openai.com/docs/api-reference/embeddings
//
// Coverage:
//   - POST /api/chat        ↔ POST /v1/chat/completions  (streaming + non-streaming)
//   - POST /api/generate    ↔ POST /v1/chat/completions  (prompt → single user message)
//   - POST /api/embeddings  ↔ POST /v1/embeddings        (legacy single-prompt shape)
//   - POST /api/embed       ↔ POST /v1/embeddings        (newer multi-input shape)
//
// The Ollama tags endpoint (/api/tags) and pull endpoint (/api/pull)
// are not translated — tags is served directly from the registry and
// pull is a deliberate stub.

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// Wire shapes — Ollama side
// ---------------------------------------------------------------------------

// ollamaChatRequest mirrors the Ollama /api/chat request body. Fields are
// intentionally permissive (json.Number, RawMessage) to forward exotic
// caller payloads without losing fidelity.
type ollamaChatRequest struct {
	Model     string                 `json:"model"`
	Messages  []ollamaChatMessage    `json:"messages"`
	Stream    *bool                  `json:"stream,omitempty"` // pointer so we can distinguish absent from false
	Options   map[string]interface{} `json:"options,omitempty"`
	Format    string                 `json:"format,omitempty"`
	KeepAlive interface{}            `json:"keep_alive,omitempty"`
	// Tools / images / etc. — passed through verbatim via Raw to avoid
	// silently dropping fields we don't yet translate.
	Raw json.RawMessage `json:"-"`
}

type ollamaChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ollamaGenerateRequest is /api/generate's body. Generate is sugar for
// "single user-role message"; we translate it into a chat call.
type ollamaGenerateRequest struct {
	Model     string                 `json:"model"`
	Prompt    string                 `json:"prompt"`
	System    string                 `json:"system,omitempty"`
	Stream    *bool                  `json:"stream,omitempty"`
	Options   map[string]interface{} `json:"options,omitempty"`
	Format    string                 `json:"format,omitempty"`
	KeepAlive interface{}            `json:"keep_alive,omitempty"`
}

// ollamaEmbedRequest accepts both wire variants: `prompt` (legacy
// /api/embeddings, single string) and `input` (newer /api/embed,
// string or []string). Whichever is set wins; if both are set,
// `input` takes precedence so callers can opt into the new shape
// without removing the old.
type ollamaEmbedRequest struct {
	Model  string      `json:"model"`
	Prompt string      `json:"prompt,omitempty"`
	Input  interface{} `json:"input,omitempty"`
	// Truncate / KeepAlive / Options are forwarded as-is via the
	// options field on OpenAI? — no, OpenAI doesn't carry them.
	// We drop them quietly (Ollama clients tolerate the omission).
}

// ---------------------------------------------------------------------------
// Wire shapes — OpenAI side
// ---------------------------------------------------------------------------

// openAIChatRequest is what we forward upstream to llama-server.
type openAIChatRequest struct {
	Model          string              `json:"model"`
	Messages       []ollamaChatMessage `json:"messages"`
	Stream         bool                `json:"stream"`
	Temperature    *float64            `json:"temperature,omitempty"`
	TopP           *float64            `json:"top_p,omitempty"`
	TopK           *int                `json:"top_k,omitempty"`
	MaxTokens      *int                `json:"max_tokens,omitempty"`
	Seed           *int                `json:"seed,omitempty"`
	Stop           []string            `json:"stop,omitempty"`
	ResponseFormat *openAIResponseFmt  `json:"response_format,omitempty"`
}

type openAIResponseFmt struct {
	Type string `json:"type"`
}

type openAIChatResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Message      ollamaChatMessage `json:"message"`
		FinishReason string            `json:"finish_reason"`
		Index        int               `json:"index"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// openAIChatStreamChunk is one SSE frame body from llama-server.
// We translate every chunk that has a content delta into one Ollama
// NDJSON line.
type openAIChatStreamChunk struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Delta        ollamaChatMessage `json:"delta"`
		FinishReason *string           `json:"finish_reason"`
		Index        int               `json:"index"`
	} `json:"choices"`
}

type openAIEmbedRequest struct {
	Model string      `json:"model"`
	Input interface{} `json:"input"`
}

type openAIEmbedResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string    `json:"object"`
		Index     int       `json:"index"`
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// ---------------------------------------------------------------------------
// Translator handlers — bound to upstream entry resolution
// ---------------------------------------------------------------------------

// handleOllamaChat translates POST /api/chat (and /api/generate) into a
// /v1/chat/completions call against the chat slot. It buffers the
// inbound request, builds the OpenAI body, dispatches via httpClient,
// then either streams the response back as Ollama NDJSON (stream=true)
// or returns one Ollama JSON object (stream=false).
//
// generateMode=true means /api/generate's `prompt` field is read instead
// of `messages` and turned into a single user-role message. `system`
// becomes a leading system-role message when set.
func (g *Gateway) handleOllamaChat(generateMode bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		g.mu.RLock()
		entry, ok := g.upstreams[KindChat]
		g.mu.RUnlock()
		if !ok || entry.proxy == nil {
			writeJSONError(w, http.StatusServiceUnavailable,
				"no chat slot configured. Check `quenchforge doctor` for status.")
			return
		}

		// Buffer body — we need to read it twice (parse, then forward).
		raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBodyBytes))
		if err != nil {
			writeJSONError(w, http.StatusBadRequest,
				fmt.Sprintf("read request body: %v", err))
			return
		}

		var (
			model      string
			messages   []ollamaChatMessage
			streamFlag bool
			options    map[string]interface{}
			format     string
		)
		if generateMode {
			var req ollamaGenerateRequest
			if err := json.Unmarshal(raw, &req); err != nil {
				writeJSONError(w, http.StatusBadRequest,
					fmt.Sprintf("decode /api/generate body: %v", err))
				return
			}
			model = req.Model
			if req.System != "" {
				messages = append(messages, ollamaChatMessage{Role: "system", Content: req.System})
			}
			messages = append(messages, ollamaChatMessage{Role: "user", Content: req.Prompt})
			if req.Stream != nil {
				streamFlag = *req.Stream
			} else {
				streamFlag = true // Ollama /api/generate streams by default
			}
			options = req.Options
			format = req.Format
		} else {
			var req ollamaChatRequest
			if err := json.Unmarshal(raw, &req); err != nil {
				writeJSONError(w, http.StatusBadRequest,
					fmt.Sprintf("decode /api/chat body: %v", err))
				return
			}
			model = req.Model
			messages = req.Messages
			if req.Stream != nil {
				streamFlag = *req.Stream
			} else {
				streamFlag = true // Ollama /api/chat streams by default
			}
			options = req.Options
			format = req.Format
		}

		if len(messages) == 0 {
			writeJSONError(w, http.StatusBadRequest,
				"request has no messages")
			return
		}

		// Translate options.* → OpenAI top-level fields. Anything we don't
		// recognize is dropped — llama-server rejects unknown fields, and
		// the alternative (forwarding raw) breaks more than it fixes.
		openaiReq := openAIChatRequest{
			Model:    model,
			Messages: messages,
			Stream:   streamFlag,
		}
		if v, ok := getFloat(options, "temperature"); ok {
			openaiReq.Temperature = &v
		}
		if v, ok := getFloat(options, "top_p"); ok {
			openaiReq.TopP = &v
		}
		if v, ok := getInt(options, "top_k"); ok {
			openaiReq.TopK = &v
		}
		// Ollama uses num_predict for max tokens; OpenAI uses max_tokens.
		if v, ok := getInt(options, "num_predict"); ok {
			openaiReq.MaxTokens = &v
		}
		if v, ok := getInt(options, "seed"); ok {
			openaiReq.Seed = &v
		}
		if stop, ok := getStringList(options, "stop"); ok {
			openaiReq.Stop = stop
		}
		if strings.EqualFold(format, "json") {
			openaiReq.ResponseFormat = &openAIResponseFmt{Type: "json_object"}
		}

		body, err := json.Marshal(openaiReq)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError,
				fmt.Sprintf("encode upstream body: %v", err))
			return
		}

		upstreamURL := strings.TrimRight(entry.url.String(), "/") + "/v1/chat/completions"
		req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, upstreamURL,
			bytes.NewReader(body))
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError,
				fmt.Sprintf("build upstream request: %v", err))
			return
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json")
		// Propagate the Authorization header if a caller set one — useful
		// when llama-server is bound to a non-loopback interface and the
		// gateway sits in front of an auth proxy. No-op locally.
		if auth := r.Header.Get("Authorization"); auth != "" {
			req.Header.Set("Authorization", auth)
		}

		resp, err := translateHTTPClient.Do(req)
		if err != nil {
			writeJSONError(w, http.StatusBadGateway,
				fmt.Sprintf("chat upstream %s unreachable: %v",
					entry.url.Host, err))
			return
		}
		defer resp.Body.Close()

		// Upstream non-2xx — pass the status code and best-effort message
		// through. llama-server returns OpenAI-style { "error": { "message", "type" } };
		// translate that into Ollama-style { "error": "..." }.
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
			writeJSONError(w, resp.StatusCode,
				extractOpenAIErrorMessage(body))
			return
		}

		if streamFlag {
			streamOllamaChat(w, resp.Body, model)
		} else {
			respondOllamaChat(w, resp.Body, model)
		}
	}
}

// handleOllamaEmbeddings translates /api/embeddings and /api/embed into
// /v1/embeddings on the embed slot. Always non-streaming.
//
// Model-name dispatch: when the request body's `model` field matches
// Config.CodeEmbedModel and a code-embed slot is registered, the call is
// routed to KindCodeEmbed instead of KindEmbed. Lets one quenchforge
// process serve a general-text and a code-tuned embedder side-by-side.
func (g *Gateway) handleOllamaEmbeddings() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBodyBytes))
		if err != nil {
			writeJSONError(w, http.StatusBadRequest,
				fmt.Sprintf("read request body: %v", err))
			return
		}

		var req ollamaEmbedRequest
		if err := json.Unmarshal(raw, &req); err != nil {
			writeJSONError(w, http.StatusBadRequest,
				fmt.Sprintf("decode embed body: %v", err))
			return
		}

		kind := g.resolveEmbedKind(req.Model)
		g.mu.RLock()
		entry, ok := g.upstreams[kind]
		g.mu.RUnlock()
		if !ok || entry.proxy == nil {
			writeJSONError(w, http.StatusServiceUnavailable,
				fmt.Sprintf("no %s slot configured. Check `quenchforge doctor` for status.", kind))
			return
		}

		// Coerce Ollama's `prompt` (string) and `input` (string|[]string)
		// into a single OpenAI `input` field. `input` wins when both are
		// set; that mirrors Ollama's own behavior of treating /api/embed
		// as the canonical newer shape.
		var input interface{} = req.Input
		// Detect "input absent or empty" — interface{} typed nil from JSON
		// stays nil; an empty string or []string{""} should also fall back
		// to prompt if prompt is non-empty.
		if isEmptyInput(input) && req.Prompt != "" {
			input = req.Prompt
		}
		if isEmptyInput(input) {
			writeJSONError(w, http.StatusBadRequest,
				"request has neither `prompt` nor `input`")
			return
		}

		body, err := json.Marshal(openAIEmbedRequest{
			Model: req.Model,
			Input: input,
		})
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError,
				fmt.Sprintf("encode upstream body: %v", err))
			return
		}

		upstreamURL := strings.TrimRight(entry.url.String(), "/") + "/v1/embeddings"
		upReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, upstreamURL,
			bytes.NewReader(body))
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError,
				fmt.Sprintf("build upstream request: %v", err))
			return
		}
		upReq.Header.Set("Content-Type", "application/json")
		upReq.Header.Set("Accept", "application/json")

		resp, err := translateHTTPClient.Do(upReq)
		if err != nil {
			writeJSONError(w, http.StatusBadGateway,
				fmt.Sprintf("embed upstream %s unreachable: %v",
					entry.url.Host, err))
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
			writeJSONError(w, resp.StatusCode,
				extractOpenAIErrorMessage(body))
			return
		}

		var openaiResp openAIEmbedResponse
		if err := json.NewDecoder(resp.Body).Decode(&openaiResp); err != nil {
			writeJSONError(w, http.StatusBadGateway,
				fmt.Sprintf("decode upstream embed response: %v", err))
			return
		}

		// Ollama wire: legacy /api/embeddings returns {embedding: [...]},
		// newer /api/embed returns {embeddings: [[...], ...]}. We always
		// emit the newer shape because it works for both single and batch
		// callers; the legacy shape is recovered by reading [0] when the
		// request was single-input.
		//
		// Compatibility note: cerid's `core.utils.embeddings` only reads
		// `embedding` (singular) on /api/embeddings calls; emit BOTH keys
		// so legacy clients keep working without an extra round-trip.
		out := map[string]interface{}{
			"model":      openaiResp.Model,
			"embeddings": extractEmbeddings(openaiResp),
		}
		if len(openaiResp.Data) == 1 {
			out["embedding"] = openaiResp.Data[0].Embedding
		}
		writeJSON(w, http.StatusOK, out)
	}
}

// ---------------------------------------------------------------------------
// Response writers
// ---------------------------------------------------------------------------

// respondOllamaChat reads one OpenAI chat JSON object and writes one
// Ollama chat JSON object. Used when the caller set stream=false.
func respondOllamaChat(w http.ResponseWriter, body io.Reader, requestedModel string) {
	var resp openAIChatResponse
	if err := json.NewDecoder(body).Decode(&resp); err != nil {
		writeJSONError(w, http.StatusBadGateway,
			fmt.Sprintf("decode upstream chat response: %v", err))
		return
	}
	model := resp.Model
	if model == "" {
		model = requestedModel
	}
	var msg ollamaChatMessage
	finishReason := "stop"
	if len(resp.Choices) > 0 {
		msg = resp.Choices[0].Message
		if resp.Choices[0].FinishReason != "" {
			finishReason = resp.Choices[0].FinishReason
		}
	}
	if msg.Role == "" {
		msg.Role = "assistant"
	}
	out := map[string]interface{}{
		"model":             model,
		"created_at":        time.Now().UTC().Format(time.RFC3339Nano),
		"message":           msg,
		"done":              true,
		"done_reason":       finishReason,
		"prompt_eval_count": resp.Usage.PromptTokens,
		"eval_count":        resp.Usage.CompletionTokens,
	}
	writeJSON(w, http.StatusOK, out)
}

// streamOllamaChat reads the upstream SSE stream and writes one
// Ollama NDJSON line per chunk, ending with a `done: true` line.
//
// Spec:
//   - Each OpenAI SSE event is "data: {json}\n\n" or "data: [DONE]\n\n"
//   - Final Ollama NDJSON line carries `done: true` and `done_reason`
//   - Empty content deltas (e.g. role-only "delta": {"role":"assistant"})
//     emit a line with content="" so caller's token accumulator stays
//     happy — matches Ollama's behavior.
func streamOllamaChat(w http.ResponseWriter, body io.Reader, requestedModel string) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no") // disable proxy buffering
	flusher, _ := w.(http.Flusher)

	scanner := bufio.NewScanner(body)
	// Default Scanner buffer is 64K. SSE event payloads with embedded
	// images / large tool-call JSON can exceed that; bump to 1 MB.
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	var (
		lastModel    = requestedModel
		finishReason = ""
		emitted      bool
	)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" {
			continue
		}
		if payload == "[DONE]" {
			break
		}
		var chunk openAIChatStreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			// Best-effort: skip malformed events rather than tearing down
			// the whole stream. Real-world llama-server output sometimes
			// includes server-side log lines on the same SSE channel.
			continue
		}
		if chunk.Model != "" {
			lastModel = chunk.Model
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		c := chunk.Choices[0]
		if c.FinishReason != nil && *c.FinishReason != "" {
			finishReason = *c.FinishReason
		}
		msg := c.Delta
		if msg.Role == "" {
			msg.Role = "assistant"
		}
		out := map[string]interface{}{
			"model":      lastModel,
			"created_at": time.Now().UTC().Format(time.RFC3339Nano),
			"message":    msg,
			"done":       false,
		}
		if err := writeNDJSONLine(w, out); err != nil {
			return // client gone
		}
		if flusher != nil {
			flusher.Flush()
		}
		emitted = true
	}

	if finishReason == "" {
		finishReason = "stop"
	}
	final := map[string]interface{}{
		"model":       lastModel,
		"created_at":  time.Now().UTC().Format(time.RFC3339Nano),
		"message":     ollamaChatMessage{Role: "assistant", Content: ""},
		"done":        true,
		"done_reason": finishReason,
	}
	_ = writeNDJSONLine(w, final)
	if flusher != nil {
		flusher.Flush()
	}
	_ = emitted // reserved for future "no-output" warning logs
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

const maxRequestBodyBytes = 8 * 1024 * 1024 // 8 MB — plenty for chat with embedded images

func writeNDJSONLine(w http.ResponseWriter, v interface{}) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	_, err = w.Write(b)
	return err
}

func getFloat(opts map[string]interface{}, key string) (float64, bool) {
	v, ok := opts[key]
	if !ok {
		return 0, false
	}
	switch x := v.(type) {
	case float64:
		return x, true
	case float32:
		return float64(x), true
	case int:
		return float64(x), true
	case int64:
		return float64(x), true
	case json.Number:
		f, err := x.Float64()
		if err != nil {
			return 0, false
		}
		return f, true
	default:
		return 0, false
	}
}

func getInt(opts map[string]interface{}, key string) (int, bool) {
	v, ok := opts[key]
	if !ok {
		return 0, false
	}
	switch x := v.(type) {
	case float64:
		return int(x), true
	case int:
		return x, true
	case int64:
		return int(x), true
	case json.Number:
		i, err := x.Int64()
		if err != nil {
			return 0, false
		}
		return int(i), true
	default:
		return 0, false
	}
}

func getStringList(opts map[string]interface{}, key string) ([]string, bool) {
	v, ok := opts[key]
	if !ok {
		return nil, false
	}
	switch x := v.(type) {
	case string:
		return []string{x}, true
	case []interface{}:
		out := make([]string, 0, len(x))
		for _, e := range x {
			if s, ok := e.(string); ok {
				out = append(out, s)
			}
		}
		return out, len(out) > 0
	default:
		return nil, false
	}
}

func isEmptyInput(v interface{}) bool {
	if v == nil {
		return true
	}
	switch x := v.(type) {
	case string:
		return x == ""
	case []interface{}:
		return len(x) == 0
	case []string:
		return len(x) == 0
	default:
		return false
	}
}

func extractEmbeddings(r openAIEmbedResponse) [][]float64 {
	out := make([][]float64, len(r.Data))
	for i, d := range r.Data {
		out[i] = d.Embedding
	}
	return out
}

// extractOpenAIErrorMessage tries to read the OpenAI-shape error body
// llama-server returns on 4xx/5xx and produce a plain string for the
// Ollama-style { "error": "..." } reply.
func extractOpenAIErrorMessage(body []byte) string {
	var env struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &env); err == nil && env.Error.Message != "" {
		return env.Error.Message
	}
	// Fall back to the raw body, truncated.
	s := strings.TrimSpace(string(body))
	if len(s) > 256 {
		s = s[:256] + "…"
	}
	if s == "" {
		s = "upstream returned an empty error body"
	}
	return s
}

// translateHTTPClient is the http.Client used to forward translated
// requests upstream. No timeout — streaming chat completions hold the
// connection open for as long as the user wants the response. The
// gateway listener has no write timeout for the same reason.
var translateHTTPClient = &http.Client{}
