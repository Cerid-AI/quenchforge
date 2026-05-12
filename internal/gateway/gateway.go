// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

// Package gateway is Quenchforge's HTTP front door.
//
// Routes:
//
//	GET  /                         — landing JSON {service, version, slots, routes}
//	GET  /health                   — liveness probe (always 200 if reachable)
//	GET  /api/tags                 — Ollama: list locally available models
//	POST /api/chat                 — Ollama: chat completion       → KindChat
//	POST /api/generate             — Ollama: text completion       → KindChat
//	POST /v1/chat/completions      — OpenAI: chat (streams SSE)    → KindChat
//	POST /api/embeddings           — Ollama: embeddings            → KindEmbed
//	POST /v1/embeddings            — OpenAI: embeddings            → KindEmbed
//	POST /api/pull                 — Ollama: model pull (stub 501 in MVP)
//
// Upstream resolution is keyed by SlotKind. The supervisor calls
// `gateway.SetUpstream(KindChat, "http://127.0.0.1:11500")` once the chat
// slot is ready; the same call for KindEmbed when the embedding slot lands.
// /api/chat against an unconfigured chat slot returns 503; same for embed.
// /api/tags reads the model registry directly so it works without any slot.
//
// v0.2 swaps the routing layer for the vendored Olla gateway in
// internal/gateway/olla/ — the public surface here is designed to be the
// shim Olla feeds into. Pinned SHA for the future vendoring:
// thushan/olla @ b11b81868504d07603e3815c6e38ddda068f862c.
package gateway

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/cerid-ai/quenchforge/internal/config"
)

// SlotKind enumerates the slot types the gateway can route to. Adding a
// new kind requires a Slot wiring in cmd/quenchforge/serve and a gateway
// route registration in Start.
type SlotKind string

const (
	KindChat     SlotKind = "chat"
	KindEmbed    SlotKind = "embed"
	KindRerank   SlotKind = "rerank"
	KindWhisper  SlotKind = "whisper"
	KindImageGen SlotKind = "imagegen"
	KindTTS      SlotKind = "tts"
)

// String implements fmt.Stringer.
func (k SlotKind) String() string { return string(k) }

// upstreamEntry holds the URL + ready-to-use proxy for one slot kind.
type upstreamEntry struct {
	url   *url.URL
	proxy *httputil.ReverseProxy
}

// Gateway is the HTTP server. Construct via New.
type Gateway struct {
	cfg config.Config

	mu        sync.RWMutex
	server    *http.Server
	upstreams map[SlotKind]upstreamEntry
	version   string
}

// New returns a Gateway bound to cfg. The server is not yet listening;
// call Start to bind and serve.
func New(cfg config.Config) *Gateway {
	return &Gateway{
		cfg:       cfg,
		version:   "0.0.0-dev",
		upstreams: make(map[SlotKind]upstreamEntry),
	}
}

// SetVersion updates the version string surfaced by the landing route.
// Typically called from cmd/quenchforge with the build-time ldflag value.
func (g *Gateway) SetVersion(v string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.version = v
}

// SetUpstream points the proxy for the given slot kind at a URL. Passing an
// empty raw URL clears the entry (chat/embed/rerank routes for that kind
// will go back to returning 503). The compatibility-friendly variant for
// the original single-slot call style is in compat.go.
func (g *Gateway) SetUpstream(kind SlotKind, raw string) error {
	if raw == "" {
		g.mu.Lock()
		delete(g.upstreams, kind)
		g.mu.Unlock()
		return nil
	}
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("gateway: parse %s upstream %q: %w", kind, raw, err)
	}
	proxy := httputil.NewSingleHostReverseProxy(u)
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		writeJSONError(w, http.StatusBadGateway,
			fmt.Sprintf("%s upstream %s unreachable: %v", kind, u.Host, err))
	}
	g.mu.Lock()
	g.upstreams[kind] = upstreamEntry{url: u, proxy: proxy}
	g.mu.Unlock()
	return nil
}

// Start binds the listener and begins serving. The call returns once the
// listener is ready; Serve runs in a goroutine. Use Stop to shut down.
//
// Returns ErrAddrInUse when ListenAddr's port is held by another process
// (so callers can render a useful "Ollama is already running on
// 127.0.0.1:11434 — run `brew services stop ollama` or set
// QUENCHFORGE_LISTEN_ADDR" message).
func (g *Gateway) Start(ctx context.Context) error {
	ln, err := net.Listen("tcp", g.cfg.ListenAddr)
	if err != nil {
		if isAddrInUse(err) {
			return fmt.Errorf("gateway: %w: %v", ErrAddrInUse, err)
		}
		return fmt.Errorf("gateway: listen on %s: %w", g.cfg.ListenAddr, err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", g.handleRoot)
	mux.HandleFunc("/health", g.handleHealth)
	mux.HandleFunc("/api/tags", g.handleTags)
	// chat (Ollama + OpenAI surfaces)
	mux.HandleFunc("/api/chat", g.proxyHandler(KindChat, ""))
	mux.HandleFunc("/api/generate", g.proxyHandler(KindChat, ""))
	mux.HandleFunc("/v1/chat/completions", g.proxyHandler(KindChat, ""))
	// embeddings (Ollama + OpenAI surfaces)
	mux.HandleFunc("/api/embeddings", g.proxyHandler(KindEmbed, ""))
	mux.HandleFunc("/api/embed", g.proxyHandler(KindEmbed, "")) // Ollama alias
	mux.HandleFunc("/v1/embeddings", g.proxyHandler(KindEmbed, ""))
	// rerank (OpenAI-style /v1/rerank; llama-server speaks its own /rerank
	// when launched with --reranking; we route both to the same slot).
	mux.HandleFunc("/v1/rerank", g.proxyHandler(KindRerank, "/rerank"))
	mux.HandleFunc("/rerank", g.proxyHandler(KindRerank, ""))
	// whisper audio transcription. OpenAI's /v1/audio/transcriptions and
	// whisper.cpp's /inference take the same multipart shape but on
	// different paths — rewrite the OpenAI path to /inference on the way
	// through. /v1/audio/translations same surface (English-only output).
	mux.HandleFunc("/v1/audio/transcriptions", g.proxyHandler(KindWhisper, "/inference"))
	mux.HandleFunc("/v1/audio/translations", g.proxyHandler(KindWhisper, "/inference"))
	mux.HandleFunc("/inference", g.proxyHandler(KindWhisper, "")) // whisper-native path
	// image generation — sd-server speaks OpenAI's /v1/images/generations
	// natively, so no path rewrite needed.
	mux.HandleFunc("/v1/images/generations", g.proxyHandler(KindImageGen, ""))
	// stable-diffusion.cpp also exposes its own SD-API surface; expose
	// /sdapi/ unchanged for clients that prefer the AUTOMATIC1111-style API.
	mux.HandleFunc("/sdapi/v1/txt2img", g.proxyHandler(KindImageGen, ""))
	mux.HandleFunc("/sdapi/v1/img2img", g.proxyHandler(KindImageGen, ""))
	// text-to-speech (bark.cpp server). Its native route is /tts (returns
	// audio/wav); OpenAI's /v1/audio/speech POSTs JSON { input, voice, ... }.
	// We pass-through; client format-translation is a v0.4 concern.
	mux.HandleFunc("/v1/audio/speech", g.proxyHandler(KindTTS, "/tts"))
	mux.HandleFunc("/tts", g.proxyHandler(KindTTS, ""))
	// pull (stub — points users at migrate-from-ollama)
	mux.HandleFunc("/api/pull", g.handlePull)

	g.mu.Lock()
	g.server = &http.Server{
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
		// No write timeout — streaming completions need to hold the
		// connection open for the duration of the response.
		BaseContext: func(_ net.Listener) context.Context { return ctx },
	}
	g.mu.Unlock()

	go func() {
		if err := g.server.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			fmt.Fprintf(os.Stderr, "quenchforge: gateway serve: %v\n", err)
		}
	}()
	return nil
}

// Stop shuts the server down with the given grace period. Idempotent.
func (g *Gateway) Stop(grace time.Duration) error {
	g.mu.Lock()
	srv := g.server
	g.mu.Unlock()
	if srv == nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), grace)
	defer cancel()
	return srv.Shutdown(ctx)
}

// ListenAddr returns the configured bind address.
func (g *Gateway) ListenAddr() string {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.cfg.ListenAddr
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

func (g *Gateway) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	g.mu.RLock()
	slots := make(map[string]any, len(g.upstreams))
	for k, v := range g.upstreams {
		slots[string(k)] = map[string]any{
			"configured": true,
			"url":        v.url.String(),
		}
	}
	// Always include known kinds in the report so consumers can see which
	// ones aren't configured.
	for _, k := range []SlotKind{KindChat, KindEmbed, KindRerank, KindWhisper, KindImageGen, KindTTS} {
		if _, ok := slots[string(k)]; !ok {
			slots[string(k)] = map[string]any{"configured": false}
		}
	}
	resp := map[string]any{
		"service": "quenchforge",
		"version": g.version,
		"slots":   slots,
		"routes": []string{
			"GET /health",
			"GET /api/tags",
			"POST /api/chat",
			"POST /api/generate",
			"POST /v1/chat/completions",
			"POST /api/embeddings",
			"POST /v1/embeddings",
			"POST /v1/rerank",
			"POST /v1/audio/transcriptions",
			"POST /v1/audio/translations",
			"POST /v1/images/generations (501 reserved)",
			"POST /v1/audio/speech (501 reserved)",
			"POST /api/pull (stub)",
		},
	}
	g.mu.RUnlock()
	writeJSON(w, http.StatusOK, resp)
}

func (g *Gateway) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (g *Gateway) handleTags(w http.ResponseWriter, _ *http.Request) {
	models, err := EnumerateModels(g.cfg.ModelsDir)
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, err.Error())
		return
	}
	// Ollama returns: {"models": [{"name", "modified_at", "size", "digest", ...}]}
	out := make([]map[string]any, 0, len(models))
	for _, m := range models {
		out = append(out, map[string]any{
			"name":        m.Name,
			"model":       m.Name,
			"modified_at": m.ModifiedAt.Format(time.RFC3339),
			"size":        m.SizeBytes,
			"digest":      m.Digest,
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": out})
}

// proxyHandler returns an http.HandlerFunc that reverse-proxies to the
// upstream registered for kind. Returns 503 when the kind has no upstream
// — common during startup when the slot hasn't finished loading the model.
//
// When rewriteTo is non-empty, the upstream request URL.Path is rewritten
// to that value before being forwarded. Used to translate OpenAI-shaped
// paths (e.g. /v1/audio/transcriptions, /v1/rerank) onto whisper-server's
// /inference and llama-server's /rerank natives.
func (g *Gateway) proxyHandler(kind SlotKind, rewriteTo string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		g.mu.RLock()
		entry, ok := g.upstreams[kind]
		g.mu.RUnlock()
		if !ok || entry.proxy == nil {
			writeJSONError(w, http.StatusServiceUnavailable,
				fmt.Sprintf("no %s slot configured. Check `quenchforge doctor` for status.", kind))
			return
		}
		if rewriteTo != "" {
			// Clone the request so concurrent handlers don't see our rewrite.
			r2 := r.Clone(r.Context())
			r2.URL.Path = rewriteTo
			r2.URL.RawPath = ""
			r = r2
		}
		entry.proxy.ServeHTTP(w, r)
	}
}

// handlePull is a deliberate stub. Ollama's /api/pull downloads a model
// from a registry; Quenchforge's v0.1 path is `quenchforge migrate-from-ollama`
// or manually placing a GGUF in the models dir. Returning 501 with a
// pointer is the right honest answer.
func (g *Gateway) handlePull(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusNotImplemented, map[string]any{
		"error": "model pull is not implemented in this version",
		"hint": "place a GGUF under " + g.cfg.ModelsDir + " or run " +
			"`quenchforge migrate-from-ollama` to symlink existing Ollama models",
		"docs": "https://github.com/cerid-ai/quenchforge#models",
	})
}

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

// Model is one entry in /api/tags.
type Model struct {
	Name       string
	Path       string
	SizeBytes  int64
	ModifiedAt time.Time
	Digest     string // SHA-256 of the path, NOT the bytes — matches Ollama's display digest
}

// EnumerateModels walks modelsDir for .gguf files and returns them as Models.
// The digest is a hash of the file path rather than contents — Ollama only
// uses it as a display identifier and hashing GBs of weights at boot is
// untenable.
func EnumerateModels(modelsDir string) ([]Model, error) {
	if _, err := os.Stat(modelsDir); err != nil {
		if os.IsNotExist(err) {
			return []Model{}, nil // empty registry, not an error
		}
		return nil, err
	}
	var out []Model
	err := filepath.WalkDir(modelsDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil // best-effort
		}
		if d.IsDir() || !strings.HasSuffix(strings.ToLower(d.Name()), ".gguf") {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		rel, _ := filepath.Rel(modelsDir, path)
		// Strip ".gguf" extension for the display name to match Ollama conventions.
		name := strings.TrimSuffix(rel, ".gguf")
		h := sha256.Sum256([]byte(path))
		out = append(out, Model{
			Name:       name,
			Path:       path,
			SizeBytes:  info.Size(),
			ModifiedAt: info.ModTime(),
			Digest:     "sha256:" + hex.EncodeToString(h[:8]), // short prefix
		})
		return nil
	})
	return out, err
}

// ---------------------------------------------------------------------------
// helpers + sentinels
// ---------------------------------------------------------------------------

// ErrAddrInUse is returned by Start when the port is held by another process.
// Callers should compare with errors.Is and render a friendly hint.
var ErrAddrInUse = errors.New("listen address already in use")

func isAddrInUse(err error) bool {
	if err == nil {
		return false
	}
	// net.OpError → os.SyscallError → EADDRINUSE
	var opErr *net.OpError
	if errors.As(err, &opErr) {
		if opErr.Err != nil && strings.Contains(opErr.Err.Error(), "address already in use") {
			return true
		}
	}
	return strings.Contains(err.Error(), "address already in use")
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeJSONError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}
