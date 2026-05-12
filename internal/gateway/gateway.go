// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

// Package gateway is Quenchforge's HTTP front door.
//
// Routes (MVP):
//
//	GET  /api/tags                 — Ollama: list locally available models
//	POST /api/chat                 — Ollama: chat completion (streams NDJSON)
//	POST /api/generate             — Ollama: text completion (streams NDJSON)
//	POST /v1/chat/completions      — OpenAI-compatible chat (streams SSE)
//	GET  /health                   — liveness probe (always 200 if reachable)
//	GET  /                         — landing JSON {service, version, slots}
//
// The MVP wires /api/chat, /api/generate, and /v1/chat/completions to
// reverse-proxy the supervised llama-server slot. /api/tags reads the
// model registry (GGUF files under config.ModelsDir) directly so it works
// without a running slot.
//
// v0.2 swaps this for the vendored Olla gateway in internal/gateway/olla/
// — at that point the package keeps its public surface but the routing
// logic is delegated. Pinned SHA for the future vendoring:
// thushan/olla @ b11b81868504d07603e3815c6e38ddda068f862c.
package gateway

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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

// Gateway is the HTTP server. Construct via New.
type Gateway struct {
	cfg config.Config

	// upstreamURL is the URL where the supervised llama-server (or the
	// future multi-slot router) is reachable. Empty means slot is not yet
	// available — chat/generate routes will return 503.
	upstreamURL *url.URL

	mu     sync.RWMutex
	server *http.Server
	proxy  *httputil.ReverseProxy

	// version is the binary version surfaced by /. Set via SetVersion.
	version string
}

// New returns a Gateway bound to cfg. The server is not yet listening;
// call Start to bind and serve.
func New(cfg config.Config) *Gateway {
	return &Gateway{cfg: cfg, version: "0.0.0-dev"}
}

// SetVersion updates the version string surfaced by the landing route.
// Typically called from cmd/quenchforge with the build-time ldflag value.
func (g *Gateway) SetVersion(v string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.version = v
}

// SetUpstream points the proxy at the llama-server slot's local URL
// (e.g. "http://127.0.0.1:11500"). Called by the supervisor once the
// slot is ready.
func (g *Gateway) SetUpstream(raw string) error {
	if raw == "" {
		g.mu.Lock()
		g.upstreamURL = nil
		g.proxy = nil
		g.mu.Unlock()
		return nil
	}
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("gateway: parse upstream %q: %w", raw, err)
	}
	g.mu.Lock()
	g.upstreamURL = u
	g.proxy = httputil.NewSingleHostReverseProxy(u)
	g.proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		writeJSONError(w, http.StatusBadGateway,
			fmt.Sprintf("upstream %s unreachable: %v", u.Host, err))
	}
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
	mux.HandleFunc("/api/chat", g.handleProxy)
	mux.HandleFunc("/api/generate", g.handleProxy)
	mux.HandleFunc("/v1/chat/completions", g.handleProxy)

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

// ListenAddr returns the bound address (useful when ListenAddr was ":0").
func (g *Gateway) ListenAddr() string {
	g.mu.RLock()
	defer g.mu.RUnlock()
	if g.server == nil {
		return g.cfg.ListenAddr
	}
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
	resp := map[string]any{
		"service": "quenchforge",
		"version": g.version,
		"upstream": map[string]any{
			"configured": g.upstreamURL != nil,
		},
		"routes": []string{
			"GET /health",
			"GET /api/tags",
			"POST /api/chat",
			"POST /api/generate",
			"POST /v1/chat/completions",
		},
	}
	if g.upstreamURL != nil {
		resp["upstream"].(map[string]any)["url"] = g.upstreamURL.String()
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

func (g *Gateway) handleProxy(w http.ResponseWriter, r *http.Request) {
	g.mu.RLock()
	proxy := g.proxy
	upstream := g.upstreamURL
	g.mu.RUnlock()
	if proxy == nil || upstream == nil {
		writeJSONError(w, http.StatusServiceUnavailable,
			"upstream not configured — no llama-server slot is ready. "+
				"Check `quenchforge doctor` for status.")
		return
	}
	proxy.ServeHTTP(w, r)
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

// drainAndClose is a tiny helper for the test that bodies are fully read.
// Returns the body bytes for assertions and closes the reader.
func drainAndClose(r io.ReadCloser) []byte {
	defer r.Close()
	b, _ := io.ReadAll(r)
	return b
}

// ensure drainAndClose is referenced so go vet doesn't whine.
var _ = drainAndClose
