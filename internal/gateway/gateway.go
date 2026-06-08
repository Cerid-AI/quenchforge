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
	"bytes"
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
	"github.com/cerid-ai/quenchforge/internal/placement"
	"github.com/cerid-ai/quenchforge/internal/scheduler"
)

// SlotKind enumerates the slot types the gateway can route to. Adding a
// new kind requires a Slot wiring in cmd/quenchforge/serve and a gateway
// route registration in Start.
type SlotKind string

const (
	KindChat  SlotKind = "chat"
	KindEmbed SlotKind = "embed"
	// KindCodeEmbed is a *second* embedding slot dedicated to code-tuned
	// models. Requests arriving at /api/embeddings or /v1/embeddings whose
	// body specifies a `model` matching Config.CodeEmbedModel are dispatched
	// here instead of KindEmbed. Lets one quenchforge process serve a
	// general-text embedder (for KB / RAG) alongside a code-tuned embedder
	// (for semantic-code-search MCPs) without forcing operators to choose.
	KindCodeEmbed SlotKind = "code-embed"
	KindRerank    SlotKind = "rerank"
	KindWhisper   SlotKind = "whisper"
	KindImageGen  SlotKind = "imagegen"
	KindTTS       SlotKind = "tts"
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
	// cpuUpstreams holds the CPU instance of a dual-placed ("auto") kind. Only
	// embedding kinds populate it today, via SetCPUUpstream; routeEmbed sends a
	// single/small request here when the placement policy routes it to the CPU.
	cpuUpstreams map[SlotKind]upstreamEntry
	version      string
	latency      *latencyTracker
	// sched, when non-nil, gates GPU-bound routes through the admission
	// scheduler so the pressure governor's concurrency ceiling reserves GPU
	// headroom for the display compositor. nil → routes run ungated.
	sched *scheduler.Scheduler
	// policy is the device-placement policy. The zero value reports every kind
	// as "gpu" (Mode default), so until SetPlacement is called the gateway
	// behaves exactly as it did before placement awareness: all kinds GPU-bound
	// and governed. autoThreshold is the input-count boundary routeEmbed uses
	// for "auto"-placed kinds.
	policy        placement.Policy
	autoThreshold int
}

// New returns a Gateway bound to cfg. The server is not yet listening;
// call Start to bind and serve.
func New(cfg config.Config) *Gateway {
	return &Gateway{
		cfg:          cfg,
		version:      "0.0.0-dev",
		upstreams:    make(map[SlotKind]upstreamEntry),
		cpuUpstreams: make(map[SlotKind]upstreamEntry),
		latency:      newLatencyTracker(),
	}
}

// SetVersion updates the version string surfaced by the landing route.
// Typically called from cmd/quenchforge with the build-time ldflag value.
func (g *Gateway) SetVersion(v string) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.version = v
}

// SetScheduler installs the admission scheduler used to gate GPU-bound
// routes. Pass nil (the default) to leave routes ungated. The pressure
// governor adjusts the scheduler's concurrency ceiling over time.
func (g *Gateway) SetScheduler(s *scheduler.Scheduler) {
	g.mu.Lock()
	g.sched = s
	g.mu.Unlock()
}

// priorityForKind maps a slot kind to its scheduler priority so streaming
// chat is admitted ahead of batch embed/rerank when GPU headroom is scarce.
func priorityForKind(kind SlotKind) scheduler.Priority {
	switch kind {
	case KindChat:
		return scheduler.PriorityChat
	case KindEmbed, KindCodeEmbed:
		return scheduler.PriorityEmbed
	case KindRerank:
		return scheduler.PriorityRerank
	default: // whisper, imagegen, tts
		return scheduler.PriorityBackground
	}
}

// gated wraps a GPU-bound handler with the admission scheduler. It is the
// single chokepoint covering BOTH forward paths — the reverse-proxy handlers
// AND the Ollama-translation handlers (which forward via their own
// http.Client) — so the governor's ceiling applies to all GPU traffic.
// Non-GPU routes (/, /health, /api/tags, /api/pull) are never wrapped. When
// no scheduler is installed the handler runs unchanged (zero overhead).
//
// Acquire blocks on the request's context, so a client that gives up (or
// times out) frees its place in line; if that happens before admission we
// return 503 + Retry-After so callers get structured backpressure instead of
// a hang.
func (g *Gateway) gated(kind SlotKind, h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// CPU-placed kinds (chat / rerank on AMD-discrete, anything an operator
		// pins to "cpu") don't contend with the display compositor for the GPU,
		// so they skip admission entirely and run at full speed. Until
		// SetPlacement is called the zero-value policy reports every kind as GPU,
		// preserving the pre-placement behaviour (all routes governed).
		if g.placementDevice(kind) == placement.CPU {
			h(w, r)
			return
		}
		g.withGPUAdmission(kind, w, r, func() { h(w, r) })
	}
}

// withGPUAdmission runs serve under the GPU admission scheduler: it acquires a
// slot (503 + Retry-After if the request's context expires while waiting),
// runs serve, then holds the slot idle for the duty-cycle cooldown before
// releasing so the GPU yields a window to the display compositor. When no
// scheduler is installed, serve runs unchanged (zero overhead). This is the
// single chokepoint for all GPU traffic — reverse-proxy routes via gated, and
// the per-request embedding routers call it directly for their GPU branch.
func (g *Gateway) withGPUAdmission(kind SlotKind, w http.ResponseWriter, r *http.Request, serve func()) {
	g.mu.RLock()
	sched := g.sched
	g.mu.RUnlock()
	if sched == nil {
		serve()
		return
	}
	release, err := sched.Acquire(r.Context(), priorityForKind(kind))
	if err != nil {
		w.Header().Set("Retry-After", "1")
		writeJSONError(w, http.StatusServiceUnavailable,
			fmt.Sprintf("%s slot saturated under GPU pressure; retry shortly", kind))
		return
	}
	defer release()
	start := time.Now()
	serve()
	// Duty-cycle cooldown: after the response is sent, keep holding the slot
	// idle for a span proportional to the GPU time this request just used,
	// so the GPU yields a window to the display compositor before the next
	// admission. This temporal gap — not concurrency capping — is what
	// prevents sustained gapless inference from starving WindowServer.
	if d := sched.DutyCycle(); d < 1.0 {
		idle := time.Duration(float64(time.Since(start)) * (1.0 - d) / d)
		if m := g.maxCooldown(); idle > m {
			idle = m
		}
		if idle > 0 {
			time.Sleep(idle)
		}
	}
}

// placementDevice reports the device the placement policy assigns to a kind.
// The zero-value policy returns GPU for everything (see Gateway.policy).
func (g *Gateway) placementDevice(kind SlotKind) placement.Device {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.policy.Device(string(kind))
}

// maxCooldown caps the per-request duty-cycle idle hold so a single long
// generation can't stall the queue for seconds. The in-generation yield for
// long requests comes from command-buffer granularity (GGML_METAL_N_CB); this
// cap bounds the inter-request gap.
func (g *Gateway) maxCooldown() time.Duration {
	ms := g.cfg.GovernorMaxCooldownMS
	if ms <= 0 {
		ms = 250
	}
	return time.Duration(ms) * time.Millisecond
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

// SetCPUUpstream points the CPU instance of a dual-placed ("auto") kind at a
// URL. Mirrors SetUpstream but stores into cpuUpstreams; routeEmbed forwards a
// single/small request here when the policy routes it to the CPU. Passing an
// empty raw URL clears the entry (routeEmbed then falls back to the GPU
// upstream for that kind).
func (g *Gateway) SetCPUUpstream(kind SlotKind, raw string) error {
	if raw == "" {
		g.mu.Lock()
		delete(g.cpuUpstreams, kind)
		g.mu.Unlock()
		return nil
	}
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("gateway: parse %s cpu upstream %q: %w", kind, raw, err)
	}
	proxy := httputil.NewSingleHostReverseProxy(u)
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		writeJSONError(w, http.StatusBadGateway,
			fmt.Sprintf("%s cpu upstream %s unreachable: %v", kind, u.Host, err))
	}
	g.mu.Lock()
	g.cpuUpstreams[kind] = upstreamEntry{url: u, proxy: proxy}
	g.mu.Unlock()
	return nil
}

// SetPlacement installs the device-placement policy and the auto-routing batch
// threshold. Called once at startup after the host profile is known. Before
// this call the gateway's zero-value policy treats every kind as GPU-bound, so
// placement awareness is dormant (no behaviour change) until it is wired.
func (g *Gateway) SetPlacement(p placement.Policy, threshold int) {
	g.mu.Lock()
	g.policy = p
	g.autoThreshold = threshold
	g.mu.Unlock()
}

// routeEmbed picks the upstream for one embedding request given the kind's
// placement mode and the request's input count:
//
//   - "cpu"  : always the (single) upstream registered for the kind, ungoverned.
//   - "gpu"  : always the kind's upstream, GPU-governed.
//   - "auto" : RouteRequest decides by batch shape. A CPU verdict routes to the
//     registered CPU instance when one exists; otherwise it falls back to the
//     GPU upstream so a missing CPU slot degrades to working-but-on-GPU rather
//     than 503. A GPU verdict always uses the GPU upstream.
//
// onGPU reports whether the chosen instance needs GPU admission; ok is false
// when the chosen upstream has no proxy (slot not configured) so the caller can
// return 503. The zero-value policy reports "gpu", so the default is the GPU
// upstream — identical to the pre-placement single-upstream path.
func (g *Gateway) routeEmbed(kind SlotKind, batchN int) (entry upstreamEntry, onGPU bool, ok bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	switch g.policy.Mode(string(kind)) {
	case placement.ModeCPU:
		entry = g.upstreams[kind]
		onGPU = false
	case placement.ModeAuto:
		if g.policy.RouteRequest(string(kind), batchN, g.autoThreshold) == placement.CPU {
			if e, exists := g.cpuUpstreams[kind]; exists && e.proxy != nil {
				return e, false, true
			}
		}
		entry = g.upstreams[kind]
		onGPU = true
	default: // ModeGPU and any unknown mode
		entry = g.upstreams[kind]
		onGPU = true
	}
	ok = entry.proxy != nil
	return entry, onGPU, ok
}

// countEmbedInputs returns the number of inputs in an embedding request, used
// by routeEmbed to classify single-vs-batch under "auto" placement. An []string
// or []interface{} counts by length; a non-empty string counts as 1; an empty
// or absent input falls back to the prompt. The result is clamped to a minimum
// of 1 so a malformed body still routes deterministically (upstream rejects it).
func countEmbedInputs(input interface{}, prompt string) int {
	switch x := input.(type) {
	case []interface{}:
		if len(x) > 0 {
			return len(x)
		}
	case []string:
		if len(x) > 0 {
			return len(x)
		}
	case string:
		if x != "" {
			return 1
		}
	}
	if prompt != "" {
		return 1
	}
	return 1
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
	// chat (Ollama + OpenAI surfaces).  llama-server only speaks the
	// OpenAI wire — /api/chat and /api/generate are translated by the
	// handlers in ollama_translate.go so Ollama clients work end-to-end.
	// /v1/chat/completions is OpenAI-native and goes through the simple
	// reverse-proxy path unchanged.
	mux.HandleFunc("/api/chat", g.gated(KindChat, g.handleOllamaChat(false)))
	mux.HandleFunc("/api/generate", g.gated(KindChat, g.handleOllamaChat(true)))
	mux.HandleFunc("/v1/chat/completions", g.gated(KindChat, g.proxyHandler(KindChat, "")))
	// embeddings (Ollama + OpenAI surfaces).  Same translation story:
	// /api/embeddings and /api/embed are Ollama wire, translated to
	// /v1/embeddings on the upstream embed slot; /v1/embeddings is
	// pass-through.
	// Embeddings self-admit: the handlers route per request (resolving the
	// embed kind and, under "auto" placement, the GPU/CPU instance) and apply
	// GPU admission only to the GPU branch — so a CPU-routed embed runs
	// ungoverned. Wrapping them in gated(KindEmbed, …) would double-admit and
	// misclassify code-embed traffic, so the route registration is bare.
	mux.HandleFunc("/api/embeddings", g.handleOllamaEmbeddings())
	mux.HandleFunc("/api/embed", g.handleOllamaEmbeddings())
	mux.HandleFunc("/v1/embeddings", g.handleOpenAIEmbeddings())
	// rerank (OpenAI-style /v1/rerank; llama-server speaks its own /rerank
	// when launched with --reranking; we route both to the same slot).
	mux.HandleFunc("/v1/rerank", g.gated(KindRerank, g.proxyHandler(KindRerank, "/rerank")))
	mux.HandleFunc("/rerank", g.gated(KindRerank, g.proxyHandler(KindRerank, "")))
	// whisper audio transcription. OpenAI's /v1/audio/transcriptions and
	// whisper.cpp's /inference take the same multipart shape but on
	// different paths — rewrite the OpenAI path to /inference on the way
	// through. /v1/audio/translations same surface (English-only output).
	mux.HandleFunc("/v1/audio/transcriptions", g.gated(KindWhisper, g.proxyHandler(KindWhisper, "/inference")))
	mux.HandleFunc("/v1/audio/translations", g.gated(KindWhisper, g.proxyHandler(KindWhisper, "/inference")))
	mux.HandleFunc("/inference", g.gated(KindWhisper, g.proxyHandler(KindWhisper, ""))) // whisper-native path
	// image generation — sd-server speaks OpenAI's /v1/images/generations
	// natively, so no path rewrite needed.
	mux.HandleFunc("/v1/images/generations", g.gated(KindImageGen, g.proxyHandler(KindImageGen, "")))
	// stable-diffusion.cpp also exposes its own SD-API surface; expose
	// /sdapi/ unchanged for clients that prefer the AUTOMATIC1111-style API.
	mux.HandleFunc("/sdapi/v1/txt2img", g.gated(KindImageGen, g.proxyHandler(KindImageGen, "")))
	mux.HandleFunc("/sdapi/v1/img2img", g.gated(KindImageGen, g.proxyHandler(KindImageGen, "")))
	// text-to-speech (bark.cpp server). Its native route is /tts (returns
	// audio/wav); OpenAI's /v1/audio/speech POSTs JSON { input, voice, ... }.
	// We pass-through; client format-translation is a v0.4 concern.
	mux.HandleFunc("/v1/audio/speech", g.gated(KindTTS, g.proxyHandler(KindTTS, "/tts")))
	mux.HandleFunc("/tts", g.gated(KindTTS, g.proxyHandler(KindTTS, "")))
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
	for _, k := range []SlotKind{KindChat, KindEmbed, KindCodeEmbed, KindRerank, KindWhisper, KindImageGen, KindTTS} {
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

// handleHealth returns the gateway's overall status plus a per-slot
// breakdown of rolling-window latency and error rate. Always 200 while
// the gateway is reachable — consumers parse the JSON to decide whether
// to throttle. The opt-in QUENCHFORGE_AUTO_BACKOFF flag is the only
// thing that turns a `critical` snapshot into an actual 503 on the
// upstream proxy paths; /health itself never blocks.
//
// Schema:
//
//	{
//	  "status": "ok",
//	  "slots": {
//	    "embed": {
//	      "kind": "embed",
//	      "samples": 312,
//	      "p50_ms": 18.4,
//	      "p99_ms": 41.2,
//	      "error_rate": 0.0,
//	      "status": "ok",
//	      "window_secs": 60
//	    },
//	    ...
//	  },
//	  "auto_backoff_enabled": false
//	}
func (g *Gateway) handleHealth(w http.ResponseWriter, _ *http.Request) {
	snaps := g.latency.Snapshot()
	slots := make(map[string]LatencySnapshot, len(snaps))
	for k, v := range snaps {
		slots[string(k)] = v
	}
	// Overall status is the worst per-slot status (so /health caller
	// gets a one-glance answer to "is anything wrong").
	overall := StatusOK
	for _, v := range snaps {
		if v.Status == StatusCritical {
			overall = StatusCritical
			break
		}
		if v.Status == StatusDegraded && overall == StatusOK {
			overall = StatusDegraded
		}
	}
	resp := map[string]any{
		"status":               string(overall),
		"slots":                slots,
		"auto_backoff_enabled": g.cfg.AutoBackoffEnabled,
	}
	g.mu.RLock()
	sched := g.sched
	g.mu.RUnlock()
	if sched != nil {
		// Live governor state — lets operators watch the admission ceiling
		// drop while a display is being driven and recover when it sleeps.
		resp["governor"] = map[string]any{
			"concurrency": sched.Concurrency(),
			"active":      sched.Active(),
			"pending":     sched.Pending(),
		}
	}
	writeJSON(w, http.StatusOK, resp)
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
//
// Latency tracking: every upstream call records duration + error-flag in
// the gateway's rolling per-kind tracker. /health surfaces the resulting
// status (ok | degraded | critical). When QUENCHFORGE_AUTO_BACKOFF is on
// and the slot is currently `critical`, the handler returns 503 +
// Retry-After before forwarding — gives consumers a structured signal to
// throttle before the family-B Metal crash tips the slot over.
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
		if g.shouldBackoff(kind) {
			w.Header().Set("Retry-After", "2")
			writeJSONError(w, http.StatusServiceUnavailable,
				fmt.Sprintf("%s slot is at critical latency — back off (Retry-After: 2s)", kind))
			return
		}
		if rewriteTo != "" {
			// Clone the request so concurrent handlers don't see our rewrite.
			r2 := r.Clone(r.Context())
			r2.URL.Path = rewriteTo
			r2.URL.RawPath = ""
			r = r2
		}
		g.serveAndTrack(kind, entry.proxy, w, r)
	}
}

// shouldBackoff reports whether the gateway should preemptively 503 a
// request to `kind`. Gated on cfg.AutoBackoffEnabled so the default
// behaviour is observability-only.
func (g *Gateway) shouldBackoff(kind SlotKind) bool {
	if !g.cfg.AutoBackoffEnabled {
		return false
	}
	return g.latency.SnapshotKind(kind).Status == StatusCritical
}

// serveAndTrack wraps ServeHTTP so the per-kind latency tracker sees
// every upstream call. The response status is captured via a thin
// ResponseWriter shim — bool flag for is-error (status >= 500 or write
// failure) feeds the per-kind error rate.
func (g *Gateway) serveAndTrack(kind SlotKind, proxy http.Handler, w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
	proxy.ServeHTTP(rec, r)
	g.latency.Record(kind, time.Since(start), rec.status >= 500)
}

// statusRecorder is the minimal wrapper that captures the status code
// the upstream proxy writes. Required because the latency tracker needs
// to count "5xx as error" to compute the per-kind error rate. We don't
// inspect the body — the goal is one bool per call, not a deep inspect.
type statusRecorder struct {
	http.ResponseWriter
	status      int
	wroteHeader bool
}

func (s *statusRecorder) WriteHeader(code int) {
	if !s.wroteHeader {
		s.status = code
		s.wroteHeader = true
	}
	s.ResponseWriter.WriteHeader(code)
}

func (s *statusRecorder) Write(b []byte) (int, error) {
	if !s.wroteHeader {
		// Per http.ResponseWriter contract, Write without WriteHeader
		// implies 200.
		s.wroteHeader = true
	}
	return s.ResponseWriter.Write(b)
}

// Flush forwards to the underlying writer if it supports streaming. Required
// because httputil.ReverseProxy uses Flusher for SSE streaming responses;
// without forwarding, /v1/chat/completions streams would buffer until the
// upstream closed the connection.
func (s *statusRecorder) Flush() {
	if f, ok := s.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// resolveEmbedKind picks the embed slot kind for an inbound request by
// matching the request's `model` field against Config.CodeEmbedModel.
//
//   - Empty CodeEmbedModel  → always KindEmbed (legacy single-slot behavior).
//   - model == CodeEmbedModel and a KindCodeEmbed upstream is registered →
//     KindCodeEmbed.
//   - Anything else → KindEmbed.
//
// The fallback to KindEmbed is deliberate: if an operator typo'd the code
// model name or the code-embed slot failed to register, callers still get
// a working general-text response instead of a 503 they can't diagnose.
func (g *Gateway) resolveEmbedKind(model string) SlotKind {
	if g.cfg.CodeEmbedModel == "" || model == "" {
		return KindEmbed
	}
	if model != g.cfg.CodeEmbedModel {
		return KindEmbed
	}
	g.mu.RLock()
	_, ok := g.upstreams[KindCodeEmbed]
	g.mu.RUnlock()
	if !ok {
		return KindEmbed
	}
	return KindCodeEmbed
}

// handleOpenAIEmbeddings is the OpenAI-native /v1/embeddings entry point.
// Peeks at the body's `model` field to dispatch between KindEmbed and
// KindCodeEmbed, then reverse-proxies to the chosen upstream. Replaces the
// static proxyHandler(KindEmbed, "") registration for this route so a
// single quenchforge process can serve two embedders on the same gateway.
func (g *Gateway) handleOpenAIEmbeddings() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		raw, err := readAllLimited(w, r)
		if err != nil {
			writeJSONError(w, http.StatusBadRequest,
				fmt.Sprintf("read request body: %v", err))
			return
		}
		// Minimal probe — `model` selects the embed kind and `input` gives the
		// batch shape for "auto" placement routing; everything else passes
		// through unchanged. We re-attach the body below.
		var probe struct {
			Model string      `json:"model"`
			Input interface{} `json:"input"`
		}
		_ = json.Unmarshal(raw, &probe) // tolerate empty/invalid bodies; let upstream reject
		kind := g.resolveEmbedKind(probe.Model)
		batchN := countEmbedInputs(probe.Input, "")
		entry, onGPU, ok := g.routeEmbed(kind, batchN)
		if !ok {
			writeJSONError(w, http.StatusServiceUnavailable,
				fmt.Sprintf("no %s slot configured. Check `quenchforge doctor` for status.", kind))
			return
		}
		// Re-attach the consumed body so the reverse-proxy can forward it.
		r.Body = io.NopCloser(bytes.NewReader(raw))
		r.ContentLength = int64(len(raw))
		if g.shouldBackoff(kind) {
			w.Header().Set("Retry-After", "2")
			writeJSONError(w, http.StatusServiceUnavailable,
				fmt.Sprintf("%s slot is at critical latency — back off (Retry-After: 2s)", kind))
			return
		}
		serve := func() { g.serveAndTrack(kind, entry.proxy, w, r) }
		if onGPU {
			g.withGPUAdmission(kind, w, r, serve)
		} else {
			serve()
		}
	}
}

// readAllLimited reads the request body with the same MaxBytesReader cap
// the Ollama-translation handlers use. Extracted so /v1/embeddings's body
// peek shares the limit.
func readAllLimited(w http.ResponseWriter, r *http.Request) ([]byte, error) {
	return io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBodyBytes))
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
