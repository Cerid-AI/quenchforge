// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

// Package gateway is Quenchforge's HTTP front door.
//
// Routes (the authoritative table is Gateway.routes, which drives both mux
// registration and the advertisement served at GET /):
//
//	GET  /                         — landing JSON {service, version, slots, routes}
//	GET  /health                   — per-slot readiness + rolling latency
//	GET  /api/tags                 — Ollama: models in the registry
//	POST /api/chat                 — Ollama: chat completion       → KindChat
//	POST /api/generate             — Ollama: text completion       → KindChat
//	POST /v1/chat/completions      — OpenAI: chat (streams SSE)    → KindChat
//	POST /api/embeddings           — Ollama: embeddings            → KindEmbed / KindCodeEmbed
//	POST /api/embed                — Ollama: embeddings (new shape)→ KindEmbed / KindCodeEmbed
//	POST /v1/embeddings            — OpenAI: embeddings            → KindEmbed / KindCodeEmbed
//	POST /v1/rerank                — OpenAI-style rerank           → KindRerank
//	POST /rerank                   — llama-server native rerank    → KindRerank
//	POST /v1/audio/transcriptions  — OpenAI: transcription         → KindWhisper
//	POST /v1/audio/translations    — OpenAI: translation           → KindWhisper
//	POST /inference                — whisper-server native         → KindWhisper
//	POST /v1/images/generations    — OpenAI: image generation      → KindImageGen
//	POST /sdapi/v1/txt2img         — sd.cpp native txt2img         → KindImageGen
//	POST /sdapi/v1/img2img         — sd.cpp native img2img         → KindImageGen
//	POST /v1/audio/speech          — OpenAI: TTS                   → KindTTS
//	POST /tts                      — bark.cpp native TTS           → KindTTS
//	POST /api/pull                 — Ollama: model pull (stub 501)
//
// The /v1/* routes are inference-endpoint compatibility only: quenchforge
// serves what its slots have loaded, so there is deliberately no OpenAI
// model-management surface (/v1/models). Use GET /api/tags for the
// registry and GET /health for what each slot can actually serve.
//
// Upstream resolution is keyed by SlotKind. `quenchforge serve` calls
// `gateway.SetUpstream(KindChat, "http://127.0.0.1:11500")` once the chat
// slot is ready, and the same call per kind as the other slots land. A kind
// with no upstream 503s and reports its readiness at GET /health; an
// upstream that proves dead is deregistered and retried after a cool-off
// (see markUpstreamUnreachable). /api/tags reads the model registry
// directly so it works without any slot.
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
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
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
	// KindBackground is a *second* chat-class slot dedicated to a
	// different model — e.g. a small model for background enrichment
	// tasks running alongside a larger interactive chat model. Requests
	// arriving at /api/chat, /api/generate, or /v1/chat/completions whose
	// `model` names Config.BackgroundModel are dispatched here instead of
	// KindChat. Unlike KindCodeEmbed's fallback-on-miss behavior, a
	// request that explicitly names the background model gets a 503 (not
	// a silent fall-through to chat) when no background upstream is
	// registered — the caller asked for a specific model.
	KindBackground SlotKind = "background"
	KindRerank     SlotKind = "rerank"
	KindWhisper    SlotKind = "whisper"
	KindImageGen   SlotKind = "imagegen"
	KindTTS        SlotKind = "tts"
)

// String implements fmt.Stringer.
func (k SlotKind) String() string { return string(k) }

// knownSlotKinds is every kind the gateway mounts routes for, in report
// order. handleRoot and handleHealth both enumerate it so a kind with no
// upstream shows up as unavailable instead of being silently absent.
var knownSlotKinds = []SlotKind{
	KindChat, KindEmbed, KindCodeEmbed, KindBackground, KindRerank, KindWhisper, KindImageGen, KindTTS,
}

// upstreamEntry holds the URL + ready-to-use proxy for one slot kind.
type upstreamEntry struct {
	url   *url.URL
	proxy *httputil.ReverseProxy
}

// downUpstream is a deregistered upstream awaiting a retry.
type downUpstream struct {
	raw    string
	since  time.Time
	reason string
}

// probedModel is one cached /v1/models answer. An empty model means the
// probe failed; it is retried after slotModelProbeRetry so a slot that was
// still loading when we first asked doesn't stay unknown forever.
type probedModel struct {
	model string
	at    time.Time
}

// Gateway is the HTTP server. Construct via New.
type Gateway struct {
	cfg config.Config

	mu        sync.RWMutex
	server    *http.Server
	upstreams map[SlotKind]upstreamEntry
	// downUpstreams parks the URL of an upstream that proved dead
	// (connection refused) after markUpstreamUnreachable deregistered it, so
	// the next request past the cool-off can re-register the same address
	// when the slot respawns on it.
	downUpstreams map[SlotKind]downUpstream
	// slotModels caches the model name each slot reports as loaded, probed
	// from the upstream's /v1/models. Cleared whenever the kind's upstream is
	// re-registered (a new registration may be a different model).
	slotModels map[SlotKind]probedModel
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

	// backoffOn remembers, per tracker key, whether auto-backoff was shedding
	// on the previous shouldBackoff evaluation — so state transitions log
	// exactly once instead of once per rejected request (or never, which is
	// what made the 2026-07-08 503 storm cost hours to diagnose).
	backoffMu sync.Mutex
	backoffOn map[SlotKind]bool
}

// New returns a Gateway bound to cfg. The server is not yet listening;
// call Start to bind and serve.
func New(cfg config.Config) *Gateway {
	return &Gateway{
		cfg:           cfg,
		version:       "0.0.0-dev",
		upstreams:     make(map[SlotKind]upstreamEntry),
		cpuUpstreams:  make(map[SlotKind]upstreamEntry),
		downUpstreams: make(map[SlotKind]downUpstream),
		slotModels:    make(map[SlotKind]probedModel),
		latency:       newLatencyTracker(),
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
	case KindChat, KindBackground:
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

// syncUpstreamTimeout bounds one embed or rerank call. Both are
// non-streaming and sub-minute, so an unbounded wait can only ever be a
// wedged slot (the AMDRadeonX5000 kernel-mutex stall in patches/README.md
// section 3) hanging the caller forever with no error, no latency sample
// and no /health movement. Chat (streams for as long as the user wants),
// transcription and image generation are deliberately NOT bounded by it.
// Package var so tests can shorten it.
var syncUpstreamTimeout = 120 * time.Second

// SetUpstream points the proxy for the given slot kind at a URL. Passing an
// empty raw URL clears the entry (chat/embed/rerank routes for that kind
// will go back to returning 503) — that is the deregistration path
// markUpstreamUnreachable uses when a slot proves dead. Registering a URL
// also clears any parked "dead" record and the cached slot model, because a
// fresh registration may be a different process serving a different model.
func (g *Gateway) SetUpstream(kind SlotKind, raw string) error {
	if raw == "" {
		g.mu.Lock()
		delete(g.upstreams, kind)
		delete(g.slotModels, kind)
		g.mu.Unlock()
		return nil
	}
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("gateway: parse %s upstream %q: %w", kind, raw, err)
	}
	proxy := httputil.NewSingleHostReverseProxy(u)
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		g.markUpstreamUnreachable(kind, err)
		if errors.Is(err, context.DeadlineExceeded) {
			writeJSONError(w, http.StatusGatewayTimeout,
				fmt.Sprintf("%s upstream %s timed out after %s", kind, u.Host, syncUpstreamTimeout))
			return
		}
		writeJSONError(w, http.StatusBadGateway,
			fmt.Sprintf("%s upstream %s unreachable: %v", kind, u.Host, err))
	}
	g.mu.Lock()
	g.upstreams[kind] = upstreamEntry{url: u, proxy: proxy}
	delete(g.downUpstreams, kind)
	delete(g.slotModels, kind)
	g.mu.Unlock()
	return nil
}

// upstreamRetryCooloff is how long a deregistered (proved-dead) upstream
// stays parked before the next request for that kind re-registers it and
// tries again. Long enough that a crash loop doesn't turn into a dial
// storm, short enough that a respawned slot is picked up without an
// operator restart. Package var so tests can shorten it.
var upstreamRetryCooloff = 15 * time.Second

// markUpstreamUnreachable deregisters an upstream that proved dead so the
// gateway stops proxying at a corpse. Only a refused connection counts:
// that means nothing is listening on the slot's port (the shape a crashed
// or never-restarted slot leaves behind). Timeouts, resets mid-response and
// client cancellations are NOT treated as death — a wedged or slow slot is
// still a process we can reach, and a cancelled request says nothing about
// the upstream.
//
// The URL is parked so the next request past upstreamRetryCooloff
// re-registers it: a slot that respawns on the same port is picked up
// without an operator restart. Until then the kind's routes return the
// documented 503 + doctor hint instead of an ErrorHandler 502, and /health
// reports the kind "unreachable".
func (g *Gateway) markUpstreamUnreachable(kind SlotKind, cause error) {
	if !isConnectionRefused(cause) {
		return
	}
	g.mu.Lock()
	entry, ok := g.upstreams[kind]
	if !ok || entry.url == nil {
		g.mu.Unlock()
		return
	}
	raw := entry.url.String()
	g.mu.Unlock()

	// Clear first, then park — SetUpstream("") drops the entry, and the
	// parked record is what makes the retry (and /health) possible.
	_ = g.SetUpstream(kind, "")
	g.mu.Lock()
	g.downUpstreams[kind] = downUpstream{raw: raw, since: time.Now(), reason: cause.Error()}
	g.mu.Unlock()
	log.Printf("quenchforge: %s upstream %s is not accepting connections (%v) — deregistered; "+
		"%s requests return 503 until the slot is back (retrying in %s)",
		kind, raw, cause, kind, upstreamRetryCooloff)
}

// lookupUpstream resolves the registered upstream for a kind, first giving
// a parked (proved-dead) upstream its retry once the cool-off has elapsed.
// ok is false when the kind has no usable upstream; callers render the
// reason with unavailableReason.
func (g *Gateway) lookupUpstream(kind SlotKind) (upstreamEntry, bool) {
	g.retryUpstreamIfDue(kind)
	g.mu.RLock()
	entry, ok := g.upstreams[kind]
	g.mu.RUnlock()
	return entry, ok && entry.proxy != nil
}

// retryUpstreamIfDue re-registers a parked upstream once upstreamRetryCooloff
// has passed since it was deregistered. If the slot is still dead the next
// request marks it unreachable again, so a crash loop costs one dial per
// cool-off rather than one per request.
func (g *Gateway) retryUpstreamIfDue(kind SlotKind) {
	g.mu.Lock()
	down, parked := g.downUpstreams[kind]
	if !parked || time.Since(down.since) < upstreamRetryCooloff {
		g.mu.Unlock()
		return
	}
	delete(g.downUpstreams, kind)
	g.mu.Unlock()
	if err := g.SetUpstream(kind, down.raw); err != nil {
		log.Printf("quenchforge: re-registering %s upstream %s failed: %v", kind, down.raw, err)
		return
	}
	log.Printf("quenchforge: retrying %s upstream %s after cool-off", kind, down.raw)
}

// unavailableReason is the 503 body for a kind with no usable upstream. It
// distinguishes "never configured" from "was configured and died", because
// the operator action differs.
func (g *Gateway) unavailableReason(kind SlotKind) string {
	g.mu.RLock()
	down, parked := g.downUpstreams[kind]
	g.mu.RUnlock()
	if parked {
		return fmt.Sprintf("%s upstream %s is unreachable (%s) — deregistered %s ago, retrying after %s. "+
			"Check `quenchforge doctor` for slot status.",
			kind, down.raw, down.reason, time.Since(down.since).Truncate(time.Second), upstreamRetryCooloff)
	}
	return fmt.Sprintf("no %s slot configured. Check `quenchforge doctor` for status.", kind)
}

// isConnectionRefused reports whether err is a refused TCP connection —
// nothing is listening on the upstream port.
func isConnectionRefused(err error) bool {
	return err != nil && errors.Is(err, syscall.ECONNREFUSED)
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
// onGPU reports whether the chosen instance needs GPU admission. track is the
// latency-tracker key for the chosen INSTANCE — the "auto" CPU twin records
// under "<kind>-cpu" so its millisecond singles and the GPU instance's
// multi-second batches never share one latency distribution (a mixed window
// made the p99/p50 classifier report a false "critical" and 503 ALL embed
// traffic during the 2026-07-08 cerid eval incident). ok is false when the
// chosen upstream has no proxy (slot not configured) so the caller can return
// 503. The zero-value policy reports "gpu", so the default is the GPU
// upstream — identical to the pre-placement single-upstream path.
func (g *Gateway) routeEmbed(kind SlotKind, batchN int) (entry upstreamEntry, onGPU bool, track SlotKind, ok bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	track = kind
	switch g.policy.Mode(string(kind)) {
	case placement.ModeCPU:
		entry = g.upstreams[kind]
		onGPU = false
	case placement.ModeAuto:
		if g.policy.RouteRequest(string(kind), batchN, g.autoThreshold) == placement.CPU {
			if e, exists := g.cpuUpstreams[kind]; exists && e.proxy != nil {
				return e, false, cpuTrackKind(kind), true
			}
		}
		entry = g.upstreams[kind]
		onGPU = true
	default: // ModeGPU and any unknown mode
		entry = g.upstreams[kind]
		onGPU = true
	}
	ok = entry.proxy != nil
	return entry, onGPU, track, ok
}

// cpuTrackKind derives the latency-tracker key for the CPU twin instance of a
// dual-placed kind. Kept as a helper so /health consumers and tests share one
// naming rule ("embed-cpu", "code-embed-cpu").
func cpuTrackKind(kind SlotKind) SlotKind {
	return kind + "-cpu"
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

// route is one mounted HTTP route. The table returned by Gateway.routes is
// the single source of truth for BOTH mux registration and the capability
// advertisement served at GET /, so what the gateway advertises cannot
// drift from what it actually serves.
type route struct {
	method  string
	pattern string
	note    string // advertisement annotation, e.g. "(stub)"
	handler http.HandlerFunc
}

// routes is the mounted HTTP surface.
//
// Chat: llama-server only speaks the OpenAI wire, so /api/chat and
// /api/generate are translated by the handlers in ollama_translate.go;
// /v1/chat/completions is OpenAI-native and takes the reverse-proxy path.
//
// Embeddings self-admit: those handlers route per request (resolving the
// embed kind and, under "auto" placement, the GPU/CPU instance) and apply
// GPU admission only to the GPU branch, so a CPU-routed embed runs
// ungoverned. Wrapping them in gated(KindEmbed, …) would double-admit and
// misclassify code-embed traffic, so their registration is bare. They are
// bounded by syncUpstreamTimeout instead — a wedged embed or rerank slot
// must fail the caller rather than hang it.
//
// Rerank, transcription, image generation and TTS are pass-through proxies;
// where the upstream's native path differs from the OpenAI path (whisper's
// /inference, llama-server's /rerank, bark's /tts) the handler rewrites it
// on the way through.
func (g *Gateway) routes() []route {
	return []route{
		{"GET", "/", "", g.handleRoot},
		{"GET", "/health", "", g.handleHealth},
		{"GET", "/api/tags", "", g.handleTags},

		// Chat self-admits: the handlers resolve KindChat or KindBackground
		// from the request's model and apply GPU admission for that kind, so
		// wrapping them in gated(KindChat, …) would misprioritise background
		// traffic.
		{"POST", "/api/chat", "", g.handleOllamaChat(false)},
		{"POST", "/api/generate", "", g.handleOllamaChat(true)},
		{"POST", "/v1/chat/completions", "", g.handleOpenAIChat()},

		// Embeddings self-admit too: GPU admission applies only to the GPU
		// branch, so a CPU-routed embed runs ungoverned.

		{"POST", "/api/embeddings", "", withRequestTimeout(g.handleOllamaEmbeddings())},
		{"POST", "/api/embed", "", withRequestTimeout(g.handleOllamaEmbeddings())},
		{"POST", "/v1/embeddings", "", withRequestTimeout(g.handleOpenAIEmbeddings())},

		{"POST", "/v1/rerank", "", withRequestTimeout(g.gated(KindRerank, g.proxyHandler(KindRerank, "/rerank")))},
		{"POST", "/rerank", "", withRequestTimeout(g.gated(KindRerank, g.proxyHandler(KindRerank, "")))},

		{"POST", "/v1/audio/transcriptions", "", g.gated(KindWhisper, g.proxyHandler(KindWhisper, "/inference"))},
		{"POST", "/v1/audio/translations", "", g.gated(KindWhisper, g.proxyHandler(KindWhisper, "/inference"))},
		{"POST", "/inference", "", g.gated(KindWhisper, g.proxyHandler(KindWhisper, ""))},

		{"POST", "/v1/images/generations", "", g.gated(KindImageGen, g.proxyHandler(KindImageGen, ""))},
		{"POST", "/sdapi/v1/txt2img", "", g.gated(KindImageGen, g.proxyHandler(KindImageGen, ""))},
		{"POST", "/sdapi/v1/img2img", "", g.gated(KindImageGen, g.proxyHandler(KindImageGen, ""))},

		{"POST", "/v1/audio/speech", "", g.gated(KindTTS, g.proxyHandler(KindTTS, "/tts"))},
		{"POST", "/tts", "", g.gated(KindTTS, g.proxyHandler(KindTTS, ""))},

		{"POST", "/api/pull", "(stub — see `quenchforge migrate-from-ollama`)", g.handlePull},
	}
}

// withRequestTimeout bounds a non-streaming route with syncUpstreamTimeout.
// Applied to the embed and rerank routes only: they are the short,
// synchronous calls where an unbounded wait can only mean a wedged slot.
// Chat streams, transcription and image generation legitimately run long
// and stay unbounded.
func withRequestTimeout(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), syncUpstreamTimeout)
		defer cancel()
		h(w, r.WithContext(ctx))
	}
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
	for _, rt := range g.routes() {
		mux.HandleFunc(rt.pattern, rt.handler)
	}

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

// slotModelNames maps each known slot kind to its configured model name (may
// be empty). Shared by handleRoot and handleTags so both routes name a
// slot's model — and decide whether a cached file is "loaded" — the same way.
func (g *Gateway) slotModelNames() map[SlotKind]string {
	return map[SlotKind]string{
		KindChat:       g.cfg.DefaultModel,
		KindBackground: g.cfg.BackgroundModel,
		KindEmbed:      g.cfg.EmbedModel,
		KindCodeEmbed:  g.cfg.CodeEmbedModel,
		KindRerank:     g.cfg.RerankModel,
	}
}

func (g *Gateway) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	g.mu.RLock()
	slotModel := g.slotModelNames()
	slots := make(map[string]any, len(g.upstreams))
	for k, v := range g.upstreams {
		entry := map[string]any{
			"configured": true,
			"url":        v.url.String(),
		}
		if m := slotModel[k]; m != "" {
			entry["model"] = strings.TrimSuffix(m, ".gguf")
		}
		slots[string(k)] = entry
	}
	// Always include known kinds in the report so consumers can see which
	// ones aren't configured.
	for _, k := range knownSlotKinds {
		if _, ok := slots[string(k)]; !ok {
			slots[string(k)] = map[string]any{"configured": false}
		}
	}
	version := g.version
	g.mu.RUnlock()
	resp := map[string]any{
		"service": "quenchforge",
		"version": version,
		"slots":   slots,
		"routes":  g.advertisedRoutes(),
	}
	writeJSON(w, http.StatusOK, resp)
}

// advertisedRoutes renders the mounted route table for GET /. Derived from
// the same table Start registers, so an operator (or install.sh, which
// tells them to curl this) reads the surface that actually exists.
func (g *Gateway) advertisedRoutes() []string {
	rts := g.routes()
	out := make([]string, 0, len(rts))
	for _, rt := range rts {
		line := rt.method + " " + rt.pattern
		if rt.note != "" {
			line += " " + rt.note
		}
		out = append(out, line)
	}
	return out
}

// handleHealth returns the gateway's overall status plus a per-slot
// breakdown: rolling-window latency and error rate for slots that are
// serving, and readiness for those that are not. Dual-placed "auto" kinds
// report their CPU twin under a separate "<kind>-cpu" key so the two
// instances' latency distributions stay legible. Always 200 while the
// gateway is reachable — consumers parse the JSON to decide whether to
// throttle or to stop retrying a lane. The opt-in QUENCHFORGE_AUTO_BACKOFF
// flag turns a critical ERROR RATE (only — never the latency ratio) into an
// actual 503 on the upstream proxy paths; /health itself never blocks.
//
// Readiness is the half this endpoint used to be blind to. It was built
// entirely from the latency tracker, which only knows about slots that have
// served traffic — so a route with no upstream (never started, or the slot
// died) produced no samples, no slots entry, and an overall "ok" while
// every request to it returned 503. Per-slot status now also carries:
//
//	unconfigured — the operator configured a model for this kind but no
//	               upstream is registered. Degrades the overall status.
//	unreachable  — an upstream was registered and proved dead; it has been
//	               deregistered and will be retried. Degrades the overall.
//	disabled     — no model configured for this kind. The route is mounted
//	               and 503s, which is the documented behaviour, so this does
//	               NOT degrade the overall status — but a consumer can still
//	               see the lane is unavailable and stop retrying it.
//
// Schema (the latency keys are unchanged; readiness keys are additive):
//
//	{
//	  "status": "degraded",
//	  "slots": {
//	    "embed": {
//	      "kind": "embed",
//	      "samples": 312,
//	      "p50_ms": 18.4,
//	      "p99_ms": 41.2,
//	      "error_rate": 0.0,
//	      "status": "ok",
//	      "window_secs": 60,
//	      "configured": true,
//	      "upstream": "http://127.0.0.1:11501"
//	    },
//	    "rerank": {
//	      "kind": "rerank",
//	      "samples": 0,
//	      "status": "unconfigured",
//	      "configured": false,
//	      "detail": "model \"bge-reranker-v2-m3\" is configured but no upstream is registered …"
//	    },
//	    ...
//	  },
//	  "auto_backoff_enabled": false
//	}
func (g *Gateway) handleHealth(w http.ResponseWriter, _ *http.Request) {
	snaps := g.latency.Snapshot()
	slots := make(map[string]SlotHealth, len(snaps)+len(knownSlotKinds))
	overall := StatusOK
	for _, kind := range knownSlotKinds {
		h := g.slotHealth(kind, snaps[kind])
		slots[string(kind)] = h
		overall = worstStatus(overall, h.Status)
		delete(snaps, kind)
	}
	// Anything left is a dual-placed CPU twin ("embed-cpu") — an instance
	// rather than a kind. It has samples, so it exists; the kind's own entry
	// above carries the readiness.
	for kind, snap := range snaps {
		slots[string(kind)] = SlotHealth{LatencySnapshot: snap, Configured: true}
		overall = worstStatus(overall, snap.Status)
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

// SlotHealth is one entry in /health's slots map: the rolling latency
// snapshot (same JSON keys as before, inlined) plus the readiness facts.
type SlotHealth struct {
	LatencySnapshot
	// Configured reports whether a usable upstream is registered for the
	// kind right now. False means every request to its route returns 503.
	Configured bool   `json:"configured"`
	Upstream   string `json:"upstream,omitempty"`
	Detail     string `json:"detail,omitempty"`
}

// slotHealth merges one kind's latency snapshot with its readiness. snap
// may be the zero value (no traffic in the window).
func (g *Gateway) slotHealth(kind SlotKind, snap LatencySnapshot) SlotHealth {
	if snap.Kind == "" {
		snap = LatencySnapshot{
			Kind:       kind,
			Status:     StatusOK,
			WindowSecs: int(latencyWindow / time.Second),
		}
	}
	g.mu.RLock()
	entry, registered := g.upstreams[kind]
	down, parked := g.downUpstreams[kind]
	g.mu.RUnlock()

	h := SlotHealth{LatencySnapshot: snap}
	switch {
	case registered && entry.proxy != nil:
		h.Configured = true
		h.Upstream = entry.url.String()
	case parked:
		h.Status = StatusUnreachable
		h.Detail = fmt.Sprintf("upstream %s stopped accepting connections %s ago (%s); retrying after %s",
			down.raw, time.Since(down.since).Truncate(time.Second), down.reason, upstreamRetryCooloff)
	case g.configuredModel(kind) != "":
		h.Status = StatusUnconfigured
		h.Detail = fmt.Sprintf("model %q is configured but no upstream is registered — "+
			"the slot is not running and every %s request returns 503",
			g.configuredModel(kind), kind)
	default:
		h.Status = StatusDisabled
		h.Detail = fmt.Sprintf("no model configured for %s; the route is mounted and returns 503", kind)
	}
	return h
}

// configuredModel is the model the operator asked this kind to serve.
// Empty means the kind was never requested, which is the difference between
// "disabled" and "unconfigured" in /health.
func (g *Gateway) configuredModel(kind SlotKind) string {
	switch kind {
	case KindChat:
		return g.cfg.DefaultModel
	case KindEmbed:
		return g.cfg.EmbedModel
	case KindCodeEmbed:
		return g.cfg.CodeEmbedModel
	case KindRerank:
		return g.cfg.RerankModel
	case KindWhisper:
		return g.cfg.WhisperModel
	case KindImageGen:
		return g.cfg.SDModel
	case KindTTS:
		return g.cfg.BarkModel
	}
	return ""
}

// worstStatus folds a per-slot status into the overall one. The overall
// vocabulary stays {ok, degraded, critical}: "unconfigured" and
// "unreachable" are faults and degrade it, "disabled" is an operator choice
// and does not (otherwise a chat-only deployment reads "degraded" forever
// and consumers learn to ignore the field).
func worstStatus(overall, s SlotStatus) SlotStatus {
	if statusRank(s) <= statusRank(overall) {
		return overall
	}
	if statusRank(s) >= statusRank(StatusCritical) {
		return StatusCritical
	}
	return StatusDegraded
}

func statusRank(s SlotStatus) int {
	switch s {
	case StatusCritical:
		return 2
	case StatusDegraded, StatusUnconfigured, StatusUnreachable:
		return 1
	default: // ok, disabled
		return 0
	}
}

// handleTags lists the GGUFs in the model directory in Ollama's tags shape.
//
// Ollama's contract is that a name here can be loaded on demand; quenchforge
// serves whatever its slots already have loaded and does not swap models per
// request. Each entry therefore carries `loaded`: true when a registered
// slot is serving that model, false when the file is merely cached. A
// request for a name with loaded=false is refused by /api/chat rather than
// silently answered by the loaded model.
func (g *Gateway) handleTags(w http.ResponseWriter, r *http.Request) {
	models, err := EnumerateModels(g.cfg.ModelsDir)
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, err.Error())
		return
	}
	loaded := g.loadedModels(r.Context())
	// Ollama returns: {"models": [{"name", "modified_at", "size", "digest", ...}]}
	out := make([]map[string]any, 0, len(models))
	for _, m := range models {
		out = append(out, map[string]any{
			"name":        m.Name,
			"model":       m.Name,
			"modified_at": m.ModifiedAt.Format(time.RFC3339),
			"size":        m.SizeBytes,
			"digest":      m.Digest,
			"loaded":      anyModelMatches(loaded, m.Name),
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": out})
}

// slotModelProbeTimeout bounds one /v1/models probe, and
// slotModelProbeRetry is how long a failed probe is cached before we ask
// again — a slot that was still loading its model when we first asked must
// not stay "unknown" for the life of the process.
const (
	slotModelProbeTimeout = 2 * time.Second
	slotModelProbeRetry   = 30 * time.Second
)

// servedModel returns the model name a slot reports as loaded, probed from
// the upstream's OpenAI /v1/models and cached until the kind is
// re-registered. Returns "" when the slot cannot be asked (still loading, or
// an upstream that does not implement the route); callers must read that as
// "no opinion", never as a mismatch.
func (g *Gateway) servedModel(ctx context.Context, kind SlotKind, entry upstreamEntry) string {
	g.mu.RLock()
	cached, ok := g.slotModels[kind]
	g.mu.RUnlock()
	if ok && (cached.model != "" || time.Since(cached.at) < slotModelProbeRetry) {
		return cached.model
	}
	model := probeUpstreamModel(ctx, entry.url)
	g.mu.Lock()
	// Only cache against the registration we probed: SetUpstream drops the
	// cache on re-registration, and a probe racing one must not resurrect a
	// name from the previous process.
	if cur, still := g.upstreams[kind]; still && cur.url != nil && entry.url != nil &&
		cur.url.String() == entry.url.String() {
		g.slotModels[kind] = probedModel{model: model, at: time.Now()}
	}
	g.mu.Unlock()
	return model
}

// loadedModels is the set of models the registered slots are serving.
func (g *Gateway) loadedModels(ctx context.Context) []string {
	// A slot that cannot be asked still serves the model it was launched
	// with, so its configured name stands in when the probe has no answer.
	configured := g.slotModelNames()
	var out []string
	for _, kind := range knownSlotKinds {
		entry, ok := g.lookupUpstream(kind)
		if !ok {
			continue
		}
		m := g.servedModel(ctx, kind, entry)
		if m == "" {
			m = configured[kind]
		}
		if m != "" {
			out = append(out, m)
		}
	}
	return out
}

// probeUpstreamModel asks a llama-server-style upstream which model it has
// loaded. Best-effort: any failure returns "".
func probeUpstreamModel(ctx context.Context, u *url.URL) string {
	if u == nil {
		return ""
	}
	ctx, cancel := context.WithTimeout(ctx, slotModelProbeTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		strings.TrimRight(u.String(), "/")+"/v1/models", nil)
	if err != nil {
		return ""
	}
	resp, err := translateHTTPClient.Do(req)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return ""
	}
	var body struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 64*1024)).Decode(&body); err != nil {
		return ""
	}
	if len(body.Data) == 0 {
		return ""
	}
	return body.Data[0].ID
}

// modelsMatch reports whether a requested model name refers to the model a
// slot has loaded. One model reaches the gateway under three spellings —
// Ollama-style "qwen2.5:7b-instruct-q4_k_m", the GGUF filename
// "qwen2.5-7b-instruct-q4_k_m.gguf", and shorthand "qwen2.5:7b" — so the
// comparison normalises separators and accepts a shorthand that is a
// component-boundary prefix of the loaded name. It does not accept an
// unrelated name; catching that is the point.
func modelsMatch(requested, served string) bool {
	rq, sv := normalizeModelName(requested), normalizeModelName(served)
	if rq == "" || sv == "" {
		return true // nothing to compare against — no opinion
	}
	if rq == sv {
		return true
	}
	return isModelPrefix(rq, sv) || isModelPrefix(sv, rq)
}

// anyModelMatches reports whether name is one of the served models.
func anyModelMatches(served []string, name string) bool {
	for _, s := range served {
		if s != "" && modelsMatch(name, s) {
			return true
		}
	}
	return false
}

// isModelPrefix reports whether short is a component-boundary prefix of
// long ("qwen2.5-7b" of "qwen2.5-7b-instruct-q4-k-m", but not "qwen2.5-7"
// of it).
func isModelPrefix(short, long string) bool {
	return len(short) < len(long) && strings.HasPrefix(long, short) && long[len(short)] == '-'
}

// normalizeModelName folds the spellings of one model onto a single key:
// lowercased, no directory prefix, no ".gguf" suffix, and ':' / '_' / ' '
// collapsed to '-'.
func normalizeModelName(name string) string {
	n := strings.ToLower(strings.TrimSpace(name))
	n = strings.TrimSuffix(n, ".gguf")
	if i := strings.LastIndex(n, "/"); i >= 0 {
		n = n[i+1:]
	}
	n = strings.NewReplacer(":", "-", "_", "-", " ", "-").Replace(n)
	return strings.Trim(n, "-")
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
// and the slot's error rate is critical (the family-B crash signature —
// SIGABRT → dead upstream → 5xx burst while AutoRespawn recovers), the
// handler returns 503 + Retry-After before forwarding, giving consumers a
// structured signal instead of a hang. Latency-ratio degradation is
// observability-only and never sheds.
func (g *Gateway) proxyHandler(kind SlotKind, rewriteTo string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		entry, ok := g.lookupUpstream(kind)
		if !ok {
			writeJSONError(w, http.StatusServiceUnavailable, g.unavailableReason(kind))
			return
		}
		if g.shouldBackoff(kind) {
			w.Header().Set("Retry-After", "2")
			writeJSONError(w, http.StatusServiceUnavailable,
				fmt.Sprintf("%s slot is shedding load (critical error rate) — back off (Retry-After: 2s)", kind))
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
// request to the tracker key `kind` (a slot kind, or an instance key like
// "embed-cpu"). Gated on cfg.AutoBackoffEnabled so the default behaviour is
// observability-only.
//
// The trigger is the ERROR RATE only — the signal that actually precedes and
// accompanies the family-B Metal crash (SIGABRT → dead upstream → 5xx burst
// while AutoRespawn brings the slot back). The p99/p50 latency ratio remains
// an observability status in /health but never sheds: it is workload-shape
// sensitive (one 26s GPU batch against millisecond singles reads as
// ratio≈1700 with zero failures — the 2026-07-08 false-critical incident),
// and the roadmap principle is "degrade to working, not 503".
func (g *Gateway) shouldBackoff(kind SlotKind) bool {
	if !g.cfg.AutoBackoffEnabled {
		return false
	}
	snap := g.latency.SnapshotKind(kind)
	on := snap.Samples >= statusMinSamples && snap.ErrorRate > statusCriticalErrorRate
	g.logBackoffTransition(kind, on, snap)
	return on
}

// logBackoffTransition emits one log line when a tracker key enters or
// leaves the shedding state. Per-request logging would flood under a storm;
// no logging at all is how the 2026-07-08 silent 503 storm cost hours to
// diagnose from the caller's side.
func (g *Gateway) logBackoffTransition(kind SlotKind, on bool, snap LatencySnapshot) {
	g.backoffMu.Lock()
	defer g.backoffMu.Unlock()
	if g.backoffOn == nil {
		g.backoffOn = make(map[SlotKind]bool)
	}
	if g.backoffOn[kind] == on {
		return
	}
	g.backoffOn[kind] = on
	if on {
		log.Printf("quenchforge: auto-backoff ON for %s (error_rate=%.2f over %d samples) — shedding with 503 + Retry-After",
			kind, snap.ErrorRate, snap.Samples)
	} else {
		log.Printf("quenchforge: auto-backoff OFF for %s (error_rate=%.2f over %d samples)",
			kind, snap.ErrorRate, snap.Samples)
	}
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
//   - Empty CodeEmbedModel  → KindEmbed (legacy single-slot behavior).
//   - model == CodeEmbedModel and a KindCodeEmbed upstream is registered →
//     KindCodeEmbed.
//   - model == CodeEmbedModel and no code-embed upstream → KindEmbed, demoted.
//   - Anything else → KindEmbed.
//
// demoted is true for the third case: the caller asked for the code model
// by name and quenchforge cannot serve it. It used to fall through to the
// general-text embedder — but vectors from a different model live in a
// different space, so the caller does not get "a working response", it gets
// silently unusable numbers that poison whatever index it writes them to.
// Callers must refuse the request instead.
func (g *Gateway) resolveEmbedKind(model string) (kind SlotKind, demoted bool) {
	if g.cfg.CodeEmbedModel == "" || model == "" {
		return KindEmbed, false
	}
	if model != g.cfg.CodeEmbedModel {
		return KindEmbed, false
	}
	if _, ok := g.lookupUpstream(KindCodeEmbed); !ok {
		return KindEmbed, true
	}
	return KindCodeEmbed, false
}

// codeEmbedUnavailable is the 503 body for a code-embed request the gateway
// refuses to answer from the general-text slot.
func (g *Gateway) codeEmbedUnavailable(model string) string {
	return fmt.Sprintf("no %s slot configured for model %q — refusing to answer from the %s slot, "+
		"whose vectors are from a different model in a different embedding space. "+
		"Set QUENCHFORGE_CODE_EMBED_MODEL and check `quenchforge doctor` for slot status.",
		KindCodeEmbed, model, KindEmbed)
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
		kind, demoted := g.resolveEmbedKind(probe.Model)
		if demoted {
			writeJSONError(w, http.StatusServiceUnavailable, g.codeEmbedUnavailable(probe.Model))
			return
		}
		batchN := countEmbedInputs(probe.Input, "")
		entry, onGPU, track, ok := g.routeEmbed(kind, batchN)
		if !ok {
			writeJSONError(w, http.StatusServiceUnavailable, g.unavailableReason(kind))
			return
		}
		// Re-attach the consumed body so the reverse-proxy can forward it.
		r.Body = io.NopCloser(bytes.NewReader(raw))
		r.ContentLength = int64(len(raw))
		// Backoff is evaluated against the INSTANCE this request routes to —
		// a struggling GPU instance must never shed traffic bound for the
		// healthy CPU twin (and vice versa).
		if g.shouldBackoff(track) {
			w.Header().Set("Retry-After", "2")
			writeJSONError(w, http.StatusServiceUnavailable,
				fmt.Sprintf("%s slot is shedding load (critical error rate) — back off (Retry-After: 2s)", track))
			return
		}
		serve := func() { g.serveAndTrack(track, entry.proxy, w, r) }
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

// modelMatchesBackground reports whether a request's `model` field names
// Config.BackgroundModel, accepting every spelling resolveModelPath (see
// cmd/quenchforge/main.go) accepts for the *configured* value plus the
// Ollama-style `:tag` suffix callers commonly send on the *request*:
//
//   - Exact match: model == bg.
//   - Both sides sans a ".gguf" extension match — bg can be configured
//     either as a bare name or a name+".gguf" (resolveModelPath tries
//     both), and a request can name either form too.
//   - The request carries an Ollama `:tag` suffix (e.g. ":latest") and
//     stripping it produces a match under either of the rules above —
//     but ONLY when bg itself has no colon. bg is allowed to legitimately
//     contain a colon (e.g. a Modelfile-style tag such as "qwen2.5:3b"
//     used as the GGUF's on-disk name); stripping the request's tag in
//     that case would wrongly collapse a colon-qualified request onto an
//     unrelated shorter model name, so the fallback does not apply.
//
// bg is compared verbatim in every branch — it is never itself stripped
// or otherwise canonicalized, so an operator's exact configured spelling
// is always one of the two things being compared, never a lossy derivative
// of it.
func modelMatchesBackground(model, bg string) bool {
	if model == "" || bg == "" {
		return false
	}
	if model == bg {
		return true
	}
	if strings.TrimSuffix(model, ".gguf") == strings.TrimSuffix(bg, ".gguf") {
		return true
	}
	if strings.Contains(bg, ":") {
		return false
	}
	i := strings.LastIndex(model, ":")
	if i < 0 {
		return false
	}
	stripped := model[:i]
	return stripped == bg || strings.TrimSuffix(stripped, ".gguf") == strings.TrimSuffix(bg, ".gguf")
}

// resolveChatKind picks the chat slot kind for an inbound request by
// matching the request's `model` field against Config.BackgroundModel
// (see modelMatchesBackground for the accepted spellings).
//
//   - Empty BackgroundModel, or no model in the request → always KindChat
//     (legacy single-slot behavior).
//   - model matches BackgroundModel → KindBackground.
//   - Anything else → KindChat.
//
// Unlike resolveEmbedKind, a match here does NOT check whether the
// background upstream is registered — that check, and the resulting 503,
// is the caller's responsibility. A caller that explicitly asked for the
// background model must not be silently redirected to the (different)
// chat model when the background slot isn't up.
func (g *Gateway) resolveChatKind(model string) SlotKind {
	if g.cfg.BackgroundModel == "" || model == "" {
		return KindChat
	}
	if !modelMatchesBackground(model, g.cfg.BackgroundModel) {
		return KindChat
	}
	return KindBackground
}

// maxChatPeekBodyBytes bounds the /v1/chat/completions body peek used to
// dispatch by model name once QUENCHFORGE_BACKGROUND_MODEL is set. Larger
// than maxRequestBodyBytes (8 MB, the Ollama-translation paths' cap)
// because OpenAI-native chat callers routinely attach long-context
// history or inline images that easily exceed 8 MB.
const maxChatPeekBodyBytes = 64 * 1024 * 1024 // 64 MB

// handleOpenAIChat is the OpenAI-native /v1/chat/completions entry point.
//
// When Config.BackgroundModel is unset (the common case), this is a
// byte-for-byte passthrough to the pre-dispatch behavior — gated(KindChat,
// proxyHandler(KindChat, "")) — with no body read at all, so unbounded
// streaming requests are unaffected.
//
// When BackgroundModel is set, it peeks the body's `model` field (capped
// at maxChatPeekBodyBytes) to dispatch between KindChat and KindBackground,
// then reverse-proxies the exact bytes read to the chosen upstream under
// GPU admission for that kind.
func (g *Gateway) handleOpenAIChat() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if g.cfg.BackgroundModel == "" {
			g.gated(KindChat, g.proxyHandler(KindChat, ""))(w, r)
			return
		}
		raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxChatPeekBodyBytes))
		if err != nil {
			writeJSONError(w, http.StatusBadRequest,
				fmt.Sprintf("read request body: %v", err))
			return
		}
		// Minimal probe — only `model` matters for routing; everything
		// else passes through unchanged. We re-attach the body below.
		var probe struct {
			Model string `json:"model"`
		}
		_ = json.Unmarshal(raw, &probe) // tolerate empty/invalid bodies; let upstream reject
		kind := g.resolveChatKind(probe.Model)
		entry, ok := g.lookupUpstream(kind)
		if !ok {
			writeJSONError(w, http.StatusServiceUnavailable, g.unavailableReason(kind))
			return
		}
		if g.shouldBackoff(kind) {
			w.Header().Set("Retry-After", "2")
			writeJSONError(w, http.StatusServiceUnavailable,
				fmt.Sprintf("%s slot is shedding load (critical error rate) — back off (Retry-After: 2s)", kind))
			return
		}
		// Re-attach the consumed body — exact bytes, correct length — so
		// the reverse-proxy forwards it unchanged.
		r.Body = io.NopCloser(bytes.NewReader(raw))
		r.ContentLength = int64(len(raw))
		g.gated(kind, func(w http.ResponseWriter, r *http.Request) {
			g.serveAndTrack(kind, entry.proxy, w, r)
		})(w, r)
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
