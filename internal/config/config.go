// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

// Package config holds the runtime configuration for `quenchforge serve`.
//
// MVP-stage approach: stdlib-only struct binding from environment variables
// plus sensible defaults. No YAML parser dependency until we have a use case
// that justifies it. The `quenchforge` binary stays a single small artifact;
// operators who need richer config can layer a wrapper script.
//
// Config keys:
//
//	QUENCHFORGE_LISTEN_ADDR  — gateway bind address. Default 127.0.0.1:11434.
//	                            Matches Ollama's default port so existing
//	                            clients work without flags.
//	QUENCHFORGE_LLAMA_BIN    — path to the patched llama-server binary.
//	                            Empty = search PATH and the default Homebrew
//	                            install location.
//	QUENCHFORGE_MODELS_DIR   — GGUF model store. Default ~/.quenchforge/models.
//	                            Operators who run `quenchforge migrate-from-ollama`
//	                            get symlinks pointing into ~/.ollama/models.
//	QUENCHFORGE_LOG_DIR      — log directory. Default ~/Library/Logs/quenchforge.
//	QUENCHFORGE_PID_DIR      — orphan-reaper scratch dir. Default
//	                            ~/.config/quenchforge/pids.
//	QUENCHFORGE_DEFAULT_MODEL — fallback model when a request omits the field.
//	                            Default qwen2.5:7b-instruct-q4_k_m.
//	QUENCHFORGE_CHAT_PORT    — supervised chat-slot port. Default 11500.
//	QUENCHFORGE_EMBED_MODEL  — GGUF model for the embedding slot. Empty = no
//	                            embed slot is started; /api/embeddings 503s.
//	QUENCHFORGE_EMBED_PORT   — supervised embed-slot port. Default 11501.
//	QUENCHFORGE_CODE_EMBED_MODEL — GGUF model for the *code-tuned* embedding
//	                            slot. Empty = no code-embed slot; embed
//	                            requests fall through to the regular embed
//	                            slot. Routed by request-body model match in
//	                            the gateway — clients (e.g. semantic-code
//	                            search MCPs) get the code-tuned slot by
//	                            asking for this model name; general-text
//	                            clients are unaffected.
//	QUENCHFORGE_CODE_EMBED_PORT — supervised code-embed-slot port. Default
//	                            11506 (11503-11505 reserved for whisper/sd/bark).
//	QUENCHFORGE_RERANK_MODEL — GGUF reranker model (BGE-reranker etc.).
//	                            Empty = no rerank slot; /v1/rerank 503s.
//	QUENCHFORGE_RERANK_PORT  — supervised rerank-slot port. Default 11502.
//	QUENCHFORGE_WHISPER_MODEL — whisper.cpp ggml model path. Empty = no
//	                            whisper slot; /v1/audio/transcriptions 503s.
//	QUENCHFORGE_WHISPER_PORT — supervised whisper-server port. Default 11503.
//	QUENCHFORGE_WHISPER_GPU  — opt-in: try Metal on whisper.cpp. Default
//	                            "off" — whisper's Metal path has additional
//	                            AMD-Mac bugs beyond what our patch fixes
//	                            (transcription regresses to garbage tokens
//	                            even with the patch applied; root cause is
//	                            in whisper-specific Metal kernels). CPU
//	                            mode on a multi-core Xeon runs comfortably
//	                            faster than real-time anyway.
//	QUENCHFORGE_MAX_CONTEXT  — max KV-cache context in tokens. Default 8192.
//	QUENCHFORGE_METAL_N_CB   — Metal command-buffer count. Default 2 — set as
//	                            a launch env var on llama-server (issue surface
//	                            of the would-be third patch).
//	QUENCHFORGE_EMBED_UBATCH_SIZE  — physical ubatch (and batch) the embed and
//	                            code-embed slots use. Zero (default) means
//	                            fall back to MaxContext so single inputs ≤
//	                            8192 tokens fit one batch. Lowering this
//	                            (e.g. to 1024) shrinks per-call Metal
//	                            staging allocations on AMD discrete —
//	                            primary mitigation for the family-B
//	                            sustained-load crash. See
//	                            `patches/README.md` section 3.
//	QUENCHFORGE_EMBED_METAL_N_CB   — per-slot GGML_METAL_N_CB for the embed
//	                            and code-embed slots. Zero (default) falls
//	                            back to the global QUENCHFORGE_METAL_N_CB.
//	                            Set to 1 on AMD discrete to serialise Metal
//	                            command-buffer submission and let the
//	                            staging-buffer pool drain between calls.
//	QUENCHFORGE_RERANK_BATCH_SIZE  — physical batch (and ubatch) the rerank
//	                            slot uses. Zero (default) keeps llama.cpp's
//	                            512-token default. Raise this when the
//	                            reranker model + workload routinely produces
//	                            (query, doc) pairs above 512 tokens
//	                            (e.g. bge-reranker-v2-m3 with 1k-2k-token
//	                            chunks). Subject to the same family-B
//	                            constraints as the embed knobs on AMD.
//	QUENCHFORGE_RERANK_METAL_N_CB  — per-slot GGML_METAL_N_CB for the rerank
//	                            slot. Same semantics as EMBED_METAL_N_CB.
//	QUENCHFORGE_AUTO_BACKOFF — opt-in: gateway returns 503 + Retry-After when
//	                            a slot's rolling latency p99 is at the
//	                            "critical" threshold (impending crash).
//	                            Lets consumers throttle before the slot
//	                            SIGABRTs. Default "off".
//	QUENCHFORGE_TELEMETRY    — opt-in: "on" or "off". Default "off".
//	QUENCHFORGE_ADVERTISE_MDNS — opt-in: advertise `_quenchforge._tcp.local.`
//	                            via the system mDNSResponder. Default "off".
//	                            First-launch triggers the documented Sonoma+
//	                            "find devices on your local network" TCC prompt.
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Config is the in-memory runtime configuration.
type Config struct {
	// ListenAddr is the gateway's bind address (host:port).
	ListenAddr string

	// LlamaBin is the path to the patched llama-server binary. Empty means
	// look it up at start time via Resolve().
	LlamaBin string

	// ModelsDir is the root of the GGUF model cache.
	ModelsDir string

	// LogDir is where slot stdout/stderr land.
	LogDir string

	// PIDDir holds per-slot pidfiles for the orphan reaper.
	PIDDir string

	// DefaultModel is what the chat slot loads when a request omits the
	// `model` field. Should be a name resolvable under ModelsDir.
	DefaultModel string

	// ChatPort is where the supervised chat slot binds (127.0.0.1:ChatPort).
	// Default 11500. One above 11434 (default ListenAddr port) so the slot
	// is never a conflict with the gateway itself.
	ChatPort int

	// EmbedModel is the GGUF the embedding slot loads. Empty means the embed
	// slot is not started and /api/embeddings returns 503.
	EmbedModel string

	// EmbedPort is where the supervised embed slot binds. Default 11501.
	EmbedPort int

	// EmbedCPUPort is where the *CPU* embed instance binds when the embed
	// kind is placed in "auto" mode (dual-placed). Default 11511. Only used
	// when PlaceEmbed == "auto"; otherwise no CPU instance is launched and
	// this port stays free.
	EmbedCPUPort int

	// CodeEmbedModel is the GGUF the *code-tuned* embedding slot loads.
	// Empty means no code-embed slot is started; embed requests for any
	// model name route to the regular embed slot. When set, the gateway
	// dispatches embed requests whose `model` field equals CodeEmbedModel
	// to this slot instead. Lets a single quenchforge process serve a
	// general-text embedder (for KB / RAG) and a code-tuned embedder (for
	// semantic-code-search MCPs) on the same gateway port.
	CodeEmbedModel string

	// CodeEmbedPort is where the supervised code-embed slot binds.
	// Default 11506 — 11503-11505 are reserved for whisper/sd/bark.
	CodeEmbedPort int

	// CodeEmbedCPUPort is where the *CPU* code-embed instance binds when the
	// code-embed kind is placed in "auto" mode (dual-placed). Default 11516.
	// Only used when PlaceCodeEmbed == "auto".
	CodeEmbedCPUPort int

	// RerankModel is the GGUF reranker model. Empty disables /v1/rerank.
	RerankModel string

	// RerankPort is where the supervised rerank slot binds. Default 11502.
	RerankPort int

	// WhisperModel is the whisper.cpp ggml model path (NOT a name under
	// ModelsDir — whisper models use a different filename convention and
	// don't all live in the same directory). Empty disables the whisper
	// slot and /v1/audio/transcriptions returns 503.
	WhisperModel string

	// WhisperPort is where the supervised whisper-server binds. Default 11503.
	WhisperPort int

	// WhisperGPU controls whether whisper-server attempts Metal acceleration.
	// Default false on darwin because whisper.cpp's Metal path has
	// additional AMD-Mac bugs beyond what the patch fixes. CPU on a Xeon
	// runs comfortably faster than real-time anyway.
	WhisperGPU bool

	// SDModel is the path to the stable-diffusion.cpp ggml/safetensors
	// model. Empty disables the image-gen slot and /v1/images/generations
	// returns 503.
	SDModel string

	// SDPort is where the supervised sd-server binds. Default 11504.
	SDPort int

	// BarkModel is the path to the bark.cpp ggml model. Empty disables the
	// TTS slot and /v1/audio/speech returns 503.
	BarkModel string

	// BarkPort is where the supervised bark server binds. Default 11505.
	BarkPort int

	// MaxContext is the KV-cache token cap exposed to clients.
	MaxContext int

	// MetalNCB is the Metal command-buffer count passed via GGML_METAL_N_CB
	// when spawning llama-server. 2 is the tested-good value for Vega II
	// (issue #15228 thread).
	MetalNCB int

	// EmbedUbatchSize, when non-zero, overrides the embed and code-embed
	// slots' --batch-size / --ubatch-size. Zero falls back to MaxContext
	// (preserves the v0.5.0 contextplus single-batch behaviour). On AMD
	// discrete, lowering this to 512-2048 caps per-call Metal staging
	// allocations and prevents the family-B sustained-load crash.
	EmbedUbatchSize int

	// EmbedMetalNCB, when non-zero, overrides GGML_METAL_N_CB for the
	// embed and code-embed slots only. Zero falls back to MetalNCB. On
	// AMD discrete, setting this to 1 serialises Metal command-buffer
	// submission and lets the staging-buffer pool drain.
	EmbedMetalNCB int

	// RerankBatchSize, when non-zero, sets --batch-size and --ubatch-size
	// on the rerank slot. Zero (default) keeps llama.cpp's 512-token
	// internal default — too small for modern rerankers fed > 510-token
	// (query, doc) pairs. Operators with bge-reranker-v2-m3 or similar
	// raise this; same family-B caveats as EmbedUbatchSize.
	RerankBatchSize int

	// RerankMetalNCB, when non-zero, overrides GGML_METAL_N_CB for the
	// rerank slot. Same semantics as EmbedMetalNCB.
	RerankMetalNCB int

	// AutoBackoffEnabled is the opt-in flag for the gateway's
	// 503+Retry-After response when a slot reaches the "critical"
	// latency threshold. Default false — observability via /health
	// works without this flag; only the back-pressure response is gated.
	AutoBackoffEnabled bool

	// GovernorEnabled turns on the GPU-pressure governor: adaptive admission
	// concurrency that reserves GPU headroom for the macOS display compositor
	// (WindowServer) while a screen is being driven, preventing sustained
	// inference from starving it into a kernel-watchdog panic. Default true;
	// it is a no-op on headless hosts (full throughput) so server users are
	// unaffected.
	GovernorEnabled bool

	// GPUConcurrencyMax is the admission ceiling when the host is headless or
	// the display is asleep — full throughput.
	GPUConcurrencyMax int

	// GPUConcurrencyDisplayActive is the admission ceiling while a display is
	// being driven. Serialized (1) by default so the duty-cycle gaps are clean.
	GPUConcurrencyDisplayActive int

	// GPUDutyCycleDisplayActive is the target GPU busy fraction (0<d<=1) while
	// a display is being driven. Below 1, the gateway inserts proportional GPU
	// idle gaps after each request so the compositor gets time slices. This is
	// the lever that actually prevents WindowServer starvation — concurrency
	// capping alone does not (sustained gapless GPU work starves it at any
	// concurrency).
	GPUDutyCycleDisplayActive float64

	// GovernorMaxCooldownMS caps the per-request duty-cycle idle hold so one
	// long generation can't stall the queue for seconds.
	GovernorMaxCooldownMS int

	// GovernorIntervalMS is how often the governor re-reads host pressure.
	GovernorIntervalMS int

	// Place{Chat,Embed,CodeEmbed,Rerank} override the per-kind device
	// placement ("gpu" | "cpu" | "auto"). Empty = use the hardware-adaptive
	// default from internal/placement (AMD-discrete: chat=cpu, others=gpu;
	// non-AMD: all gpu). "auto" dual-places and routes per request by batch.
	PlaceChat      string
	PlaceEmbed     string
	PlaceCodeEmbed string
	PlaceRerank    string

	// AutoBatchThreshold is the input-count boundary the gateway uses to
	// route "auto"-placed embedding requests: a request whose input count is
	// > threshold (bulk/throughput) routes to the GPU instance; <= threshold
	// (single/small, latency-bound) routes to the CPU instance. Default 1, so
	// single-input requests go CPU and any batch goes GPU. Treated as 1 when
	// below 1.
	AutoBatchThreshold int

	// TelemetryEnabled is opt-in. Wired in v0.2 once the consent screen ships.
	TelemetryEnabled bool

	// AdvertiseMDNS controls Bonjour advertisement of `_quenchforge._tcp.local.`
	// via the system mDNSResponder. Off by default — flipping to true triggers
	// the documented Sonoma+ TCC prompt for local-network access.
	AdvertiseMDNS bool
}

// Default returns a Config populated with defaults that don't require any
// environment variables to be set.
func Default() (Config, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return Config{}, fmt.Errorf("config: resolve $HOME: %w", err)
	}
	return Config{
		ListenAddr:         "127.0.0.1:11434",
		LlamaBin:           "", // resolved lazily
		ModelsDir:          filepath.Join(home, ".quenchforge", "models"),
		LogDir:             filepath.Join(home, "Library", "Logs", "quenchforge"),
		PIDDir:             filepath.Join(home, ".config", "quenchforge", "pids"),
		DefaultModel:       "qwen2.5:7b-instruct-q4_k_m",
		ChatPort:           11500,
		EmbedModel:         "", // opt-in
		EmbedPort:          11501,
		EmbedCPUPort:       11511,
		CodeEmbedModel:     "", // opt-in
		CodeEmbedPort:      11506,
		CodeEmbedCPUPort:   11516,
		RerankModel:        "", // opt-in
		RerankPort:         11502,
		WhisperModel:       "", // opt-in
		WhisperPort:        11503,
		WhisperGPU:         false, // opt-in; see config docstring
		SDModel:            "",    // opt-in
		SDPort:             11504,
		BarkModel:          "", // opt-in
		BarkPort:           11505,
		MaxContext:         8192,
		MetalNCB:           2,
		EmbedUbatchSize:    0, // 0 = inherit MaxContext (preserves v0.5.0 behaviour)
		EmbedMetalNCB:      0, // 0 = inherit MetalNCB
		RerankBatchSize:    0, // 0 = use llama.cpp's 512-token internal default
		RerankMetalNCB:     0, // 0 = inherit MetalNCB
		AutoBackoffEnabled: false,

		GovernorEnabled:             true,
		GPUConcurrencyMax:           6,
		GPUConcurrencyDisplayActive: 1,
		GPUDutyCycleDisplayActive:   0.5,
		GovernorMaxCooldownMS:       250,
		GovernorIntervalMS:          3000,

		AutoBatchThreshold: 1,
	}, nil
}

// Load returns Default() overlaid with any QUENCHFORGE_* env-var overrides.
// Unknown env vars are ignored — operators get typo-detection from
// `quenchforge doctor`, not from a strict-parse refusal at boot.
func Load() (Config, error) {
	cfg, err := Default()
	if err != nil {
		return Config{}, err
	}

	cfg.ListenAddr = envOr("QUENCHFORGE_LISTEN_ADDR", cfg.ListenAddr)
	cfg.LlamaBin = envOr("QUENCHFORGE_LLAMA_BIN", cfg.LlamaBin)
	cfg.ModelsDir = envOr("QUENCHFORGE_MODELS_DIR", cfg.ModelsDir)
	cfg.LogDir = envOr("QUENCHFORGE_LOG_DIR", cfg.LogDir)
	cfg.PIDDir = envOr("QUENCHFORGE_PID_DIR", cfg.PIDDir)
	cfg.DefaultModel = envOr("QUENCHFORGE_DEFAULT_MODEL", cfg.DefaultModel)
	cfg.ChatPort = envIntOr("QUENCHFORGE_CHAT_PORT", cfg.ChatPort)
	cfg.EmbedModel = envOr("QUENCHFORGE_EMBED_MODEL", cfg.EmbedModel)
	cfg.EmbedPort = envIntOr("QUENCHFORGE_EMBED_PORT", cfg.EmbedPort)
	cfg.EmbedCPUPort = envIntOr("QUENCHFORGE_EMBED_CPU_PORT", cfg.EmbedCPUPort)
	cfg.CodeEmbedModel = envOr("QUENCHFORGE_CODE_EMBED_MODEL", cfg.CodeEmbedModel)
	cfg.CodeEmbedPort = envIntOr("QUENCHFORGE_CODE_EMBED_PORT", cfg.CodeEmbedPort)
	cfg.CodeEmbedCPUPort = envIntOr("QUENCHFORGE_CODE_EMBED_CPU_PORT", cfg.CodeEmbedCPUPort)
	cfg.RerankModel = envOr("QUENCHFORGE_RERANK_MODEL", cfg.RerankModel)
	cfg.RerankPort = envIntOr("QUENCHFORGE_RERANK_PORT", cfg.RerankPort)
	cfg.WhisperModel = envOr("QUENCHFORGE_WHISPER_MODEL", cfg.WhisperModel)
	cfg.WhisperPort = envIntOr("QUENCHFORGE_WHISPER_PORT", cfg.WhisperPort)
	cfg.WhisperGPU = envBoolOr("QUENCHFORGE_WHISPER_GPU", cfg.WhisperGPU)
	cfg.SDModel = envOr("QUENCHFORGE_SD_MODEL", cfg.SDModel)
	cfg.SDPort = envIntOr("QUENCHFORGE_SD_PORT", cfg.SDPort)
	cfg.BarkModel = envOr("QUENCHFORGE_BARK_MODEL", cfg.BarkModel)
	cfg.BarkPort = envIntOr("QUENCHFORGE_BARK_PORT", cfg.BarkPort)
	cfg.MaxContext = envIntOr("QUENCHFORGE_MAX_CONTEXT", cfg.MaxContext)
	cfg.MetalNCB = envIntOr("QUENCHFORGE_METAL_N_CB", cfg.MetalNCB)
	cfg.EmbedUbatchSize = envIntOr("QUENCHFORGE_EMBED_UBATCH_SIZE", cfg.EmbedUbatchSize)
	cfg.EmbedMetalNCB = envIntOr("QUENCHFORGE_EMBED_METAL_N_CB", cfg.EmbedMetalNCB)
	cfg.RerankBatchSize = envIntOr("QUENCHFORGE_RERANK_BATCH_SIZE", cfg.RerankBatchSize)
	cfg.RerankMetalNCB = envIntOr("QUENCHFORGE_RERANK_METAL_N_CB", cfg.RerankMetalNCB)
	cfg.AutoBackoffEnabled = envBoolOr("QUENCHFORGE_AUTO_BACKOFF", cfg.AutoBackoffEnabled)
	cfg.GovernorEnabled = envBoolOr("QUENCHFORGE_GOVERNOR", cfg.GovernorEnabled)
	cfg.GPUConcurrencyMax = envIntOr("QUENCHFORGE_GPU_CONCURRENCY_MAX", cfg.GPUConcurrencyMax)
	cfg.GPUConcurrencyDisplayActive = envIntOr("QUENCHFORGE_GPU_CONCURRENCY_DISPLAY_ACTIVE", cfg.GPUConcurrencyDisplayActive)
	cfg.GPUDutyCycleDisplayActive = envFloatOr("QUENCHFORGE_GPU_DUTY_DISPLAY_ACTIVE", cfg.GPUDutyCycleDisplayActive)
	cfg.GovernorMaxCooldownMS = envIntOr("QUENCHFORGE_GOVERNOR_MAX_COOLDOWN_MS", cfg.GovernorMaxCooldownMS)
	cfg.GovernorIntervalMS = envIntOr("QUENCHFORGE_GOVERNOR_INTERVAL_MS", cfg.GovernorIntervalMS)
	cfg.PlaceChat = envOr("QUENCHFORGE_PLACE_CHAT", cfg.PlaceChat)
	cfg.PlaceEmbed = envOr("QUENCHFORGE_PLACE_EMBED", cfg.PlaceEmbed)
	cfg.PlaceCodeEmbed = envOr("QUENCHFORGE_PLACE_CODE_EMBED", cfg.PlaceCodeEmbed)
	cfg.PlaceRerank = envOr("QUENCHFORGE_PLACE_RERANK", cfg.PlaceRerank)
	cfg.AutoBatchThreshold = envIntOr("QUENCHFORGE_AUTO_BATCH_THRESHOLD", cfg.AutoBatchThreshold)
	cfg.TelemetryEnabled = envBoolOr("QUENCHFORGE_TELEMETRY", false)
	cfg.AdvertiseMDNS = envBoolOr("QUENCHFORGE_ADVERTISE_MDNS", false)

	return cfg, cfg.Validate()
}

// Validate checks invariants that callers care about — non-empty paths, a
// listen address that includes a port, a sane context size. Returns the
// first violation as an error.
func (c Config) Validate() error {
	if c.ListenAddr == "" {
		return fmt.Errorf("config: ListenAddr is empty")
	}
	if !strings.Contains(c.ListenAddr, ":") {
		return fmt.Errorf("config: ListenAddr %q must include a port", c.ListenAddr)
	}
	if c.ModelsDir == "" {
		return fmt.Errorf("config: ModelsDir is empty")
	}
	if c.MaxContext < 512 {
		return fmt.Errorf("config: MaxContext %d is below the 512-token floor", c.MaxContext)
	}
	if c.MetalNCB < 1 {
		return fmt.Errorf("config: MetalNCB %d must be >= 1", c.MetalNCB)
	}
	if c.EmbedUbatchSize < 0 {
		return fmt.Errorf("config: EmbedUbatchSize %d must be >= 0 (0 = inherit MaxContext)", c.EmbedUbatchSize)
	}
	if c.EmbedMetalNCB < 0 {
		return fmt.Errorf("config: EmbedMetalNCB %d must be >= 0 (0 = inherit MetalNCB)", c.EmbedMetalNCB)
	}
	if c.RerankBatchSize < 0 {
		return fmt.Errorf("config: RerankBatchSize %d must be >= 0 (0 = llama.cpp default)", c.RerankBatchSize)
	}
	if c.RerankMetalNCB < 0 {
		return fmt.Errorf("config: RerankMetalNCB %d must be >= 0 (0 = inherit MetalNCB)", c.RerankMetalNCB)
	}
	if c.ChatPort < 1 || c.ChatPort > 65535 {
		return fmt.Errorf("config: ChatPort %d outside valid TCP port range", c.ChatPort)
	}
	if c.EmbedPort < 1 || c.EmbedPort > 65535 {
		return fmt.Errorf("config: EmbedPort %d outside valid TCP port range", c.EmbedPort)
	}
	for _, p := range []struct {
		name string
		v    int
	}{
		{"EmbedCPUPort", c.EmbedCPUPort},
		{"CodeEmbedPort", c.CodeEmbedPort},
		{"CodeEmbedCPUPort", c.CodeEmbedCPUPort},
		{"RerankPort", c.RerankPort},
		{"WhisperPort", c.WhisperPort},
		{"SDPort", c.SDPort},
		{"BarkPort", c.BarkPort},
	} {
		if p.v < 1 || p.v > 65535 {
			return fmt.Errorf("config: %s %d outside valid TCP port range", p.name, p.v)
		}
	}
	// No two slot ports can collide.
	ports := map[string]int{
		"ChatPort":         c.ChatPort,
		"EmbedPort":        c.EmbedPort,
		"EmbedCPUPort":     c.EmbedCPUPort,
		"CodeEmbedPort":    c.CodeEmbedPort,
		"CodeEmbedCPUPort": c.CodeEmbedCPUPort,
		"RerankPort":       c.RerankPort,
		"WhisperPort":      c.WhisperPort,
		"SDPort":           c.SDPort,
		"BarkPort":         c.BarkPort,
	}
	seen := map[int]string{}
	for name, p := range ports {
		if other, ok := seen[p]; ok {
			return fmt.Errorf("config: %s and %s both set to %d — must differ", other, name, p)
		}
		seen[p] = name
	}
	return nil
}

// EnsureDirs creates ModelsDir, LogDir, and PIDDir if they don't exist.
// Idempotent. Returns the first creation error.
func (c Config) EnsureDirs() error {
	for _, p := range []string{c.ModelsDir, c.LogDir, c.PIDDir} {
		if err := os.MkdirAll(p, 0o755); err != nil {
			return fmt.Errorf("config: mkdir %q: %w", p, err)
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func envOr(name, fallback string) string {
	if v, ok := os.LookupEnv(name); ok && v != "" {
		return v
	}
	return fallback
}

func envIntOr(name string, fallback int) int {
	v, ok := os.LookupEnv(name)
	if !ok || v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}
	return n
}

func envFloatOr(name string, fallback float64) float64 {
	v, ok := os.LookupEnv(name)
	if !ok || v == "" {
		return fallback
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return fallback
	}
	return f
}

func envBoolOr(name string, fallback bool) bool {
	v, ok := os.LookupEnv(name)
	if !ok {
		return fallback
	}
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	}
	return fallback
}
