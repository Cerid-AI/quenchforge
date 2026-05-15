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
		ListenAddr:   "127.0.0.1:11434",
		LlamaBin:     "", // resolved lazily
		ModelsDir:    filepath.Join(home, ".quenchforge", "models"),
		LogDir:       filepath.Join(home, "Library", "Logs", "quenchforge"),
		PIDDir:       filepath.Join(home, ".config", "quenchforge", "pids"),
		DefaultModel: "qwen2.5:7b-instruct-q4_k_m",
		ChatPort:     11500,
		EmbedModel:     "", // opt-in
		EmbedPort:      11501,
		CodeEmbedModel: "", // opt-in
		CodeEmbedPort:  11506,
		RerankModel:    "", // opt-in
		RerankPort:   11502,
		WhisperModel: "", // opt-in
		WhisperPort:  11503,
		WhisperGPU:   false, // opt-in; see config docstring
		SDModel:      "",    // opt-in
		SDPort:       11504,
		BarkModel:    "", // opt-in
		BarkPort:     11505,
		MaxContext:   8192,
		MetalNCB:     2,
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
	cfg.CodeEmbedModel = envOr("QUENCHFORGE_CODE_EMBED_MODEL", cfg.CodeEmbedModel)
	cfg.CodeEmbedPort = envIntOr("QUENCHFORGE_CODE_EMBED_PORT", cfg.CodeEmbedPort)
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
		{"CodeEmbedPort", c.CodeEmbedPort},
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
		"ChatPort":      c.ChatPort,
		"EmbedPort":     c.EmbedPort,
		"CodeEmbedPort": c.CodeEmbedPort,
		"RerankPort":    c.RerankPort,
		"WhisperPort":   c.WhisperPort,
		"SDPort":        c.SDPort,
		"BarkPort":      c.BarkPort,
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
