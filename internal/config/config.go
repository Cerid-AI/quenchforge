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
//	QUENCHFORGE_MAX_CONTEXT  — max KV-cache context in tokens. Default 8192.
//	QUENCHFORGE_METAL_N_CB   — Metal command-buffer count. Default 2 — set as
//	                            a launch env var on llama-server (issue surface
//	                            of the would-be third patch).
//	QUENCHFORGE_TELEMETRY    — opt-in: "on" or "off". Default "off".
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

	// DefaultModel is what the gateway falls back to when a request omits
	// the `model` field. Should be a name resolvable under ModelsDir.
	DefaultModel string

	// MaxContext is the KV-cache token cap exposed to clients.
	MaxContext int

	// MetalNCB is the Metal command-buffer count passed via GGML_METAL_N_CB
	// when spawning llama-server. 2 is the tested-good value for Vega II
	// (issue #15228 thread).
	MetalNCB int

	// TelemetryEnabled is opt-in. Wired in v0.2 once the consent screen ships.
	TelemetryEnabled bool
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
	cfg.MaxContext = envIntOr("QUENCHFORGE_MAX_CONTEXT", cfg.MaxContext)
	cfg.MetalNCB = envIntOr("QUENCHFORGE_METAL_N_CB", cfg.MetalNCB)
	cfg.TelemetryEnabled = envBoolOr("QUENCHFORGE_TELEMETRY", false)

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
