// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDefaultIsValid(t *testing.T) {
	cfg, err := Default()
	if err != nil {
		t.Fatalf("Default: %v", err)
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Default config failed Validate: %v", err)
	}
	if cfg.ListenAddr != "127.0.0.1:11434" {
		t.Errorf("ListenAddr = %q, want 127.0.0.1:11434", cfg.ListenAddr)
	}
	if cfg.MetalNCB != 2 {
		t.Errorf("MetalNCB = %d, want 2", cfg.MetalNCB)
	}
	if cfg.MaxContext != 8192 {
		t.Errorf("MaxContext = %d, want 8192", cfg.MaxContext)
	}
}

func TestLoadAppliesEnvOverrides(t *testing.T) {
	t.Setenv("QUENCHFORGE_LISTEN_ADDR", "0.0.0.0:8080")
	t.Setenv("QUENCHFORGE_MAX_CONTEXT", "16384")
	t.Setenv("QUENCHFORGE_METAL_N_CB", "4")
	t.Setenv("QUENCHFORGE_TELEMETRY", "on")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.ListenAddr != "0.0.0.0:8080" {
		t.Errorf("ListenAddr = %q, want 0.0.0.0:8080", cfg.ListenAddr)
	}
	if cfg.MaxContext != 16384 {
		t.Errorf("MaxContext = %d, want 16384", cfg.MaxContext)
	}
	if cfg.MetalNCB != 4 {
		t.Errorf("MetalNCB = %d, want 4", cfg.MetalNCB)
	}
	if !cfg.TelemetryEnabled {
		t.Error("TelemetryEnabled = false, want true (env said 'on')")
	}
}

func TestValidateRejectsBadConfigs(t *testing.T) {
	// Helper to build a config that's valid in every field except one — so
	// each case isolates the field under test.
	base := func() Config {
		return Config{
			ListenAddr:  "127.0.0.1:11434",
			ModelsDir:   "/tmp/m",
			MaxContext:  8192,
			MetalNCB:    2,
			ChatPort:    11500,
			EmbedPort:   11501,
			RerankPort:  11502,
			WhisperPort: 11503,
			SDPort:      11504,
			BarkPort:    11505,
		}
	}
	cases := []struct {
		name    string
		mutate  func(*Config)
		wantSub string
	}{
		{"no port", func(c *Config) { c.ListenAddr = "127.0.0.1" }, "must include a port"},
		{"empty addr", func(c *Config) { c.ListenAddr = "" }, "ListenAddr is empty"},
		{"empty models", func(c *Config) { c.ModelsDir = "" }, "ModelsDir is empty"},
		{"low ctx", func(c *Config) { c.MaxContext = 100 }, "below the 512"},
		{"low ncb", func(c *Config) { c.MetalNCB = 0 }, "MetalNCB"},
		{"chat port 0", func(c *Config) { c.ChatPort = 0 }, "ChatPort"},
		{"chat port out of range", func(c *Config) { c.ChatPort = 70000 }, "ChatPort"},
		{"embed port 0", func(c *Config) { c.EmbedPort = 0 }, "EmbedPort"},
		{"rerank port 0", func(c *Config) { c.RerankPort = 0 }, "RerankPort"},
		{"whisper port 0", func(c *Config) { c.WhisperPort = 0 }, "WhisperPort"},
		{"same chat embed port", func(c *Config) { c.EmbedPort = c.ChatPort }, "must differ"},
		{"same chat rerank port", func(c *Config) { c.RerankPort = c.ChatPort }, "must differ"},
		{"same embed whisper port", func(c *Config) { c.WhisperPort = c.EmbedPort }, "must differ"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := base()
			tc.mutate(&cfg)
			err := cfg.Validate()
			if err == nil {
				t.Fatal("Validate succeeded, want error")
			}
			if !strings.Contains(err.Error(), tc.wantSub) {
				t.Errorf("Validate error = %q, want substring %q", err, tc.wantSub)
			}
		})
	}
}

func TestEnsureDirsCreatesAll(t *testing.T) {
	tmp := t.TempDir()
	cfg := Config{
		ListenAddr:   "127.0.0.1:11434",
		ModelsDir:    filepath.Join(tmp, "models"),
		LogDir:       filepath.Join(tmp, "logs"),
		PIDDir:       filepath.Join(tmp, "pids"),
		DefaultModel: "x",
		MaxContext:   8192,
		MetalNCB:     2,
		ChatPort:     11500,
		EmbedPort:    11501,
		RerankPort:   11502,
		WhisperPort:  11503,
		SDPort:       11504,
		BarkPort:     11505,
	}
	if err := cfg.EnsureDirs(); err != nil {
		t.Fatalf("EnsureDirs: %v", err)
	}
	for _, p := range []string{cfg.ModelsDir, cfg.LogDir, cfg.PIDDir} {
		if st, err := os.Stat(p); err != nil || !st.IsDir() {
			t.Errorf("expected directory at %q, err=%v stat=%v", p, err, st)
		}
	}
	// Idempotent
	if err := cfg.EnsureDirs(); err != nil {
		t.Fatalf("EnsureDirs second call: %v", err)
	}
}

func TestEnvBoolHelpers(t *testing.T) {
	cases := []struct {
		val  string
		want bool
	}{
		{"1", true}, {"true", true}, {"yes", true}, {"on", true},
		{"0", false}, {"false", false}, {"no", false}, {"off", false},
	}
	for _, tc := range cases {
		t.Setenv("X_TEST_FLAG", tc.val)
		if got := envBoolOr("X_TEST_FLAG", !tc.want); got != tc.want {
			t.Errorf("envBoolOr(%q) = %v, want %v", tc.val, got, tc.want)
		}
	}
}
