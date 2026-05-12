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
	cases := []struct {
		name    string
		cfg     Config
		wantSub string
	}{
		{"no port", Config{ListenAddr: "127.0.0.1", ModelsDir: "/tmp/m", MaxContext: 8192, MetalNCB: 2}, "must include a port"},
		{"empty addr", Config{ModelsDir: "/tmp/m", MaxContext: 8192, MetalNCB: 2}, "ListenAddr is empty"},
		{"empty models", Config{ListenAddr: "127.0.0.1:11434", MaxContext: 8192, MetalNCB: 2}, "ModelsDir is empty"},
		{"low ctx", Config{ListenAddr: "127.0.0.1:11434", ModelsDir: "/tmp/m", MaxContext: 100, MetalNCB: 2}, "below the 512"},
		{"low ncb", Config{ListenAddr: "127.0.0.1:11434", ModelsDir: "/tmp/m", MaxContext: 8192, MetalNCB: 0}, "MetalNCB"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cfg.Validate()
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
