// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cerid-ai/quenchforge/internal/config"
)

// TestSlotExecutablesIncludesTheConfiguredLlamaBinary guards the evidence the
// orphan reaper's legacy-pidfile path depends on: hand it an empty list and
// the grace is silently inert, which is the upgrade hazard it exists to fix.
func TestSlotExecutablesIncludesTheConfiguredLlamaBinary(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "llama-server")
	if err := os.WriteFile(bin, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	got := slotExecutables(config.Config{LlamaBin: bin})
	for _, p := range got {
		if p == bin {
			return
		}
	}
	t.Errorf("slotExecutables = %v, want it to contain the configured llama-server %q — "+
		"without it the reaper has no evidence for a pidfile written by an older build", got, bin)
}
