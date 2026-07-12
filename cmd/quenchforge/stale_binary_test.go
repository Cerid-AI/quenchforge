// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestStaleBinaryStatusPassOnSamePath(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "quenchforge")
	if err := os.WriteFile(bin, []byte("x"), 0o755); err != nil {
		t.Fatal(err)
	}
	got := staleBinaryStatus(bin, bin, false)
	if !strings.HasPrefix(got, "PASS") {
		t.Errorf("same path: got %q, want PASS", got)
	}
}

func TestStaleBinaryStatusPassThroughSymlink(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "Cellar-quenchforge")
	if err := os.WriteFile(target, []byte("x"), 0o755); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "opt-quenchforge")
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
	// The running server resolved to the Cellar target; doctor was
	// invoked via the opt symlink. Same install — must be PASS.
	got := staleBinaryStatus(target, link, false)
	if !strings.HasPrefix(got, "PASS") {
		t.Errorf("symlinked same install: got %q, want PASS", got)
	}
}

func TestStaleBinaryStatusCriticalWhenRunningPathGone(t *testing.T) {
	dir := t.TempDir()
	current := filepath.Join(dir, "quenchforge-new")
	if err := os.WriteFile(current, []byte("x"), 0o755); err != nil {
		t.Fatal(err)
	}
	gone := filepath.Join(dir, "quenchforge-removed")
	got := staleBinaryStatus(gone, current, false)
	if !strings.HasPrefix(got, "CRITICAL") {
		t.Errorf("removed running path: got %q, want CRITICAL", got)
	}
	if !strings.Contains(got, "kickstart") {
		t.Errorf("CRITICAL message must carry the restart remediation, got %q", got)
	}
}

func TestStaleBinaryStatusWarnOnVersionSkew(t *testing.T) {
	dir := t.TempDir()
	oldBin := filepath.Join(dir, "quenchforge-0.9.0")
	newBin := filepath.Join(dir, "quenchforge-0.10.0")
	for _, p := range []string{oldBin, newBin} {
		if err := os.WriteFile(p, []byte("x"), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	got := staleBinaryStatus(oldBin, newBin, false)
	if !strings.HasPrefix(got, "WARN") {
		t.Errorf("version skew: got %q, want WARN", got)
	}
	if !strings.Contains(got, "kickstart") {
		t.Errorf("WARN message must carry the restart remediation, got %q", got)
	}
}
