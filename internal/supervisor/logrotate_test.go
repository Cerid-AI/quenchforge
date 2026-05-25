// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package supervisor

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRotatingWriter_RotatesAtMaxBytes(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "chat.log")

	// 1 KiB max, 3 backups for fast iteration.
	w, err := NewRotatingWriter(path, 1024, 3)
	if err != nil {
		t.Fatalf("NewRotatingWriter: %v", err)
	}
	defer w.Close()

	// 600 bytes — under the threshold; no rotation expected.
	if _, err := w.Write(bytes.Repeat([]byte("a"), 600)); err != nil {
		t.Fatalf("write 600: %v", err)
	}
	if fileExists(t, path+".1") {
		t.Errorf("rotation fired below threshold")
	}

	// Another 600 bytes — crosses the 1024 threshold. Rotation expected.
	if _, err := w.Write(bytes.Repeat([]byte("b"), 600)); err != nil {
		t.Fatalf("write 600 more: %v", err)
	}
	if !fileExists(t, path+".1") {
		t.Errorf(".1 backup missing after threshold crossed")
	}

	primary, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read primary: %v", err)
	}
	if len(primary) > 1024 {
		t.Errorf("primary file size %d > maxBytes 1024", len(primary))
	}
}

func TestRotatingWriter_KeepsAtMostNBackups(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "embed.log")

	w, err := NewRotatingWriter(path, 100, 2) // 100 B max, 2 backups
	if err != nil {
		t.Fatalf("NewRotatingWriter: %v", err)
	}
	defer w.Close()

	for i := 0; i < 10; i++ {
		if _, err := w.Write([]byte(strings.Repeat("x", 150))); err != nil {
			t.Fatalf("write %d: %v", i, err)
		}
	}

	// .1 and .2 should exist; .3 must NOT.
	if !fileExists(t, path+".1") {
		t.Errorf(".1 missing")
	}
	if !fileExists(t, path+".2") {
		t.Errorf(".2 missing")
	}
	if fileExists(t, path+".3") {
		t.Errorf(".3 exists but backups=2")
	}
}

func TestRotatingWriter_ZeroMaxBytesMeansNoRotation(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "x.log")

	w, err := NewRotatingWriter(path, 0, 5)
	if err != nil {
		t.Fatalf("NewRotatingWriter: %v", err)
	}
	defer w.Close()

	for i := 0; i < 100; i++ {
		if _, err := w.Write([]byte(strings.Repeat("z", 1024))); err != nil {
			t.Fatalf("write %d: %v", i, err)
		}
	}
	if fileExists(t, path+".1") {
		t.Errorf("rotation fired with maxBytes=0 (disabled)")
	}
}

func fileExists(t *testing.T, p string) bool {
	t.Helper()
	_, err := os.Stat(p)
	if err == nil {
		return true
	}
	if os.IsNotExist(err) {
		return false
	}
	t.Fatalf("stat %s: %v", p, err)
	return false
}
