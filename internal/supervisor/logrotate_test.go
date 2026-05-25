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

// TestRotatingWriter_RotateFailureLeavesWriterUsable guards against the
// regression where rotateLocked nil-ed w.f and then aborted on a rename
// failure, leaving the next Write() to dereference a nil file pointer
// and panic in production. The rotator MUST keep the writer in a usable
// state across any rotation failure mode.
//
// Reproduction: seed the backup chain so the shift loop trips an
// os.Rename failure. On macOS, renaming any file to the path of a
// non-empty directory fails with EEXIST/ENOTEMPTY. We place a real
// .2 backup file and a non-empty directory at .3 so the very first
// shift iteration (i=2: .2 → .3) errors out, leaving w.f nil on the
// pre-fix code path and triggering the panic on the next Write.
func TestRotatingWriter_RotateFailureLeavesWriterUsable(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "chat.log")

	// .2 is a real file the shift loop will try to move to .3.
	if err := os.WriteFile(path+".2", []byte("old-backup"), 0o644); err != nil {
		t.Fatalf("seed .2 backup: %v", err)
	}
	// .3 is a non-empty directory — the rename of .2 → .3 will fail
	// because os.Rename cannot replace a non-empty directory.
	blocker := path + ".3"
	if err := os.Mkdir(blocker, 0o755); err != nil {
		t.Fatalf("seed .3 blocker dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(blocker, "occupant"), []byte("x"), 0o644); err != nil {
		t.Fatalf("seed .3 blocker contents: %v", err)
	}

	w, err := NewRotatingWriter(path, 1024, 3)
	if err != nil {
		t.Fatalf("NewRotatingWriter: %v", err)
	}
	defer w.Close()

	// Fill primary close to the threshold without triggering rotation.
	if _, err := w.Write(bytes.Repeat([]byte("a"), 600)); err != nil {
		t.Fatalf("write 600: %v", err)
	}

	// Cross the threshold — rotation will fire and the .2 → .3 rename
	// will fail because .3 is a non-empty directory. The Write must
	// return the error rather than panic.
	_, rotErr := w.Write(bytes.Repeat([]byte("b"), 600))
	if rotErr == nil {
		t.Fatalf("expected rotation error from blocked rename, got nil")
	}

	// The contract: subsequent writes must not panic the process and
	// must succeed (the writer recovers to a usable state, even if
	// rotation is still wedged). defer/recover catches the nil-deref
	// panic case so a regression surfaces as a t.Fatalf instead of
	// crashing the test runner.
	var recoveryWriteN int
	var recoveryWriteErr error
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("subsequent Write panicked after rotate failure: %v", r)
			}
		}()
		recoveryWriteN, recoveryWriteErr = w.Write([]byte("c"))
	}()
	if recoveryWriteErr != nil {
		t.Fatalf("post-rotate-failure Write returned err=%v (writer not usable)", recoveryWriteErr)
	}
	if recoveryWriteN != 1 {
		t.Fatalf("post-rotate-failure Write n=%d want 1", recoveryWriteN)
	}

	// Recovery state must include the byte we just wrote — either still
	// in the recovered primary file, or already shifted into a rotated
	// backup if the next rotation succeeded.
	primary, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read primary after recovery: %v", err)
	}
	if !bytes.Contains(primary, []byte("c")) && !bytes.Contains(readAll(t, path+".1"), []byte("c")) {
		t.Errorf("post-recovery byte 'c' not found in primary or .1 (writer silently dropped data)")
	}
}

func readAll(t *testing.T, p string) []byte {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		return nil
	}
	return b
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
