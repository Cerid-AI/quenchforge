// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package supervisor

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestStartRejectsEmptyName(t *testing.T) {
	s := &Slot{BinPath: "/bin/sh", LogDir: t.TempDir(), PIDDir: t.TempDir()}
	if err := s.Start(context.Background()); err == nil || !strings.Contains(err.Error(), "Name is empty") {
		t.Fatalf("Start: got err=%v, want 'Name is empty'", err)
	}
}

func TestStartRejectsEmptyBinPath(t *testing.T) {
	s := NewSlot("chat")
	s.LogDir = t.TempDir()
	s.PIDDir = t.TempDir()
	if err := s.Start(context.Background()); err == nil || !strings.Contains(err.Error(), "BinPath is empty") {
		t.Fatalf("Start: got err=%v, want 'BinPath is empty'", err)
	}
}

func TestStartRejectsMissingDirs(t *testing.T) {
	s := NewSlot("chat")
	s.BinPath = "/bin/sh"
	if err := s.Start(context.Background()); err == nil || !strings.Contains(err.Error(), "LogDir and Slot.PIDDir") {
		t.Fatalf("Start: got err=%v, want 'LogDir and Slot.PIDDir'", err)
	}
}

func TestStartStopRoundTrip(t *testing.T) {
	if _, err := os.Stat("/bin/sleep"); err != nil {
		t.Skip("/bin/sleep not present")
	}
	logDir := t.TempDir()
	pidDir := t.TempDir()
	s := NewSlot("test")
	s.BinPath = "/bin/sleep"
	s.Args = []string{"30"}
	s.LogDir = logDir
	s.PIDDir = pidDir

	if err := s.Start(context.Background()); err != nil {
		t.Fatalf("Start: %v", err)
	}
	if !s.Running() {
		t.Fatal("Running() = false after Start")
	}
	if s.PID() == 0 {
		t.Fatal("PID() = 0 after Start")
	}
	if up := s.Uptime(); up <= 0 {
		t.Errorf("Uptime() = %v, want > 0", up)
	}

	// pidfile written
	pidPath := filepath.Join(pidDir, "test.pid")
	if _, err := os.Stat(pidPath); err != nil {
		t.Fatalf("pidfile missing: %v", err)
	}

	if err := s.Stop(2 * time.Second); err == nil {
		// /bin/sleep terminated by SIGTERM returns a non-nil error
		// (signal: terminated). Logging only — exact semantics are fine.
		t.Log("Stop returned nil — process exited cleanly")
	}
	if s.Running() {
		t.Error("Running() = true after Stop")
	}
	if _, err := os.Stat(pidPath); !os.IsNotExist(err) {
		t.Errorf("pidfile should be removed after Stop, got err=%v", err)
	}
}

func TestStopOnUnstartedSlotIsNoOp(t *testing.T) {
	s := NewSlot("test")
	if err := s.Stop(time.Second); err != nil {
		t.Errorf("Stop on unstarted slot: %v, want nil", err)
	}
	if s.Running() {
		t.Error("Running() = true on unstarted slot")
	}
}

func TestPIDFileRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.pid")
	if err := writePIDFile(path, 12345); err != nil {
		t.Fatalf("writePIDFile: %v", err)
	}
	pid, err := readPIDFile(path)
	if err != nil {
		t.Fatalf("readPIDFile: %v", err)
	}
	if pid != 12345 {
		t.Errorf("readPIDFile: got %d, want 12345", pid)
	}
}

func TestReadPIDFileBadContent(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.pid")
	if err := os.WriteFile(path, []byte("not a number"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := readPIDFile(path); err == nil {
		t.Error("readPIDFile: want error on non-numeric content")
	}
}

func TestReapOrphansHandlesMissingDir(t *testing.T) {
	// Reaping a non-existent dir should return an empty list, not panic.
	res := ReapOrphans("/nonexistent/quenchforge/pids")
	if len(res) != 0 {
		t.Errorf("ReapOrphans on missing dir returned %d results, want 0", len(res))
	}
}

func TestReapOrphansHandlesStaleNonMatchingPID(t *testing.T) {
	pidDir := t.TempDir()
	// Use PID 1 (init) — not a quenchforge child, must be skipped.
	if err := writePIDFile(filepath.Join(pidDir, "stale.pid"), 1); err != nil {
		t.Fatal(err)
	}
	res := ReapOrphans(pidDir)
	if len(res) != 1 {
		t.Fatalf("ReapOrphans: %d results, want 1", len(res))
	}
	if res[0].Action != "skip" {
		t.Errorf("ReapOrphans action = %q, want skip", res[0].Action)
	}
	// pidfile removed regardless
	if _, err := os.Stat(filepath.Join(pidDir, "stale.pid")); !os.IsNotExist(err) {
		t.Errorf("stale pidfile not removed after reap, err=%v", err)
	}
}

func TestReapOrphansHandlesUnreadablePID(t *testing.T) {
	pidDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(pidDir, "bad.pid"), []byte("garbage"), 0o644); err != nil {
		t.Fatal(err)
	}
	res := ReapOrphans(pidDir)
	if len(res) != 1 || res[0].Action != "skip" {
		t.Errorf("ReapOrphans bad pidfile: %+v", res)
	}
	if _, err := os.Stat(filepath.Join(pidDir, "bad.pid")); !os.IsNotExist(err) {
		t.Errorf("unreadable pidfile not removed after reap")
	}
}
