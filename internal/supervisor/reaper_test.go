// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

//go:build darwin || linux

package supervisor

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"syscall"
	"testing"
	"time"
)

// decoyProcess starts a harmless long-lived process whose command line
// contains "quenchforge" — the shape of any operator shell, editor or second
// quenchforge instance that inherits a recycled PID. It runs in its own
// process group so a mis-aimed group signal from the reaper cannot reach the
// test runner.
func decoyProcess(t *testing.T) *exec.Cmd {
	t.Helper()
	cmd := exec.Command("/bin/sh", "-c", "while :; do sleep 1; done", "quenchforge-decoy")
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		t.Fatalf("start decoy: %v", err)
	}
	t.Cleanup(func() {
		_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
		_ = cmd.Wait()
	})
	// Give the shell a moment to appear in the process table.
	time.Sleep(100 * time.Millisecond)
	return cmd
}

func alive(pid int) bool {
	return syscall.Kill(pid, 0) == nil
}

func TestReapOrphans_DoesNotKillAProcessItCannotVerify(t *testing.T) {
	decoy := decoyProcess(t)
	pidDir := t.TempDir()
	// A pidfile from a crashed supervisor that recorded nothing but the PID:
	// the PID may since have been recycled, so the reaper cannot prove the
	// process is one of its children.
	if err := os.WriteFile(filepath.Join(pidDir, "chat.pid"),
		[]byte(strconv.Itoa(decoy.Process.Pid)+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	res := ReapOrphans(pidDir)
	if len(res) != 1 {
		t.Fatalf("ReapOrphans: %d results, want 1", len(res))
	}
	if res[0].Action != "skip" {
		t.Errorf("reaper action = %q, want skip — it cannot prove pid %d is its child",
			res[0].Action, decoy.Process.Pid)
	}
	time.Sleep(100 * time.Millisecond)
	if !alive(decoy.Process.Pid) {
		t.Fatalf("the reaper SIGKILLed pid %d, an unrelated process whose command line "+
			"merely contains \"quenchforge\"", decoy.Process.Pid)
	}
}

func TestReapOrphans_SkipsRecycledPID(t *testing.T) {
	decoy := decoyProcess(t)
	pid := decoy.Process.Pid
	_, execPath, err := processIdentity(pid)
	if err != nil {
		t.Fatalf("processIdentity(%d): %v", pid, err)
	}
	pidDir := t.TempDir()
	// Same PID, same executable — but the recorded child started earlier, so
	// this process is a different one wearing a recycled PID.
	if err := writePIDRecord(filepath.Join(pidDir, "chat.pid"), pidRecord{
		PID:   pid,
		Start: "Mon Jan  1 00:00:00 2001",
		Exec:  execPath,
	}); err != nil {
		t.Fatal(err)
	}

	res := ReapOrphans(pidDir)
	if len(res) != 1 || res[0].Action != "skip" {
		t.Errorf("ReapOrphans = %+v, want a single skip", res)
	}
	time.Sleep(100 * time.Millisecond)
	if !alive(pid) {
		t.Fatalf("the reaper killed pid %d, whose start time does not match the pidfile", pid)
	}
}

// TestReapOrphans_KillsAMatchingChild is the vacuity guard for the two skip
// tests above: with a fully matching identity the reaper must still fire.
func TestReapOrphans_KillsAMatchingChild(t *testing.T) {
	decoy := decoyProcess(t)
	pid := decoy.Process.Pid
	start, execPath, err := processIdentity(pid)
	if err != nil {
		t.Fatalf("processIdentity(%d): %v", pid, err)
	}
	pidDir := t.TempDir()
	if err := writePIDRecord(filepath.Join(pidDir, "chat.pid"), pidRecord{
		PID: pid, Start: start, Exec: execPath,
	}); err != nil {
		t.Fatal(err)
	}

	res := ReapOrphans(pidDir)
	if len(res) != 1 || res[0].Action != "killed" {
		t.Fatalf("ReapOrphans = %+v, want a single kill", res)
	}
	done := make(chan struct{})
	go func() {
		_ = decoy.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Errorf("pid %d matched the recorded identity but survived the reaper", pid)
	}
}
