// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

// Package supervisor manages the lifecycle of child `llama-server` (and
// future whisper-server) processes that do the actual inference work.
//
// Responsibilities:
//
//  1. Spawn child processes with a clean environment, log redirection,
//     and explicit GGML_METAL_* defaults derived from the hardware profile.
//  2. Track each child as a Slot with a pidfile in ~/.config/quenchforge/pids/
//     so an orphan reaper can clean up survivors if the supervisor itself
//     crashes (kill -9, panic, OS upgrade reboot, etc).
//  3. Run children in their own process group so a single `kill -PGID`
//     cleanly tears down hung children plus any subprocesses they spawned.
//  4. Stream stdout/stderr to per-slot log files and to a ring buffer the
//     gateway can expose for the doctor command.
//
// MVP-stage scope: single chat slot. The Slot abstraction is designed
// to support N slots (chat + embed + rerank + whisper) but the supervisor
// only wires the chat slot for v0.1. v0.2 lights up the remaining slots.
package supervisor

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// Slot is a single supervised llama-protocol process. The zero value is not
// valid; construct via NewSlot.
type Slot struct {
	// Name is a short identifier ("chat", "embed", "rerank", "whisper").
	// Used as the pidfile filename and the log file prefix.
	Name string

	// BinPath is the absolute path to the binary (typically llama-server).
	BinPath string

	// Args are the arguments handed to the binary. The supervisor adds
	// nothing implicit — callers control the full command line.
	Args []string

	// Env are extra environment variables layered onto the parent's env.
	// Use this for GGML_METAL_N_CB, MTL_DEVICE_INDEX, etc.
	Env []string

	// LogDir is where stdout/stderr land. One file per slot:
	//   ${LogDir}/${Name}.log
	LogDir string

	// PIDDir is where the orphan-reaper pidfile lives:
	//   ${PIDDir}/${Name}.pid
	PIDDir string

	// internal state — only valid after Start
	mu      sync.Mutex
	cmd     *exec.Cmd
	logFile *os.File
	pidPath string
	started time.Time
}

// NewSlot returns an unstarted Slot. The caller is expected to set BinPath,
// Args, Env, LogDir, and PIDDir before calling Start.
func NewSlot(name string) *Slot {
	return &Slot{Name: name}
}

// Start spawns the process and writes its pidfile. Idempotent: a second
// call while the slot is running returns nil. Returns an error if BinPath
// is empty or the binary fails to launch.
func (s *Slot) Start(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cmd != nil && s.cmd.Process != nil && s.cmd.ProcessState == nil {
		return nil // already running
	}
	if s.Name == "" {
		return errors.New("supervisor: Slot.Name is empty")
	}
	if s.BinPath == "" {
		return errors.New("supervisor: Slot.BinPath is empty")
	}
	if s.LogDir == "" || s.PIDDir == "" {
		return errors.New("supervisor: Slot.LogDir and Slot.PIDDir must be set")
	}

	if err := os.MkdirAll(s.LogDir, 0o755); err != nil {
		return fmt.Errorf("supervisor: mkdir log dir: %w", err)
	}
	if err := os.MkdirAll(s.PIDDir, 0o755); err != nil {
		return fmt.Errorf("supervisor: mkdir pid dir: %w", err)
	}

	logPath := filepath.Join(s.LogDir, s.Name+".log")
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("supervisor: open log: %w", err)
	}

	cmd := exec.CommandContext(ctx, s.BinPath, s.Args...)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	cmd.Env = append(os.Environ(), s.Env...)
	cmd.SysProcAttr = newProcAttr() // platform-specific: process group on unix

	if err := cmd.Start(); err != nil {
		_ = logFile.Close()
		return fmt.Errorf("supervisor: start %s: %w", s.Name, err)
	}

	s.cmd = cmd
	s.logFile = logFile
	s.started = time.Now()
	s.pidPath = filepath.Join(s.PIDDir, s.Name+".pid")
	if err := writePIDFile(s.pidPath, cmd.Process.Pid); err != nil {
		// PID-file write failure is non-fatal — orphan reaper can't clean
		// up this one if we crash, but the slot itself still works.
		// Surfacing via a returned error would refuse to start over a
		// hygiene issue. Log instead.
		fmt.Fprintf(os.Stderr, "quenchforge: warn: write pidfile %q: %v\n", s.pidPath, err)
	}
	return nil
}

// Stop sends SIGTERM, waits up to grace, then SIGKILLs and reaps. Returns
// the process's wait error (often *exec.ExitError) so callers can decide
// whether the exit was clean.
func (s *Slot) Stop(grace time.Duration) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cmd == nil || s.cmd.Process == nil {
		return nil
	}
	defer s.cleanupLocked()

	// Try graceful first
	if err := signalGroup(s.cmd.Process.Pid, syscall.SIGTERM); err != nil {
		return fmt.Errorf("supervisor: SIGTERM %s: %w", s.Name, err)
	}
	done := make(chan error, 1)
	go func() { done <- s.cmd.Wait() }()
	select {
	case err := <-done:
		return err
	case <-time.After(grace):
		_ = signalGroup(s.cmd.Process.Pid, syscall.SIGKILL)
		return <-done
	}
}

// Running returns true if the child is still up.
func (s *Slot) Running() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.cmd != nil && s.cmd.Process != nil && s.cmd.ProcessState == nil
}

// PID returns the child's PID, or 0 if not running.
func (s *Slot) PID() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cmd == nil || s.cmd.Process == nil {
		return 0
	}
	return s.cmd.Process.Pid
}

// Uptime returns how long the child has been running. Zero if not started.
func (s *Slot) Uptime() time.Duration {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.started.IsZero() {
		return 0
	}
	return time.Since(s.started)
}

func (s *Slot) cleanupLocked() {
	if s.logFile != nil {
		_ = s.logFile.Close()
		s.logFile = nil
	}
	if s.pidPath != "" {
		_ = os.Remove(s.pidPath)
		s.pidPath = ""
	}
}

// ---------------------------------------------------------------------------
// Orphan reaper
// ---------------------------------------------------------------------------

// ReapOrphans walks pidDir, reads each pidfile, and SIGKILLs the recorded PID
// if it's still running and has our `quenchforge` ancestor signature. Always
// removes the pidfile.
//
// Called at startup so a previous supervisor crash doesn't leave dangling
// llama-server children chewing GPU memory.
//
// We're deliberately permissive about who we kill: if a PID in our pidfile
// has been reused by an unrelated process, we still skip it because we
// double-check the command line includes "llama-server" or "whisper-server"
// before sending the signal.
func ReapOrphans(pidDir string) []ReapResult {
	results := make([]ReapResult, 0)
	entries, err := os.ReadDir(pidDir)
	if err != nil {
		return results
	}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".pid") {
			continue
		}
		full := filepath.Join(pidDir, e.Name())
		pid, err := readPIDFile(full)
		if err != nil {
			results = append(results, ReapResult{
				File:   e.Name(),
				Action: "skip",
				Note:   fmt.Sprintf("unreadable: %v", err),
			})
			_ = os.Remove(full)
			continue
		}

		res := ReapResult{File: e.Name(), PID: pid}
		if !pidLooksLikeOurChild(pid) {
			res.Action = "skip"
			res.Note = "pid not running or not a quenchforge child"
		} else {
			if err := signalGroup(pid, syscall.SIGKILL); err == nil {
				res.Action = "killed"
			} else {
				res.Action = "skip"
				res.Note = err.Error()
			}
		}
		_ = os.Remove(full)
		results = append(results, res)
	}
	return results
}

// ReapResult is a one-line audit record from ReapOrphans.
type ReapResult struct {
	File   string
	PID    int
	Action string // "killed" | "skip"
	Note   string
}

// pidLooksLikeOurChild returns true when /proc-style introspection (or ps
// shell-out on darwin) reports the PID's command line includes a name we
// recognize. Used by the reaper to avoid blasting unrelated processes.
func pidLooksLikeOurChild(pid int) bool {
	if pid <= 1 {
		return false
	}
	cmdline, err := commandLineForPID(pid)
	if err != nil || cmdline == "" {
		return false
	}
	lower := strings.ToLower(cmdline)
	return strings.Contains(lower, "llama-server") ||
		strings.Contains(lower, "whisper-server") ||
		strings.Contains(lower, "quenchforge")
}

// ---------------------------------------------------------------------------
// pidfile helpers
// ---------------------------------------------------------------------------

func writePIDFile(path string, pid int) error {
	return os.WriteFile(path, []byte(strconv.Itoa(pid)+"\n"), 0o644)
}

func readPIDFile(path string) (int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	s := strings.TrimSpace(string(data))
	n, err := strconv.Atoi(s)
	if err != nil {
		return 0, fmt.Errorf("pidfile %q: %w", path, err)
	}
	return n, nil
}

// AppendToLog is a convenience for tests that want to write to the slot's
// log file without spawning a child. Returns the number of bytes written.
func AppendToLog(logDir, name string, r io.Reader) (int64, error) {
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		return 0, err
	}
	f, err := os.OpenFile(filepath.Join(logDir, name+".log"),
		os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	return io.Copy(f, r)
}
