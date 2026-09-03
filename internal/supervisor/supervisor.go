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
//  5. Reap every child on exit via an unconditional watcher (no zombies,
//     whatever the RestartPolicy) and auto-respawn when policy allows.
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

// RestartPolicy controls whether the supervisor brings the slot back
// after a non-zero exit (typically a Metal SIGABRT on AMD discrete
// embed/rerank slots — the family-B sustained-load crash leaves the
// slot dead and the gateway 503s every subsequent request until a
// manual `launchctl kickstart`. With auto-respawn, the slot returns
// within 30s and the gateway recovers without operator intervention).
type RestartPolicy int

const (
	// PolicyNone disables auto-respawn. The slot exits and stays
	// exited; callers see the death via Slot.Running() / PID().
	// This is the default (zero value) for backward compatibility.
	PolicyNone RestartPolicy = iota

	// PolicyExpBackoff retries Start after 2s, 4s, 8s on each
	// successive crash within a rolling 60-second window. After 3
	// attempts inside the window the slot cools off for
	// crashStormCooloff (default 5m) and then probes again with a
	// fresh window — a supervised slot never stays dead permanently.
	// A jetsam kill-storm on a memory-pressured host passes; a slot
	// that gave up permanently turned the 2026-07-11 transient crunch
	// into a days-long dead chat slot. Resets to fast backoff when
	// crashes stop clustering.
	PolicyExpBackoff
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

	// RestartPolicy controls whether the supervisor auto-respawns the
	// slot on non-zero exit. Default PolicyNone (no respawn).
	RestartPolicy RestartPolicy

	// MaxLogBytes bounds the per-slot log file. When a write would
	// push the file past this threshold, the rotator closes the
	// current file, renames it to ${Name}.log.1 (shifting older
	// backups by one), and reopens a fresh ${Name}.log.
	// Zero (the default) disables rotation — writes append forever,
	// preserving prior unbounded-log behaviour for callers that
	// haven't opted in.
	MaxLogBytes int64

	// LogBackups is the number of rotated backups to retain (.1 …
	// .LogBackups). Ignored when MaxLogBytes is zero.
	LogBackups int

	// internal state — only valid after Start
	mu          sync.Mutex
	cmd         *exec.Cmd
	logFile     io.WriteCloser
	pidPath     string
	started     time.Time
	waitCh      chan error  // watcher publishes the child's Wait result; closed after reap
	stopping    bool        // Stop() in progress — watcher must not respawn
	exited      bool        // watcher reaped the current cmd (avoids racing on cmd.ProcessState)
	respawnMu   sync.Mutex  // serialises respawn attempts
	respawnHist []time.Time // crash timestamps within last 60s
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
	if s.cmd != nil && s.cmd.Process != nil && !s.exited {
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
	logFile, err := NewRotatingWriter(logPath, s.MaxLogBytes, s.LogBackups)
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
	// Record the child's identity alongside its PID. Without it the reaper
	// cannot tell our child from whatever inherits the PID after a crash, and
	// it refuses to signal what it cannot identify.
	rec := pidRecord{PID: cmd.Process.Pid}
	if start, execPath, idErr := processIdentity(rec.PID); idErr == nil {
		rec.Start, rec.Exec = start, execPath
	} else {
		fmt.Fprintf(os.Stderr, "quenchforge: warn: identify %s child pid %d: %v "+
			"(orphan reaping disabled for this slot)\n", s.Name, rec.PID, idErr)
	}
	if err := writePIDRecord(s.pidPath, rec); err != nil {
		// PID-file write failure is non-fatal — orphan reaper can't clean
		// up this one if we crash, but the slot itself still works.
		// Surfacing via a returned error would refuse to start over a
		// hygiene issue. Log instead.
		fmt.Fprintf(os.Stderr, "quenchforge: warn: write pidfile %q: %v\n", s.pidPath, err)
	}

	// Every slot gets a watcher that Wait()s the child — reaping is
	// unconditional. A PolicyNone slot whose child dies (jetsam SIGKILL,
	// Metal SIGABRT) must not linger as a zombie: the 2026-07-11 incident
	// left a dead chat llama-server unreaped for days because only
	// respawning slots were ever Wait()ed. The watcher also owns the
	// respawn decision when policy allows.
	s.stopping = false
	s.exited = false
	s.waitCh = make(chan error, 1)
	go s.watch(ctx, cmd, s.waitCh)
	return nil
}

// crashStormCooloff is how long a slot that exceeded the crash-storm cap
// (3 respawns in a rolling 60-second window) waits before probing again.
// respawnBackoffUnit scales the fast-path exponential backoff (2·unit,
// 4·unit, 8·unit). Package vars so tests can shrink them.
var (
	crashStormCooloff  = 5 * time.Minute
	respawnBackoffUnit = time.Second
)

// watch reaps the supervised process when it exits, publishes the Wait
// result to waitCh (consumed by Stop), and applies the RestartPolicy.
// Exactly one watcher runs per spawned process; a respawn starts a new
// process and with it a new watcher.
func (s *Slot) watch(ctx context.Context, cmd *exec.Cmd, waitCh chan<- error) {
	err := cmd.Wait()
	waitCh <- err
	close(waitCh)

	s.mu.Lock()
	stopping := s.stopping
	if s.cmd == cmd {
		s.exited = true
		if !stopping {
			// The child died on its own — release its log handle and
			// pidfile now rather than leaking them until the next
			// Start/Stop.
			s.cleanupLocked()
		}
	}
	s.mu.Unlock()

	// Stop() owns shutdown-path cleanup; a cancelled ctx means the
	// supervisor itself is going down; a clean exit (code 0) is operator
	// intent. None of the three respawns.
	if stopping || ctx.Err() != nil || err == nil {
		return
	}
	if s.RestartPolicy == PolicyNone {
		fmt.Fprintf(os.Stderr,
			"quenchforge: slot %s exited (%v); no restart policy — slot stays down\n",
			s.Name, err)
		return
	}
	s.respawnAfterCrash(ctx, err)
}

// respawnAfterCrash applies exponential backoff (2s, 4s, 8s) within a
// rolling 60-second window. Exceeding 3 attempts in the window means a
// crash storm (permanently-wedged GPU, jetsam kill-storm): rather than
// leaving the slot dead forever, cool off for crashStormCooloff and
// probe again with a fresh window.
func (s *Slot) respawnAfterCrash(ctx context.Context, waitErr error) {
	s.respawnMu.Lock()

	// Trim crash history older than 60s.
	now := time.Now()
	cutoff := now.Add(-60 * time.Second)
	keep := s.respawnHist[:0]
	for _, t := range s.respawnHist {
		if t.After(cutoff) {
			keep = append(keep, t)
		}
	}
	s.respawnHist = keep

	var delay time.Duration
	if len(s.respawnHist) >= 3 {
		delay = crashStormCooloff
		s.respawnHist = s.respawnHist[:0]
		fmt.Fprintf(os.Stderr,
			"quenchforge: slot %s crashed more than 3 times in 60s — cooling off %s "+
				"before the next attempt. Last error: %v\n",
			s.Name, delay, waitErr)
	} else {
		// 2·unit on first crash this window, 4·unit on second, 8·unit
		// on third (unit is 1s in production).
		delay = time.Duration(2<<len(s.respawnHist)) * respawnBackoffUnit
		s.respawnHist = append(s.respawnHist, now)
		fmt.Fprintf(os.Stderr,
			"quenchforge: slot %s exited (%v); respawn in %s (%d/3 in window)\n",
			s.Name, waitErr, delay, len(s.respawnHist))
	}
	s.respawnMu.Unlock()

	select {
	case <-ctx.Done():
		return
	case <-time.After(delay):
	}

	// Reset state under the slot's mutex so the next Start() sees a
	// clean slate. Skip if an operator started a Stop during the wait.
	s.mu.Lock()
	if s.stopping {
		s.mu.Unlock()
		return
	}
	s.cmd = nil
	s.logFile = nil
	s.mu.Unlock()

	if err := s.Start(ctx); err != nil {
		fmt.Fprintf(os.Stderr,
			"quenchforge: slot %s respawn failed: %v\n", s.Name, err)
	}
}

// Stop sends SIGTERM to the child's process group, waits up to grace for
// the watcher to reap it, then SIGKILLs. Returns the child's Wait error
// (often *exec.ExitError) so callers can decide whether the exit was
// clean. The watcher owns Wait(); Stop consumes its result via waitCh —
// never double-Waiting and never signalling a child that already exited
// (the source of the "SIGTERM: operation not permitted" shutdown noise
// when slots had died earlier in the run).
func (s *Slot) Stop(grace time.Duration) error {
	s.mu.Lock()
	if s.cmd == nil || s.cmd.Process == nil {
		s.mu.Unlock()
		return nil
	}
	s.stopping = true
	cmd := s.cmd
	waitCh := s.waitCh
	pid := cmd.Process.Pid
	exited := s.exited
	s.mu.Unlock()

	var err error
	switch {
	case exited:
		// Already reaped by the watcher; collect the buffered result
		// (a closed, drained channel yields nil).
		if waitCh != nil {
			err = <-waitCh
		}
	case waitCh == nil:
		// No watcher exists (hand-constructed Slot); reap directly so a
		// nil channel can't block Stop forever.
		_ = signalGroup(pid, syscall.SIGTERM)
		err = cmd.Wait()
	default:
		// Signal errors (ESRCH on an already-gone group) are not fatal:
		// the watcher's result below is the ground truth.
		_ = signalGroup(pid, syscall.SIGTERM)
		select {
		case err = <-waitCh:
		case <-time.After(grace):
			_ = signalGroup(pid, syscall.SIGKILL)
			select {
			case err = <-waitCh:
			case <-time.After(2 * time.Second):
				// A hung child must not block supervisor exit.
				err = fmt.Errorf("supervisor: %s (pid %d) did not exit after SIGKILL",
					s.Name, pid)
			}
		}
	}

	s.mu.Lock()
	s.cleanupLocked()
	s.mu.Unlock()
	return err
}

// Running returns true if the child is still up.
func (s *Slot) Running() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.cmd != nil && s.cmd.Process != nil && !s.exited
}

// PID returns the child's PID, or 0 if not running. A reaped child
// reports 0 rather than its stale PID so status surfaces (doctor,
// startup banners) never point operators at a dead process.
func (s *Slot) PID() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cmd == nil || s.cmd.Process == nil || s.exited {
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

// ReapOrphans walks pidDir, reads each pidfile, and SIGKILLs the recorded
// process group when — and only when — the running process is provably the
// child we spawned. Always removes the pidfile.
//
// Called at startup so a previous supervisor crash doesn't leave dangling
// llama-server children chewing GPU memory.
//
// Proof is the (start time, executable path) pair recorded at spawn time. It
// used to be a substring test over the command line, which matched any
// process whose argv mentioned "quenchforge" — an operator shell, an editor,
// a second instance — and the signal goes to the whole process group. macOS
// recycles PIDs freely and pidfiles survive a crash, so that was a live risk,
// not a theoretical one. A pidfile without a recorded identity (written by an
// older build) is never acted on.
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
		rec, err := readPIDRecord(full)
		if err != nil {
			results = append(results, ReapResult{
				File:   e.Name(),
				Action: "skip",
				Note:   fmt.Sprintf("unreadable: %v", err),
			})
			_ = os.Remove(full)
			continue
		}

		res := ReapResult{File: e.Name(), PID: rec.PID}
		if ours, why := pidIsOurChild(rec); !ours {
			res.Action = "skip"
			res.Note = why
		} else {
			if err := signalGroup(rec.PID, syscall.SIGKILL); err == nil {
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

// pidIsOurChild reports whether the process now holding rec.PID is the same
// process the supervisor recorded, and why not when it isn't. The identity is
// the (start time, executable path) pair captured at spawn: a recycled PID
// fails the start-time comparison, and an unrelated process that happens to
// start at the same second fails the executable comparison.
//
// Anything unproven is a skip. The signal this gates is a process-group
// SIGKILL; a false positive takes out whatever else shares that group.
func pidIsOurChild(rec pidRecord) (bool, string) {
	if rec.PID <= 1 {
		return false, "invalid pid"
	}
	if rec.Start == "" || rec.Exec == "" {
		return false, "pidfile carries no process identity (older build); refusing to signal"
	}
	start, execPath, err := processIdentity(rec.PID)
	if err != nil {
		return false, "pid not running"
	}
	if start != rec.Start {
		return false, fmt.Sprintf("pid reused: started %q, pidfile recorded %q", start, rec.Start)
	}
	if execPath != rec.Exec {
		return false, fmt.Sprintf("pid reused: running %q, pidfile recorded %q", execPath, rec.Exec)
	}
	return true, ""
}

// ---------------------------------------------------------------------------
// pidfile helpers
// ---------------------------------------------------------------------------

// pidRecord is what a pidfile carries: the child's PID plus the identity the
// reaper needs to prove the PID has not been recycled since.
type pidRecord struct {
	PID   int
	Start string // opaque process start-time token from ps
	Exec  string // executable path as ps reports it
}

// writePIDRecord writes the pidfile as `key=value` lines. The first line is
// `pid=N` so a human `cat` still answers the obvious question first.
func writePIDRecord(path string, rec pidRecord) error {
	body := fmt.Sprintf("pid=%d\nstart=%s\nexec=%s\n", rec.PID, rec.Start, rec.Exec)
	return os.WriteFile(path, []byte(body), 0o644)
}

// readPIDRecord parses a pidfile. A bare-number file (written by a build
// before identity recording) yields a record with an empty identity, which
// pidIsOurChild refuses to act on.
func readPIDRecord(path string) (pidRecord, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return pidRecord{}, err
	}
	body := strings.TrimSpace(string(data))
	if !strings.Contains(body, "=") {
		n, err := strconv.Atoi(body)
		if err != nil {
			return pidRecord{}, fmt.Errorf("pidfile %q: %w", path, err)
		}
		return pidRecord{PID: n}, nil
	}
	var rec pidRecord
	for _, line := range strings.Split(body, "\n") {
		key, value, ok := strings.Cut(strings.TrimSpace(line), "=")
		if !ok {
			continue
		}
		switch key {
		case "pid":
			n, err := strconv.Atoi(value)
			if err != nil {
				return pidRecord{}, fmt.Errorf("pidfile %q: %w", path, err)
			}
			rec.PID = n
		case "start":
			rec.Start = value
		case "exec":
			rec.Exec = value
		}
	}
	if rec.PID == 0 {
		return pidRecord{}, fmt.Errorf("pidfile %q: no pid", path)
	}
	return rec, nil
}
