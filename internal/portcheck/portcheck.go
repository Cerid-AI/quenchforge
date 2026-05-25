// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

// Package portcheck inspects whether a TCP port on localhost is held
// by another process before quenchforge attempts to bind it. The
// motivating case is the Ollama.app login agent on macOS: when both
// quenchforge and Ollama try to bind 127.0.0.1:11434, whichever wins
// the race owns the port and the other crashes its supervisor loop.
//
// portcheck.Check returns a structured Result describing the holder
// (if any), so the caller can render an actionable error and exit
// cleanly — which pairs with the plist KeepAlive=<dict><SuccessfulExit
// false/></dict> setting that suppresses launchd respawns on clean
// exits with non-zero status.
package portcheck

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// Verdict classifies the port state.
type Verdict int

const (
	// VerdictFree means the port is available — quenchforge can bind.
	VerdictFree Verdict = iota

	// VerdictHeldByOllama means a process named "Ollama" / "ollama" /
	// "ollama serve" is the listener. Caller should print the canonical
	// Ollama deconfliction guidance and exit 0.
	VerdictHeldByOllama

	// VerdictHeldByStaleQuenchforge means an existing quenchforge
	// listener is present. Caller may attempt graceful takeover via
	// SIGTERM before binding.
	VerdictHeldByStaleQuenchforge

	// VerdictHeldByOther means some other process holds the port. The
	// Result.Holder fields describe it. Caller should log and exit 0.
	VerdictHeldByOther

	// VerdictUnknown means we detected SOMETHING on the port but could
	// not identify the holder (lsof + netstat fallback both failed).
	// Caller should log a "could not identify holder" and exit 0.
	VerdictUnknown
)

// Holder describes the process currently listening on the port.
type Holder struct {
	PID         int
	CommandName string // e.g. "Ollama", "quenchforge", "ollama"
	ExecPath    string // e.g. "/Applications/Ollama.app/Contents/MacOS/Ollama"
}

// Result is the outcome of a Check call.
type Result struct {
	Verdict Verdict
	Holder  Holder // zero value when Verdict == VerdictFree
}

// Check probes addr (e.g. "127.0.0.1:11434") to determine whether
// it's free, held by Ollama, held by a stale quenchforge, held by
// something else, or in an indeterminate state.
//
// The probe is non-destructive: it never sends data, never holds a
// connection longer than necessary, and never sends signals to the
// holder. Caller decides what to do with the verdict.
func Check(ctx context.Context, addr string) (Result, error) {
	// First try to bind. If we can bind, the port is free.
	ln, err := tryListen(addr)
	if err == nil {
		_ = ln.Close()
		return Result{Verdict: VerdictFree}, nil
	}
	if !isAddrInUse(err) {
		// Bind failed for some other reason — permission, name
		// resolution, etc. Propagate.
		return Result{}, fmt.Errorf("portcheck: probe-bind %s: %w", addr, err)
	}

	// Bind failed with EADDRINUSE — identify the holder.
	port, perr := parsePort(addr)
	if perr != nil {
		return Result{}, perr
	}

	holder, ok := identifyHolder(ctx, port)
	if !ok {
		return Result{Verdict: VerdictUnknown}, nil
	}

	return Result{
		Verdict: classifyHolder(holder),
		Holder:  holder,
	}, nil
}

// classifyHolder maps a Holder to a Verdict.
func classifyHolder(h Holder) Verdict {
	cn := strings.ToLower(h.CommandName)
	switch {
	case strings.HasPrefix(cn, "ollama"), strings.Contains(h.ExecPath, "Ollama.app"):
		return VerdictHeldByOllama
	case strings.HasPrefix(cn, "quenchfor"): // lsof truncates to 9 chars
		return VerdictHeldByStaleQuenchforge
	default:
		return VerdictHeldByOther
	}
}

// tryListen attempts a TCP listen on addr and immediately closes if
// successful. Returns the listener (so the caller can verify) or the
// bind error.
func tryListen(addr string) (net.Listener, error) {
	return net.Listen("tcp", addr)
}

// isAddrInUse reports whether err is a syscall-level "address in use".
//
// Preferred path: unwrap into *net.OpError → *os.SyscallError →
// syscall.EADDRINUSE. That's stable against Go-runtime and macOS
// error-string changes. Substring fallback covers unusual wrapping
// (e.g. errors wrapped past the OpError boundary).
func isAddrInUse(err error) bool {
	if err == nil {
		return false
	}
	var opErr *net.OpError
	if errors.As(err, &opErr) {
		if se, ok := opErr.Err.(*os.SyscallError); ok {
			if se.Err == syscall.EADDRINUSE {
				return true
			}
		}
	}
	s := err.Error()
	return strings.Contains(s, "address already in use") ||
		strings.Contains(s, "EADDRINUSE")
}

// parsePort extracts the numeric port from "host:port".
func parsePort(addr string) (int, error) {
	_, p, err := net.SplitHostPort(addr)
	if err != nil {
		return 0, fmt.Errorf("portcheck: parse addr %q: %w", addr, err)
	}
	port, err := strconv.Atoi(p)
	if err != nil {
		return 0, fmt.Errorf("portcheck: parse port %q: %w", p, err)
	}
	return port, nil
}

// identifyHolder runs `lsof -i :PORT -sTCP:LISTEN -F pcn` first; on
// failure (lsof missing, permission denied) falls back to a `netstat
// -anv -p tcp` parse. Returns (zero, false) if neither yields a hit.
func identifyHolder(ctx context.Context, port int) (Holder, bool) {
	if h, ok := identifyViaLsof(ctx, port); ok {
		return h, true
	}
	if h, ok := identifyViaNetstat(ctx, port); ok {
		return h, true
	}
	return Holder{}, false
}

// identifyViaLsof parses `lsof -i :PORT -sTCP:LISTEN -F pcn` output.
// The -F flag emits one field per line, each prefixed with its type:
//
//	p<PID>
//	c<command>
//	n<host:port>->...
//
// We grab the first p/c pair that matches our port. ExecPath is
// resolved via `ps -p PID -o comm=` when available.
func identifyViaLsof(ctx context.Context, port int) (Holder, bool) {
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "lsof", "-i", fmt.Sprintf(":%d", port), "-sTCP:LISTEN", "-F", "pcn")
	out, err := cmd.Output()
	if err != nil {
		return Holder{}, false
	}

	var h Holder
	for _, line := range strings.Split(string(out), "\n") {
		if len(line) == 0 {
			continue
		}
		switch line[0] {
		case 'p':
			if pid, err := strconv.Atoi(line[1:]); err == nil {
				h.PID = pid
			}
		case 'c':
			h.CommandName = line[1:]
		}
	}
	if h.PID == 0 {
		return Holder{}, false
	}
	h.ExecPath = resolveExecPath(ctx, h.PID)
	return h, true
}

// identifyViaNetstat is a fallback for sandboxed environments where
// lsof is unavailable. macOS netstat -anv does not expose PID columns
// reliably (the BSD netstat output has no stable PID field), so we
// cannot identify the holder here — we can only confirm SOMETHING is
// listening. Returning (Holder{}, false) routes the caller to
// VerdictUnknown, which is honest about the lack of identification.
func identifyViaNetstat(ctx context.Context, port int) (Holder, bool) {
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "netstat", "-anv", "-p", "tcp")
	out, err := cmd.Output()
	if err != nil {
		return Holder{}, false
	}

	needle := fmt.Sprintf(".%d ", port)
	for _, line := range strings.Split(string(out), "\n") {
		if strings.Contains(line, "LISTEN") && strings.Contains(line, needle) {
			// Confirmed in use; identity unknown.
			return Holder{}, false
		}
	}
	return Holder{}, false
}

// resolveExecPath returns the absolute executable path for pid, or
// "" if it can't be resolved.
func resolveExecPath(ctx context.Context, pid int) string {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "ps", "-p", strconv.Itoa(pid), "-o", "comm=").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// CanonicalOllamaMessage returns the canonical actionable error text
// quenchforge prints when VerdictHeldByOllama. Held as an exported
// constant so doctor and serve render it identically.
const CanonicalOllamaMessage = `quenchforge: port %s is held by Ollama.app (pid %d).
  Two ways forward:
    1. Disable Ollama's login agent and use quenchforge:
         launchctl bootout gui/$(id -u)/com.ollama.ollama
    2. Coexist on different ports — set QUENCHFORGE_LISTEN_ADDR=:11435
       in your LaunchAgent plist's <EnvironmentVariables> dict, then:
         launchctl kickstart -k gui/$(id -u)/com.cerid.quenchforge
       (clients pointing at :11434 will hit Ollama; clients pointing at
       :11435 will hit quenchforge.)
  See: quenchforge doctor --explain
`

// FormatOllamaMessage fills the canonical template with addr + pid.
func FormatOllamaMessage(addr string, pid int) string {
	return fmt.Sprintf(CanonicalOllamaMessage, addr, pid)
}
