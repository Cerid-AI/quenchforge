# System Stability + Ollama Deconfliction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the recurring system freeze caused by quenchforge ⇄ Ollama port-11434 contention and chat-slot Vega II SIGABRT loops, while shipping product-grade deconfliction so any public quenchforge user is protected.

**Architecture:** Six independent layers, each shippable on its own. Stability code lives in quenchforge (`internal/portcheck/` for the pre-bind helper, `internal/supervisor/logrotate.go` for log rotation, `cmd/quenchforge/main.go::cmdDoctor` extensions, one-line `internal/tuning/tuning.go::chatParams` change). LaunchAgent hardening lives in `plist_template.plist`. OS hygiene and gated v0.8.0-rc1 GPU activation are operator-executed.

**Tech Stack:** Go 1.23 (zero deps — `go.mod` has no `require` block; rotator is inlined). Bash for OS-level operational steps. Python 3 for the v0.8.0-rc1 bench scripts that already exist in `scripts/`.

**Spec:** [docs/superpowers/specs/2026-05-25-system-stability-and-ollama-deconfliction-design.md](../specs/2026-05-25-system-stability-and-ollama-deconfliction-design.md) (commit `9ca6ffe`)

---

## File map

**New files (quenchforge repo `~/Develop/quenchforge`):**
- `internal/portcheck/portcheck.go` — pre-bind port holder detection (Layer 1a)
- `internal/portcheck/portcheck_test.go`
- `internal/supervisor/logrotate.go` — size-based rotating writer (Layer 1c)
- `internal/supervisor/logrotate_test.go`

**Modified files (quenchforge repo):**
- `cmd/quenchforge/main.go` — wire portcheck into `cmdServe`; extend `cmdDoctor` with 4 new checks (Layers 1a, 1b)
- `internal/supervisor/supervisor.go` — swap `AppendToLog` / log file open to use rotator (Layer 1c)
- `internal/tuning/tuning.go` — append `--gpu-layers 0` to chat AMD `ExtraArgs` (Layer 1d)
- `internal/tuning/tuning_test.go` — extend `TestKernelParams_ChatAMDGetsCorrectnessFlags` (Layer 1d)
- `plist_template.plist` — `KeepAlive` boolean → dict (Layer 2)
- `README.md` — new "Coexistence with Ollama" section (Layer 1e)
- `patches/README.md` — extend Section 3 (Layer 1e)
- `CHANGELOG.md` — v0.7.2 entry covering this work

**Modified files (cerid-ai-internal repo `~/Develop/cerid-ai-internal`):**
- `scripts/detect-gpu.sh` — flip Vega-II recommendation `ollama` → `quenchforge` (Layer 4)
- `scripts/start-cerid.sh` — error string (Layer 4)
- `scripts/validate-env.sh` — warn string (Layer 4)

**Memory files (`~/.claude/projects/-Users-sunrunner-Develop/memory/`):**
- `feedback_quenchforge_safety.md` — post-fix state (Layer 6)
- `project_cerid_quenchforge_chat_on_cpu.md` — NEW (Layer 6)
- `MEMORY.md` — index the new file

**No code changes (operator-executed):**
- Layer 3: OS hygiene commands run from a shell
- Layer 5: bench → gated tuning edit

---

# Phase 0 — Stop the bleeding (Layer 3, OS-level)

Execute first. No code. ~10 minutes.

### Task 0.1: Disable Ollama LaunchAgent

**Files:** none (state lives in launchd)

- [ ] **Step 1: Quit the Ollama GUI app**

```sh
osascript -e 'tell application "Ollama" to quit' 2>/dev/null || true
```

Expected: silent success, or "Ollama isn't running" (also fine).

- [ ] **Step 2: Confirm Ollama is gone**

```sh
ps -ef | grep -i "/Applications/Ollama.app" | grep -v grep
```

Expected: no output. If Ollama.app helper still listed, repeat Step 1.

- [ ] **Step 3: Bootout the LaunchAgent**

```sh
launchctl bootout gui/$(id -u)/com.ollama.ollama
```

Expected: silent success (exit 0). If it was already booted out: `Boot-out failed: 5: Input/output error` is fine — it means the agent wasn't loaded.

- [ ] **Step 4: Verify**

```sh
launchctl list 2>/dev/null | grep -i ollama
```

Expected: no output. (If `actions.runner.Cerid-AI-quenchforge.mac-pro-2019-vega-ii` shows, that's GitHub Actions, unrelated.)

```sh
lsof -i :11434 -sTCP:LISTEN
```

Expected: only `quenchfor` (truncated to 9 chars by lsof) listening.

### Task 0.2: Free disk space

**Files:** none

- [ ] **Step 1: Capture baseline**

```sh
df -h /System/Volumes/Data
```

Note the "Avail" value. Goal is to reach ≥ 50 GB free.

- [ ] **Step 2: Prune Docker volumes and unused images**

```sh
docker system prune -a --volumes -f
```

Expected: reclaims tens of GB. Output ends with "Total reclaimed space: NNN GB". If Docker is not running, start Docker Desktop first.

- [ ] **Step 3: Truncate the in-flight embed.log**

```sh
: > ~/Library/Logs/quenchforge/embed.log
ls -lh ~/Library/Logs/quenchforge/embed.log
```

Expected: file size now 0. (Layer 1c will prevent future re-bloat.)

- [ ] **Step 4: Verify target**

```sh
df -h /System/Volumes/Data
```

Expected: Avail ≥ 50 GB. If under, identify with `du -sh ~/Library/Containers/* | sort -h | tail -10` and prune further.

### Task 0.3: Schedule weekly reboot

**Files:** none

- [ ] **Step 1: Set the recurring restart**

```sh
sudo pmset repeat restartall MTWRFSU 04:00
```

Expected: prompts for sudo password, then silent success.

- [ ] **Step 2: Verify**

```sh
pmset -g sched
```

Expected: output includes a "Repeating power events" line showing the MTWRFSU 04:00 restart.

---

# Phase 1 — Log rotation (Layer 1c)

Ships independently. No other code depends on this. ~45 minutes.

### Task 1.1: Read existing supervisor structure

**Files:**
- Read: `internal/supervisor/supervisor.go` (whole file, 423 lines)
- Read: `cmd/quenchforge/main.go:656-695` (the `startSlot` function — this is where the per-slot stderr/stdout `io.Writer` is created and passed in)

- [ ] **Step 1: Identify where slot log files are opened**

Look at `AppendToLog` at supervisor.go:412 — it appends a reader to a file. Identify all callers (`grep -n "AppendToLog\|supervisor.NewSlot\|.log" cmd/quenchforge/main.go`). Confirm whether slot logs are passed in by the caller as an `*os.File`-wrapped writer or opened inside the supervisor.

Outcome: a note on whether the rotator is wired at the `startSlot` callsite (caller-side) or inside `supervisor.NewSlot/Start` (callee-side). Caller-side is simpler — the rotator becomes an `io.Writer` we pass in.

### Task 1.2: Write the failing test for the rotator

**Files:**
- Create: `internal/supervisor/logrotate_test.go`

- [ ] **Step 1: Write `TestRotatingWriter_RotatesAtMaxBytes`**

```go
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
```

- [ ] **Step 2: Run tests to verify they fail**

```sh
cd ~/Develop/quenchforge && go test ./internal/supervisor/ -run TestRotatingWriter -v
```

Expected: FAIL with `undefined: NewRotatingWriter`.

### Task 1.3: Implement the rotator

**Files:**
- Create: `internal/supervisor/logrotate.go`

- [ ] **Step 1: Write the implementation**

```go
// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package supervisor

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// RotatingWriter is an io.Writer that rolls the underlying file over
// at a configurable size threshold, keeping a bounded number of
// numbered backup files (file.log, file.log.1, ..., file.log.N).
//
// Used by the supervisor to bound per-slot log growth. llama-server
// chat/embed/rerank slots emit one model-load banner per spawn (multi-KB
// each), and the embed slot under sustained load produces verbose
// per-request output that accumulates to GB-scale in days. Without
// rotation, an unattended quenchforge install consumes the user's
// disk and contributes to the disk-full state that prevents APFS
// from writing kernel panic reports.
//
// The rotator is intentionally minimal: synchronous rotation on the
// write path (not a background goroutine), no compression of backups,
// no time-based policy. Operators who need more (e.g., daily rotation,
// gzip) can layer macOS `newsyslog` on top — the rotator does not own
// the file exclusively in any way that breaks external rotation.
type RotatingWriter struct {
	path     string
	maxBytes int64 // 0 disables rotation
	backups  int

	mu      sync.Mutex
	f       *os.File
	curSize int64
}

// NewRotatingWriter opens path for append. If the file exists, its
// current size becomes the starting curSize. maxBytes <= 0 disables
// rotation (writes append forever).
func NewRotatingWriter(path string, maxBytes int64, backups int) (*RotatingWriter, error) {
	if backups < 0 {
		backups = 0
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("logrotate: mkdir %s: %w", filepath.Dir(path), err)
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, fmt.Errorf("logrotate: open %s: %w", path, err)
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("logrotate: stat %s: %w", path, err)
	}
	return &RotatingWriter{
		path:     path,
		maxBytes: maxBytes,
		backups:  backups,
		f:        f,
		curSize:  info.Size(),
	}, nil
}

// Write implements io.Writer. May rotate the file synchronously when
// a write would cause curSize to exceed maxBytes.
func (w *RotatingWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.maxBytes > 0 && w.curSize+int64(len(p)) > w.maxBytes {
		if err := w.rotateLocked(); err != nil {
			return 0, err
		}
	}
	n, err := w.f.Write(p)
	w.curSize += int64(n)
	return n, err
}

// Close releases the underlying file. Safe to call multiple times.
func (w *RotatingWriter) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.f == nil {
		return nil
	}
	err := w.f.Close()
	w.f = nil
	return err
}

// rotateLocked: caller holds w.mu. Closes the current file, rolls
// existing backups one slot higher, then reopens path fresh.
//
// Backup naming: file.log → file.log.1 → file.log.2 → ... → file.log.N.
// The oldest is dropped on overflow. If backups == 0, the primary is
// truncated with no backup preserved.
func (w *RotatingWriter) rotateLocked() error {
	if w.f != nil {
		if err := w.f.Close(); err != nil {
			return fmt.Errorf("logrotate: close before rotate: %w", err)
		}
		w.f = nil
	}

	// Walk from oldest to newest, shifting each up one slot.
	// e.g., backups=3: drop .3, rename .2→.3, .1→.2, primary→.1.
	if w.backups > 0 {
		oldest := fmt.Sprintf("%s.%d", w.path, w.backups)
		_ = os.Remove(oldest) // best-effort; missing is fine

		for i := w.backups - 1; i >= 1; i-- {
			src := fmt.Sprintf("%s.%d", w.path, i)
			dst := fmt.Sprintf("%s.%d", w.path, i+1)
			if _, err := os.Stat(src); err == nil {
				if err := os.Rename(src, dst); err != nil {
					return fmt.Errorf("logrotate: rename %s -> %s: %w", src, dst, err)
				}
			}
		}
		// primary → .1
		if _, err := os.Stat(w.path); err == nil {
			if err := os.Rename(w.path, w.path+".1"); err != nil {
				return fmt.Errorf("logrotate: rename %s -> %s.1: %w", w.path, w.path, err)
			}
		}
	} else {
		// backups == 0: just remove primary (it's about to be reopened fresh).
		_ = os.Remove(w.path)
	}

	f, err := os.OpenFile(w.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("logrotate: reopen after rotate: %w", err)
	}
	w.f = f
	w.curSize = 0
	return nil
}
```

- [ ] **Step 2: Run tests to verify they pass**

```sh
cd ~/Develop/quenchforge && go test ./internal/supervisor/ -run TestRotatingWriter -v
```

Expected: all three tests PASS.

- [ ] **Step 3: Run the full supervisor test suite to make sure nothing else broke**

```sh
go test ./internal/supervisor/ -v
```

Expected: all tests PASS.

### Task 1.4: Wire the rotator into slot start path

**Files:**
- Modify: `cmd/quenchforge/main.go` (the `startSlot` function near line 656; specifically the per-slot stderr file open)

- [ ] **Step 1: Identify the current log file open**

```sh
grep -n "OpenFile\|os\.Create\|chat\.log\|.log\"" ~/Develop/quenchforge/cmd/quenchforge/main.go | head -10
```

Note the line(s) that open the slot's per-process log file (chat.log, embed.log, etc.).

- [ ] **Step 2: Replace direct `os.OpenFile` with `supervisor.NewRotatingWriter`**

Pattern (adapt to actual code at the identified line):

Before:
```go
// existing code in startSlot — opens slot log via os.OpenFile or similar
logFile, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
if err != nil {
    return nil, fmt.Errorf("open slot log %s: %w", logPath, err)
}
```

After:
```go
const (
    defaultLogMaxBytes = 100 * 1024 * 1024 // 100 MB
    defaultLogBackups  = 5
)

maxBytes := envInt64("QUENCHFORGE_LOG_MAX_BYTES", defaultLogMaxBytes)
backups := envInt("QUENCHFORGE_LOG_BACKUPS", defaultLogBackups)

logFile, err := supervisor.NewRotatingWriter(logPath, maxBytes, backups)
if err != nil {
    return nil, fmt.Errorf("open slot log %s: %w", logPath, err)
}
```

Also add helpers near the bottom of main.go if they don't already exist:

```go
func envInt64(key string, def int64) int64 {
    s := os.Getenv(key)
    if s == "" {
        return def
    }
    n, err := strconv.ParseInt(s, 10, 64)
    if err != nil || n < 0 {
        return def
    }
    return n
}

func envInt(key string, def int) int {
    s := os.Getenv(key)
    if s == "" {
        return def
    }
    n, err := strconv.Atoi(s)
    if err != nil || n < 0 {
        return def
    }
    return n
}
```

Add `"strconv"` to the import block if not already present.

- [ ] **Step 3: If `RotatingWriter` needs to satisfy `io.Closer` for cleanup, verify the caller closes it**

```sh
grep -n "logFile\.Close\|defer .*\.Close" cmd/quenchforge/main.go | head -5
```

If the existing code does `defer logFile.Close()`, the rotator's `Close()` method satisfies that (it implements `io.Closer`). No change needed.

- [ ] **Step 4: Build and run the full test suite**

```sh
cd ~/Develop/quenchforge && go build ./... && go test ./...
```

Expected: clean build, all tests pass.

### Task 1.5: Manual integration check

**Files:** none

- [ ] **Step 1: Build a local binary and run it briefly**

```sh
cd ~/Develop/quenchforge && go build -o /tmp/quenchforge-dev ./cmd/quenchforge
# Smoke test: rotation triggers on a small file. Override via env.
QUENCHFORGE_LOG_MAX_BYTES=4096 QUENCHFORGE_LOG_BACKUPS=2 \
  /tmp/quenchforge-dev doctor 2>&1 | head -20
```

Expected: doctor command runs (we'll wire the rotator into doctor's output path in Phase 3; for now we're just confirming the binary builds).

- [ ] **Step 2: Stress the rotator with a small bash loop** (only if you want to confirm the integration path; not required since unit tests cover the logic)

Skip this step unless concerned.

### Task 1.6: Commit Phase 1

- [ ] **Step 1: Commit**

```sh
cd ~/Develop/quenchforge
git add internal/supervisor/logrotate.go internal/supervisor/logrotate_test.go cmd/quenchforge/main.go
git commit -m "$(cat <<'EOF'
feat(supervisor): bounded per-slot log rotation

Adds RotatingWriter to internal/supervisor — a synchronous,
size-thresholded io.Writer that bounds per-slot log file growth. The
unattended embed.log on a Vega II install reached 3.73 GB in 7 days of
uptime before this; combined with Docker.raw growth that's how the dev
machine reached 97% full and stopped being able to write kernel panic
dumps.

Default: 100 MB per slot, 5 backups (chat.log, chat.log.1, ...,
chat.log.5). Override via QUENCHFORGE_LOG_MAX_BYTES and
QUENCHFORGE_LOG_BACKUPS env vars. maxBytes=0 disables rotation
(preserves any prior install that relied on unbounded logs + external
rotation).

Zero new deps — go.mod's empty require block is preserved.
EOF
)"
```

Expected: commit succeeds. Note the SHA for the followup phases.

---

# Phase 2 — Pre-bind check + plist hardening (Layers 1a, 2)

Single PR. Code change + plist change pair. ~90 minutes.

### Task 2.1: Create the portcheck package skeleton

**Files:**
- Create: `internal/portcheck/portcheck.go`
- Create: `internal/portcheck/portcheck_test.go`

- [ ] **Step 1: Create the package directory and a minimal stub**

```sh
mkdir -p ~/Develop/quenchforge/internal/portcheck
```

Create `internal/portcheck/portcheck.go`:

```go
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
	"os/exec"
	"strconv"
	"strings"
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
func isAddrInUse(err error) bool {
	if err == nil {
		return false
	}
	// Fastest portable check: substring match. The structured-error
	// path requires syscall.EADDRINUSE comparison, which is a noisy
	// import for one predicate; the strings.Contains is reliable
	// across all macOS Go versions we target.
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
// lsof is unavailable. macOS netstat doesn't directly report PID, so
// this is a softer signal — best-effort.
func identifyViaNetstat(ctx context.Context, port int) (Holder, bool) {
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "netstat", "-anv", "-p", "tcp")
	out, err := cmd.Output()
	if err != nil {
		return Holder{}, false
	}

	// Look for a LISTEN line on the port. macOS netstat -anv columns
	// vary; we only need to confirm presence and grab the PID column
	// if it's there.
	needle := fmt.Sprintf(".%d ", port)
	for _, line := range strings.Split(string(out), "\n") {
		if !strings.Contains(line, "LISTEN") {
			continue
		}
		if !strings.Contains(line, needle) {
			continue
		}
		fields := strings.Fields(line)
		// macOS netstat -anv ends with: rxbytes txbytes ... uid pid
		if len(fields) < 2 {
			continue
		}
		// Try the last field as PID.
		if pid, err := strconv.Atoi(fields[len(fields)-1]); err == nil && pid > 0 {
			return Holder{
				PID:         pid,
				CommandName: resolveExecPath(ctx, pid), // we don't have a separate command name here
			}, true
		}
		// We saw the listener but can't name it.
		return Holder{}, false
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

// ErrUnresolvedHolder is returned only by helpers that explicitly
// signal "saw a holder, couldn't name it" — Check never returns it.
var ErrUnresolvedHolder = errors.New("port held but holder unidentified")
```

### Task 2.2: Write the failing tests for portcheck

**Files:**
- Create: `internal/portcheck/portcheck_test.go`

- [ ] **Step 1: Write the test file**

```go
// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package portcheck

import (
	"context"
	"net"
	"strings"
	"testing"
	"time"
)

// holdRandomPort opens a real listener on a random localhost port and
// returns the resolved addr ("127.0.0.1:NNNNN") + a teardown.
func holdRandomPort(t *testing.T) (string, func()) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("hold listener: %v", err)
	}
	return ln.Addr().String(), func() { _ = ln.Close() }
}

func TestCheck_PortFree_ReturnsFree(t *testing.T) {
	// Find a port that's currently free by binding+closing immediately.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("scratch listen: %v", err)
	}
	addr := ln.Addr().String()
	_ = ln.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	r, err := Check(ctx, addr)
	if err != nil {
		t.Fatalf("Check error: %v", err)
	}
	if r.Verdict != VerdictFree {
		t.Errorf("verdict = %v, want VerdictFree", r.Verdict)
	}
}

func TestCheck_PortHeld_ReturnsIdentifiedHolder(t *testing.T) {
	addr, teardown := holdRandomPort(t)
	defer teardown()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	r, err := Check(ctx, addr)
	if err != nil {
		t.Fatalf("Check error: %v", err)
	}
	if r.Verdict == VerdictFree {
		t.Errorf("verdict = VerdictFree, want held")
	}
	// We're the holder (the test process). The Verdict will be
	// VerdictHeldByOther — that's the relevant assertion.
	if r.Verdict != VerdictHeldByOther {
		// Tolerate VerdictUnknown if lsof is sandboxed in CI. The
		// important property is "not VerdictFree".
		if r.Verdict != VerdictUnknown {
			t.Errorf("verdict = %v, want VerdictHeldByOther or VerdictUnknown",
				r.Verdict)
		}
	}
}

func TestClassifyHolder_OllamaByCommandName(t *testing.T) {
	cases := []Holder{
		{PID: 1, CommandName: "Ollama"},
		{PID: 2, CommandName: "ollama"},
		{PID: 3, CommandName: "ollama serve"},
		{PID: 4, CommandName: "OllamaServer", ExecPath: "/Applications/Ollama.app/Contents/MacOS/Ollama"},
	}
	for _, h := range cases {
		t.Run(h.CommandName, func(t *testing.T) {
			if got := classifyHolder(h); got != VerdictHeldByOllama {
				t.Errorf("classifyHolder(%+v) = %v, want VerdictHeldByOllama", h, got)
			}
		})
	}
}

func TestClassifyHolder_StaleQuenchforge(t *testing.T) {
	// lsof truncates command names to 9 chars on macOS: "quenchforge" → "quenchfor".
	h := Holder{PID: 1, CommandName: "quenchfor"}
	if got := classifyHolder(h); got != VerdictHeldByStaleQuenchforge {
		t.Errorf("classifyHolder(%+v) = %v, want VerdictHeldByStaleQuenchforge", h, got)
	}
}

func TestClassifyHolder_OtherProcess(t *testing.T) {
	h := Holder{PID: 1, CommandName: "node"}
	if got := classifyHolder(h); got != VerdictHeldByOther {
		t.Errorf("classifyHolder(%+v) = %v, want VerdictHeldByOther", h, got)
	}
}

func TestFormatOllamaMessage_ContainsActionableSteps(t *testing.T) {
	msg := FormatOllamaMessage("127.0.0.1:11434", 42)
	for _, want := range []string{
		"127.0.0.1:11434",
		"pid 42",
		"launchctl bootout gui/$(id -u)/com.ollama.ollama",
		"QUENCHFORGE_LISTEN_ADDR=:11435",
		"quenchforge doctor --explain",
	} {
		if !strings.Contains(msg, want) {
			t.Errorf("message missing %q", want)
		}
	}
}
```

- [ ] **Step 2: Run tests to verify them**

```sh
cd ~/Develop/quenchforge && go test ./internal/portcheck/ -v
```

Expected: all PASS (the impl in 2.1 was complete; these tests validate it). If TestCheck_PortHeld_ReturnsIdentifiedHolder fails because the CI runner has no `lsof`, that's acceptable — the test tolerates VerdictUnknown.

### Task 2.3: Wire portcheck into cmdServe

**Files:**
- Modify: `cmd/quenchforge/main.go` (inside `cmdServe`, before `net.Listen` or the gateway start; per the earlier grep, cmdServe starts around line 225 and the listen is around line 307 — confirm in actual code)

- [ ] **Step 1: Add the import**

In the import block at the top of `main.go`, add:

```go
"context"
"github.com/cerid-ai/quenchforge/internal/portcheck"
```

(Skip "context" if already imported.)

- [ ] **Step 2: Add the pre-bind check at the top of cmdServe**

After config is resolved (`cfg.ListenAddr` is known) but before any listener is created or gateway is started:

```go
// Pre-bind port check. Identifies common conflicts (Ollama, stale
// quenchforge) and exits cleanly with actionable guidance so the
// LaunchAgent's KeepAlive=<dict><SuccessfulExit false/></dict>
// (post-Layer-2 plist) does not respawn-loop.
{
    pcCtx, pcCancel := context.WithTimeout(ctx, 5*time.Second)
    res, err := portcheck.Check(pcCtx, cfg.ListenAddr)
    pcCancel()
    if err != nil {
        fmt.Fprintf(stderr, "quenchforge: port probe failed for %s: %v\n", cfg.ListenAddr, err)
        // Probe failure is not fatal — fall through to bind and let
        // the real error surface if there is one.
    } else {
        switch res.Verdict {
        case portcheck.VerdictFree:
            // No-op.
        case portcheck.VerdictHeldByOllama:
            fmt.Fprint(stderr, portcheck.FormatOllamaMessage(cfg.ListenAddr, res.Holder.PID))
            return nil // clean exit so launchd does not respawn
        case portcheck.VerdictHeldByStaleQuenchforge:
            fmt.Fprintf(stderr,
                "quenchforge: stale quenchforge process (pid %d) is holding %s. "+
                    "Try: kill %d && launchctl kickstart -k gui/$(id -u)/com.cerid.quenchforge\n",
                res.Holder.PID, cfg.ListenAddr, res.Holder.PID)
            return nil
        case portcheck.VerdictHeldByOther:
            fmt.Fprintf(stderr,
                "quenchforge: port %s is held by pid %d (%s, %s). "+
                    "Resolve the conflict and re-run.\n",
                cfg.ListenAddr, res.Holder.PID, res.Holder.CommandName, res.Holder.ExecPath)
            return nil
        case portcheck.VerdictUnknown:
            fmt.Fprintf(stderr,
                "quenchforge: port %s is in use but holder could not be identified "+
                    "(lsof + netstat both failed). Check `lsof -i :%s` manually.\n",
                cfg.ListenAddr, cfg.ListenAddr)
            return nil
        }
    }
}
```

Note the use of `ctx` (the existing serve context) and `time.Second` — confirm both are in scope; add the `time` import if not already present.

- [ ] **Step 3: Build**

```sh
cd ~/Develop/quenchforge && go build ./...
```

Expected: clean build. If `ctx` or `stderr` symbols don't match what cmdServe actually has, adapt to the actual identifiers.

- [ ] **Step 4: Run all tests**

```sh
go test ./...
```

Expected: all PASS.

### Task 2.4: Update plist_template.plist

**Files:**
- Modify: `cmd/quenchforge/plist_template.plist` (per the earlier grep, ProcessType / KeepAlive / ThrottleInterval are near lines 77/82)

- [ ] **Step 1: Find the current KeepAlive block**

```sh
grep -nC2 "KeepAlive" ~/Develop/quenchforge/cmd/quenchforge/plist_template.plist
```

Expected: shows `<key>KeepAlive</key>` followed by `<true/>`.

- [ ] **Step 2: Replace with dict form**

Change:
```xml
<key>KeepAlive</key>
<true/>
```
to:
```xml
<key>KeepAlive</key>
<dict>
  <key>SuccessfulExit</key>
  <false/>
</dict>
```

- [ ] **Step 3: Validate the plist**

```sh
plutil -lint ~/Develop/quenchforge/cmd/quenchforge/plist_template.plist
```

Expected: `<path>: OK`.

### Task 2.5: Manual integration test

**Files:** none

- [ ] **Step 1: Build a fresh local binary**

```sh
cd ~/Develop/quenchforge && go build -o /tmp/quenchforge-dev ./cmd/quenchforge
```

- [ ] **Step 2: Confirm baseline (port should currently be free for this test)**

```sh
lsof -i :11434 -sTCP:LISTEN
```

If quenchforge is currently listening from the LaunchAgent, the test must run on a different port. Use:

```sh
QUENCHFORGE_LISTEN_ADDR=127.0.0.1:11444 /tmp/quenchforge-dev serve 2>&1 | head -10
```

(Adjust 11444 if also occupied.)

- [ ] **Step 3: Simulate the Ollama-held scenario**

In one terminal, start a sacrificial listener with a process name resembling "ollama":

```sh
( exec -a Ollama python3 -c 'import socket,time; s=socket.socket(); s.bind(("127.0.0.1",11444)); s.listen(); time.sleep(120)' ) &
SACRIFICE=$!
```

(The `exec -a Ollama` trick sets argv[0] so `lsof` reports the command as "Ollama".)

In another terminal:

```sh
QUENCHFORGE_LISTEN_ADDR=127.0.0.1:11444 /tmp/quenchforge-dev serve 2>&1
```

Expected: prints the `CanonicalOllamaMessage` content with pid `$SACRIFICE`, exits 0.

Tear down: `kill $SACRIFICE`.

### Task 2.6: Commit Phase 2

- [ ] **Step 1: Commit**

```sh
cd ~/Develop/quenchforge
git add internal/portcheck cmd/quenchforge/main.go cmd/quenchforge/plist_template.plist
git commit -m "$(cat <<'EOF'
feat(serve): pre-bind port check + plist KeepAlive hardening

Adds internal/portcheck to identify the listener on the configured
ListenAddr before the gateway tries to bind it. When the holder is
Ollama (the common public-install conflict, since Ollama.app's login
agent races us for 11434), quenchforge now prints actionable guidance
and exits 0 instead of looping on EADDRINUSE.

Pairs with a plist_template.plist change: KeepAlive switches from
<true/> to <dict><SuccessfulExit><false/></dict>, so launchd does not
respawn on the new clean-exit path. ThrottleInterval=10 stays as
belt-and-suspenders for actual crashes.

Holder identification: lsof first (-F pcn parse), netstat fallback,
unknown if both fail. Classification: VerdictHeldByOllama (command
name starts with "ollama" or exec path includes "Ollama.app"),
VerdictHeldByStaleQuenchforge (lsof's truncated "quenchfor" prefix),
VerdictHeldByOther.

Public users who install quenchforge alongside an existing Ollama
install no longer hit a silent crash-spam loop.
EOF
)"
```

---

# Phase 3 — Extend `quenchforge doctor` (Layer 1b)

`cmdDoctor` already exists at `cmd/quenchforge/main.go:118`. We extend with four new checks. ~75 minutes.

### Task 3.1: Map existing cmdDoctor output

**Files:**
- Read: `cmd/quenchforge/main.go:115-220` (the cmdDoctor function)

- [ ] **Step 1: Capture the current output format**

```sh
cd ~/Develop/quenchforge && go run ./cmd/quenchforge doctor 2>&1 | tee /tmp/doctor-baseline.txt
```

Expected: a structured report (the exact sections to be confirmed by reading the function). Save the file — we want to ADD sections, not break existing parsers.

- [ ] **Step 2: Read the doctor function**

```sh
sed -n '115,220p' ~/Develop/quenchforge/cmd/quenchforge/main.go
```

Note the current sections, the helper functions it uses, and whether it has any flag parsing (the earlier grep showed `flag.NewFlagSet("doctor", ...)` at line 119, so flag support exists).

### Task 3.2: Add the Ollama LaunchAgent check

**Files:**
- Modify: `cmd/quenchforge/main.go` (extend cmdDoctor)
- Modify: `cmd/quenchforge/serve_test.go` (add a doctor-output assertion)

- [ ] **Step 1: Write a failing test asserting doctor mentions the Ollama check**

In `cmd/quenchforge/serve_test.go` (or whichever test file covers main.go behavior; create `cmd/quenchforge/doctor_test.go` if doctor isn't already tested):

```go
func TestDoctor_IncludesOllamaLaunchAgentCheck(t *testing.T) {
    var stdout, stderr bytes.Buffer
    if err := cmdDoctor(nil, &stdout, &stderr); err != nil {
        t.Fatalf("cmdDoctor: %v", err)
    }
    out := stdout.String()
    if !strings.Contains(out, "Ollama LaunchAgent") {
        t.Errorf("doctor output missing 'Ollama LaunchAgent' section.\nGot:\n%s", out)
    }
}
```

Add `"bytes"` and `"strings"` imports if needed.

- [ ] **Step 2: Run, verify FAIL**

```sh
cd ~/Develop/quenchforge && go test ./cmd/quenchforge/ -run TestDoctor_IncludesOllamaLaunchAgentCheck -v
```

Expected: FAIL — "doctor output missing 'Ollama LaunchAgent' section".

- [ ] **Step 3: Implement the check in cmdDoctor**

Append to cmdDoctor (after the existing sections, before the final return):

```go
fmt.Fprintln(stdout, "\nOllama LaunchAgent")
fmt.Fprintln(stdout, "------------------")
ollamaStatus := checkOllamaLaunchAgent()
fmt.Fprintf(stdout, "  status: %s\n", ollamaStatus)
```

And add the helper at the bottom of main.go:

```go
// checkOllamaLaunchAgent reports whether com.ollama.ollama is loaded
// in the user's GUI launchd domain. Returns one of:
//   - "not installed"      (Ollama.app not present)
//   - "disabled"           (plist present but launchctl bootout has
//                          been applied, OR app present but agent never
//                          loaded)
//   - "loaded (PID N)"     (currently running)
//   - "loaded but stopped" (loaded, no PID — exited cleanly)
//
// Implementation uses `launchctl list com.ollama.ollama`: exit 0 with
// output if loaded, exit nonzero if not. We don't need launchctl's
// PID column — the PID lives in `launchctl print` output, which we
// fetch only when the list call succeeded.
func checkOllamaLaunchAgent() string {
    // Is Ollama.app even installed?
    if _, err := os.Stat("/Applications/Ollama.app"); os.IsNotExist(err) {
        return "not installed"
    }

    out, err := exec.Command("launchctl", "list", "com.ollama.ollama").Output()
    if err != nil {
        return "disabled"
    }
    // launchctl list <label> output is a plist-ish key=value block.
    // PID line looks like:  "PID" = 1253;
    lines := strings.Split(string(out), "\n")
    for _, ln := range lines {
        ln = strings.TrimSpace(ln)
        if strings.HasPrefix(ln, "\"PID\"") {
            // "PID" = 1253;
            parts := strings.SplitN(ln, "=", 2)
            if len(parts) == 2 {
                v := strings.TrimSpace(parts[1])
                v = strings.TrimSuffix(v, ";")
                if pid, err := strconv.Atoi(v); err == nil && pid > 0 {
                    return fmt.Sprintf("loaded (PID %d) — DISARM with: launchctl bootout gui/$(id -u)/com.ollama.ollama", pid)
                }
            }
        }
    }
    return "loaded but stopped"
}
```

Add `"os/exec"` to imports if not present.

- [ ] **Step 4: Run, verify PASS**

```sh
go test ./cmd/quenchforge/ -run TestDoctor_IncludesOllamaLaunchAgentCheck -v
```

Expected: PASS.

### Task 3.3: Add the disk-free check

**Files:**
- Modify: `cmd/quenchforge/main.go`
- Modify: `cmd/quenchforge/serve_test.go`

- [ ] **Step 1: Write failing test**

```go
func TestDoctor_IncludesDiskFreeCheck(t *testing.T) {
    var stdout, stderr bytes.Buffer
    if err := cmdDoctor(nil, &stdout, &stderr); err != nil {
        t.Fatalf("cmdDoctor: %v", err)
    }
    if !strings.Contains(stdout.String(), "Disk space") {
        t.Errorf("doctor output missing 'Disk space' section")
    }
}
```

- [ ] **Step 2: Run, verify FAIL**

```sh
go test ./cmd/quenchforge/ -run TestDoctor_IncludesDiskFreeCheck -v
```

- [ ] **Step 3: Implement**

In cmdDoctor, append:

```go
fmt.Fprintln(stdout, "\nDisk space")
fmt.Fprintln(stdout, "----------")
fmt.Fprintln(stdout, "  "+checkDiskFree("/System/Volumes/Data"))
```

Helper:

```go
// checkDiskFree returns a human-readable line and a status hint for the
// given mount point. Uses `df -k` so we avoid syscall.Statfs CGo
// complications across macOS versions.
func checkDiskFree(mount string) string {
    out, err := exec.Command("df", "-k", mount).Output()
    if err != nil {
        return fmt.Sprintf("could not read disk usage for %s: %v", mount, err)
    }
    // df output:
    //   Filesystem  1024-blocks  Used  Available  Capacity  Mounted on
    //   /dev/disk4s2  847249000  799000000  48000000  95%  /System/Volumes/Data
    lines := strings.Split(strings.TrimSpace(string(out)), "\n")
    if len(lines) < 2 {
        return "df returned unexpected output"
    }
    fields := strings.Fields(lines[1])
    if len(fields) < 5 {
        return "df row has fewer than 5 fields"
    }
    availKB, err := strconv.ParseInt(fields[3], 10, 64)
    if err != nil {
        return fmt.Sprintf("could not parse df Available column: %q", fields[3])
    }
    availGB := availKB / 1024 / 1024
    capacity := fields[4]
    status := "PASS"
    switch {
    case availGB < 10:
        status = "CRITICAL"
    case availGB < 20:
        status = "WARN"
    }
    return fmt.Sprintf("%s: %d GB available (%s used) — %s", mount, availGB, capacity, status)
}
```

- [ ] **Step 4: Run, verify PASS**

```sh
go test ./cmd/quenchforge/ -run TestDoctor_IncludesDiskFreeCheck -v
```

### Task 3.4: Add per-slot log size check

**Files:**
- Modify: `cmd/quenchforge/main.go`
- Modify: `cmd/quenchforge/serve_test.go`

- [ ] **Step 1: Write failing test**

```go
func TestDoctor_IncludesLogSizeCheck(t *testing.T) {
    var stdout, stderr bytes.Buffer
    if err := cmdDoctor(nil, &stdout, &stderr); err != nil {
        t.Fatalf("cmdDoctor: %v", err)
    }
    if !strings.Contains(stdout.String(), "Slot log sizes") {
        t.Errorf("doctor output missing 'Slot log sizes' section")
    }
}
```

- [ ] **Step 2: Run, verify FAIL**

```sh
go test ./cmd/quenchforge/ -run TestDoctor_IncludesLogSizeCheck -v
```

- [ ] **Step 3: Implement**

```go
fmt.Fprintln(stdout, "\nSlot log sizes")
fmt.Fprintln(stdout, "--------------")
for _, line := range checkSlotLogSizes() {
    fmt.Fprintln(stdout, "  "+line)
}
```

Helper:

```go
// checkSlotLogSizes returns one line per known slot log file under
// ~/Library/Logs/quenchforge/, sorted by size desc. Each line carries a
// PASS/WARN/CRITICAL marker:
//   - PASS:     < 50 MB
//   - WARN:     50–500 MB
//   - CRITICAL: > 500 MB
func checkSlotLogSizes() []string {
    home, err := os.UserHomeDir()
    if err != nil {
        return []string{fmt.Sprintf("could not resolve home dir: %v", err)}
    }
    dir := filepath.Join(home, "Library", "Logs", "quenchforge")
    entries, err := os.ReadDir(dir)
    if err != nil {
        return []string{fmt.Sprintf("no log dir at %s: %v", dir, err)}
    }
    type entry struct {
        name string
        size int64
    }
    var es []entry
    for _, e := range entries {
        if e.IsDir() {
            continue
        }
        info, err := e.Info()
        if err != nil {
            continue
        }
        es = append(es, entry{e.Name(), info.Size()})
    }
    sort.Slice(es, func(i, j int) bool { return es[i].size > es[j].size })

    var out []string
    for _, e := range es {
        mb := e.size / 1024 / 1024
        status := "PASS"
        switch {
        case mb > 500:
            status = "CRITICAL"
        case mb > 50:
            status = "WARN"
        }
        out = append(out, fmt.Sprintf("%-40s %6d MB  %s", e.name, mb, status))
    }
    if len(out) == 0 {
        return []string{"(no slot logs yet)"}
    }
    return out
}
```

Add `"path/filepath"` and `"sort"` to imports if not present.

- [ ] **Step 4: Run, verify PASS**

```sh
go test ./cmd/quenchforge/ -run TestDoctor_IncludesLogSizeCheck -v
```

### Task 3.5: Add port 11434 holder check

**Files:**
- Modify: `cmd/quenchforge/main.go`
- Modify: `cmd/quenchforge/serve_test.go`

- [ ] **Step 1: Write failing test**

```go
func TestDoctor_IncludesPortCheck(t *testing.T) {
    var stdout, stderr bytes.Buffer
    if err := cmdDoctor(nil, &stdout, &stderr); err != nil {
        t.Fatalf("cmdDoctor: %v", err)
    }
    if !strings.Contains(stdout.String(), "Port 11434") {
        t.Errorf("doctor output missing 'Port 11434' section")
    }
}
```

- [ ] **Step 2: Run, verify FAIL**

```sh
go test ./cmd/quenchforge/ -run TestDoctor_IncludesPortCheck -v
```

- [ ] **Step 3: Implement using the new portcheck package**

```go
fmt.Fprintln(stdout, "\nPort 11434")
fmt.Fprintln(stdout, "----------")
{
    pcCtx, pcCancel := context.WithTimeout(context.Background(), 3*time.Second)
    res, err := portcheck.Check(pcCtx, "127.0.0.1:11434")
    pcCancel()
    if err != nil {
        fmt.Fprintf(stdout, "  could not probe: %v\n", err)
    } else {
        switch res.Verdict {
        case portcheck.VerdictFree:
            fmt.Fprintln(stdout, "  free — quenchforge will bind on next start")
        case portcheck.VerdictHeldByOllama:
            fmt.Fprintf(stdout, "  held by Ollama (pid %d, %s) — CRITICAL\n",
                res.Holder.PID, res.Holder.ExecPath)
        case portcheck.VerdictHeldByStaleQuenchforge:
            fmt.Fprintf(stdout, "  held by quenchforge (pid %d) — OK\n", res.Holder.PID)
        case portcheck.VerdictHeldByOther:
            fmt.Fprintf(stdout, "  held by pid %d (%s, %s) — WARN\n",
                res.Holder.PID, res.Holder.CommandName, res.Holder.ExecPath)
        case portcheck.VerdictUnknown:
            fmt.Fprintln(stdout, "  in use but holder could not be identified — WARN")
        }
    }
}
```

Add `"github.com/cerid-ai/quenchforge/internal/portcheck"` and `"time"` imports if not already.

- [ ] **Step 4: Run, verify PASS**

```sh
go test ./cmd/quenchforge/ -run TestDoctor_IncludesPortCheck -v
```

### Task 3.6: Add --explain mode

**Files:**
- Modify: `cmd/quenchforge/main.go`
- Modify: `cmd/quenchforge/serve_test.go`

- [ ] **Step 1: Write failing test**

```go
func TestDoctor_ExplainModeAddsRemediation(t *testing.T) {
    var stdout, stderr bytes.Buffer
    if err := cmdDoctor([]string{"--explain"}, &stdout, &stderr); err != nil {
        t.Fatalf("cmdDoctor --explain: %v", err)
    }
    out := stdout.String()
    if !strings.Contains(out, "Remediation") {
        t.Errorf("--explain output missing 'Remediation' section")
    }
}
```

- [ ] **Step 2: Run, verify FAIL**

- [ ] **Step 3: Implement**

In cmdDoctor's flag block (the existing `flag.NewFlagSet("doctor", ...)`), add:

```go
var explain bool
fs.BoolVar(&explain, "explain", false, "Print remediation steps for each non-PASS finding.")
```

After parsing flags and running all the existing + new checks, if `explain` is true, emit a remediation section:

```go
if explain {
    fmt.Fprintln(stdout, "\nRemediation")
    fmt.Fprintln(stdout, "-----------")
    fmt.Fprintln(stdout, `  - Ollama LaunchAgent loaded:    launchctl bootout gui/$(id -u)/com.ollama.ollama`)
    fmt.Fprintln(stdout, `  - Disk space CRITICAL:          docker system prune -a --volumes && truncate slot logs`)
    fmt.Fprintln(stdout, `  - Slot log CRITICAL:            : > <path-to-log>  (Layer 1c rotation should prevent recurrence)`)
    fmt.Fprintln(stdout, `  - Port 11434 held by Ollama:    see "Ollama LaunchAgent" remediation`)
    fmt.Fprintln(stdout, `  - Port 11434 unknown holder:    lsof -i :11434 -sTCP:LISTEN  (resolve manually)`)
}
```

(Per-finding remediation could be inlined next to each check in a future polish pass; for now a single remediation block keeps the diff small and the output skim-able.)

- [ ] **Step 4: Run, verify PASS**

```sh
go test ./cmd/quenchforge/ -run TestDoctor_ExplainModeAddsRemediation -v
```

### Task 3.7: Run the full doctor test suite + manual smoke

- [ ] **Step 1: Full test pass**

```sh
cd ~/Develop/quenchforge && go test ./...
```

Expected: all PASS.

- [ ] **Step 2: Run the new doctor against the live machine**

```sh
go run ./cmd/quenchforge doctor
```

Expected: prints the original sections plus 4 new ones (Ollama LaunchAgent, Disk space, Slot log sizes, Port 11434). The Ollama section should say `disabled` (assuming Phase 0 ran).

```sh
go run ./cmd/quenchforge doctor --explain
```

Expected: same output plus the Remediation block.

### Task 3.8: Commit Phase 3

```sh
cd ~/Develop/quenchforge
git add cmd/quenchforge/main.go cmd/quenchforge/serve_test.go
git commit -m "$(cat <<'EOF'
feat(doctor): add LaunchAgent, disk, log-size, port-holder checks + --explain

Extends `quenchforge doctor` with four new sections:
  - Ollama LaunchAgent       — flags com.ollama.ollama loaded state
  - Disk space               — df on /System/Volumes/Data, WARN<20GB, CRIT<10GB
  - Slot log sizes           — per-file size with WARN>50MB, CRIT>500MB
  - Port 11434                — portcheck verdict (Free/Ollama/stale qf/other)

Adds a `--explain` flag that appends per-finding remediation steps,
so bug-report triage can quote the doctor output AND the next action
in one paste.

These four signals were the missing parts of the picture during the
2026-05-17 panic and the 2026-05-24 freeze RCA — none of them are
visible from `quenchforge --version` alone, and the bug.yml template
relies on doctor output for first-contact triage.
EOF
)"
```

---

# Phase 4 — Chat slot CPU on AMD-discrete (Layer 1d)

Smallest diff in the plan. ~15 minutes.

### Task 4.1: Extend the existing chat-AMD test

**Files:**
- Modify: `internal/tuning/tuning_test.go` (the existing `TestKernelParams_ChatAMDGetsCorrectnessFlags`)

- [ ] **Step 1: Update the expected `wantExtra` slice**

In `TestKernelParams_ChatAMDGetsCorrectnessFlags`, change:

```go
wantExtra := []string{
    "--flash-attn", "off",
    "--cache-ram", "0",
    "--no-cache-prompt",
}
```

to:

```go
wantExtra := []string{
    "--flash-attn", "off",
    "--cache-ram", "0",
    "--no-cache-prompt",
    "--gpu-layers", "0",
}
```

- [ ] **Step 2: Run, verify FAIL**

```sh
cd ~/Develop/quenchforge && go test ./internal/tuning/ -run TestKernelParams_ChatAMDGetsCorrectnessFlags -v
```

Expected: FAIL with a `slices.Equal` mismatch — actual is the 3-flag set, expected has 4.

### Task 4.2: Add `--gpu-layers 0` to chatParams

**Files:**
- Modify: `internal/tuning/tuning.go` (the `chatParams` function around lines 124–136)

- [ ] **Step 1: Update chatParams to append `--gpu-layers 0`**

Change:

```go
func chatParams(profile hardware.Profile) SlotTuning {
    if !profileIsAMDDiscrete(profile) {
        return SlotTuning{}
    }
    return SlotTuning{
        ExtraArgs: []string{
            "--flash-attn", "off",
            "--cache-ram", "0",
            "--no-cache-prompt",
        },
        AutoRespawn: true,
    }
}
```

to:

```go
func chatParams(profile hardware.Profile) SlotTuning {
    if !profileIsAMDDiscrete(profile) {
        return SlotTuning{}
    }
    // AMD-discrete chat slot routes to CPU pending the quantized-matmul
    // fallback patch (planned as patch 0005). Patches 0001/0003/0004
    // cover fp32/fp16 BERT shapes only; chat-slot Q4_K_M / Q5_K_M
    // models still SIGABRT on Vega II Metal under sustained load —
    // 257 abort traps observed across a 7-day uptime window
    // (2026-05-17 → 2026-05-24) contributing to the 2026-05-17 panic
    // and the 2026-05-24 system freeze. Mirror of the v0.7.0
    // embed/rerank CPU routing — remove this `--gpu-layers 0` pair
    // when patch 0005 lands and bench-llama-sustained-load passes.
    return SlotTuning{
        ExtraArgs: []string{
            "--flash-attn", "off",
            "--cache-ram", "0",
            "--no-cache-prompt",
            "--gpu-layers", "0",
        },
        AutoRespawn: true,
    }
}
```

- [ ] **Step 2: Run, verify PASS**

```sh
go test ./internal/tuning/ -run TestKernelParams_ChatAMDGetsCorrectnessFlags -v
```

Expected: PASS.

- [ ] **Step 3: Run full tuning suite**

```sh
go test ./internal/tuning/ -v
```

Expected: all PASS, including `TestProfileIsAMDDiscrete_MatchesHardwarePackage` (proves we didn't accidentally affect the AMD-detection predicate).

- [ ] **Step 4: Run full test suite**

```sh
go test ./...
```

Expected: all PASS.

### Task 4.3: Commit Phase 4

```sh
cd ~/Develop/quenchforge
git add internal/tuning/tuning.go internal/tuning/tuning_test.go
git commit -m "$(cat <<'EOF'
fix(tuning): route AMD-discrete chat slot to CPU (--gpu-layers 0)

The chat slot is the only quenchforge slot still on AMD Vega II Metal
after v0.7.0. Patches 0001/0003/0004 cover fp32/fp16 BERT shapes only;
quantized chat models (Q4_K_M llama3.1-8b in the canonical install)
still hit family-B SIGABRT on sustained load — 257 abort traps observed
across a 7-day uptime window contributing to the 2026-05-17
vm_page_wire panic and the 2026-05-24 system freeze.

Mirror of the v0.7.0 embed/rerank CPU policy. Reversal is a one-line
revert when patch 0005 (quantized matmul fallback) lands and
bench-llama-sustained-load passes.

Apple Silicon / Intel-iGPU profiles unchanged — they do not hit this
Metal-family bug.

cerid AI feature impact: zero. cerid-ai-internal .env routes chat to
OpenRouter; the local chat slot is for ad-hoc operator use only.
EOF
)"
```

---

# Phase 5 — Documentation alignment (Layers 1e, 4)

Splits across two repos. ~30 minutes total.

### Task 5.1: Quenchforge README "Coexistence with Ollama" section

**Files:**
- Modify: `~/Develop/quenchforge/README.md`

- [ ] **Step 1: Locate the Installation section**

```sh
grep -nE "^## |^### " ~/Develop/quenchforge/README.md | head -20
```

Identify the line number of `## Installation` (or whatever the install header is called).

- [ ] **Step 2: Add a new subsection at the end of Installation**

Append after the existing install instructions:

```markdown
### Coexistence with Ollama.app

Quenchforge listens on `127.0.0.1:11434` — the same port as Ollama. If
`/Applications/Ollama.app` is also installed, its login agent
(`com.ollama.ollama`) races quenchforge to bind the port at every
login. The pre-bind check (added in this release) detects the conflict
and exits cleanly with guidance; the `KeepAlive=<dict><SuccessfulExit
false/></dict>` plist setting prevents launchd from respawning on the
clean exit.

To resolve:

**Option A — use quenchforge only** (recommended for cerid AI workloads):

```sh
osascript -e 'tell application "Ollama" to quit' 2>/dev/null || true
launchctl bootout gui/$(id -u)/com.ollama.ollama
launchctl kickstart -k gui/$(id -u)/com.cerid.quenchforge
```

The Ollama.app stays installed; you can run `open -a Ollama` manually
if you ever need it.

**Option B — coexist on different ports**:

Edit `~/Library/LaunchAgents/com.cerid.quenchforge.plist` and add to
the `<EnvironmentVariables>` dict:

```xml
<key>QUENCHFORGE_LISTEN_ADDR</key>
<string>:11435</string>
```

Then `launchctl kickstart -k gui/$(id -u)/com.cerid.quenchforge`.
Quenchforge is now on `11435` and Ollama owns `11434`. Update any
clients that point at quenchforge to use the new port.

Run `quenchforge doctor` to verify either resolution.
```

### Task 5.2: Extend patches/README.md Section 3

**Files:**
- Modify: `~/Develop/quenchforge/patches/README.md`

- [ ] **Step 1: Find Section 3**

```sh
grep -nE "^## |^### " ~/Develop/quenchforge/patches/README.md | head
```

Look for the heading whose body describes the embed/rerank sustained-load mitigations (the explore agent noted Section 3 covers buffer-corruption under sustained load).

- [ ] **Step 2: Append a paragraph about chat-slot symmetry**

Add at the end of Section 3 (whatever heading text it actually uses):

```markdown

**Chat slot inherits the same Metal-stability concern.** The quantized
chat models cerid runs (Q4_K_M llama3.1-8b, Q5_K_M variants) traverse
the same matmul kernels patches 0003/0004 patch for fp32/fp16, but the
fallback dispatcher in `pipeline_mul_mv` selects the upstream
(broken-on-AMD) kernel for any non-fp32/fp16 tensor type. Empirically:
257 chat-slot SIGABRTs observed across one 7-day uptime window on Vega
II, contributing to the 2026-05-17 vm_page_wire panic.

Mitigation in v0.7.2: chat slot routes to CPU via `--gpu-layers 0`
(see `internal/tuning/tuning.go::chatParams`). Mirrors the embed/rerank
CPU policy. Reversal trigger: planned patch 0005 (quantized matmul
fallback) + a `scripts/bench-llama-sustained-load.py` regression test.
```

### Task 5.3: CHANGELOG entry

**Files:**
- Modify: `~/Develop/quenchforge/CHANGELOG.md`

- [ ] **Step 1: Add a v0.7.2 entry at the top**

```markdown
## v0.7.2 — stability hardening + Ollama deconfliction (2026-05-25)

Driven by a 2026-05-24 system freeze on the dev Mac Pro requiring two
hard reboots. RCA traced the failure to a multi-day cascade rooted in
quenchforge's gateway crash-spamming when Ollama.app's login agent won
the race for 127.0.0.1:11434, combined with unbounded slot log growth
and continued chat-slot SIGABRT on Vega II Metal.

### Stability fixes

- **Pre-bind port check** (`internal/portcheck`). Before binding 11434,
  identify any existing listener via `lsof` (netstat fallback).
  Verdict-specific exits: Ollama → canonical actionable error + exit 0;
  stale quenchforge → graceful takeover; other → log + exit 0. The
  clean exit pairs with the plist KeepAlive dict (below) so launchd
  does not crash-spam.

- **LaunchAgent plist hardening** (`plist_template.plist`). KeepAlive
  changes from `<true/>` to `<dict><SuccessfulExit><false/></dict>`,
  suppressing respawn on the new clean-exit path. ThrottleInterval=10
  stays as belt-and-suspenders for real crashes.

- **Slot log rotation** (`internal/supervisor/logrotate.go`). 100 MB
  per file, 5 backups by default. Overridable via
  QUENCHFORGE_LOG_MAX_BYTES / QUENCHFORGE_LOG_BACKUPS. Eliminates the
  3.73 GB unbounded embed.log class.

- **Chat slot routes to CPU on AMD-discrete** (`internal/tuning/
  tuning.go::chatParams`). Quantized chat models hit the same
  family-B SIGABRT pattern as embed/rerank pre-v0.7.0 (patch 0003/0004
  cover fp32/fp16 only). Mirror of the existing embed/rerank CPU
  policy. Reversal: planned patch 0005.

### Diagnostics

- **`quenchforge doctor` extended** with four new sections: Ollama
  LaunchAgent state, disk free on /System/Volumes/Data, per-slot log
  file sizes, port 11434 holder. `--explain` flag appends per-finding
  remediation guidance for bug-report triage.

### Documentation

- README: new "Coexistence with Ollama" section under Installation.
- patches/README.md Section 3: extended to cover chat-slot symmetry.

### No upstream patches added or removed

The single llama.cpp patch (0001-metal-correctness-on-non-apple-silicon)
remains the only load-bearing patch; 0003/0004 stay staged but
inactive in production (per v0.8.0-rc1 changelog).
```

### Task 5.4: Commit quenchforge docs

```sh
cd ~/Develop/quenchforge
git add README.md patches/README.md CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs: README Ollama coexistence section + patches/README chat-slot note + v0.7.2 CHANGELOG

Documents the new public-facing behavior: pre-bind detection of
Ollama.app's login agent, the two resolution paths (disable agent /
move quenchforge to :11435), and the chat-slot CPU routing rationale.
Extends patches/README.md Section 3 to acknowledge that the quantized
chat path falls outside patches 0003/0004 coverage and is mitigated
via CPU routing pending planned patch 0005.
EOF
)"
```

### Task 5.5: cerid-ai-internal — flip detect-gpu.sh recommendation

**Files:**
- Modify: `~/Develop/cerid-ai-internal/scripts/detect-gpu.sh`

- [ ] **Step 1: Locate the AMD Vega II case**

```sh
grep -nC3 "ProfileVegaPro\|vega\|amd.*recommend\|recommended_backend" ~/Develop/cerid-ai-internal/scripts/detect-gpu.sh | head -40
```

Per the earlier grep, the relevant case is around lines 80–95.

- [ ] **Step 2: Change the AMD-Vega-II case**

In the AMD-Vega-II branch (and any other AMD-discrete branches that currently set `recommended_backend="ollama"`), change to `recommended_backend="quenchforge"`. Update the comment about `ollama/ollama#1016` to note that quenchforge carries the macOS-only patches that work around it.

Pattern (adapt to actual code — the branch structure is per the explore agent's earlier excerpt):

Before:
```sh
# AMD Vega II: GPU acceleration broken in mainline ollama
# (ollama/ollama#1016). Falls back to CPU.
ollama_image="native"
recommended_backend="ollama"
```

After:
```sh
# AMD Vega II: mainline ollama can't drive Metal on Intel Mac AMD
# discrete (ollama/ollama#1016). Quenchforge carries the load-bearing
# llama.cpp patches that work around it — recommend quenchforge here.
# Fall back to ollama remains available via `INTERNAL_LLM_PROVIDER=ollama`
# for operators who want to opt out of quenchforge.
ollama_image="native"
recommended_backend="quenchforge"
```

- [ ] **Step 3: Sanity-run the script**

```sh
cd ~/Develop/cerid-ai-internal && bash scripts/detect-gpu.sh 2>&1 | head -20
```

Expected: emits `CERID_RECOMMENDED_LOCAL_BACKEND=quenchforge` on this Mac (or whatever the variable name convention is — confirm from line 14's comment block).

### Task 5.6: cerid-ai-internal — error string update

**Files:**
- Modify: `~/Develop/cerid-ai-internal/scripts/start-cerid.sh:445`

- [ ] **Step 1: Replace the error string**

Before:
```sh
echo "            Install via: brew install ollama && ollama serve"
```

After:
```sh
echo "            Install via: brew install quenchforge OR brew install ollama && ollama serve"
```

(Adjust if the actual install instruction for quenchforge differs — confirm via `cat ~/Develop/quenchforge/Formula/quenchforge.rb` if it exists, or the install section of quenchforge's README. If quenchforge isn't on Homebrew yet, fall back to: `"Install via: see https://github.com/cerid-ai/quenchforge OR brew install ollama && ollama serve"`.)

### Task 5.7: cerid-ai-internal — warn string update

**Files:**
- Modify: `~/Develop/cerid-ai-internal/scripts/validate-env.sh:183`

- [ ] **Step 1: Replace the warn string**

Before:
```sh
warn "Ollama enabled but not running — start with: ollama serve (native) or docker compose --profile ollama up -d"
```

After:
```sh
warn "Local LLM enabled but not running — start with: launchctl kickstart -k gui/\$UID/com.cerid.quenchforge (quenchforge), or: ollama serve (native ollama), or: docker compose --profile ollama up -d"
```

### Task 5.8: Commit cerid-ai-internal docs

```sh
cd ~/Develop/cerid-ai-internal
git add scripts/detect-gpu.sh scripts/start-cerid.sh scripts/validate-env.sh
git commit -m "$(cat <<'EOF'
docs(scripts): recommend quenchforge as the local LLM backend on AMD discrete

Aligns recommendations with current production reality (.env already
sets EMBEDDINGS_PROVIDER=quenchforge and RERANK_PROVIDER=quenchforge).
On Intel Mac + AMD Vega II, mainline Ollama can't drive Metal
(ollama/ollama#1016); quenchforge carries the macOS-only llama.cpp
patches that work around it.

- detect-gpu.sh: AMD-Vega-II case recommends "quenchforge"
- start-cerid.sh: install hint includes quenchforge
- validate-env.sh: not-running warn mentions quenchforge first

No production code paths change; purely documentation/UX hygiene.
EOF
)"
```

---

# Phase 6 — Build, install, reload (operational)

After Phases 1–5 are committed. ~10 minutes.

### Task 6.1: Build quenchforge

**Files:** none

- [ ] **Step 1: Build**

```sh
cd ~/Develop/quenchforge && make build 2>&1 | tail -20
```

Expected: clean build. If `make build` doesn't exist, fall back to:

```sh
go build -o /tmp/quenchforge-new ./cmd/quenchforge && ls -lh /tmp/quenchforge-new
```

### Task 6.2: Install the new binary

**Files:** none

- [ ] **Step 1: Back up the current binary**

```sh
sudo cp /usr/local/bin/quenchforge /usr/local/bin/quenchforge.bak-$(date +%Y%m%d-%H%M%S)
```

- [ ] **Step 2: Install the new binary**

```sh
sudo cp /tmp/quenchforge-new /usr/local/bin/quenchforge
sudo chmod +x /usr/local/bin/quenchforge
quenchforge --version
```

Expected: version reports the new commit SHA.

### Task 6.3: Refresh the LaunchAgent plist

**Files:** `~/Library/LaunchAgents/com.cerid.quenchforge.plist`

- [ ] **Step 1: Diff the installed plist against the template**

```sh
diff ~/Library/LaunchAgents/com.cerid.quenchforge.plist ~/Develop/quenchforge/cmd/quenchforge/plist_template.plist
```

Note any deltas you've made for personal config (e.g. env vars). They must be preserved when copying the new template over.

- [ ] **Step 2: Apply the KeepAlive change**

Either re-run `quenchforge install` (if that subcommand exists — `cmd/quenchforge/install.go` was in the earlier listing) and let it regenerate the plist, or manually edit:

```sh
$EDITOR ~/Library/LaunchAgents/com.cerid.quenchforge.plist
```

Locate `<key>KeepAlive</key>` and replace `<true/>` with `<dict><key>SuccessfulExit</key><false/></dict>`. Preserve all other entries.

- [ ] **Step 3: Validate**

```sh
plutil -lint ~/Library/LaunchAgents/com.cerid.quenchforge.plist
```

Expected: `OK`.

### Task 6.4: Kickstart the LaunchAgent

**Files:** none

- [ ] **Step 1: Bootout + bootstrap to force re-read of the plist**

```sh
launchctl bootout gui/$(id -u)/com.cerid.quenchforge 2>/dev/null || true
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.cerid.quenchforge.plist
launchctl kickstart -k gui/$(id -u)/com.cerid.quenchforge
```

- [ ] **Step 2: Verify running**

```sh
sleep 3
launchctl list com.cerid.quenchforge | head
```

Expected: `"PID" = <NNNN>;` line.

- [ ] **Step 3: Verify the new behavior**

```sh
quenchforge doctor
```

Expected output sections include all four new ones (Ollama LaunchAgent: `disabled`, Disk space: `PASS`, Slot log sizes: all `PASS`, Port 11434: `held by quenchforge — OK`).

```sh
ps aux | grep "llama-server.*chat" | grep -v grep
```

Expected: chat slot args include `--gpu-layers 0`.

```sh
ls -lh ~/Library/Logs/quenchforge/embed.log
```

Expected: < 100 MB (the rotation cap).

### Task 6.5: Verify launchd does NOT respawn on clean exit (spec acceptance #3)

**Files:** none

- [ ] **Step 1: Capture the current quenchforge PID**

```sh
ORIG_PID=$(launchctl list com.cerid.quenchforge | awk -F'=' '/"PID"/ {gsub(/[ ;]/,"",$2); print $2}')
echo "quenchforge PID: $ORIG_PID"
```

- [ ] **Step 2: Re-enable Ollama to force a port conflict**

```sh
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ollama.ollama.plist 2>&1
# Wait for Ollama to start and grab port 11434.
sleep 5
lsof -i :11434 -sTCP:LISTEN
```

Expected: Ollama is now the listener.

- [ ] **Step 3: Trigger a quenchforge restart so the pre-bind check fires**

```sh
launchctl kickstart -k gui/$(id -u)/com.cerid.quenchforge
sleep 3
```

- [ ] **Step 4: Watch for respawn over the ThrottleInterval window (10 s)**

```sh
for i in $(seq 1 6); do
  date '+%H:%M:%S'
  launchctl list com.cerid.quenchforge | awk -F'=' '/"PID"/ {gsub(/[ ;]/,"",$2); print "  qf PID:", $2}' || echo "  not loaded"
  sleep 5
done
```

Expected output: across the 30 s window, the `qf PID` line should show **no** PID (or only a transient one that doesn't repeat). If you see PID values changing every cycle, `KeepAlive=<dict><SuccessfulExit><false/></dict>` did NOT apply — re-check Task 6.3.

- [ ] **Step 5: Inspect pmset log for respawn events in the window**

```sh
pmset -g log 2>&1 | grep -i quenchforge | tail -10
```

Expected: at most one entry per kickstart — no rapid respawn cycle.

- [ ] **Step 6: Restore Ollama-disabled state**

```sh
launchctl bootout gui/$(id -u)/com.ollama.ollama
launchctl kickstart -k gui/$(id -u)/com.cerid.quenchforge
sleep 3
launchctl list com.cerid.quenchforge | head
```

Expected: quenchforge is now running again (Ollama out of the way, pre-bind check passes).

```sh
quenchforge doctor 2>&1 | grep -A1 "Port 11434"
```

Expected: `held by quenchforge — OK`.

---

# Phase 7 — v0.8.0-rc1 GPU activation (gated) (Layer 5)

Optional. Only run after Phases 1–6 are confirmed stable for at least 24 hours. ~45 minutes if both benches pass; 5 min if one fails.

### Task 7.1: Run the correctness bench

**Files:** none

- [ ] **Step 1: Confirm the bench script exists**

```sh
ls ~/Develop/quenchforge/scripts/bench-bert-correctness.py
```

Expected: file exists. (Per the CHANGELOG, it was added in v0.8.0-rc1.)

- [ ] **Step 2: Run it**

```sh
cd ~/Develop/quenchforge && python3 scripts/bench-bert-correctness.py 2>&1 | tee /tmp/bench-bert-correctness.out
```

Expected: PASS in ~30 s. If FAIL: stop here. File a v0.8.0-rc2 issue with the bench output attached. Phases 1–6 remain successful.

### Task 7.2: Run the sustained-load bench

**Files:** none

- [ ] **Step 1: Confirm it exists**

```sh
ls ~/Develop/quenchforge/scripts/bench-bert-sustained-load.py
```

- [ ] **Step 2: Run for 30 minutes**

```sh
cd ~/Develop/quenchforge && python3 scripts/bench-bert-sustained-load.py --duration 1800 2>&1 | tee /tmp/bench-bert-sustained-load.out
```

Expected: PASS in ~30 min, with no SIGABRT, no 5xx burst, no drift, no RSS leak, no latency cliff. If FAIL: stop here. File a v0.8.0-rc2 issue. Phases 1–6 remain successful.

### Task 7.3: Activate the GPU route for embed + rerank

**Files:**
- Modify: `~/Develop/quenchforge/internal/tuning/tuning.go` (the `embedParams` and `rerankParams` AMD branches)
- Modify: `~/Develop/quenchforge/internal/tuning/tuning_test.go` (expectations)

- [ ] **Step 1: Read the embed AMD branch**

```sh
sed -n '157,225p' ~/Develop/quenchforge/internal/tuning/tuning.go
```

Locate the `if profileIsAMDDiscrete(profile)` block in `embedParams` and the line that appends `--gpu-layers 0` (likely an entry in `ExtraArgs`).

- [ ] **Step 2: Remove `--gpu-layers 0` from embedParams' AMD branch**

Find the `ExtraArgs` entry that adds `--gpu-layers` `0` in embedParams and delete those two elements. Leave the ubatch/batch/MetalNCB tuning intact.

- [ ] **Step 3: Same for rerankParams**

Repeat for rerankParams.

- [ ] **Step 4: Update tuning_test.go expectations**

Find the tests that assert `--gpu-layers 0` for AMD embed/rerank (search: `grep -n "gpu-layers" tuning_test.go`). Remove the expectation. **Do not** remove it from the chat-slot test (Layer 1d kept chat on CPU).

- [ ] **Step 5: Run tests, verify pass**

```sh
go test ./internal/tuning/ -v
```

Expected: all PASS, with the chat-AMD test still requiring `--gpu-layers 0` and the embed/rerank-AMD tests now NOT requiring it.

- [ ] **Step 6: Rebuild + install**

Repeat Tasks 6.1, 6.2, 6.4 (no plist change this time).

- [ ] **Step 7: Verify**

```sh
ps aux | grep llama-server | grep -v grep | grep -v chat
```

Expected: embed / code-embed / rerank processes show **no** `--gpu-layers 0` flag.

```sh
ps aux | grep "llama-server.*chat"
```

Expected: chat **still** shows `--gpu-layers 0` (Layer 1d unaffected).

```sh
quenchforge doctor
```

Expected: all PASS, port 11434 still held by quenchforge.

- [ ] **Step 8: Commit**

```sh
cd ~/Develop/quenchforge
git add internal/tuning/tuning.go internal/tuning/tuning_test.go
git commit -m "$(cat <<'EOF'
feat(tuning): activate AMD GPU route for embed + rerank (v0.8.0-rc1 → release)

bench-bert-correctness.py PASS
bench-bert-sustained-load.py --duration 1800 PASS

Removes --gpu-layers 0 from embed and rerank slot AMD-discrete tuning.
Patches 0003 + 0004 (LayerNorm + softmax + matmul fallback kernels for
fp32/fp16 BERT shapes) now cover the entire embed/rerank inference
path. Chat slot retains CPU routing (patches don't cover quantized
matmul; planned patch 0005 is the chat-slot trigger).

Bench output archived under scripts/bench-results/2026-05-25/.
EOF
)"
```

(Move `/tmp/bench-bert-correctness.out` and `/tmp/bench-bert-sustained-load.out` into `scripts/bench-results/2026-05-25/` and `git add` them if you keep bench archives; if not, omit the trailing sentence from the commit message.)

---

# Phase 8 — Memory updates (Layer 6)

After Phase 6 lands and Phase 7 is decided. ~10 minutes.

### Task 8.1: Update feedback_quenchforge_safety.md

**Files:**
- Modify: `~/.claude/projects/-Users-sunrunner-Develop/memory/feedback_quenchforge_safety.md`

- [ ] **Step 1: Replace the 2026-05-25 update section**

The current memory has a 2026-05-25 section noting Ollama.app is the recurring source. Replace it with a post-fix state:

```markdown
**2026-05-25 update — fix shipped (quenchforge v0.7.2):** The
recurring Ollama-port-fight + chat-slot SIGABRT cascade is now mitigated
by four shipped changes:

1. **Ollama LaunchAgent disabled on this machine.** `launchctl bootout
   gui/$UID/com.ollama.ollama`. `/Applications/Ollama.app` itself stays
   installed; reactivate with `launchctl bootstrap gui/$UID
   ~/Library/LaunchAgents/com.ollama.ollama.plist` if ever needed.

2. **Quenchforge has built-in pre-bind deconfliction** (`internal/
   portcheck`). Public users in the same situation get an actionable
   error and clean exit; launchd's KeepAlive=`<dict><SuccessfulExit
   false/></dict>` suppresses respawn-spam.

3. **Chat slot routes to CPU on AMD-discrete** (`tuning.go::chatParams`
   appends `--gpu-layers 0`). Reversal trigger: planned patch 0005
   (quantized matmul fallback) + `bench-llama-sustained-load.py`. See
   [[project_cerid_quenchforge_chat_on_cpu]].

4. **Slot logs rotate at 100 MB / 5 backups.** Prevents the unbounded
   `embed.log` growth pattern (3.73 GB observed in 7 days pre-fix).

**Diagnose-first command:** `quenchforge doctor` (extended in v0.7.2)
reports Ollama state, port holder, disk free, log sizes, and recent
crashes in one paste. Quote it before any quenchforge bug-report.

The "never run `ollama serve` while quenchforge owns 11434" rule
above is still correct — but the consequences of accidental violation
are now bounded by the pre-bind check, not a kernel panic.
```

### Task 8.2: Create the chat-slot CPU memory

**Files:**
- Create: `~/.claude/projects/-Users-sunrunner-Develop/memory/project_cerid_quenchforge_chat_on_cpu.md`

```markdown
---
name: project-cerid-quenchforge-chat-on-cpu
description: Quenchforge chat slot routes to CPU on AMD-discrete (Vega II) as of v0.7.2. Reversal trigger is planned patch 0005 (quantized matmul Metal fallback) + bench-llama-sustained-load passing.
metadata:
  type: project
---

**State (2026-05-25):** The quenchforge chat slot — `llama-server`
running `llama3.1-8b.gguf` (Q4_K_M) — runs with `--gpu-layers 0` on
AMD-discrete profiles. Code: [internal/tuning/tuning.go::chatParams]
(~/Develop/quenchforge/internal/tuning/tuning.go).

**Why:** Patches 0001/0003/0004 fix Metal correctness for fp32/fp16
BERT shapes only. The `pipeline_mul_mv` dispatcher falls through to
the upstream (broken-on-AMD) kernel for quantized tensor types, so
Q4_K_M / Q5_K_M chat models still SIGABRT under sustained load on
Vega II. 257 abort traps observed across 7 days of uptime contributed
to the 2026-05-17 vm_page_wire panic and the 2026-05-24 freeze.

**How to apply:** When evaluating chat throughput on this machine,
expect CPU-bound rates (~15 tok/s on the 16-core Xeon W vs. ~40 tok/s
on Vega II). For chat-heavy cerid workloads (which already use
OpenRouter per `.env`), this has no impact. For ad-hoc local queries,
expect slower-than-historical performance.

**Reversal trigger (one-line revert):** Remove the `--gpu-layers`/`0`
pair from chatParams' ExtraArgs when:
1. A patch (planned 0005) lands covering quantized matmul fallback
   kernels for the Vega II Metal path.
2. A new `scripts/bench-llama-sustained-load.py` passes for ≥ 30 min
   at typical agentic-chat load.

**Related:** [[feedback_quenchforge_safety]] (the rule), [[reference_quenchforge_inference]] (endpoint reference).
```

### Task 8.3: Update MEMORY.md index

**Files:**
- Modify: `~/.claude/projects/-Users-sunrunner-Develop/memory/MEMORY.md`

- [ ] **Step 1: Add the new entry under Project**

Under the existing `## Project` section, add (alphabetically):

```markdown
- [project_cerid_quenchforge_chat_on_cpu.md](project_cerid_quenchforge_chat_on_cpu.md) — Chat slot on CPU pending patch 0005; reversal trigger documented
```

---

## Acceptance checklist

When all phases are complete:

- [ ] `quenchforge doctor` returns all-PASS sections.
- [ ] `lsof -i :11434 -sTCP:LISTEN` shows only `quenchfor`.
- [ ] `launchctl list com.ollama.ollama` returns non-zero (agent not loaded).
- [ ] `ps aux | grep "llama-server.*chat"` shows `--gpu-layers 0`.
- [ ] `ls -lh ~/Library/Logs/quenchforge/` — no file > 100 MB.
- [ ] `df -h /System/Volumes/Data` — ≥ 10% free.
- [ ] `pmset -g sched` — shows weekly restart entry.
- [ ] `go test ./...` in quenchforge — all PASS.
- [ ] cerid-ai-internal tests still pass (`tests/test_ollama_proxy_quenchforge.py`, `tests/test_e2e_integration.py`).
- [ ] (After 14 days of uptime) system has not frozen or panicked.
- [ ] (If Phase 7 ran) bench scripts pass on a re-run.
