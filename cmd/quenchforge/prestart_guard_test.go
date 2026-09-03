// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"
)

// runGuard executes the shipped prestart guard against a test port with
// launchd stubbed out (an exported bash function shadows launchctl, so the
// guard never touches the real gui domain) and /bin/echo standing in for the
// quenchforge binary. extraStubs is bash sourced before the guard runs.
func runGuard(t *testing.T, port int, extraStubs string) (stdout string, err error) {
	t.Helper()
	guard := filepath.Join("prestart-guard.sh")
	if _, statErr := os.Stat(guard); statErr != nil {
		t.Fatalf("guard script: %v", statErr)
	}
	script := "launchctl() { return 1; }; export -f launchctl; " + extraStubs +
		" exec bash " + guard + " serve"
	cmd := exec.Command("/bin/bash", "-c", script)
	cmd.Env = append(os.Environ(),
		fmt.Sprintf("QUENCHFORGE_GUARD_PORT=%d", port),
		"QUENCHFORGE_BIN=/bin/echo",
	)
	out, err := cmd.CombinedOutput()
	return string(out), err
}

// squatter starts a listener on port, as a process named "ollama" (the only
// kind the guard evicts), that either dies on SIGTERM or ignores it. It is
// this test binary, copied under that name and re-run as
// TestHelperSquatter.
func squatter(t *testing.T, port int, ignoreTerm bool) {
	t.Helper()
	self, err := os.Executable()
	if err != nil {
		t.Fatalf("test binary: %v", err)
	}
	data, err := os.ReadFile(self)
	if err != nil {
		t.Fatalf("read test binary: %v", err)
	}
	bin := filepath.Join(t.TempDir(), "ollama")
	if err := os.WriteFile(bin, data, 0o755); err != nil {
		t.Fatalf("write squatter binary: %v", err)
	}
	cmd := exec.Command(bin, "-test.run=^TestHelperSquatter$")
	cmd.Env = append(os.Environ(),
		fmt.Sprintf("QF_GUARD_SQUATTER_PORT=%d", port),
		fmt.Sprintf("QF_GUARD_SQUATTER_IGNORE_TERM=%t", ignoreTerm),
	)
	if err := cmd.Start(); err != nil {
		t.Fatalf("start squatter: %v", err)
	}
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	})
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if c, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), 100*time.Millisecond); err == nil {
			_ = c.Close()
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("squatter never bound port %d", port)
}

// TestHelperSquatter is the squatter process body, not a test: it runs only
// when squatter() launches it with QF_GUARD_SQUATTER_PORT set.
func TestHelperSquatter(t *testing.T) {
	port := os.Getenv("QF_GUARD_SQUATTER_PORT")
	if port == "" {
		t.Skip("helper process for the prestart guard tests")
	}
	if os.Getenv("QF_GUARD_SQUATTER_IGNORE_TERM") == "true" {
		signal.Ignore(syscall.SIGTERM)
	}
	ln, err := net.Listen("tcp", "127.0.0.1:"+port)
	if err != nil {
		os.Exit(2)
	}
	defer ln.Close()
	time.Sleep(60 * time.Second)
	os.Exit(0)
}

// portHeld reports whether the port is still bound. It probes by binding
// rather than dialling: a listener with a full accept queue refuses connects
// while still holding the port.
func portHeld(port int) bool {
	ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		return true
	}
	_ = ln.Close()
	return false
}

func skipUnlessGuardTestable(t *testing.T) {
	t.Helper()
	if runtime.GOOS != "darwin" {
		t.Skip("the prestart guard is macOS-only")
	}
	// Never run this where an Ollama job could be evicted for real.
	if out, err := exec.Command("pgrep", "-l", "ollama").Output(); err == nil && len(out) > 0 {
		t.Skip("an ollama process is running on this host")
	}
	home, err := os.UserHomeDir()
	if err == nil {
		if _, err := os.Stat(filepath.Join(home, "Library", "LaunchAgents", "com.ollama.ollama.plist")); err == nil {
			t.Skip("an Ollama LaunchAgent is installed on this host")
		}
	}
}

func TestPrestartGuard_ExecsWhenPortIsFree(t *testing.T) {
	skipUnlessGuardTestable(t)
	port := freeTestPort(t)
	out, err := runGuard(t, port, "")
	if err != nil {
		t.Fatalf("guard exited %v with a free port:\n%s", err, out)
	}
	if !strings.Contains(out, "serve") {
		t.Errorf("guard did not exec the server with its args:\n%s", out)
	}
}

func TestPrestartGuard_EscalatesToSIGKILL(t *testing.T) {
	skipUnlessGuardTestable(t)
	port := freeTestPort(t)
	squatter(t, port, true) // ignores SIGTERM

	out, err := runGuard(t, port, "")
	if err != nil {
		t.Fatalf("guard exited %v; it should have escalated and started the server:\n%s", err, out)
	}
	if portHeld(port) {
		t.Errorf("guard handed off with :%d still held by a squatter that ignored SIGTERM:\n%s",
			port, out)
	}
}

func TestPrestartGuard_FailsWhenThePortIsNeverReleased(t *testing.T) {
	skipUnlessGuardTestable(t)
	port := freeTestPort(t)
	squatter(t, port, true)

	// Neutralise the signals so the port is never released: the guard must
	// report failure rather than exec into quenchforge's exit-0 yield path,
	// which launchd reads as a clean exit and never restarts.
	out, err := runGuard(t, port, "kill() { return 0; }; export -f kill;")
	if err == nil {
		t.Fatalf("guard exited 0 with the port still held; launchd will leave "+
			"quenchforge dead:\n%s", out)
	}
	if strings.Contains(out, "serve") {
		t.Errorf("guard exec'd the server anyway:\n%s", out)
	}
}
