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
