// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"fmt"
	"net"
	"sync"
	"testing"
	"time"
)

// syncBuffer is a mutex-guarded bytes.Buffer — the readiness prober writes
// from its own goroutine.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

// freeTestPort returns a port with nothing listening on it.
func freeTestPort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()
	return port
}

func TestWaitPortReady_ListeningPort(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	port := ln.Addr().(*net.TCPAddr).Port

	if !waitPortReady(context.Background(), port, 2*time.Second) {
		t.Errorf("port %d has a listener; waitPortReady said otherwise", port)
	}
}

func TestWaitPortReady_DeadPort(t *testing.T) {
	port := freeTestPort(t)
	if waitPortReady(context.Background(), port, 300*time.Millisecond) {
		t.Errorf("nothing listens on port %d; waitPortReady said it was ready", port)
	}
}

func TestRegisterWhenReady_HoldsRegistrationUntilTheSlotAcceptsConnections(t *testing.T) {
	prev := slotReadyPoll
	slotReadyPoll = 20 * time.Millisecond
	defer func() { slotReadyPoll = prev }()

	port := freeTestPort(t)
	var mu sync.Mutex
	var registered []string
	set := func(u string) error {
		mu.Lock()
		defer mu.Unlock()
		registered = append(registered, u)
		return nil
	}
	seen := func() int {
		mu.Lock()
		defer mu.Unlock()
		return len(registered)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var out syncBuffer
	registerWhenReady(ctx, "rerank", port, set, &out)

	time.Sleep(120 * time.Millisecond)
	if n := seen(); n != 0 {
		t.Fatalf("upstream registered %d time(s) before the slot bound its port — "+
			"the gateway would proxy into a closed socket and return 502", n)
	}

	ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		t.Skipf("port %d was taken between probes: %v", port, err)
	}
	defer ln.Close()

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if seen() > 0 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if seen() != 1 {
		t.Fatalf("want exactly one registration once the slot is listening, got %d", seen())
	}
	mu.Lock()
	got := registered[0]
	mu.Unlock()
	if want := fmt.Sprintf("http://127.0.0.1:%d", port); got != want {
		t.Errorf("registered %q, want %q", got, want)
	}
}

func TestRegisterWhenReady_ReportsSetterErrors(t *testing.T) {
	prev := slotReadyPoll
	slotReadyPoll = 20 * time.Millisecond
	defer func() { slotReadyPoll = prev }()

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	port := ln.Addr().(*net.TCPAddr).Port

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var out syncBuffer
	registerWhenReady(ctx, "embed", port, func(string) error {
		return fmt.Errorf("bad upstream")
	}, &out)

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if out.String() != "" {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if got := out.String(); got == "" {
		t.Errorf("a rejected upstream registration must be reported, not discarded")
	}
}
