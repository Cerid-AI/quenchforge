// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

// Slot readiness probing.
//
// startSlot returns as soon as cmd.Start() succeeds — llama-server has not
// bound its port yet, let alone loaded the model. Registering the upstream at
// that instant means the gateway proxies into a closed socket for the whole
// model-load window and callers get httputil's 502 "upstream unreachable"
// instead of the documented 503 + doctor hint. cerid reads 503 as "lane not
// ready" and 502 as a transport fault, and those 502s also seed the error
// rate that drives auto-backoff.
//
// So: probe first, register second. The probe runs in the background so a
// slow model load never delays the rest of startup — an unregistered lane
// 503s, which is the documented and expected shape.

package main

import (
	"context"
	"fmt"
	"io"
	"net"
	"time"

	"github.com/cerid-ai/quenchforge/internal/gateway"
)

// slotReadyPoll is the interval between readiness probes. Package var so
// tests can shrink it.
var slotReadyPoll = 250 * time.Millisecond

// slotReadyDialTimeout bounds one connect attempt.
const slotReadyDialTimeout = 500 * time.Millisecond

// defaultSlotReadySeconds is how long a slot may take to bind its port before
// the supervisor calls it out. Model load on a 7B over a cold page cache is
// tens of seconds; the ceiling is generous because the cost of being wrong is
// a false alarm in the log, not a dead lane.
const defaultSlotReadySeconds = 120

// slotReadyTimeout is the deadline before an unready slot is reported.
// Override with QUENCHFORGE_SLOT_READY_TIMEOUT_SEC.
func slotReadyTimeout() time.Duration {
	return time.Duration(envInt("QUENCHFORGE_SLOT_READY_TIMEOUT_SEC", defaultSlotReadySeconds)) * time.Second
}

// portAccepts reports whether something is accepting TCP connections on
// 127.0.0.1:port.
func portAccepts(port int) bool {
	conn, err := net.DialTimeout("tcp", fmt.Sprintf("127.0.0.1:%d", port), slotReadyDialTimeout)
	if err != nil {
		return false
	}
	_ = conn.Close()
	return true
}

// waitPortReady polls until 127.0.0.1:port accepts a connection. Returns
// false when ctx is cancelled or timeout elapses first; a non-positive
// timeout means "until ctx is done".
func waitPortReady(ctx context.Context, port int, timeout time.Duration) bool {
	var deadline time.Time
	if timeout > 0 {
		deadline = time.Now().Add(timeout)
	}
	for {
		if portAccepts(port) {
			return true
		}
		if !deadline.IsZero() && !time.Now().Before(deadline) {
			return false
		}
		select {
		case <-ctx.Done():
			return false
		case <-time.After(slotReadyPoll):
		}
	}
}

// registerWhenReady registers the slot's upstream once the slot is accepting
// connections. Returns immediately; the probe runs in the background.
//
// A slot that misses the readiness deadline is reported and then kept under
// probe until ctx ends — a slot that loads slowly must still end up
// registered, otherwise a slow start becomes a permanently dead lane.
func registerWhenReady(ctx context.Context, name string, port int, set func(string) error, w io.Writer) {
	upstream := fmt.Sprintf("http://127.0.0.1:%d", port)
	go func() {
		if !waitPortReady(ctx, port, slotReadyTimeout()) {
			if ctx.Err() != nil {
				return
			}
			fmt.Fprintf(w,
				"quenchforge: ERROR: %s slot has not accepted a connection on 127.0.0.1:%d "+
					"within %s — its lane returns 503 until it does.\n"+
					"  Check the slot log for a model-load failure; the probe keeps running.\n",
				name, port, slotReadyTimeout())
			if !waitPortReady(ctx, port, 0) {
				return
			}
			fmt.Fprintf(w, "quenchforge: %s slot is ready after all; registering %s\n", name, upstream)
		}
		if err := set(upstream); err != nil {
			fmt.Fprintf(w, "quenchforge: ERROR: %s slot upstream %s rejected: %v\n", name, upstream, err)
		}
	}()
}

// upstreamSetter adapts the gateway's per-kind upstream registration to the
// setter registerWhenReady calls once the slot answers.
func upstreamSetter(g *gateway.Gateway, kind gateway.SlotKind) func(string) error {
	return func(upstream string) error { return g.SetUpstream(kind, upstream) }
}

// cpuUpstreamSetter is upstreamSetter for the CPU twin of an "auto"-placed kind.
func cpuUpstreamSetter(g *gateway.Gateway, kind gateway.SlotKind) func(string) error {
	return func(upstream string) error { return g.SetCPUUpstream(kind, upstream) }
}
