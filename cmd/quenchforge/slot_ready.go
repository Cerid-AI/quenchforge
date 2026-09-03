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
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
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
// connections AND has identified itself as serving expectModel. Returns
// immediately; the probe runs in the background.
//
// A slot that misses the readiness deadline is reported and then kept under
// probe until ctx ends — a slot that loads slowly must still end up
// registered, otherwise a slow start becomes a permanently dead lane.
//
// expectModel is the model the slot was started with. Accepting a connection
// proves only that something holds the port, and after an unclean restart
// that something can be an orphaned llama-server from the previous run still
// serving the previous model: registering it points the lane's live traffic
// at a stale process while /health reports ok. An empty expectModel means the
// slot's server has no /v1/models to ask (sd-server, bark, whisper-server),
// and TCP accept is all the evidence available.
func registerWhenReady(ctx context.Context, name string, port int, expectModel string, set func(string) error, w io.Writer) {
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
			fmt.Fprintf(w, "quenchforge: %s slot is accepting connections after all\n", name)
		}
		if !waitModelConfirmed(ctx, name, port, expectModel, w) {
			return
		}
		if err := set(upstream); err != nil {
			fmt.Fprintf(w, "quenchforge: ERROR: %s slot upstream %s rejected: %v\n", name, upstream, err)
		}
	}()
}

// slotModelProbeTimeout bounds one /v1/models request.
const slotModelProbeTimeout = 2 * time.Second

var slotProbeClient = &http.Client{Timeout: slotModelProbeTimeout}

// waitModelConfirmed blocks until whatever is listening on the slot port
// reports expectModel as loaded, and reports true when it does. An empty
// expectModel confirms immediately.
//
// A port that answers with a different model is never adopted, however long
// it holds the port: that is exactly the orphan case, and serving the wrong
// model is less recoverable than serving nothing — a 503 lane is visible,
// a lane answering from the previous model is not. Probing continues, so the
// lane comes up on its own once the real slot wins the port.
func waitModelConfirmed(ctx context.Context, name string, port int, expectModel string, w io.Writer) bool {
	if expectModel == "" {
		return true
	}
	deadline := time.Now().Add(slotReadyTimeout())
	var reportedMismatch string
	var reportedSilence bool
	for {
		switch served := probeSlotModel(ctx, port); {
		case served == "":
			if time.Now().After(deadline) && !reportedSilence {
				reportedSilence = true
				fmt.Fprintf(w,
					"quenchforge: ERROR: %s slot accepts connections on 127.0.0.1:%d but does not "+
						"report a loaded model at /v1/models — its lane returns 503 until it does.\n"+
						"  Check the slot log for a model-load failure; the probe keeps running.\n",
					name, port)
			}
		case slotModelMatches(expectModel, served):
			return true
		case served != reportedMismatch:
			reportedMismatch = served
			fmt.Fprintf(w,
				"quenchforge: ERROR: 127.0.0.1:%d serves %q, but the %s slot was started with %q — "+
					"refusing to register it as the %s upstream.\n"+
					"  Most likely an orphaned server from a previous run still holds the port; "+
					"stop it and restart quenchforge. The lane returns 503 until the port is ours.\n",
				port, served, name, expectModel, name)
		}
		select {
		case <-ctx.Done():
			return false
		case <-time.After(slotReadyPoll):
		}
	}
}

// probeSlotModel asks the process listening on the slot port which model it
// loaded, over the OpenAI /v1/models route llama-server serves. Returns ""
// when it cannot be asked — still loading, or not a server that has the
// route — which is never read as a match.
func probeSlotModel(ctx context.Context, port int) string {
	ctx, cancel := context.WithTimeout(ctx, slotModelProbeTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		fmt.Sprintf("http://127.0.0.1:%d/v1/models", port), nil)
	if err != nil {
		return ""
	}
	resp, err := slotProbeClient.Do(req)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return ""
	}
	var body struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 64*1024)).Decode(&body); err != nil || len(body.Data) == 0 {
		return ""
	}
	return body.Data[0].ID
}

// slotModelMatches reports whether the model a slot reports loaded is the one
// the slot was started with. One model reaches us under several spellings —
// the resolved GGUF path llama-server echoes back, the configured filename,
// and the Ollama-style "qwen2.5:7b" shorthand — so the comparison normalises
// separators and accepts a shorthand that is a component-boundary prefix.
// Unlike the gateway's request-time contract, an unreadable name is not a
// match: nothing is registered on no evidence.
func slotModelMatches(expected, served string) bool {
	e, s := normalizeSlotModelName(expected), normalizeSlotModelName(served)
	if e == "" || s == "" {
		return false
	}
	if e == s {
		return true
	}
	return modelNameIsPrefix(e, s) || modelNameIsPrefix(s, e)
}

// modelNameIsPrefix reports whether short is a component-boundary prefix of
// long ("qwen2.5-7b" of "qwen2.5-7b-instruct-q4-k-m", but not "qwen2.5-7").
func modelNameIsPrefix(short, long string) bool {
	return len(short) < len(long) && strings.HasPrefix(long, short) && long[len(short)] == '-'
}

// normalizeSlotModelName folds the spellings of one model onto a single key:
// lowercased, no directory prefix, no ".gguf" suffix, ':' / '_' / ' '
// collapsed to '-'. Mirrors the gateway's normalisation so the readiness
// check and the request-time model contract agree on what one model is.
func normalizeSlotModelName(name string) string {
	n := strings.ToLower(strings.TrimSpace(name))
	n = strings.TrimSuffix(n, ".gguf")
	if i := strings.LastIndex(n, "/"); i >= 0 {
		n = n[i+1:]
	}
	n = strings.NewReplacer(":", "-", "_", "-", " ", "-").Replace(n)
	return strings.Trim(n, "-")
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
