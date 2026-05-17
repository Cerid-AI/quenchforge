// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"sort"
	"syscall"
	"time"
)

// cmdSustainedEmbed runs the `sustained-embed` subcommand and returns
// the process exit code. See main.go's package comment for the code
// table.
//
// Behaviour summary:
//
//  1. Parse flags + verify the gateway is reachable.
//  2. Loop /v1/embeddings POSTs with rotating sample texts.
//  3. Every progressInterval, emit a one-line progress record
//     ("samples=N p50=… p99=… status=…").
//  4. On any HTTP 502, connection refused, or 5xx, stop and report
//     the wall-time-to-crash.
//  5. At the end of duration (or on crash), emit a final summary line.
func cmdSustainedEmbed(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("sustained-embed", flag.ContinueOnError)
	fs.SetOutput(stderr)

	gatewayURL := fs.String("gateway", defaultGatewayURL(), "quenchforge gateway base URL")
	model := fs.String("model", "", "model name to embed against (required)")
	duration := fs.Duration("duration", 5*time.Minute, "stop after this much wall time")
	progressEvery := fs.Duration("progress-every", 30*time.Second, "emit a progress line at this interval")
	requestTimeout := fs.Duration("timeout", 30*time.Second, "per-request HTTP timeout")
	batchSize := fs.Int("batch", 1, "number of texts per request (1 simulates per-call ingest; >1 simulates ChromaDB batch ingest)")
	concurrency := fs.Int("concurrency", 1, "concurrent in-flight requests (1 = serial; default mirrors ChromaDB)")
	pauseBetween := fs.Duration("pause", 0, "sleep this long between requests (sustained load defaults to 0)")
	criticalRatio := fs.Float64("critical-ratio", 5.0, "p99/p50 ratio above which the run is classified degraded at exit")

	if err := fs.Parse(args); err != nil {
		// flag package already printed the error.
		return 64
	}
	if *model == "" {
		fmt.Fprintln(stderr, "quenchforge-bench: --model is required")
		fs.Usage()
		return 64
	}
	if *batchSize < 1 || *concurrency < 1 {
		fmt.Fprintln(stderr, "quenchforge-bench: --batch and --concurrency must be >= 1")
		return 64
	}

	client := &http.Client{Timeout: *requestTimeout}

	// Pre-flight: hit /health so we fail fast if the gateway isn't up.
	if err := probeGateway(client, *gatewayURL); err != nil {
		fmt.Fprintf(stderr, "quenchforge-bench: gateway probe failed: %v\n", err)
		return 3
	}

	ctx, stop := signal.NotifyContext(context.Background(),
		syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	deadline := time.Now().Add(*duration)

	samples := newSampleStore(8192)
	stats := newRunStats()

	fmt.Fprintf(stdout, "[bench] sustained-embed start gateway=%s model=%s duration=%s "+
		"concurrency=%d batch=%d\n",
		*gatewayURL, *model, *duration, *concurrency, *batchSize)

	progress := time.NewTicker(*progressEvery)
	defer progress.Stop()

	// Worker pool — each goroutine runs requests in a tight loop.
	results := make(chan callResult, *concurrency*2)
	for i := 0; i < *concurrency; i++ {
		go runWorker(ctx, deadline, client, *gatewayURL, *model,
			*batchSize, *pauseBetween, results)
	}

	startedAt := time.Now()
	var crashed bool
	var crashAt time.Time
	var crashReason string

drain:
	for {
		select {
		case <-ctx.Done():
			break drain
		case <-progress.C:
			p50, p99 := samples.percentiles()
			fmt.Fprintf(stdout,
				"[bench] t=%5s n=%d p50=%.1fms p99=%.1fms err=%d/%d "+
					"ratio=%.2f\n",
				time.Since(startedAt).Round(time.Second),
				stats.total, p50, p99,
				stats.errs, stats.total,
				safeRatio(p99, p50))
		case r := <-results:
			if r.fatal {
				crashed = true
				crashAt = time.Now()
				crashReason = r.err.Error()
				stop()
				continue
			}
			samples.add(r.dur)
			stats.recordSample(r.isError)
			if time.Now().After(deadline) {
				stop()
			}
		}
	}

	// Drain any in-flight workers so reported counts are stable.
	drainTimer := time.After(2 * time.Second)
drainLoop:
	for {
		select {
		case <-drainTimer:
			break drainLoop
		case r := <-results:
			if !r.fatal {
				samples.add(r.dur)
				stats.recordSample(r.isError)
			}
		}
	}

	wall := time.Since(startedAt)
	p50, p99 := samples.percentiles()
	ratio := safeRatio(p99, p50)

	if crashed {
		fmt.Fprintf(stdout,
			"[bench] CRASHED at t=%s reason=%q (n=%d p50=%.1fms p99=%.1fms)\n",
			crashAt.Sub(startedAt).Round(time.Second).String(),
			crashReason, stats.total, p50, p99)
		return 1
	}

	status := "ok"
	exit := 0
	if ratio >= *criticalRatio {
		status = "degraded"
		exit = 2
	}

	fmt.Fprintf(stdout,
		"[bench] DONE wall=%s n=%d errs=%d p50=%.1fms p99=%.1fms ratio=%.2f status=%s\n",
		wall.Round(time.Second), stats.total, stats.errs, p50, p99, ratio, status)
	return exit
}

// runWorker is the per-goroutine request loop. Sends results down the
// channel until ctx is cancelled or the deadline is reached. Reports
// fatal errors (connection refused, 502, sustained 5xx) so the main
// loop can stop the run promptly.
func runWorker(ctx context.Context, deadline time.Time, client *http.Client,
	gateway, model string, batch int, pause time.Duration, out chan<- callResult,
) {
	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return
		default:
		}
		r := embedOnce(ctx, client, gateway, model, batch)
		select {
		case out <- r:
		case <-ctx.Done():
			return
		}
		if r.fatal {
			return
		}
		if pause > 0 {
			select {
			case <-ctx.Done():
				return
			case <-time.After(pause):
			}
		}
	}
}

// embedOnce performs a single /v1/embeddings POST and returns the
// observed outcome. Fatal errors (502, refused, server error) abort
// the whole run; non-fatal errors are still counted in the error rate.
func embedOnce(ctx context.Context, client *http.Client,
	gateway, model string, batch int,
) callResult {
	body := buildEmbedBody(model, batch)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		gateway+"/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return callResult{err: err, fatal: true}
	}
	req.Header.Set("Content-Type", "application/json")

	t0 := time.Now()
	resp, err := client.Do(req)
	dur := time.Since(t0)
	if err != nil {
		// Connection refused / EOF / TLS errors / timeouts: treat as fatal
		// because in practice they signal the slot is dead, not just
		// slow.
		return callResult{err: fmt.Errorf("transport: %w", err), dur: dur, fatal: true}
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body) // drain so keep-alive works

	if resp.StatusCode == http.StatusBadGateway {
		return callResult{
			err:   fmt.Errorf("HTTP 502 — upstream slot likely crashed"),
			dur:   dur,
			fatal: true,
		}
	}
	if resp.StatusCode >= 500 {
		return callResult{
			err:     fmt.Errorf("HTTP %d", resp.StatusCode),
			dur:     dur,
			isError: true,
		}
	}
	if resp.StatusCode >= 400 {
		// 4xx — operator error (wrong model name etc.); also fatal
		// because retrying doesn't help.
		return callResult{
			err:   fmt.Errorf("HTTP %d (client error)", resp.StatusCode),
			dur:   dur,
			fatal: true,
		}
	}
	return callResult{dur: dur}
}

// callResult is one observation emitted by a worker to the main loop.
type callResult struct {
	dur     time.Duration
	err     error
	isError bool // non-fatal error (5xx); still counts in error rate
	fatal   bool // run-aborting error (502, refused, 4xx)
}

// runStats is the running counter the main loop maintains. Worker
// goroutines emit into a channel; the main loop is the only mutator,
// so no lock is needed.
type runStats struct {
	total int
	errs  int
}

func newRunStats() *runStats { return &runStats{} }

func (s *runStats) recordSample(isError bool) {
	s.total++
	if isError {
		s.errs++
	}
}

// sampleStore is a bounded ring buffer of latency observations the bench
// uses to compute p50/p99 at each progress tick + final summary. The
// cap mirrors the gateway tracker for behavioural parity.
type sampleStore struct {
	cap     int
	samples []time.Duration
}

func newSampleStore(cap int) *sampleStore {
	return &sampleStore{cap: cap}
}

func (s *sampleStore) add(d time.Duration) {
	if len(s.samples) >= s.cap {
		s.samples = s.samples[1:]
	}
	s.samples = append(s.samples, d)
}

func (s *sampleStore) percentiles() (p50, p99 float64) {
	if len(s.samples) == 0 {
		return 0, 0
	}
	cp := make([]time.Duration, len(s.samples))
	copy(cp, s.samples)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	p50 = float64(cp[len(cp)/2].Nanoseconds()) / float64(time.Millisecond)
	pidx := int(float64(len(cp)-1) * 0.99)
	p99 = float64(cp[pidx].Nanoseconds()) / float64(time.Millisecond)
	return p50, p99
}

// buildEmbedBody constructs an OpenAI /v1/embeddings request body. The
// sample texts are intentionally mid-length (~300 tokens each) so each
// call exercises a non-trivial Metal staging-buffer transfer.
func buildEmbedBody(model string, batch int) []byte {
	texts := make([]string, batch)
	for i := range batch {
		texts[i] = sampleTexts[i%len(sampleTexts)]
	}
	body, _ := json.Marshal(map[string]any{
		"model": model,
		"input": texts,
	})
	return body
}

// probeGateway verifies the gateway is reachable. Returns nil on success.
func probeGateway(client *http.Client, gateway string) error {
	req, err := http.NewRequest(http.MethodGet, gateway+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("gateway /health returned HTTP %d", resp.StatusCode)
	}
	return nil
}

func defaultGatewayURL() string {
	if u := os.Getenv("QUENCHFORGE_URL"); u != "" {
		return u
	}
	return "http://127.0.0.1:11434"
}

func safeRatio(num, denom float64) float64 {
	if denom <= 0 {
		return 0
	}
	return num / denom
}

// sampleTexts are mid-length English snippets used as embedding inputs.
// Variety in length and topic prevents the embed model from any trivial
// caching benefit; chosen to land around 200-400 tokens each.
var sampleTexts = []string{
	"The Metal shading language is Apple's GPU programming language; it shares syntactic ancestry with C++ and Open CL. Apple-Silicon GPUs and Intel-Mac AMD-discrete GPUs both implement the API, but the underlying drivers differ substantially in their memory-management model — unified memory on Apple Silicon versus a dedicated VRAM pool on AMD.",
	"Quenchforge is a community fork of Ollama focused on Intel Mac plus AMD discrete GPU configurations that are not first-class targets of upstream llama.cpp. The project carries a single load-bearing patch series against the Metal kernels plus a Go supervisor that wires up the runtime configuration knobs needed for stable inference.",
	"Sustained-load Metal correctness on AMD discrete GPUs requires care in the staging-buffer pool size — small allocations under tight loops can exhaust the pool faster than the driver releases. Workarounds include reducing the GGML_METAL_N_CB command-buffer count or lowering the per-batch token budget so each allocation is smaller.",
	"The LongMemEval benchmark probes long-context memory retrieval across six question types. Strong systems combine an embedding model with structured memory extraction, query decomposition for multi-session questions, and a cross-encoder reranker for the final top-k selection. Baseline vector-only retrieval lands around 0.43; tuned systems reach 0.85 or higher.",
	"Cross-encoder reranking takes a query and a candidate document, encodes them together, and emits a single relevance score. Compared to bi-encoder retrieval, the reranker pays in latency but rewards in accuracy because the model sees both texts at once and can reason about their relationship rather than computing similarity in a fixed embedding space.",
	"Faithfulness scoring with natural-language inference decomposes a generated answer into atomic claims and tests each claim for entailment against the retrieved context. The DeBERTa-v3 family is a common choice; smaller variants are cheaper but exhibit lower precision on long-tail paraphrases and pronoun resolutions, with the gap visible in the rolling p50/p99 of the metric.",
}

// errInterrupted is a sentinel for ctx-cancellation, kept distinct from
// transport errors so the caller can differentiate operator stops from
// real failures.
var errInterrupted = errors.New("interrupted by signal")

// _ ensures the sentinel is referenced; future error-translation in
// embedOnce will compare against it for the SIGINT-during-Do case.
var _ = errInterrupted
