// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

// Command quenchforge-bench drives synthetic load against a running
// quenchforge gateway and reports latency + stability characteristics.
// Primary use case: empirically tuning the AMD-discrete embed/rerank
// slot env knobs (QUENCHFORGE_EMBED_UBATCH_SIZE,
// QUENCHFORGE_EMBED_METAL_N_CB, etc.) to find values that survive
// sustained load without hitting the family-B Metal staging-buffer
// crash documented in `patches/README.md`.
//
// Subcommands (v0.6):
//
//	sustained-embed   POST-loop /v1/embeddings until duration or crash.
//	                  Pre-flight: probes the gateway, fails fast if down.
//	                  Reports per-30s progress + final p50/p99 + crash time.
//
// Future subcommands (deferred):
//
//	sustained-rerank  Same shape against /v1/rerank.
//	throughput        Concurrent-N max QPS without exceeding target p99.
//
// Exit codes:
//
//	0   the run completed for the full duration without errors
//	1   the slot crashed mid-run (HTTP 502, connection refused, > error
//	    threshold)
//	2   the slot degraded but did not crash (p99/p50 > critical-ratio at
//	    end of run)
//	3   the gateway was not reachable at start
//	64  command line error
package main

import (
	"fmt"
	"os"
)

// Version is injected at build time. Local builds carry the zero value.
var Version = "0.6.0-dev"

func main() {
	if len(os.Args) < 2 {
		usage(os.Stderr)
		os.Exit(64)
	}
	switch os.Args[1] {
	case "sustained-embed":
		os.Exit(cmdSustainedEmbed(os.Args[2:], os.Stdout, os.Stderr))
	case "version", "--version", "-v":
		fmt.Println("quenchforge-bench", Version)
	case "help", "--help", "-h":
		usage(os.Stdout)
	default:
		fmt.Fprintf(os.Stderr, "quenchforge-bench: unknown command %q\n", os.Args[1])
		usage(os.Stderr)
		os.Exit(64)
	}
}

func usage(w *os.File) {
	fmt.Fprintln(w, "quenchforge-bench — synthetic load + stability harness")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Usage:")
	fmt.Fprintln(w, "  quenchforge-bench sustained-embed [flags]")
	fmt.Fprintln(w, "  quenchforge-bench version")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Flags: run `quenchforge-bench <subcommand> --help` for details.")
}
