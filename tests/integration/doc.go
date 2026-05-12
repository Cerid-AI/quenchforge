// Copyright (c) 2026 Cerid AI and the Quenchforge Contributors.
// SPDX-License-Identifier: Apache-2.0

// Package integration holds end-to-end tests that require real hardware.
//
// All tests in this package are gated on `//go:build amd_gpu` or
// `//go:build apple_silicon` so the regular `go test ./...` on contributor
// laptops and stdlib-only CI runners skips them. They are exercised by:
//
//  1. The self-hosted runner on the maintainer's Mac Pro 2019, labelled
//     `[amd-gpu]` — runs `go test -tags=amd_gpu ./tests/integration/...`
//     as the merge gate for any change that touches the patch series.
//
//  2. Local validation by anyone with the target hardware. Run:
//     bash scripts/apply-patches.sh
//     bash scripts/build-llama.sh
//     go test -tags=amd_gpu ./tests/integration/...
//
// The tests deliberately assume the patched llama-server binary lives at
// the standard build-script output path; if you've installed it elsewhere
// override with QUENCHFORGE_LLAMA_BIN.
//
// To run against an alternative model, set TEST_MODEL_PATH to the GGUF.
// Otherwise the harness looks for the well-known Ollama llama3.2:3b blob
// (which is what the patch was validated against).
package integration
