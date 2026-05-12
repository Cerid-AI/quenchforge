# CLAUDE.md — Quenchforge

> **Extends:** `sunrunnerfire/dotfiles` — global workflow orchestration, core
> principles, commit policy, and task-management rules apply here. This file
> adds Quenchforge-specific agent directives.
>
> **Human contributor?** Start with [README.md](README.md) and
> [CONTRIBUTING.md](CONTRIBUTING.md).

## Mission and scope

Quenchforge ships first-class local inference for **Intel Mac + AMD discrete
GPU** configurations that are not served by upstream Ollama / llama.cpp on
macOS. Apple Silicon is a non-degraded secondary target. Linux and Windows
are explicitly out of scope.

The repo's mission is to leave behind a useful artifact for the community —
the patch series, the launchd glue, the hardware-detection table — not to
become a generic inference framework.

## Absolute rules

1. **No AI attribution.** Never add `Co-Authored-By: Claude`, never mention
   Claude / Anthropic / any AI tool in commit messages, PR titles, PR
   descriptions, issue comments, code comments, or any file committed to this
   repo. This is enforced for both maintainers and external contributors.
2. **macOS-only.** No CUDA, no ROCm, no Vulkan, no Linux-specific code paths.
   Apple Silicon and Intel Mac are the only targets. If a PR adds non-Darwin
   code, the answer is no.
3. **Two patches, not three.** The llama.cpp patch series is exactly the two
   load-bearing patches in `patches/`. `GGML_METAL_N_CB` is set via env, not
   a code patch. Adding a third patch requires a written rationale in
   `patches/README.md` and review.
4. **No `quenchforge doctor` paste = no bug-report triage.** The
   `.github/ISSUE_TEMPLATE/bug.yml` form makes this a hard requirement. Maintainer
   replies to triage-incomplete issues with the doctor-paste request and
   labels `triage-blocked` until the user provides it.
5. **No accepting hardware-access PRs without CI proof.** The self-hosted
   `[amd-gpu]` runner is the gate. "Works on my machine" PRs from
   contributors-with-hardware must pass `[amd-gpu]` before merge — no
   exceptions, even for the maintainer.

## Layout

```
cmd/{quenchforge,quenchforge-bench,quenchforge-preflight}/
internal/
  hardware/   — IOKit GPU/CPU/RAM detect; named profiles (vega-pro, w6800x, …)
  tuning/     — KernelParams from HardwareProfile; first-launch micro-bench
  supervisor/ — process group, orphan reaper, slot lifecycle
  scheduler/  — priority queue (chat > embed > rerank)
  gateway/    — vendored Olla at pinned SHA, Ollama+OpenAI route translation
  registry/   — GGUF pull (HF Etag integrity), disk preflight
  config/     — YAML config loader
  migrate/    — ~/.ollama/models/ symlink-import
  obs/        — Prometheus, Sentry, structured JSON logs
llama.cpp/    — git submodule
whisper.cpp/  — git submodule (v0.2)
patches/
  0001-private-vram-force.patch
  0002-disable-simdgroup-mm-non-apple-silicon.patch
scripts/
  apply-patches.sh       — idempotent: git am --reject each patch
  build-llama.sh         — CMake invocation per target triple
  rebase-upstream.sh     — fetch upstream, rebase patches, run tests
  imatrix-recalibrate.sh — quantization calibration on user query distribution
Formula/quenchforge.rb   — Homebrew formula with service block
.github/workflows/{ci,release,rebase-upstream,runner-health}.yml
.github/ISSUE_TEMPLATE/{bug.yml,hardware_profile.yml,feature.yml}
tests/integration/       — gated on [amd-gpu] self-hosted runner
```

## Toolchain

- **Go 1.23+** for cmd/ and internal/. Single static binary via
  `goreleaser`.
- **C/C++ via vendored llama.cpp + whisper.cpp submodules.** CMake builds
  driven by `scripts/build-llama.sh`. The patch series is applied
  pre-configure by `scripts/apply-patches.sh`.
- **No CGo outside `internal/hardware/detect_darwin.go`.** That one file
  links IOKit / Metal headers via cgo for GPU enumeration. All other CGo
  use requires a written rationale.
- **Build matrix:** Intel Mac x86_64 with Metal+AMD, Intel Mac x86_64
  CPU-only, arm64 Apple Silicon. Universal binary via `lipo`. Self-hosted
  CI runner on the maintainer's Mac Pro 2019 covers x86_64 + AMD-GPU; GitHub
  Actions `macos-latest` covers arm64 unit tests.

## Canonical invocations

| Action | Command |
|---|---|
| Build | `make build` (drives `scripts/build-llama.sh` then `go build ./...`) |
| Unit tests | `go test ./...` |
| AMD-GPU integration tests | `go test -tags=amd_gpu ./tests/integration/...` (self-hosted runner only) |
| Lint | `golangci-lint run` |
| Apply patches to submodule | `bash scripts/apply-patches.sh` |
| Rebase patches against upstream master | `bash scripts/rebase-upstream.sh` |
| Quick hardware probe | `./bin/quenchforge doctor` |

## Patch maintenance

The two patches in `patches/` are re-derived from the public llama.cpp issues
[#15228](https://github.com/ggml-org/llama.cpp/issues/15228) and
[#19563](https://github.com/ggml-org/llama.cpp/issues/19563). No third-party
gist text is copied verbatim — both patches are original re-implementations
that cite the issue threads in their commit messages.

Weekly the `rebase-upstream.yml` action fetches `ggml-org/llama.cpp` master,
runs `git am -3` against the patch series, and opens a PR with conflict
hunks if anything fails to apply. The PR is blocked from merge unless both
the arm64 unit-test job AND the `[amd-gpu]` integration-test job are green.

If a patch starts to bit-rot beyond a manageable conflict surface, file the
patch as a draft PR upstream first — `#15228` was previously closed
"not planned", but new maintainer review or new evidence can flip that.

## Telemetry policy

Both Sentry error reporting AND the anonymous benchmark dashboard at
`bench.quenchforge.dev` are **opt-in only** via a first-launch consent
screen. Never send anything beyond hardware profile, tokens/sec, and
latency. Never send prompts, model outputs, file paths, or anything
user-identifiable. The consent screen copy lives in
[`internal/obs/consent.go`](internal/obs/consent.go) (TBD); changes to
that copy require a maintainer review.

## Anti-patterns to reject

- Adding a Linux build tag for any non-test file
- "Just add a tiny Vulkan path" — Metal is the moat; Vulkan-via-MoltenVK is
  slower and has correctness bugs on Intel Mac AMD configs
  ([llama.cpp#20104](https://github.com/ggml-org/llama.cpp/issues/20104))
- Default-on telemetry of any kind
- Bundling a GGUF model into the release artifact
- Carrying a third patch without rationale (see absolute rule #3)
- Adding any feature behind a paywall or auth gate

## Sentry

Error monitoring under Sentry org `cerid-ai`, project `quenchforge`.
DSN is set at runtime via `QUENCHFORGE_SENTRY_DSN` and is **not in the
default config**. Operators who do not opt in produce zero Sentry traffic.
