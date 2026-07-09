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
3. **Minimise the patch surface.** The patch series is exactly the
   load-bearing change(s) in `patches/<submodule>/`. As of 2026-07-08 that is:
   - `llama.cpp`: `0001-metal-correctness-on-non-apple-silicon` +
     `0002-metal-staging-buffer-pool` + `0003-metal-amd-bert-fallback-kernels`
     + `0004-metal-amd-bert-matmul-fallback` (0003/0004 un-parked and
     correctness-validated on Vega II 2026-07-08 — roadmap R1)
   - `whisper.cpp`, `sd.cpp`, `bark.cpp`: `0001-metal-correctness-on-non-apple-silicon` each
   `GGML_METAL_N_CB` is set via env, not a code patch. Adding any new patch
   requires a written rationale in `patches/README.md`, a public upstream
   issue link, and review.
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
  discovery/  — mDNS/Bonjour LAN advertisement of the Ollama API
  portcheck/  — prestart :11434 reclaim / Ollama-squatter deconfliction
llama.cpp/    — git submodule
whisper.cpp/  — git submodule
sd.cpp/       — git submodule
bark.cpp/     — git submodule
patches/<submodule>/NNNN-*.patch   — see patches/README.md for the live series
scripts/
  apply-patches.sh    — idempotent patch application across all submodules
  build-{llama,whisper,sd,bark}.sh — CMake invocation per target triple
  rebase-upstream.sh  — fetch each submodule's upstream, replay patches, regenerate series
  bench-{bert,llama}-{correctness,sustained-load}.py — slot validation harnesses
Formula/quenchforge.rb   — Homebrew formula with service block
.github/workflows/{ci,release,rebase-upstream}.yml
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

The single patch in `patches/` addresses bug
[#19563](https://github.com/ggml-org/llama.cpp/issues/19563) — Apple-Silicon-only
Metal kernels (simdgroup-reduction + bfloat) miscompile on AMD discrete /
Intel iGPU and produce garbage tokens. The patch is re-derived from live
debugging on the Mac Pro 2019 + Radeon Pro Vega II; no third-party gist text
is copied. Provenance, the live-verified reproducer, and the opt-in env vars
(`GGML_METAL_FORCE_SIMDGROUP_REDUCTION=1`, `GGML_METAL_FORCE_BF16=1`) are
documented in [`patches/README.md`](patches/README.md).

The originally-planned VRAM-force and disable-simdgroup_mm patches were
verified obsolete against the current upstream — see `patches/README.md`
"Why only one" for the diagnostic detail.

Weekly the `rebase-upstream.yml` action fetches `ggml-org/llama.cpp` master,
runs `git am -3` against the patch series, and opens a PR with conflict
hunks if anything fails to apply. The PR is blocked from merge unless both
the arm64 unit-test job AND the `[amd-gpu]` integration-test job are green.

If the patch starts to bit-rot beyond a manageable conflict surface, file
it as a draft PR upstream — new maintainer review or new evidence can flip
a previously-closed-not-planned decision.

## Telemetry policy

**Status: not yet implemented.** Quenchforge currently ships **no** telemetry,
error reporting, or analytics code — there is no `internal/obs/` package and
no network reporting path. The rules below are the binding policy for *when*
any such code is added, not a description of current behaviour.

If/when telemetry lands, both error reporting AND any anonymous benchmark
dashboard must be **opt-in only** via a first-launch consent screen. Never
send anything beyond hardware profile, tokens/sec, and latency. Never send
prompts, model outputs, file paths, or anything user-identifiable. The consent
flow and its copy must land in `internal/obs/` with a maintainer review.

## Operational gotchas

0. **Never issue rapid successive restarts (`kickstart -k` twice, or any
   overlapping restart) while slots are loading models.** Each restart
   starts GPU model loads (embed slots push all layers onto the card);
   overlapping load cycles on a display-active AMD-discrete Mac starve the
   compositor — observed 2026-07-08 as a WindowServer
   `userspace_watchdog_timeout` spin (GUI session crash; user-visible as a
   "system crash"), the userspace cousin of the 2026-05-14 unload+load
   kernel panic. One restart at a time; wait for `gateway listening` +
   all `slot pid=` lines in `quenchforge.out.log` before issuing another.
   Also: `launchctl` gui-domain `bootstrap`/`print` fail with "Domain does
   not support specified action" from Background-session shells (agent
   tools, SSH) — if a restart race unloads the job, recovery needs a
   command from the user's own Aqua session.

1. **Chat-slot AMD safety args do not apply to embed/rerank slots.**
   Sections 1 + 2 of `patches/README.md` document `--flash-attn off`,
   `--cache-ram 0`, and `--no-cache-prompt` as chat-specific (they
   address LCP-prompt-save and FA-CPU-fallback bugs absent on
   embed/rerank). Adding them to embed/rerank would force a sub-optimal
   attention path with no safety win.

2. **Embed/rerank slots have their own AMD safety surface — section 3.**
   The family-B graph-compute buffer-corruption crash hits embed/rerank
   under sustained batch load (eval suites, bulk KB ingest, sustained
   MCP retrieval). As of **v0.8.0 the embed ubatch + context ceiling are
   VRAM-tier-adaptive** (`internal/tuning/tuning.go::amdSizing`): the
   detected `GPUVRAMGB` picks 1024/none (≥ 12 GB), 512/4096 (8 GB), or
   256/2048 (4 GB) automatically, so small cards no longer need
   hand-tuning. Operators only set the env vars below to *override* the
   tier (force a value, or apply the same safety knobs on a profile the
   detector classified as non-AMD):

   ```
   QUENCHFORGE_EMBED_UBATCH_SIZE=1024   # override tier ubatch — caps Metal staging-buffer pressure
   QUENCHFORGE_EMBED_METAL_N_CB=1       # serialise command-buffer submission
   QUENCHFORGE_AUTO_BACKOFF=true        # auto-503 on critical ERROR RATE (crash signature) — v0.9.1: latency ratio is observability-only, never sheds
   ```

   The ≥ 12 GB tier keeps the bench-validated Vega II values verbatim;
   a VRAM probe miss (0/unknown) is treated as high tier so detection
   failures never throttle the validated path. `quenchforge-bench
   sustained-embed` remains the empirical tuning tool for new families.

3. **`internal/tuning/` is the sole owner of per-(profile, kind) slot
   tuning.** `cmd/quenchforge/main.go::buildSlotArgs` and `slotEnv`
   delegate to it. Adding a new per-profile or per-slot-kind flag
   means updating `tuning.go::KernelParams` + `tuning_test.go`, not
   adding a new `if hwInfo.IsAMDDiscrete() && spec.Kind == ...` block
   to `buildSlotArgs`.

## Anti-patterns to reject

- Adding a Linux build tag for any non-test file
- "Just add a tiny Vulkan path" — Metal is the moat; Vulkan-via-MoltenVK is
  slower and has correctness bugs on Intel Mac AMD configs
  ([llama.cpp#20104](https://github.com/ggml-org/llama.cpp/issues/20104))
- Default-on telemetry of any kind
- Bundling a GGUF model into the release artifact
- Carrying any new patch without rationale (see absolute rule #3)
- Adding any feature behind a paywall or auth gate

## Sentry

**Status: not yet wired into the code.** No Sentry SDK is imported anywhere in
`cmd/` or `internal/` today. The intended design — should it be added — is
error monitoring under Sentry org `cerid-ai`, project `quenchforge`, with the
DSN supplied at runtime via `QUENCHFORGE_SENTRY_DSN` and **never** baked into
the default config, so operators who do not opt in produce zero Sentry traffic.
