# CLAUDE.md — Quenchforge

Quenchforge is local inference for Intel Macs with AMD discrete GPUs, where stock llama.cpp Metal kernels produce garbage tokens and Ollama falls back to CPU. It serves the Ollama API on `:11434`. It is a finished bugfix and a community artifact, not a general inference framework.

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

Code comments cite these by number; keep the numbering stable.

1. **No AI attribution.** Never add `Co-Authored-By: Claude`, never mention
   Claude / Anthropic / any AI tool in commit messages, PR titles, PR
   descriptions, issue comments, code comments, or any file committed to this
   repo. This is enforced for both maintainers and external contributors.
2. **macOS-only.** No CUDA, no ROCm, no Vulkan, no Linux-specific code paths.
   Apple Silicon and Intel Mac are the only targets. If a PR adds non-Darwin
   code, the answer is no.
3. **Minimise the patch surface.** The patch series is exactly the
   load-bearing changes in `patches/<submodule>/`:
   - `llama.cpp`: `0001-metal-correctness-on-non-apple-silicon`,
     `0002-metal-staging-buffer-pool`, `0003-metal-amd-bert-fallback-kernels`,
     `0004-metal-amd-bert-matmul-fallback`, `0005-metal-serial-dispatch-non-uma`
   - `whisper.cpp`, `sd.cpp`, `bark.cpp`: `0001-metal-correctness-on-non-apple-silicon` each

   `GGML_METAL_N_CB` is set via env, not a code patch. Adding any new patch
   requires a written rationale in `patches/README.md`, a public upstream
   issue link, and review.
4. **No `quenchforge doctor` paste = no bug-report triage.** The
   `.github/ISSUE_TEMPLATE/bug.yml` form makes this a hard requirement. Maintainer
   replies to triage-incomplete issues with the doctor-paste request and
   labels `triage-blocked` until the user provides it. The doctor section
   headers are part of that contract; renaming one is a breaking change.
5. **No accepting hardware-access PRs without CI proof.** The self-hosted
   `[amd-gpu]` runner is the gate. "Works on my machine" PRs from
   contributors-with-hardware must pass `[amd-gpu]` before merge — no
   exceptions, even for the maintainer.

## Layout

```
cmd/{quenchforge,quenchforge-bench,quenchforge-preflight}/
internal/
  hardware/   — sysctl + system_profiler GPU/CPU/RAM detect; named profiles (vega-pro, w6800x, …)
  tuning/     — KernelParams from HardwareProfile; first-launch micro-bench
  placement/  — decides whether a workload runs on the GPU or the CPU
  pressure/   — adapts GPU admission concurrency to host pressure so the compositor is never starved
  supervisor/ — process group, orphan reaper, slot lifecycle
  scheduler/  — priority queue (chat > embed > rerank)
  gateway/    — HTTP front door; Ollama + OpenAI route translation (Gateway.routes)
  registry/   — GGUF pull (HF Etag integrity), list, remove, disk preflight
  config/     — runtime configuration from environment
  discovery/  — mDNS/Bonjour LAN advertisement of the Ollama API
  portcheck/  — prestart :11434 reclaim / Ollama-squatter deconfliction
llama.cpp/ whisper.cpp/ sd.cpp/ bark.cpp/   — git submodules
patches/<submodule>/NNNN-*.patch   — see patches/README.md for the live series
scripts/
  apply-patches.sh    — idempotent patch application across all submodules
  build-{llama,whisper,sd,bark}.sh — CMake invocation per target triple
  rebase-upstream.sh  — fetch each submodule's upstream, replay patches, regenerate series
  bench-{bert,llama}-{correctness,sustained-load}.py — slot validation harnesses
  check-release-artifacts.sh — release artifact checks run in CI
Formula/quenchforge.rb   — Homebrew formula with service block
.github/workflows/{ci,release,rebase-upstream}.yml
.github/ISSUE_TEMPLATE/{bug.yml,hardware_profile.yml,feature.yml}
tests/integration/       — gated on [amd-gpu] self-hosted runner
tests/prestart_guard_test.sh
```

## Toolchain

- **Go 1.23+** for cmd/ and internal/. Single static binary via
  `goreleaser`.
- **C/C++ via the vendored llama.cpp, whisper.cpp, sd.cpp and bark.cpp
  submodules.** CMake builds are driven by `scripts/build-*.sh`. The patch
  series is applied pre-configure by `scripts/apply-patches.sh`.
- **No CGo.** The Go binary builds with `CGO_ENABLED=0` (`.goreleaser.yaml`);
  hardware detection uses `sysctlbyname` and shells out to `system_profiler`.
  Adding CGo requires a written rationale.
- **Build matrix:** Intel Mac x86_64 with Metal+AMD, Intel Mac x86_64
  CPU-only, arm64 Apple Silicon. Universal binary via `lipo`. A self-hosted
  CI runner on a Mac Pro 2019 covers x86_64 + AMD-GPU; GitHub Actions
  `macos-latest` covers arm64 unit tests.

## Canonical invocations

| Action | Command |
|---|---|
| Build | `make build` (applies patches, runs `scripts/build-llama.sh`, then builds the Go binaries) |
| Unit tests | `make test` (`go test ./...` plus `tests/prestart_guard_test.sh`) |
| AMD-GPU integration tests | `go test -tags=amd_gpu ./tests/integration/...` (self-hosted runner only) |
| Lint | `make lint` (`gofmt -l` on cmd/ and internal/, then `go vet ./...`) |
| Apply patches to submodule | `bash scripts/apply-patches.sh` |
| Rebase patches against upstream | `bash scripts/rebase-upstream.sh` |
| Quick hardware probe | `./bin/quenchforge doctor` |

## Patch maintenance

`0001` in each submodule addresses
[ggml-org/llama.cpp#19563](https://github.com/ggml-org/llama.cpp/issues/19563):
Apple-Silicon-only Metal kernels (simdgroup-reduction + bfloat) miscompile on
AMD discrete / Intel iGPU and produce garbage tokens. The llama.cpp series
adds a staging-buffer pool (`0002`), BERT fallback kernels and matmul for AMD
(`0003`, `0004`), and serial Metal dispatch by default on non-UMA devices
(`0005`, opt back in with `GGML_METAL_CONCURRENCY_FORCE=1`). The patches are
re-derived from live debugging on a Mac Pro 2019 + Radeon Pro Vega II; no
third-party gist text is copied. Provenance, reproducers, the opt-in env vars
(`GGML_METAL_FORCE_SIMDGROUP_REDUCTION=1`, `GGML_METAL_FORCE_BF16=1`) and the
rationale for each patch are in [`patches/README.md`](patches/README.md).
Fixes that belong at the supervisor layer are flags in `internal/tuning/`,
not patches.

Weekly the `rebase-upstream.yml` action fetches each submodule's upstream,
runs `git am -3` against the patch series, and opens a PR with conflict
hunks if anything fails to apply. The PR is blocked from merge unless both
the arm64 unit-test job AND the `[amd-gpu]` integration-test job are green.

If a patch starts to bit-rot beyond a manageable conflict surface, file
it as a draft PR upstream — new maintainer review or new evidence can flip
a previously-closed-not-planned decision.

## Telemetry and error reporting policy

**Status: not implemented.** Quenchforge ships **no** telemetry, error
reporting, Sentry SDK, or analytics code — there is no `internal/obs/`
package and no network reporting path. The rules below bind *when* any such
code is added; they do not describe current behaviour.

- Error reporting and any anonymous benchmark dashboard must be **opt-in
  only**, via a first-launch consent screen. The consent flow and its copy
  land in `internal/obs/` with a maintainer review.
- Never send anything beyond hardware profile, tokens/sec, and latency.
  Never send prompts, model outputs, file paths, or anything
  user-identifiable.
- A Sentry DSN, if one is ever added, is supplied at runtime via
  `QUENCHFORGE_SENTRY_DSN` and never baked into the default config, so an
  operator who does not opt in produces zero traffic.

## Operational gotchas

Code comments cite these by number; keep the numbering stable.

0. **One restart at a time while slots are loading models.** Never issue
   rapid successive restarts (`kickstart -k` twice, or any overlapping
   restart). Each restart starts GPU model loads (embed slots push all
   layers onto the card), and overlapping load cycles on a display-active
   AMD-discrete Mac starve the compositor: a WindowServer
   `userspace_watchdog_timeout` (GUI session crash), the userspace cousin
   of an unload+load kernel panic. Wait for `gateway listening` and all
   `slot pid=` lines in `quenchforge.out.log` before issuing another.
   `launchctl` gui-domain `bootstrap`/`print` fail with "Domain does not
   support specified action" from Background-session shells (agent tools,
   SSH); if a restart race unloads the job, recovery needs a command from
   the user's own Aqua session.

1. **The three chat-slot AMD safety flags are retired.** `--flash-attn off`,
   `--cache-ram 0` and `--no-cache-prompt` are gone from
   `tuning.go::chatParams`, with regression tests pinning their absence. Do
   not re-add them to any slot without new evidence; sections 1 and 2 of
   `patches/README.md` carry the retirement data.

2. **Embed/rerank slots have their own AMD safety surface (section 3 of
   `patches/README.md`).** The graph-compute buffer-corruption crash hits
   embed/rerank under sustained batch load. The embed ubatch and context
   ceiling are VRAM-tier-adaptive (`internal/tuning/tuning.go::amdSizing`):
   the detected `GPUVRAMGB` picks 1024/none (≥ 12 GB), 512/4096 (8 GB), or
   256/2048 (4 GB). A VRAM probe miss (0/unknown) is treated as the high
   tier, so a detection failure never throttles the validated path.
   Operators set these env vars only to override the tier, or to apply the
   same knobs on a profile the detector classified as non-AMD:

   ```
   QUENCHFORGE_EMBED_UBATCH_SIZE=1024   # override tier ubatch — caps Metal staging-buffer pressure
   QUENCHFORGE_EMBED_METAL_N_CB=1       # serialise command-buffer submission
   QUENCHFORGE_AUTO_BACKOFF=true        # auto-503 on critical error rate (crash signature); latency ratio is observability-only and never sheds
   ```

   `quenchforge-bench sustained-embed` is the empirical tuning tool for new
   GPU families.

3. **`internal/tuning/` is the sole owner of per-(profile, kind) slot
   tuning.** `cmd/quenchforge/main.go::buildSlotArgs` and `slotEnv`
   delegate to it. Adding a new per-profile or per-slot-kind flag
   means updating `tuning.go::KernelParams` + `tuning_test.go`, not
   adding a new `if hwInfo.IsAMDDiscrete() && spec.Kind == ...` block
   to `buildSlotArgs`.

4. **Restart immediately after `brew upgrade` or a reinstall.** Replacing
   the Cellar binary under the live process invalidates its codesigned text
   pages; the next page-in SIGKILLs it with "Code Signature Invalid", and
   until then the server silently keeps running the old version. After any
   upgrade, issue exactly one `launchctl kickstart -k
   gui/$(id -u)/com.cerid.quenchforge` (gotcha 0 applies). `quenchforge
   doctor` reports a stale or removed exec path under "Running server
   binary".

## Anti-patterns to reject

- Adding a Linux build tag for any non-test file
- "Just add a tiny Vulkan path" — Metal is the moat; Vulkan-via-MoltenVK is
  slower and has correctness bugs on Intel Mac AMD configs
  ([llama.cpp#20104](https://github.com/ggml-org/llama.cpp/issues/20104))
- Default-on telemetry of any kind
- Bundling a GGUF model into the release artifact
- Carrying any new patch without rationale (see absolute rule 3)
- Adding any feature behind a paywall or auth gate
