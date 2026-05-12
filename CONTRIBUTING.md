# Contributing to Quenchforge

Quenchforge ships GPU-accelerated local inference for **Intel Mac + AMD discrete GPU** configurations that upstream Ollama / llama.cpp do not serve on macOS. The repo is intentionally narrow — we cover one hardware axis and we cover it well.

## What we want

- **Hardware profiles** for AMD GPUs we don't have. The Mac Pro 2019 maintainer has a Vega II Duo only. Profiles for W6800X, W6900X, RDNA1 (5500M / 5700), and RDNA2 (6700M) are gold.
- **Performance reports.** Tokens-per-second on different GPUs at different model sizes / quantizations. Open a `hardware_profile` issue with your `quenchforge doctor` output and a short benchmark table.
- **macOS integration polish.** Homebrew formula tweaks, launchd plist quirks, Sparkle 2.x updater (when we add the menubar app).
- **Patch-series improvements.** If you have a cleaner re-derivation of the load-bearing patches than what's in `patches/`, please open a PR.
- **Documentation.** Especially: a clear explanation of the `MTLCopyAllDevices` + `!isLowPower` device-selection trap that bites every Mac Pro user.

## What we don't want

- **Linux or Windows support.** Use Ollama / vLLM / koboldcpp upstream. These platforms are already well-served.
- **CUDA / ROCm / Vulkan / OpenCL backends.** Metal-only is the moat. MoltenVK is documented as slower with correctness bugs on this hardware ([llama.cpp#20104](https://github.com/ggml-org/llama.cpp/issues/20104)).
- **Custom GGUF formats or model re-hosting.** Use the upstream format verbatim; pull models from Hugging Face.
- **Default-on telemetry of any kind.** All telemetry is opt-in via the first-launch consent screen.
- **Vendored third-party code without license attribution.** Anything we vendor lands in `internal/<area>/` with the upstream license + commit SHA recorded in `third_party/LICENSES.md`.

## Setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/cerid-ai/quenchforge.git
cd quenchforge

# Apply patch series to llama.cpp submodule
bash scripts/apply-patches.sh

# Build everything
make build

# Quick check
./bin/quenchforge doctor
```

You need:
- macOS Sonoma 14.0+ (Metal 3 + argument-buffers tier 2 floor)
- Go 1.23+
- CMake 3.30+
- An Apple Developer ID if you want to test the signed-binary path (otherwise the unsigned binary will work locally — Gatekeeper will warn on first run)

## Filing a hardware-profile issue

Use the `hardware_profile` issue template (`.github/ISSUE_TEMPLATE/hardware_profile.yml`). It requires:

1. `quenchforge doctor` output (full paste — includes Metal log capture)
2. macOS version
3. Mac model
4. A quick benchmark table: model + quantization + tokens/sec
5. Whether the hardware is Apple-original or Hackintosh (no judgment; we tag telemetry differently)

We can't merge a profile-targeting PR without this data — the project's claim is "hardware-aware tuning," so untested profiles aren't shipped.

## PR conventions

- **DCO sign-off** on every commit. We use the standard Linux-kernel-style `Signed-off-by:` trailer. No CLA.
- **No AI attribution.** Do not add `Co-Authored-By: Claude` / `Anthropic` / any other AI tool line. This applies even if you used an AI assistant during development. Commits must be authored by humans.
- **Conventional Commits** for commit messages: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, etc.
- **One concern per PR.** If your PR touches the gateway, the supervisor, AND the patch series, it's three PRs.
- **CI must be green.** The `[amd-gpu]` self-hosted runner is mandatory for any change touching `internal/hardware/`, `internal/tuning/`, `patches/`, `scripts/build-llama.sh`, or `internal/supervisor/`.
- **Tests required for new code paths.** Bug fixes should include a regression test. New features should include unit + (if it touches the AMD path) integration coverage.

## Testing without AMD hardware

If you don't have an Intel Mac + AMD GPU, you can still contribute meaningfully:

- Documentation edits
- Bug-report triage
- Patch-series cleanups (the patches are pure C; correctness review is welcome)
- Apple Silicon side of the codebase (the non-degraded secondary target)
- Homebrew formula and launchd plist work
- The Olla gateway integration (HTTP routing, SSE streaming, API translation)
- `quenchforge migrate-from-ollama` and `quenchforge-preflight` (no AMD needed)

The CI matrix runs on arm64 + a Linux runner with the CGo IOKit shim stubbed out, so most logic is exercisable without the real hardware.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Contributor Covenant 2.1.

## Security

See [SECURITY.md](SECURITY.md). Use the disclosure process — do not file public issues for vulnerabilities.

## License

By contributing, you agree your contributions are licensed under the [Apache License 2.0](LICENSE), the same license as the project.
