# Changelog

All notable changes to Quenchforge are tracked here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/).
Versions follow [SemVer](https://semver.org/) — minor bumps add features,
patch bumps fix bugs or polish without behaviour change.

---

## v0.4.0 — model registry + VRAM pre-flight (2026-05-14)

The first-run UX gap vs Ollama is closed. Operators no longer need to
manually place GGUFs under `~/.quenchforge/models/` or symlink from an
existing Ollama install — `quenchforge pull <alias>` does the work.

### Added — model registry (`internal/registry/`)

New subcommands on the `quenchforge` CLI:

```sh
# Catalog alias (curated list of well-tested AMD-Mac picks)
quenchforge pull llama3.2:3b
quenchforge pull qwen2.5:7b
quenchforge pull nomic-embed:v1.5

# Explicit HF repo + quant
quenchforge pull bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M

# Explicit HF repo + full filename
quenchforge pull bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Print the catalog
quenchforge pull --list

# List installed GGUFs (size, mtime, symlink-vs-file)
quenchforge list

# Remove installed GGUF (symlink-safe — never touches the target file)
quenchforge rm llama3.2:3b
```

**Features:**
- **Atomic downloads** via `.qf-partial` tmpfile + fsync + rename. Partial
  downloads resume via HTTP Range.
- **SHA256 verification** against HuggingFace's reported LFS hash. Refuses
  to install on mismatch.
- **Idempotency** — re-pulling a model that's already present with the
  correct SHA short-circuits without downloading.
- **HF_TOKEN support** for private / gated repos.
- **Progress bar** with bytes / total / rate; suppressible via `--no-progress`.
- **Curated catalog** of 8 well-tested (alias, repo, quant) tuples keyed
  to the VRAM tiers in cerid-ai's
  `docs/AMD_GPU_MODEL_RECOMMENDATIONS.md`. Operators who need other
  quants pass the full `<repo>:<quant>` spec; the catalog is the
  "did you mean..." landing pad.
- **Helpful errors** — if your quant string doesn't match any file in
  the repo, the error lists what IS available.
- **Tests** — 11 unit tests against a mock HF server cover happy path,
  resume, SHA mismatch, 404, no-matching-file, symlink List+Remove,
  path-traversal injection rejection.

### Added — VRAM pre-flight (`cmd/quenchforge/vram_check.go`)

Before spawning any llama-server slot, `quenchforge serve` now sums the
on-disk size of every model that will load and compares against the
detected GPU VRAM. Refuses to start with a helpful multi-line error
when configured slots would oversubscribe VRAM.

```
quenchforge: configured slots exceed available VRAM:
  GPU:         AMD Radeon Pro Vega II
  VRAM:        32.00 GB available
  configured:  38.40 GB (model weights + per-slot overhead + 15% headroom)
  per-slot:
    chat     28.50 GB (qwen2.5-32b-instruct-q4_k_m.gguf)
    embed    1.10 GB  (nomic-embed-text-v1.5.gguf)
    rerank   1.40 GB  (bge-reranker-v2-m3.gguf)

  to fix, either:
    - unset one slot's model env var (e.g. `unset QUENCHFORGE_RERANK_MODEL`)
    - swap to a smaller model (`quenchforge pull --list` shows sizes)
    - reduce --ctx-size (lower KV cache footprint)
    - override the check: set QUENCHFORGE_VRAM_CHECK_DISABLE=1 (use at your own risk)
```

**Skips correctly when:**
- Host has no Metal GPU (CPU-only path — no VRAM constraint)
- VRAM size couldn't be detected (warn + continue, don't block)
- No slots configured (--no-slot, all opt-in vars empty)
- `QUENCHFORGE_VRAM_CHECK_DISABLE=1` is set

**Tests** — 7 unit tests cover non-Metal, unknown-VRAM, no-slots,
fits-comfortably, oversubscription with helpful breakdown,
missing-model handled gracefully, env-var disable.

### Changed

- Default version string in `cmd/quenchforge/main.go` bumped from
  `0.3.4-dev` to `0.4.0-dev`. Release builds keep stamping via
  goreleaser ldflags as before.
- README "Quickstart" now points at `quenchforge pull` as the canonical
  first model on-ramp; `migrate-from-ollama` framed as the
  upgrade-from-Ollama alternative.

### Not yet shipped (deferred to v0.4.1 / v0.5)

Per the audit in `docs/PUBLIC_CONSUMPTION_HARDENING.md`:
- **Web dashboard at `/dashboard`** — MEDIUM priority. Slot status,
  per-route latency p50/p95, recent requests, VRAM usage. Vanilla
  HTML + SSE, ~150 LOC. Deferred to v0.4.1.
- **Sparkle 2.x auto-updater** + macOS status-bar app — LOW priority.
  Substantial work; defer until adoption justifies it.
- **Telemetry consent flow + bench.quenchforge.dev** — no plan yet.
  `QUENCHFORGE_TELEMETRY` env var is reserved but no code shipped;
  the default config has zero network traffic.

---

## v0.3.4 — public-consumption hardening (2026-05-13)

Polish release. No new GPU kernel work (the v0.3.4 attempt at re-enabling
`simdgroup_mm` + `simdgroup_reduction` crashed the maintainer's Mac Pro 2019
three times during testing; the safe wins for AMD-Mac inference are
already in v0.3.3).

- README header re-framed: "Ollama for Mac users who care about correctness"
- Status block bumped from "v0.3.1 pre-release" to "v0.3.3 shipped"
- Hardware compatibility matrix grows "Known incompatible" row for
  Mac Pro 2013 + AMD FirePro D-series
- Configuration table grew from 7 env vars to 14, plus 4 `GGML_METAL_*`
  operator overrides
- Image-gen + TTS slots clarified as "wired but AMD-Mac correctness unverified"
- Top-level `Makefile` honoring the `make build` contract referenced in
  CLAUDE.md and CONTRIBUTING.md. Stamps Version / Commit / BuildDate
  via ldflags from `git describe --tags --always --dirty`.
- Telemetry promise rewritten as "reserved, no code shipped"
- `docs/PUBLIC_CONSUMPTION_HARDENING.md` captures the audit + v0.4 backlog

---

## v0.3.3 — AMD chat works end-to-end (2026-05-13)

Two production gaps closed at the right architectural level — no second
llama.cpp patch required, preserving the "one patch per submodule" rule.

### Supervisor: hardware-aware chat-slot args (`cmd/quenchforge/main.go::buildSlotArgs`)

When the detected profile is one of the four AMD discrete buckets
(Vega Pro, W6800X, RDNA1/2) the chat slot launches with three
additional flags:

- `--flash-attn off` — keeps standard attention GPU-resident on AMD
  instead of the FA-tensor-on-CPU per-decode-step ferry
- `--cache-ram 0` — disables the server-side LCP-similarity slot cache
  that triggers the `GGML_ASSERT(buf_dst)` crash on Vega II
- `--no-cache-prompt` — belt-and-suspenders companion

Embed / rerank / Apple-Silicon paths unchanged.

### Gateway: Ollama-wire ↔ OpenAI-wire body translation (`internal/gateway/ollama_translate.go`)

`/api/chat`, `/api/generate`, `/api/embeddings`, `/api/embed` now do
full body translation (request + response, streaming + non-streaming)
so Ollama clients work end-to-end against llama-server (which only
speaks OpenAI-wire).

### Packaging

- `packaging/macos/com.cerid.quenchforge.plist` — LaunchAgent template
  for from-source installs (Homebrew users get this auto-generated via
  the formula's service block).

### Tests

- `cmd/quenchforge/serve_test.go` — `buildSlotArgs` per-profile coverage
- `internal/hardware/hardware_test.go` — `IsAMDDiscrete` coverage
- `internal/gateway/ollama_translate_test.go` — chat non-streaming,
  streaming SSE→NDJSON, generate, legacy embed, batch embed, error
  mapping, 400-on-empty

---

## v0.3.0 — v0.3.2

Earlier 0.3.x releases shipped:
- Initial llama.cpp + whisper.cpp + sd.cpp + bark.cpp submodule
  vendoring
- The metal-correctness-on-non-apple-silicon patch series
- Embed / rerank / chat / whisper / image-gen / TTS slot infrastructure
- Goreleaser + Homebrew tap setup
- `quenchforge doctor` diagnostic command
- `quenchforge migrate-from-ollama` symlink-importer
- mDNS / Bonjour advertisement (opt-in)
- IOKit-driven hardware profile detection (vega-pro, w6800x, rdna1,
  rdna2, apple-silicon, igpu, cpu, unknown)
