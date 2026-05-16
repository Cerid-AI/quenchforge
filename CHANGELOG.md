# Changelog

All notable changes to Quenchforge are tracked here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/).
Versions follow [SemVer](https://semver.org/) — minor bumps add features,
patch bumps fix bugs or polish without behaviour change.

---

## v0.5.1 — `quenchforge install` LaunchAgent helper (UNRELEASED)

Operator-experience polish. Adds the missing CLI step in the "auto-drop
the LaunchAgent plist" story that was tracked as the last remaining
v0.95.x cerid-ai follow-on. From-source operators no longer need to
`cp packaging/macos/...` + edit `REPLACE_ME` by hand.

- **New subcommand: `quenchforge install`.** Copies the LaunchAgent
  plist into `~/Library/LaunchAgents/com.cerid.quenchforge.plist` with
  the operator's `$USER` substituted into the `REPLACE_ME` placeholders
  automatically. Refuses to overwrite an existing plist unless `--force`
  is passed. Prints the `launchctl bootstrap` next-step instructions
  on success.
- **Single canonical plist source.** The plist now lives at
  `cmd/quenchforge/plist_template.plist` and is embedded into the
  binary via `//go:embed`. The previous duplicate at
  `packaging/macos/com.cerid.quenchforge.plist` is removed — operators
  who want to inspect the template can read the canonical file or run
  `quenchforge install --print-path` to confirm the target path.
- **`packaging/macos/README.md` rewritten** to lead with the install
  command + show the `--force`, `--skip-user-substitution`, and
  `--print-path` flags.

Flags:

- `--force` overwrites an existing plist (default: refuse with a
  helpful uninstall hint).
- `--skip-user-substitution` leaves `REPLACE_ME` untouched (for
  operators who want to edit by hand).
- `--print-path` prints the resolved target and exits without writing
  (useful for `make` integrations).

Non-macOS platforms get a clear "macOS only" error instead of a
silent no-op.

## v0.5.0 — second embed slot + embed-batch fix (UNRELEASED)

Feature release. Adds a dedicated **code-tuned embedding slot** that
runs alongside the existing general-text embed slot in one quenchforge
process, plus a long-standing batch-size bug fix that was blocking any
embedding input over 512 tokens.

- **New slot kind: `code-embed`.** Opt-in via `QUENCHFORGE_CODE_EMBED_MODEL`
  (port: `QUENCHFORGE_CODE_EMBED_PORT`, default `11506`). Lets one
  quenchforge process serve a general-text embedder (Nomic, Snowflake,
  etc. — for KB / RAG workloads) alongside a code-tuned embedder
  (CodeRankEmbed, jina-embeddings-v2-base-code, etc. — for
  semantic-code-search MCPs like `contextplus`).
- **Model-name dispatch.** The gateway peeks at the `model` field of
  inbound `/api/embeddings`, `/api/embed`, and `/v1/embeddings`
  requests. When it matches `Config.CodeEmbedModel` AND a `code-embed`
  upstream is registered, the call routes to the code-embed slot;
  otherwise it falls through to the regular embed slot (so callers that
  don't know about the code slot keep working). Transparent to clients
  — no URL change, no API change.
- **Embed slots now pin `--batch-size` and `--ubatch-size` to
  `MaxContext`.** llama-server's default `ubatch=512` rejected any
  embedding input over 512 tokens with `input (N tokens) is too large
  to process. increase the physical batch size`. Code-search MCPs send
  chunks in the 600–2000 token range and tripped this every call. The
  fix applies to both `embed` and `code-embed` slots; VRAM cost is
  small for typical embed models (~138 MB nomic-embed at Q8, ~280 MB
  CodeRankEmbed at Q8).
- **`quenchforge doctor` now reports per-slot config.** New `slots:`
  section shows model + port for each kind, with `(opt-in; port=N)`
  for slots whose model env var is unset — so operators can verify
  their config from a single command without `env | grep`.
- **Tests:** new coverage for `buildSlotArgs` (embed batch override,
  non-embed kinds unaffected), `resolveEmbedKind` dispatch
  (model-match → code-embed, fallback → regular embed, unknown model
  → regular embed), and config port-collision rules for the new port.

Migration: existing operators see zero behaviour change. To opt into
the new slot, set `QUENCHFORGE_CODE_EMBED_MODEL=<gguf-name>` in your
`com.cerid.quenchforge.plist` (or wherever you launch quenchforge from)
and `launchctl kickstart -k gui/$(id -u)/com.cerid.quenchforge`.

## v0.4.1 — docs + Homebrew tap auto-push (2026-05-14)

Polish + supply-chain release. No behaviour change in the binary; the
tag exists primarily to flip the Homebrew tap from manual updates to
automated goreleaser pushes now that `HOMEBREW_TAP_GITHUB_TOKEN` is
configured.

- **`HOMEBREW_TAP_GITHUB_TOKEN` repo secret is set.** Future tags
  automatically push the updated formula to
  [`Cerid-AI/homebrew-tap`](https://github.com/Cerid-AI/homebrew-tap).
  The manual sync recipe at `docs/APPLE_DEVELOPER_ID.md` § 5 is no
  longer required.
- **README status block bumped** to v0.4.0 (now also covers v0.4.1).
- **`third_party/LICENSES.md` created.** Previously a broken link from
  the README and NOTICE; now exists with full upstream license text
  for the four submodules (llama.cpp + whisper.cpp + sd.cpp + bark.cpp,
  all MIT) and modification provenance.
- **NOTICE updated.** Removed the stale Olla reference (was design-time
  intent that never shipped — `internal/gateway/` is fully home-grown).
  Added the two submodules NOTICE was missing (sd.cpp + bark.cpp).
- **`.goreleaser.yaml` brews scaffold audit-clean.** `brew audit
  --strict --new` flagged 4 nits on the auto-generated formula; 2 were
  fixed in the goreleaser scaffold (desc starts with capital,
  `shell_output` redundant `, 0` arg removed) and 2 in the tap formula
  directly (version-before-license ordering, `macos:` hash syntax).
  `brew audit --strict --new cerid-ai/tap/quenchforge` now exits 0.
- **`docs/APPLE_DEVELOPER_ID.md` status flipped to LIVE.** Status table
  reflects that all 5 Apple GitHub secrets + the Homebrew tap PAT are
  set. v0.3.3 / v0.3.4 / v0.4.0 all shipped signed + notarized.

This is also the first release to verify-end-to-end that the auto tap
push works: pre-v0.4.1 the tap was manually synced because the
`HOMEBREW_TAP_GITHUB_TOKEN` secret wasn't configured. The release of
this tag is itself the verification.

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
