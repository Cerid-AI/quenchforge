# Quenchforge

**Ollama for Mac users who care about correctness.** Single Go binary that supervises [`llama.cpp`](https://github.com/ggml-org/llama.cpp) + [`whisper.cpp`](https://github.com/ggml-org/whisper.cpp), exposing Ollama-API + OpenAI-API HTTP on `127.0.0.1:11434`. Drop-in for Ollama clients. Runs unchanged on Apple Silicon (same defaults as Ollama there) and on Intel Mac + AMD discrete (where Ollama falls back to CPU and llama.cpp produces garbage tokens). Same binary, same wire formats, same GGUF models.

## The bug Quenchforge exists to fix

`ggml`'s Metal backend enables Apple-Silicon-specific kernels on any device that supports `MTLGPUFamilyMetal3`, including **AMD discrete GPUs on Intel Mac** (Vega II / W6800X / RDNA1+2) and **Intel iGPUs**. On those devices the simdgroup-reduction and bfloat ops *compile* but produce wrong arithmetic at runtime, so models emit garbage tokens forever ([ggml-org/llama.cpp#19563](https://github.com/ggml-org/llama.cpp/issues/19563)). Stock Ollama on the same hardware silently falls back to CPU without surfacing the underlying bug ([ollama/ollama#1016](https://github.com/ollama/ollama/issues/1016), open since 2023).

Quenchforge carries a single load-bearing patch per submodule that gates the buggy kernels to Apple Silicon only, restoring **correct output on AMD Mac Metal**. The patches are re-derived from the public issue, not copied from third-party gists, and applied at build time via `scripts/apply-patches.sh` — the submodule SHAs stay clean.

## Why ggml, not just LLMs

`ggml` is a compute library, not an LLM library. The same Metal bug affects every ggml consumer on Intel Mac + AMD discrete:

| Workload | ggml consumer | Status in Quenchforge |
|---|---|---|
| **Chat / completion** | `llama.cpp` | ✅ shipped, live-verified on Vega II (4.1 tok/s patched vs garbage stock) |
| **Embeddings** (BGE-M3, e5, GTE — *not LLMs*) | `llama.cpp --embedding` | ✅ shipped, gateway routes `/api/embeddings` + `/v1/embeddings` |
| **Reranking** (BGE-reranker, cross-encoders) | `llama.cpp --reranking` | ✅ shipped, gateway route `/v1/rerank` |
| **Speech-to-text** | `whisper.cpp` | ✅ shipped, CPU mode default — correct transcription at 12.8× real-time on Xeon W-3245 |
| **Image generation** | `stable-diffusion.cpp` | ✅ shipped in v0.3.1 — sd-server supervised slot, `/v1/images/generations` proxies through gateway |
| **Text-to-speech** | `bark.cpp` | ✅ shipped in v0.3.1 — bark server supervised slot, `/v1/audio/speech` → `/tts` path-rewrite |

> **Status:** v0.8.1 (2026-05-31), signed + notarized. Production-stable for **chat + embeddings + code-embeddings + reranking — all GPU-resident** — on Mac Pro 2019 + Radeon Pro Vega II (32 GB HBM2), and VRAM-tier-adaptive across the rest of the Intel-Mac AMD range. Whisper transcription ships CPU-mode (correct, fast). Image-gen + TTS slots are wired but **AMD-Mac correctness is unverified** — treat as experimental until a hardware-profile report confirms. Signed + notarized release binaries (Developer ID Application: Justin Michaels, team `4A5VDRMRB8`, hardened-runtime, Apple-notarized) are on the [releases page](https://github.com/Cerid-AI/quenchforge/releases); `brew install cerid-ai/tap/quenchforge` works.
>
> **Recent highlights** (full history in [CHANGELOG.md](CHANGELOG.md)):
> - **v0.8.1** — prestart port guard: the install-generated LaunchAgent reclaims `:11434` from an Ollama squatter on every start/login, so the two coexist with no manual eviction.
> - **v0.8.0** — AMD-discrete **GPU mode shipped for all four slots** (chat, embed, code-embed, rerank) via two ggml Metal patches (kernel correctness + a staging-buffer pool for sustained load), plus **VRAM-tier-adaptive sizing** so cards from 4 GB MacBook Pro dGPUs to 32 GB Vega II run out-of-the-box.
> - **v0.5.0** — dedicated **code-embed slot** (route code-tuned embedders independently of general-text), model registry (`pull` / `list` / `rm` with HuggingFace + SHA-256 verification), and VRAM pre-flight that refuses to oversubscribe.

## Hardware compatibility matrix

| Configuration | Status | Notes |
|---|---|---|
| Intel Mac (Mac Pro 2019, iMac Pro, MacBook Pro 2018+) + AMD Vega II / Vega II Duo | **Primary** | The target this project exists to serve |
| Intel Mac + AMD W6800X / W6900X (RDNA2 Pro) | **Primary** | Apple-supported MPX modules |
| Intel Mac + AMD RDNA1/RDNA2 (5500M / 5700 / 6700M consumer) | Supported | Same patch surface; smaller HBM/GDDR |
| Apple Silicon (M1/M2/M3/M4/M5) | Supported (non-degraded) | Patches runtime-gated; effectively stock on this arch |
| Intel Mac, Intel iGPU only (Iris Plus, etc.) | Supported (CPU-class) | Metal available but very small VRAM — auto fallback to CPU |
| Intel Mac Pro 2013 + AMD FirePro D300/D500/D700 | **Known incompatible** | Reported gibberish-output ([llama.cpp#20104](https://github.com/ggml-org/llama.cpp/issues/20104)); not Metal3 |
| Linux / Windows | **Out of scope** | Use stock Ollama with CUDA / ROCm / DirectML; that path is already well-served |
| Hackintosh + AMD | Community best-effort | Tagged in telemetry as non-genuine; no SLA |

## Versus the alternatives (Intel Mac + AMD discrete)

On Apple Silicon, Ollama / LM Studio / llama.cpp all work fine — use any of them. This table is specifically the **Intel-Mac + AMD-discrete** axis quenchforge exists to serve:

| | GPU on AMD-Mac Metal | Correct output | Drop-in Ollama API | Embed + rerank + STT, one port | Local-only |
|---|---|---|---|---|---|
| **Quenchforge** | ✅ patched | ✅ | ✅ | ✅ | ✅ |
| Ollama | ❌ silent CPU fallback ([#1016](https://github.com/ollama/ollama/issues/1016)) | ✅ (CPU, slow) | — (it *is* the API) | partial | ✅ |
| llama.cpp (stock) | ⚠️ runs but **garbage tokens** ([#19563](https://github.com/ggml-org/llama.cpp/issues/19563)) | ❌ | ❌ | ✅ (manual, multi-process) | ✅ |
| LM Studio | ❌ no AMD-Mac GPU path¹ | ✅ (CPU) | partial | partial | ✅ |

¹ LM Studio is llama.cpp-based, so its Metal path inherits the same `#19563` bug class on AMD discrete; we haven't independently benchmarked it.

## Honesty about Metal on AMD

| Workload | llama.cpp Metal on Vega II | whisper.cpp Metal on Vega II |
|---|---|---|
| Stock (no patches) | garbage tokens (#19563 repro) | garbage tokens |
| Quenchforge patched | **correct, ~4.1 tok/s** | still buggy beyond what the patch covers — root cause is in whisper-specific Metal kernels we haven't fully audited |
| Quenchforge CPU fallback | n/a (chat needs GPU) | **correct, 12.8× real-time on Xeon W-3245** |

That's why the whisper slot defaults to `--no-gpu` on Intel Mac + AMD. Flip `QUENCHFORGE_WHISPER_GPU=true` if you're on hardware where it works (or want to help us debug).

## What's in the box

| Component | Description |
|---|---|
| `quenchforge serve` | Supervisor + HTTP gateway. Spawns chat / embed / rerank / whisper slots as configured, fronts Ollama + OpenAI APIs on `127.0.0.1:11434`, reaps orphan children on startup, mDNS-advertises `_quenchforge._tcp.local.` when opted in |
| `quenchforge doctor` | Hardware profile + config + binary lookup + model registry, all in one paste-safe blob for bug reports (`--redacted` swaps `$HOME` → `~`) |
| `quenchforge migrate-from-ollama` | Symlink-imports `~/.ollama/models/` blobs into the quenchforge model dir so existing Ollama users don't redownload |
| `quenchforge-preflight` | One-line `curl ... | sh` gating binary that emits `KEY=VALUE` for install scripts. Refuses to install on unsupported macOS / hardware |
| `scripts/build-llama.sh` | Builds patched `llama-server` (Metal, dual-arch, universal lipo) |
| `scripts/build-whisper.sh` | Builds patched `whisper-server` (same patch shape, different submodule) |
| `patches/llama.cpp/` and `patches/whisper.cpp/` | The actual diffs against each submodule. `scripts/apply-patches.sh` is idempotent + `--check` + `--reset` |

## Quickstart

### Building from source (today)

```sh
git clone --recursive https://github.com/Cerid-AI/quenchforge
cd quenchforge

# Apply patches + build both binaries
bash scripts/apply-patches.sh
bash scripts/build-llama.sh
bash scripts/build-whisper.sh   # only if you want the transcription slot

# Build the quenchforge supervisor + CLI
go build -o /usr/local/bin/quenchforge ./cmd/quenchforge
go build -o /usr/local/bin/quenchforge-preflight ./cmd/quenchforge-preflight

# Sanity check
quenchforge-preflight                       # status=ok on supported Mac
quenchforge doctor                          # hardware + config + registry

# Pull a model from HuggingFace and serve (v0.4.0+)
quenchforge pull llama3.2:3b              # see `quenchforge pull --list` for the curated catalog
quenchforge list                          # what's installed
QUENCHFORGE_DEFAULT_MODEL=llama-3.2-3b-instruct-q4_k_m quenchforge serve

# Or, if you already have Ollama with models cached locally:
quenchforge migrate-from-ollama           # symlinks ~/.ollama/models/ blobs in (no redownload)
QUENCHFORGE_DEFAULT_MODEL=llama3.2-3b quenchforge serve
```

### Use it as a drop-in for Ollama clients

```sh
# Already-configured Ollama clients just work
curl -X POST http://127.0.0.1:11434/api/chat \
  -d '{"model":"llama3.2-3b","messages":[{"role":"user","content":"hi"}]}'

# Transcribe audio (OpenAI-shaped /v1/audio/transcriptions)
curl -X POST http://127.0.0.1:11434/v1/audio/transcriptions \
  -F file=@speech.wav -F response_format=json
```

### Homebrew (recommended)

```sh
brew install cerid-ai/tap/quenchforge
quenchforge install            # writes the LaunchAgent (+ prestart port guard) to ~/Library/LaunchAgents
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.cerid.quenchforge.plist
curl http://127.0.0.1:11434/   # verify
```

Release binaries are signed with a Developer ID and Apple-notarized.

### Coexistence with Ollama.app

Quenchforge listens on `127.0.0.1:11434` — the same port as Ollama. If
`/Applications/Ollama.app` is installed, its login agent
(`com.ollama.ollama`) would otherwise race quenchforge to bind the port
at every login.

**This is handled automatically (v0.8.1+).** `quenchforge install`
writes a LaunchAgent whose `ProgramArguments[0]` is a prestart guard
(`~/.config/quenchforge/prestart-guard.sh`): on every start and at login
it boots out `com.ollama.ollama` and evicts any non-quenchforge listener
on `:11434` before starting the server. So quenchforge reliably owns the
canonical Ollama-API port with no manual eviction. Ollama.app stays
installed for its GUI — `open -a Ollama` still works; it just won't
auto-serve on `:11434`.

**Prefer them on separate ports instead?** Set `QUENCHFORGE_LISTEN_ADDR`
in the LaunchAgent's `<EnvironmentVariables>` (e.g. `:11435`), restart,
and point quenchforge clients at the new port; Ollama keeps `11434`.

Run `quenchforge doctor` to verify.

## First-launch prompts to expect

- **"Quenchforge would like to find and connect to devices on your local network"** — Sonoma+ TCC prompt for mDNS / Bonjour advertisement (`_quenchforge._tcp.local.`). Only shown when `QUENCHFORGE_ADVERTISE_MDNS=true`. Allowing this lets cerid-ai and other LAN clients auto-discover the service.
- **Telemetry** — none. Zero network traffic in the default config. The CLAUDE.md design contract reserves `QUENCHFORGE_TELEMETRY` for a future opt-in benchmark dashboard at `bench.quenchforge.dev`, but no code is shipped for that yet. Setting `QUENCHFORGE_SENTRY_DSN` enables Sentry error reporting for operators who explicitly want it; absent that env var, no Sentry traffic.
- **Gatekeeper** — once signed/notarized binaries ship, `quenchforge --version` is the first run that triggers a one-time online check.

## Configuration

All settings have sensible defaults. Selected env vars:

| Env var | Default | What |
|---|---|---|
| `QUENCHFORGE_LISTEN_ADDR` | `127.0.0.1:11434` | Gateway bind |
| `QUENCHFORGE_DEFAULT_MODEL` | `qwen2.5:7b-instruct-q4_k_m` | Chat slot model name (resolved under `QUENCHFORGE_MODELS_DIR`) |
| `QUENCHFORGE_EMBED_MODEL` | unset | Embed slot opt-in (BERT-family GGUF; produces dense embeddings on `/v1/embeddings`) |
| `QUENCHFORGE_RERANK_MODEL` | unset | Rerank slot opt-in (cross-encoder GGUF; serves `/v1/rerank`) |
| `QUENCHFORGE_WHISPER_MODEL` | unset | Whisper slot opt-in (ggml model path; serves `/v1/audio/transcriptions`) |
| `QUENCHFORGE_WHISPER_GPU` | `false` | Try Metal for whisper (currently buggy on AMD Mac; CPU default is 12.8× real-time on Xeon W-3245) |
| `QUENCHFORGE_SD_MODEL` | unset | Image-gen slot opt-in (stable-diffusion.cpp; serves `/v1/images/generations`) |
| `QUENCHFORGE_BARK_MODEL` | unset | TTS slot opt-in (bark.cpp; serves `/v1/audio/speech`) |
| `QUENCHFORGE_MODELS_DIR` | `~/.quenchforge/models` | Where Quenchforge looks for GGUFs |
| `QUENCHFORGE_LOG_DIR` | `~/Library/Logs/quenchforge` | Per-slot log files land here |
| `QUENCHFORGE_PID_DIR` | `~/.config/quenchforge/pids` | Orphan-reaper pidfile dir |
| `QUENCHFORGE_MAX_CONTEXT` | `8192` | `--ctx-size` passed to every slot. On AMD-discrete cards ≤ 11 GB this is auto-capped by VRAM tier (4096 on 8 GB, 2048 on 4 GB) so the KV cache fits; the cap only lowers, never raises, your value. ≥ 12 GB cards use it verbatim. |
| `QUENCHFORGE_METAL_N_CB` | `2` | Metal command-buffer count (`GGML_METAL_N_CB`); global default — per-slot overrides below |
| `QUENCHFORGE_EMBED_UBATCH_SIZE` | `0` (auto) | Per-call `--batch-size` / `--ubatch-size` for embed and code-embed slots. Zero auto-sizes by VRAM tier on AMD discrete (1024 on ≥ 12 GB, 512 on 8 GB, 256 on 4 GB) to cap Metal staging-buffer pressure and prevent the family-B sustained-load SIGABRT (`patches/README.md` section 3); non-AMD inherits MaxContext. An explicit value overrides the tier. |
| `QUENCHFORGE_EMBED_METAL_N_CB` | `0` (inherit `METAL_N_CB`) | Per-slot `GGML_METAL_N_CB` for embed and code-embed. Set to `1` on AMD discrete to serialise Metal command-buffer submission. |
| `QUENCHFORGE_RERANK_BATCH_SIZE` | `0` (llama.cpp's 512-token default) | Rerank slot `--batch-size` and `--ubatch-size`. Raise this when the reranker takes (query, doc) pairs longer than 510 tokens (e.g. `bge-reranker-v2-m3` with ≥ 1k-token chunks). |
| `QUENCHFORGE_RERANK_METAL_N_CB` | `0` (inherit `METAL_N_CB`) | Per-slot `GGML_METAL_N_CB` for the rerank slot. |
| `QUENCHFORGE_AUTO_BACKOFF` | `false` | Opt-in: gateway returns `HTTP 503` + `Retry-After: 2` on `/v1/embeddings` etc. when the slot's rolling p99 latency is `critical` (5× p50 or error rate > 5%). Default off — observability via `/health` works without this flag. |
| `QUENCHFORGE_ADVERTISE_MDNS` | `false` | Bonjour advertisement (`_quenchforge._tcp.local.`) |

**Operator overrides** (escape hatches over the AMD-Mac-safe defaults the patches + tuning apply automatically):
| Env var | Default | What |
|---|---|---|
| `GGML_METAL_FORCE_SIMDGROUP_REDUCTION` | unset | Re-enables the AMD-buggy reduction kernel — for diagnostic use only |
| `GGML_METAL_FORCE_BF16` | unset | Re-enables the AMD-buggy bfloat path |
| `GGML_METAL_BF16_DISABLE` | unset | Hard-disable bfloat regardless of profile |
| `GGML_METAL_CONCURRENCY_DISABLE` | unset | Serial encoder dispatch (slower but more predictable) |

Full list in `internal/config/config.go`.

## Why this exists

`cerid-ai` (an upstream project) needed real inference performance on a 2019 Mac Pro with a Radeon Pro Vega II while bridging to Apple-Silicon hardware. There's no maintained project that bridges this gap. The patches and tuning live here, in the open, so any other Intel-Mac + AMD user gets the same benefit without depending on cerid-ai. Sponsored by Cerid AI; license, governance, and roadmap are community-friendly.

## License

[Apache License 2.0](LICENSE). Third-party attributions in [NOTICE](NOTICE) and [third_party/LICENSES.md](third_party/LICENSES.md). Patch provenance in [patches/README.md](patches/README.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Hardware profile reports for GPUs we don't have (RDNA1 5700, W6800X Duo, anything Hackintosh) are especially welcome — open a `hardware_profile` issue with your `quenchforge doctor` output. Self-hosted CI runner setup is in [docs/SELF_HOSTED_RUNNER.md](docs/SELF_HOSTED_RUNNER.md).

## Security

See [SECURITY.md](SECURITY.md) for the disclosure process.
