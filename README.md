# Quenchforge

Open-standards LLM, embedding, reranker, and Whisper inference for **Intel Mac + AMD discrete GPU**, behind Ollama-compatible and OpenAI-compatible HTTP APIs.

Stock Ollama on a 2019 Mac Pro with an AMD Radeon Pro Vega II GPU does not accelerate on the GPU — it silently falls back to CPU ([ollama/ollama#1016](https://github.com/ollama/ollama/issues/1016), open since 2023). Building stock `llama.cpp` against Metal on the same hardware *does* enumerate the GPU and load a model, but the simdgroup-reduction and bfloat kernels miscompile on non-Apple Metal devices and the sampler emits garbage tokens forever ([ggml-org/llama.cpp#19563](https://github.com/ggml-org/llama.cpp/issues/19563)).

Quenchforge carries one small patch on top of `llama.cpp` that gates those kernels to Apple-Silicon-only, restoring **correct output on AMD Mac Metal**. Same Ollama HTTP wire format, same single Go binary supervisor — drop-in for any client that already speaks Ollama or OpenAI.

> **Status:** v0.0 pre-release. End-to-end inference verified live on Mac Pro 2019 + Radeon Pro Vega II (32 GB HBM2) with llama3.2:3b-Q4_K_M: **72.6 tok/s prompt, 4.1 tok/s generate**, coherent output. Binary distribution (signed Homebrew tap) is pending.

## Hardware compatibility matrix

| Configuration | Status | Notes |
|---|---|---|
| Intel Mac (Mac Pro 2019, iMac Pro, MacBook Pro 2018+) + AMD Vega II / Vega II Duo | **Primary** | The target this project exists to serve |
| Intel Mac + AMD W6800X / W6900X (RDNA2 Pro) | **Primary** | Apple-supported MPX modules |
| Intel Mac + AMD RDNA1/RDNA2 (5500M / 5700 / 6700M consumer) | Supported | Same patch surface; smaller HBM/GDDR |
| Apple Silicon (M1/M2/M3/M4/M5) | Supported (non-degraded) | Patches runtime-gated; effectively stock llama.cpp on this arch |
| Intel Mac, Intel iGPU only (Iris Plus, etc.) | Supported (CPU-class) | Metal available but very small VRAM — fallback to CPU dispatch is automatic |
| Linux / Windows | **Out of scope** | Use stock Ollama with CUDA / ROCm / DirectML; that path is already well-served |
| Hackintosh + AMD | Community best-effort | Tagged in telemetry as non-genuine; no SLA |

## What's in the box

- `quenchforge serve` — supervisor that owns one or more `llama-server` child processes pinned to detected AMD GPUs, fronted by the Olla HTTP gateway speaking both Ollama and OpenAI APIs on `127.0.0.1:11434`
- `quenchforge doctor` — hardware profile + recent Metal log capture, formatted for one-paste bug reports
- `quenchforge migrate-from-ollama` — symlink-imports `~/.ollama/models/` into our cache so existing users don't re-download GGUFs
- `quenchforge-preflight` — one-line `curl | sh` script that checks your hardware before any install
- `quenchforge bench` — first-launch micro-benchmark that finds the right `--n-gpu-layers` for your VRAM

## Quickstart

### Building from source (today)

```bash
git clone --recursive https://github.com/Cerid-AI/quenchforge
cd quenchforge

# Apply the llama.cpp patch + build the patched llama-server
bash scripts/apply-patches.sh
bash scripts/build-llama.sh

# Build the quenchforge supervisor + CLI
go build -o /usr/local/bin/quenchforge ./cmd/quenchforge
go build -o /usr/local/bin/quenchforge-preflight ./cmd/quenchforge-preflight

# Sanity check
quenchforge-preflight                 # should print status=ok on a supported Mac
quenchforge doctor                    # hardware profile, llama-server lookup, model registry
quenchforge migrate-from-ollama       # symlink existing Ollama GGUFs (optional)
quenchforge serve                     # gateway on 127.0.0.1:11434 + supervised chat slot
```

The supervisor automatically finds `llama-server` under `./llama.cpp/build-*/bin/` after a `build-llama.sh` run, or at `/usr/local/bin/llama-server` if you installed it system-wide.

### Homebrew tap (coming with v0.1)

```bash
# brew install cerid-ai/homebrew-tap/quenchforge
# brew services start quenchforge
# quenchforge doctor
```

Signed + notarized bottles are pending Apple Developer ID configuration.

## First-launch prompts to expect

- **"Quenchforge would like to find and connect to devices on your local network"** — Sonoma+ TCC prompt for mDNS / Bonjour advertisement (`_quenchforge._tcp.local.`). Allowing this lets cerid-ai and other LAN clients auto-discover the service. Decline if you only want loopback access.
- **Telemetry consent** — first launch shows a one-screen consent flow. Both error reports (Sentry) and the anonymous benchmark dashboard at `bench.quenchforge.dev` are off until you opt in. Hardware profile, tokens/sec, and latency are the only things ever sent.
- **Gatekeeper** — the binary is signed and notarized. If you see "downloaded from the Internet" the first time you run it, that's normal; Gatekeeper checks notarization online once and remembers.

## Why this exists

cerid-ai (an upstream project) needed real inference performance on a 2019 Mac Pro with a Radeon Pro Vega II while bridging to Apple-Silicon hardware. Stock Ollama is CPU-only on this combination. There's nothing else maintained that bridges this gap. The patches and tuning live here, in the open, so any other Intel-Mac + AMD user gets the same benefit without depending on cerid-ai. The project is sponsored by Cerid AI but the license, governance, and roadmap are community-friendly.

## License

[Apache License 2.0](LICENSE). Third-party attributions in [NOTICE](NOTICE) and [third_party/LICENSES.md](third_party/LICENSES.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Hardware profiles for GPUs we don't have are especially welcome — open a `hardware_profile` issue with your `quenchforge doctor` output.

## Security

See [SECURITY.md](SECURITY.md) for the disclosure process.
