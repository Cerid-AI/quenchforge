# Quenchforge

Open-standards LLM, embedding, reranker, and Whisper inference for **Intel Mac + AMD discrete GPU**, behind Ollama-compatible and OpenAI-compatible HTTP APIs.

Stock Ollama on a 2019 Mac Pro with an AMD Radeon Pro Vega II GPU does not accelerate on the GPU — it silently falls back to CPU ([ollama/ollama#1016](https://github.com/ollama/ollama/issues/1016), open since 2023). Quenchforge carries a small patch series on top of `llama.cpp` that fixes private-VRAM buffer allocation on discrete AMD Metal devices and disables the Apple-Silicon-tuned simdgroup matrix-multiply codepath on non-Apple-Silicon GPUs. Same single Go binary, same Ollama HTTP wire format, drop-in for any client that already speaks Ollama or OpenAI.

> **Status:** pre-release. MVP targets `quenchforge serve` with a single chat slot on Mac Pro 2019 + Vega II. See [docs/ROADMAP.md](docs/ROADMAP.md) once it exists.

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

> **Not yet available.** Coming with v0.1 once the brew tap is signed and notarized.

```bash
# brew install cerid-ai/homebrew-tap/quenchforge
# brew services start quenchforge
# quenchforge doctor
```

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
