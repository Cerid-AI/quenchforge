# Advanced routing, stream management & fleet futureproofing — roadmap

> Status: design / proposal. Sister documents: [`patches/README.md`](../patches/README.md),
> [`docs/METAL_AMD_BERT_CORRECTNESS.md`](METAL_AMD_BERT_CORRECTNESS.md),
> [`docs/SELF_HOSTED_RUNNER.md`](SELF_HOSTED_RUNNER.md).

## Purpose

Quenchforge today is a **single-host supervisor**: it spawns local `llama-server` /
`whisper-server` children (`internal/supervisor`), fronts them with a per-kind
reverse proxy (`internal/gateway`), and decides GPU-vs-CPU placement per workload
(`internal/placement`) with a duty-cycle GPU governor (`internal/scheduler` +
`internal/pressure`). That design exists to make the **AMD-Mac Metal correctness
fix** usable on one box.

The deployment is outgrowing one box. The near-term topology adds a **headless
Linux + CUDA node** (e.g. RTX 3090) for fast/correct chat and bulk embeddings,
keeps the Mac Pro's Vega II for overflow embeddings and its 160 GB RAM as a
large in-RAM index host, and — medium term — promotes an Apple-Silicon Mac Studio
to primary. Quenchforge's role shifts from *"AMD-Metal correctness shim on one
Mac"* to *"heterogeneous-fleet gateway."*

This document specifies the features and a phased TODO to make that shift without
regressing the single-Mac path or the Apple-Silicon no-op behaviour.

## Design principles

1. **Non-regression first.** With no fleet config set, behaviour is byte-identical
   to today: local children, loopback upstreams, single-host governor.
2. **The gateway is the stable contract.** Clients keep talking OpenAI + Ollama on
   one address; everything below the gateway can move between nodes.
3. **Decouple the three roles** Quenchforge currently fuses: (a) *AMD-Metal patched
   llama-server host*, (b) *fleet gateway/router*, (c) *GPU governor owner*. After
   this work they can run on different nodes.
4. **Minimal blast radius.** The reverse-proxy plumbing already accepts arbitrary
   upstream URLs (`gateway.SetUpstream(kind, rawURL)` →
   `httputil.NewSingleHostReverseProxy`). Most of the keystone work is config +
   health + governor-scoping, not a rewrite.
5. **No overbuild.** Ship the keystone (remote backends) and stop; layer the rest
   only when a concrete topology needs it.

---

## Feature areas

### F0 — AMD-Metal GPU reclamation via custom kernel interfacing (CORE)

**Motivation.** The systemic integration of the AMD GPU on Intel-Mac
architecture is quenchforge's founding deliverable — the vendors are leaving
this debt unpatched on both sides (Apple's Metal driver for non-UMA AMD is
frozen; upstream ggml's Apple-Silicon-only kernel assumptions were closed
not-planned in [#19563](https://github.com/ggml-org/llama.cpp/issues/19563)).
The patch series IS the moat. But the current operating posture is a
**partial retreat**: correctness was won by *disabling* the broken kernel
paths (patch 0001 gates + `GGML_METAL_CONCURRENCY_DISABLE` + `--flash-attn
off`), which made the GPU slow enough that chat, rerank, and whisper all
moved to CPU. Today the Vega II earns its keep only on batched embeddings.
Reclamation means replacing each disabled path with a **correct custom
kernel**, then letting placement move work back to the GPU on merit.

**Current device coverage (vega-pro, v0.9.1):**

| Kind | Device | Why |
|---|---|---|
| chat | CPU | GPU 27s vs CPU ~3.8s / 32 tok under constrained kernels; 257 SIGABRTs/7-day window pre-retreat (quantized matmul path) |
| embed / code-embed | GPU (batched) + CPU twin (singles) | v0.8.0 throughput win; ubatch cap + NCB=1 + patch 0002 pool |
| rerank | CPU | latency + (pre-v0.9.1) the 512-token batch bug; GPU now has a safe batch default → benchable |
| whisper | CPU | deeper unpatched encoder kernel bug; CPU is 12.8× realtime |

**Reclamation ladder (R-track).** Each rung has a written exit criterion and
is gated on the `[amd-gpu]` self-hosted runner + the sustained-load benches;
every fixed kernel gets offered upstream as a PR referencing #19563 (the
mission is leaving the artifact for the community, per CLAUDE.md).

- **R1 — Unbreak the parked `_fb` fallback kernels (patches 0003/0004).**
  Fix the Metal template signature that parked them
  (`helper_mv_reduce_and_write_fb<NR0=2>` fails to compile), land the
  LayerNorm/softmax/matmul fallback dispatchers that honour
  `has_simdgroup_reduction`. Exit: BERT correctness bench green with the
  fallback dispatchers active; measure whether `GGML_METAL_CONCURRENCY_DISABLE`
  and the ubatch caps can be relaxed (staged, one knob at a time).
- **R2 — Patch 0005: quantized-matmul fallback (chat's documented reversal
  trigger, `patches/README.md`).** Q4_K/Q5_K `mul_mv` fallback kernels for
  non-simdgroup devices. Exit: `bench-llama-sustained-load` p50 ≤ CPU
  baseline AND zero SIGABRTs over a 7-day soak → flip `chatParams` back to
  GPU on AMD-discrete.
- **R3 — Flash-attention fallback + prompt-cache re-enable.** Remove
  `--flash-attn off` (FA path safe without simdgroup reduction) and
  root-cause the LCP prompt-cache `GGML_ASSERT` so `--cache-ram 0 /
  --no-cache-prompt` can go. Prefill already runs 72.6 tok/s on the patched
  GPU — decode is the reclamation target.
- **R4 — Rerank GPU A/B under v0.9.1 defaults.** Batched rerank (20-doc
  requests are the norm) on GPU vs CPU; if GPU wins on batches, extend the
  embed-style "auto" split to rerank (batchN = document count — the routing
  plumbing already exists).
- **R5 — Multi-device Metal.** The scheduler's v0.2 note: round-robin across
  `MTLCopyAllDevices` with per-device admission (Vega II Duo / multi-MPX).
  Pairs with F2's `Target` generalisation — a second local GPU is just
  another target.
- **R6 — Whisper encoder kernel debug.** The "necessary but not sufficient"
  bug in whisper-specific kernels (`patches/README.md` honesty note). Lowest
  priority — CPU is already 12.8× realtime.

**Effort/risk.** R1/R2 are deep Metal-kernel work (weeks, not days) with the
highest payoff: they convert the correctness moat into a performance moat on
hardware nobody else serves. R3 carries crash-regression risk (two
historical crash families live there) — never ship without the 7-day soak.
R4 is cheap (bench + placement table change). The R-track runs on a
different layer from F1–F9 and can proceed in parallel.

### F1 — Remote backends (keystone)

**Motivation.** A backend should be either a *supervised local process* (today) or
a *remote endpoint* (the CUDA box, later the Studio). Today every upstream is
`127.0.0.1:<port>` (`internal/config/config.go`: `ChatPort 11500`, `EmbedPort
11501`, `EmbedCPUPort 11511`, …) and is assumed up because the supervisor owns its
lifecycle.

**Design.**
- Introduce a `Backend` descriptor: `{ kind, device, endpoint URL, supervised
  bool, health }`. Local backends keep a `Supervisor` handle; remote backends
  carry only a URL + health state.
- Config: `QUENCHFORGE_<KIND>_UPSTREAM` (e.g. `QUENCHFORGE_CHAT_UPSTREAM=http://10.0.0.20:11500`).
  When set, the gateway registers a **remote** backend for that kind via the
  existing `SetUpstream`, and the supervisor **does not** spawn a local child for
  it.
- Scope the GPU governor (`withGPUAdmission` in `internal/gateway`) to **local GPU
  backends only** — a remote kind must skip admission entirely (its GPU contention
  lives on another node, and the WindowServer/display-power sensor in
  `internal/pressure` is meaningless for it).
- Health: add active health-checking + a circuit breaker per remote backend
  (parallel to the local `AutoRespawn` path in `internal/supervisor`). On open
  breaker, fail fast or fall back (see F2).

**Effort/risk.** Small-medium. Highest leverage change in this doc; everything else
builds on it. Risk: error semantics for an unreachable remote vs. a crashed local
child must be unified in the gateway's 5xx/translation paths
(`internal/gateway/ollama_translate.go`).

### F2 — Multi-target, model-aware routing (beyond the GPU/CPU binary)

**Motivation.** `placement.Device` is `{GPU, CPU}` and `Policy.Mode` is
`gpu|cpu|auto` (`internal/placement/placement.go`). The fleet needs **N targets per
kind** (local-CPU, local-Vega-II, remote-3090, remote-Studio) and routing by more
than batch shape.

**Design.**
- Generalise the routing target from `Device` to a `Target = {node, device,
  backendID}`. Keep `Device` as a compatibility shim.
- Routing inputs, in priority order:
  1. **Model name** — small chat model → 3090; huge model (≥ N params / ≥ M GB) →
     local Xeon+RAM. A gateway-level model→backend table, or delegate to each
     backend's `--models-preset`.
  2. **Batch shape** — already implemented (`RouteRequest(kind, batchN,
     autoThreshold)`); extend so the "GPU" verdict can mean a *remote* GPU.
  3. **Backend health/load** — prefer the healthy, least-loaded backend (latency
     tracker already exists in `internal/gateway/latency.go`).
- **Failover chains** per kind: ordered backend list with automatic fallback
  (remote 3090 down → local CPU; remote embed down → local Vega II). Make
  "degrade to working, not 503" the default, matching the existing
  `placement_routing_test.go` intent.

**Effort/risk.** Medium. Keep the `placement` package import-cycle-free (it already
avoids cycles by using string kinds).

### F3 — Advanced stream management

**Motivation.** The gateway proxies SSE today via `httputil.ReverseProxy` + Flusher
(`internal/gateway/gateway.go`). Across a network hop to a remote backend, naive
proxying leaks slots and strands generations.

**Design.**
- **Client-disconnect → upstream cancel.** Propagate client `context` cancellation
  to the upstream request so a remote 3090 stops generating and frees its slot when
  the caller hangs up. (Today a local child would keep decoding.)
- **Stall/idle-timeout detection** on the SSE stream with a configurable first-token
  and inter-token deadline; surface a clean error instead of a hung connection.
- **Keep-alive / heartbeat** frames for long generations through intermediary
  proxies.
- **Per-backend concurrency caps + a small admission queue** with priority
  (interactive chat > batch embed). This generalises the single-host governor into
  a fleet-aware queue; reuse the `internal/scheduler` admission chokepoint.
- **No mid-stream replay.** Be explicit: a dropped upstream mid-generation fails the
  request; only *pre-first-token* failures are retried against the failover chain
  (F2).

**Effort/risk.** Medium. This is where remote backends earn their reliability;
under-investing here shows up as stranded GPU memory on the CUDA node.

### F4 — Health, resilience & observability

**Motivation.** `/health` already reports per-slot `ok|degraded|critical` and the
latency tracker drives `QUENCHFORGE_AUTO_BACKOFF`. Extend to the fleet.

**Design.**
- Per-backend health aggregation (local + remote + node reachability) in `/health`.
- Circuit breaker + exponential backoff per remote backend (mirror local
  `AutoRespawn`: 2s/4s/8s, capped).
- **Prometheus `/metrics`**: per-backend p50/p95/p99, queue depth, tokens/sec, GPU
  duty cycle, breaker state. Feeds the fleet observability dashboard.
- Structured request tracing across the gateway→backend hop (request ID header).

**Effort/risk.** Small-medium, incremental.

### F5 — Service discovery & fleet manifest

**Motivation.** mDNS *advertise* already exists (`QUENCHFORGE_ADVERTISE_MDNS`,
`_quenchforge._tcp.local.`). Add the *discovery* side so nodes find each other.

**Design.**
- mDNS **browse** for peer backends; auto-register discovered `_quenchforge._tcp`
  endpoints as candidate backends (gated behind explicit opt-in + an allowlist;
  never auto-trust the network).
- Optional declarative **fleet manifest** (`fleet.yaml`): nodes, their backends,
  routing/failover policy, governor scope. Env overrides still win for one-offs.

**Effort/risk.** Medium. Security-sensitive — see F8.

### F6 — Pluggable backend adapters (futureproofing for MLX / vLLM)

**Motivation.** Today a backend is implicitly a patched `llama-server` speaking the
OpenAI surface (`internal/supervisor` checks for `llama-server`/`whisper-server` in
the command line). The fleet will include **vLLM** on the 3090 and **MLX** (e.g.
`mlx_lm`-class servers) on the Mac Studio.

**Design.**
- A `BackendAdapter` interface abstracting: how to (optionally) supervise it, its
  health endpoint, and its **capability metadata** (models served, max context,
  embedding dims, supports-streaming, supports-rerank).
- Adapters: `llama-server` (today), `vllm` (remote, unsupervised), `mlx` (Apple
  Silicon). Capability negotiation lets the router avoid sending a request to a
  backend that can't serve it.
- The orphan-reaper command-line check in `internal/supervisor/supervisor.go`
  becomes adapter-provided, not a hardcoded `llama-server`/`whisper-server` string
  match.

**Effort/risk.** Medium. Pure futureproofing — implement the interface now, add
non-llama adapters when the hardware lands.

### F7 — llama.cpp RPC orchestration (memory pooling)

**Motivation.** A single oversized model can span the 3090's VRAM + the Mac Pro's
160 GB over 10 GbE via llama.cpp's RPC backend (`rpc-server` workers).

**Design.**
- Optional supervised `rpc-server` worker role; the chat backend launches with
  `--rpc <worker-list>`. Surface as a placement option (`device: rpc`) for a kind.
- Document the tradeoff in-tree: RPC-pooled inference runs at the **slowest tier's**
  pace — "make it fit," not "make it fast." Default off.

**Effort/risk.** Medium-high, niche. Lowest priority; only build when a model that
won't fit any single node is actually needed.

### F8 — Security/auth for networked backends

**Motivation.** Everything today assumes loopback (`127.0.0.1`); the translation
layer even notes loopback assumptions (`internal/gateway/ollama_translate.go`).
Multi-node means traffic crosses the LAN.

**Design.**
- Backends bind to a chosen interface deliberately, not 0.0.0.0 by default.
- Optional shared-secret bearer token between gateway and backends; optional TLS, or
  an explicit "trusted segmented VLAN" mode documented as the deployment assumption.
- The mDNS auto-discovery (F5) must require an allowlist/token before routing to a
  discovered peer.

**Effort/risk.** Small-medium, but **gates** any production multi-node rollout. Do
it alongside F1, not after.

### F9 — Gateway relocatability (futureproofing the Studio transition)

**Motivation.** When the Mac Studio becomes primary, the **gateway role moves to
it** while the Vega II / patched `llama-server` stays on the Mac Pro as a remote
backend. Today gateway, supervisor, and governor are co-located and partly assume
the AMD host.

**Design.**
- Make the gateway node-agnostic: it must run on a host with **no local GPU
  backend** and route entirely to remote backends (the inverse of today).
- The governor (`internal/pressure` display-power sensor) becomes a property of the
  *node that owns a local GPU*, not of the gateway — already implied by F1's
  governor-scoping, made explicit here.
- On Apple Silicon, confirm the patches remain runtime-gated no-ops and the MLX
  adapter (F6) is the chat backend.

**Effort/risk.** Small once F1 + F6 exist; mostly removing co-location assumptions.

---

## Phased TODO

### R-track — AMD-Metal GPU reclamation (parallel to P0–P3; different layer)
- [x] R1 (core, 2026-07-08): compile failure root-caused (helper above the
      FC_mul_mv_* declarations + threadgroup variable in non-kernel scope) and
      fixed; 0003 + 0004 landed as live patches (drafts removed);
      `bench-bert-correctness` all 4 probes PASS on Vega II GPU
      (determinism cos_sim 1.000000 vs 0.07–0.29 broken baseline).
- [x] R1 (soak + deploy, 2026-07-08): 4-phase display-asleep matrix on the
      fb kernels — **A** prod-parity 30-min gate PASS (1033 reqs, p50 5.10s /
      p95 9.89s, 0.59 req/s, RSS 2.38× < 4× floor, drift steady 0.9989);
      **B** relaxation answer: **NO — serialization is load-bearing.**
      Without `GGML_METAL_CONCURRENCY_DISABLE=1` output is instant garbage
      (cos_sim 0.117) even with fb kernels ⇒ the concurrent-dispatch
      command-buffer ordering bug on non-UMA AMD is an INDEPENDENT defect
      from the simdgroup miscompile, with a 60-second one-env-var
      reproducer; **C** ubatch 8192 (the v0.5.x ~2-min-crash config) 15-min
      PASS — crash class closed, but no throughput win (p50 7.02s vs 5.10s;
      tighter p95 7.67s) ⇒ keep 1024 default, caps are now perf-tuning not
      crash-guards; **D** bge-reranker deterministic on GPU ⇒ R4 unblocked.
      New llama-server (upstream a9883db + 4 patches) DEPLOYED to
      production; gateway embed/rerank/chat + cerid e2e verified.
- [x] R1.5 (2026-07-08): patch 0005 landed — `use_concurrency` defaults to
      false on `!has_unified_memory` devices (ggml-metal-context.m), with
      `GGML_METAL_CONCURRENCY_FORCE=1` as the test escape hatch. Verified:
      no-env correctness 4/4 PASS (was 3/4 FAIL pre-patch);
      FORCE=1 reproduces the failure on demand. 5-patch series round-trips
      clean. Upstream submission (vs #19563) still to be filed.
- [ ] R2: patch 0005 quantized-matmul fallback; `bench-llama-sustained-load` p50 ≤ CPU + 7-day zero-SIGABRT soak → `chatParams` back to GPU.
- [ ] R3: FA fallback kernel + LCP prompt-cache root-cause; remove the three chat safety flags one at a time, each behind the soak gate.
- [ ] R4: rerank GPU-vs-CPU batched A/B (v0.9.1 batch defaults make GPU rerank runnable); extend "auto" placement to rerank if GPU wins.
- [ ] R5: multi-device Metal scheduling (`MTLCopyAllDevices` round-robin, per-device admission) — with F2's `Target`.
- [ ] R6: whisper encoder kernel debug (opt-in `QUENCHFORGE_WHISPER_GPU` stays until fixed).
- [ ] Each landed kernel: upstream PR referencing #19563.

### P0 — Keystone (unblocks the whole fleet)
- [ ] F1: `Backend` descriptor; `QUENCHFORGE_<KIND>_UPSTREAM` remote registration; supervisor skips local spawn for remote kinds.
- [ ] F1: scope `withGPUAdmission` / governor to local GPU backends only.
- [ ] F8: interface-bind + optional shared-secret token; document the trusted-VLAN assumption.
- [ ] F4: per-remote health check + circuit breaker; surface in `/health`.
- [ ] Tests: remote-upstream routing, governor-bypass for remote, breaker open/close. Non-regression: unset config ⇒ identical single-host behaviour.

### P1 — Routing & streams
- [ ] F2: `Target` generalisation; model-name routing table; failover chains per kind.
- [ ] F3: client-disconnect → upstream cancel; stall/idle timeouts; per-backend concurrency cap + priority queue.
- [ ] F4: Prometheus `/metrics`; request-ID tracing.

### P2 — Fleet ergonomics & adapters
- [ ] F6: `BackendAdapter` interface; refactor the supervisor command-match behind it; `vllm` + `mlx` adapters (stubs until hardware).
- [ ] F5: mDNS discovery + `fleet.yaml` manifest with allowlist.
- [ ] F9: gateway-without-local-GPU mode; remove AMD-host co-location assumptions.

### P3 — Optional / niche
- [ ] F7: `rpc-server` worker role + `device: rpc` placement; document the speed tradeoff.
- [ ] Role presets: `single-mac` (today), `fleet` (gateway + remote), `studio-primary` (Phase 2).

---

## Non-regression & compatibility guarantees

- **Single-Mac unchanged:** with no `*_UPSTREAM`, no manifest, and no discovery, the
  supervisor spawns the same local children on the same loopback ports and the
  governor behaves exactly as today.
- **Apple Silicon stays a no-op:** the Metal patches remain runtime-gated; on an
  Apple-Silicon gateway the MLX adapter serves chat and the AMD-specific governor
  scope simply has no local GPU to govern.
- **Wire formats are frozen:** OpenAI + Ollama surfaces on `:11434` do not change;
  this work is entirely below the gateway contract.

## Open questions / decisions

- **Model→backend mapping:** centralise in the gateway, or delegate to each
  backend's `--models-preset` and route only by `model` name? (Leaning: gateway
  table for cross-node, preset for within-node.)
- **Manifest vs. env:** how much topology belongs in `fleet.yaml` vs. staying
  env-driven? Keep env as the override layer regardless.
- **Quenchforge vs. generic router at Phase 2:** once the AMD-Metal correctness work
  is no longer the differentiator, is the placement/governor/registry worth keeping
  over LiteLLM/llama-swap? This roadmap assumes *keep* — the fleet-aware placement +
  governor + GGUF registry are the differentiators — but the adapter interface (F6)
  keeps that reversible.
