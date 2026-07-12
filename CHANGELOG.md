# Changelog

All notable changes to Quenchforge are tracked here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/).
Versions follow [SemVer](https://semver.org/) — minor bumps add features,
patch bumps fix bugs or polish without behaviour change.

---

## v0.10.1 — supervisor lifecycle hardening (post-incident) (2026-07-12)

Driven by the 2026-07-11 incident on the reference Mac Pro: a jetsam-class
kill during a host-wide memory crunch left the chat slot's llama-server dead
— and, because only auto-respawn slots were ever `Wait()`ed, unreaped as a
zombie for days while the gateway kept routing chat requests at a corpse.

- **CPU-placed slots get auto-respawn.** `cpuTuning` treated AutoRespawn
  as a GPU-safety knob and never set it, so the placement policy's CPU
  routes — chat and rerank on the AMD-discrete defaults — ran with no
  restart policy at all. This, not the GPU path, is what left the
  incident's chat slot dead; caught live when a SIGKILLed rerank slot
  logged "no restart policy — slot stays down" during deploy validation.
- **Every slot child is reaped by an unconditional watcher**, whatever its
  `RestartPolicy`. A dead, un-respawned slot also releases its log handle
  and removes its pidfile immediately instead of leaking both until the
  next Start/Stop.
- **The crash-storm cap no longer gives up forever.** After 3 respawns in
  the rolling 60-second window the slot cools off (5 minutes) and probes
  again with a fresh window — transient host memory pressure no longer
  converts into a permanent slot outage.
- **`Stop()` consumes the watcher's wait result** instead of double-
  `Wait()`ing and signalling already-dead children (the source of the
  "SIGTERM: operation not permitted" shutdown noise). A Stop that lands
  during a pending respawn cancels the respawn.
- **`Slot.PID()` reports 0 for a reaped child** instead of the stale PID,
  so status surfaces never point operators at a dead process.
- **`doctor` gained a "Running server binary" section** that flags the
  upgrade-while-running footgun: `brew upgrade` replaces the Cellar binary
  under the live server, whose next text-page fault SIGKILLs it with
  "Code Signature Invalid" (observed 2026-07-08) — and until that lands it
  silently keeps executing the old version. Formula caveats and the ops
  gotchas document the one-restart remediation.
- **Fixed the AMD-discrete startup banner** that still announced the three
  retired chat safety flags (R3, 2026-07-08) long after `tuning.go`
  stopped applying them.

## v0.10.0 — AMD-Metal reclamation: fallback kernels, intrinsic serial dispatch, measured placement (2026-07-09)

The two-month-parked `_fb` fallback kernels are live (roadmap R1, the first
rung of the F0 GPU-reclamation track). The park-blocking compile failure in
`helper_mv_reduce_and_write_fb<NR0=2>` was two MSL rules, not the template:
the helper sat textually above the `FC_mul_mv_*` function constants it reads,
and a `threadgroup` array was declared in non-kernel scope. Fixed by
relocating the helper and hoisting a fixed 8 KiB `threadgroup` buffer into
each `_fb` kernel entry (passed down as a parameter; the dynamic `shmem`
entry parameter — the gfx-rs/wgpu#4500 hazard — stays unused by design).

Validated on Mac Pro 2019 + Vega II: full patched Metal source compiles on
device; `bench-bert-correctness` passes all 4 probes ON GPU with the
fallback dispatchers active — same-batch and separate-call determinism
cos_sim = 1.000000 (broken baseline: 0.07–0.29), paraphrase 0.9551 ≫
unrelated 0.4430, L2 = 1.0000. The 3-way rebase onto current upstream
(`a9883db`) is folded into the regenerated patches.

Soak matrix (same day, display-asleep, isolated server; full data under the
session's bench report):

- **A — prod-parity 30-min gate: PASS.** 1033 reqs, 0 SIGABRTs, p50 5.10s /
  p95 9.89s @ concurrency 4 × batch 8 (0.59 req/s ≈ 4.7 embeds/s), RSS
  216→516 MB (2.38×, under the 4× leak floor), interleaved drift probe
  steady at 0.9989 over 30 min (no cumulative GPU-state corruption).
- **B — relaxation answer: NO.** Without `GGML_METAL_CONCURRENCY_DISABLE=1`
  output is immediate garbage (cos_sim 0.117) even with the fb kernels —
  the non-UMA AMD concurrent-dispatch command-buffer ordering bug is an
  INDEPENDENT defect from the simdgroup miscompile. One-env-var, 60-second
  reproducer; serialization stays load-bearing. Follow-up (roadmap R1.5):
  make the workaround intrinsic by gating `MTLDispatchTypeSerial` on
  `!has_unified_memory` in ggml-metal-device.m.
- **C — ubatch 8192 (the v0.5.x ~2-min-crash config): 15-min PASS.** Crash
  class closed by pool + fb kernels; no throughput win vs 1024 (p50 7.02s
  vs 5.10s, tighter p95) — the VRAM-tier caps are now performance tuning,
  not crash guards.
- **D — bge-reranker deterministic on GPU** — the rerank GPU A/B (R4) is
  unblocked.

### Added — patch 0005: serial dispatch by default on non-UMA devices

The Phase-B finding, made intrinsic: `use_concurrency` now defaults to
false when the Metal device lacks unified memory, so BERT correctness on
AMD-Mac no longer depends on operators setting
`GGML_METAL_CONCURRENCY_DISABLE`. `GGML_METAL_CONCURRENCY_FORCE=1` opts
back in for testing (verified to reproduce the cos_sim ~0.117 failure on
demand). The series is now 5 patches and round-trips clean from pristine
upstream.

### Changed — R3: the three chat-slot AMD safety flags retired

Measurement on the 5-patch build inverted both rationales. FA: `--flash-attn
auto` now decodes 3.7–3.8 tok/s vs 2.6 with `off` (+42%), deterministic and
GPU-resident — the historical CPU-fallback throttle is gone upstream.
Prompt cache: the LCP prompt-save `GGML_ASSERT(buf_dst)` crash was the same
staging-allocation class patch 0002 pools — 6 LCP-similar requests run
clean, the cache works (prompt_n 83→17 on a shared prefix), and an 8-min
sustained chat run passed (56 reqs, 0 failures, RSS 1.00×).
`chatParams` now passes only `--gpu-layers 999`; regression tests pin the
retirement. GPU decode remains serial-dispatch-capped, so chat stays
CPU-placed by default on AMD-discrete.

The new llama-server (upstream `a9883db` + the 4-patch series) is deployed
to production: gateway embed (4/4 probes), rerank determinism, chat
completion, and a cerid end-to-end ingest all verified post-restart.

---

## v0.9.1 — CPU-slot batch sizing + instance-scoped error-rate backoff (2026-07-08)

Root-caused from a downstream eval incident (cerid `/agent/query` burst):
three interacting defects turned a routine retrieval workload into
deterministic 500s and a 60-second 503 storm.

### Fixed — CPU-placed slots reverted to llama-server's 512-token batch

`cpuTuning` returned only `--gpu-layers 0`, so every CPU-placed slot (rerank
+ the "auto" embed/code-embed CPU twins — CPU placement shipped in v0.9.0)
ran at llama-server's 512 default physical batch. The embedding and rerank
pooling paths reject any single sequence longer than n_ubatch, so every
>512-token (query, doc) pair or single input returned HTTP 500 "input too
large to process" — reintroducing the v0.5.0 512-token-limit bug on the CPU
path, and silently dropping the documented `QUENCHFORGE_RERANK_BATCH_SIZE` /
`QUENCHFORGE_EMBED_UBATCH_SIZE` overrides (dead knobs on CPU-placed slots).

Now: CPU-placed embed/code-embed/rerank slots size the physical batch to the
context (no Metal staging pressure exists on CPU; fits-context ⇒ fits-batch,
the same principle the non-AMD embed path always used), and operator
overrides reach CPU-placed slots. GPU-placed rerank also gains a real
default (non-AMD: context-sized; AMD-discrete: the bench-validated amdSizing
ubatch as a staging cap) — the old "no default" shipped llama-server's 512,
a strictly-worse deterministic 500 generator.

### Fixed — auto-backoff false positives: 503 storms from healthy slots

The v0.6.0 p99/p50-ratio classifier assumed one homogeneous instance per
kind. v0.9.0's "auto" placement made embed latency bimodal by design
(millisecond CPU singles + multi-second governed GPU batches) — one mixed
60-second window pushed the ratio past 5x and flipped "critical" with a 0.00
error rate, after which `QUENCHFORGE_AUTO_BACKOFF=true` shed ALL embed
traffic (including requests bound for the idle, healthy CPU twin) with 503s.

Now: (a) the dual-placed CPU twin records under its own tracker key
(`embed-cpu` / `code-embed-cpu`, visible in `/health`), so the two
instances' distributions never mix and backoff is evaluated against the
instance a request actually routes to; (b) shedding fires on a critical
ERROR RATE only — the actual family-B crash signature — while latency-ratio
degradation remains observability-only in `/health` ("degrade to working,
not 503"); (c) backoff state transitions log once (`auto-backoff ON/OFF for
<slot>`), so the next storm is diagnosable from the gateway's own log.

---

## v0.9.0 — GPU resource-control framework: placement + governor (2026-06-08)

A two-part control plane so sustained inference on a single-GPU Mac can't
starve the display compositor (WindowServer) into a kernel-watchdog panic, and
so each workload runs on the device that actually serves it best.

### Placement — where work runs (`internal/placement`)

Per-kind device policy `gpu | cpu | auto`, hardware-adaptive defaults, operator
overrides via `QUENCHFORGE_PLACE_{CHAT,EMBED,CODE_EMBED,RERANK}`.

- On AMD-discrete (e.g. Mac Pro 2019 + Vega II), **chat and rerank default to
  CPU** — measured single-request latency is far better on CPU than the AMD
  Metal path (chat ~4–5s vs ~27s for 32 tokens; corroborated by
  `bench-llama-sustained-load` p50≈29s) — which also removes them from
  compositor GPU contention. Non-AMD (Apple Silicon UMA) keeps everything on GPU.
- embed / code-embed default to GPU (the v0.8.0 batched-throughput win) and may
  run **`auto`**: a dual GPU+CPU instance where the gateway routes each request
  by input count — single → CPU (latency), batched → GPU (throughput). CPU
  instances bind `QUENCHFORGE_{EMBED,CODE_EMBED}_CPU_PORT` (11511/11516);
  `QUENCHFORGE_AUTO_BATCH_THRESHOLD` (default 1) is the split point.

### Governor — how much GPU (`internal/pressure` + `internal/scheduler` + gateway)

Adaptive GPU admission: a concurrency cap **plus duty-cycle idle gaps** — the
key lever, because concurrency capping alone does not prevent starvation
(sustained gapless command buffers do, at any concurrency). Driven by a
display-power + memory-pressure sensor: full throughput when headless or the
display is asleep; reserve compositor headroom when a screen is driven. Env:
`QUENCHFORGE_GOVERNOR`, `_GPU_CONCURRENCY_{MAX,DISPLAY_ACTIVE}`,
`_GPU_DUTY_DISPLAY_ACTIVE`, `_GOVERNOR_MAX_COOLDOWN_MS`.

### Safety / compatibility

- Dormant by default: with no kind set to `auto`, no second instance launches and
  the single-upstream path is unchanged; the governor is a no-op when headless.
- Single GPU-admission chokepoint (`withGPUAdmission`); CPU-placed kinds skip
  admission entirely. CPU-routed work runs ungoverned at full speed.
- Note: `Device Utilization %` from ioreg is unreliable on AMD-discrete Macs —
  validate GPU contention with Instruments' Metal System Trace.

### Also in this release

- Public-surface consolidation: drift fixes, leak redaction, internal-doc removal.
- `rebase-upstream`: apply the committed patch series directly onto upstream.
- README refreshed; `docs/APPLE_DEVELOPER_ID.md` made maintainer-local.

---

## v0.8.2 — one-line curl installer (2026-06-02)

### One-line installer

Added `install.sh` (served from `main`):

```sh
curl -fsSL https://raw.githubusercontent.com/Cerid-AI/quenchforge/main/install.sh | sh
```

Resolves the latest release, downloads the universal `darwin_all` tarball,
**verifies its SHA-256 against `checksums.txt`**, installs both binaries to
`/usr/local/bin`, gates on `quenchforge-preflight` (`status=ok`), and writes
the LaunchAgent + prestart port guard via `quenchforge install`. Knobs:
`QUENCHFORGE_VERSION` to pin, `QUENCHFORGE_PREFIX` to relocate,
`QUENCHFORGE_NO_SERVICE=1` for binaries-only. macOS-only and installs a
per-user LaunchAgent on purpose (the daemon needs an Aqua session for Metal
GPU access). Also fixed the GitHub-release footer, which pointed at a
non-existent `quenchforge-preflight_*_darwin_universal.tar.gz` asset.

---

## v0.8.1 — Prestart port guard reclaims :11434 from Ollama (2026-05-31)

`quenchforge install` now also writes a prestart guard to
`~/.config/quenchforge/prestart-guard.sh` and points the generated plist's
`ProgramArguments[0]` at it. Before exec'ing `quenchforge serve` the guard
boots out Ollama's launchd job and evicts any non-quenchforge listener on
port 11434, so quenchforge authoritatively reclaims the canonical
Ollama-API port on every (re)start and at login.

Fixes the recurring contention where Ollama.app's auto-launched
`ollama serve` grabbed 11434 during a quenchforge restart window: because
the pre-bind check yields (exits 0) on a held port and
`KeepAlive.SuccessfulExit=false` then leaves the job dead, the squatter
would win and quenchforge stayed down until hand-evicted. The guard
removes the manual step. It only kills the actual squatter (never a
running quenchforge / `llama-server`) and is a no-op when Ollama isn't
present. Source: `cmd/quenchforge/prestart-guard.sh`; covered by
`TestInstall_WritesPlistAndPrestartGuard`.

---

## v0.8.0 — AMD-discrete GPU mode + VRAM-tier-adaptive sizing (2026-05-31)

Promotes the `v0.8.0-rc2` AMD-discrete GPU-mode revival (below) to a
final release and adds **VRAM-tier-adaptive slot sizing** so the full
range of Intel-Mac AMD GPUs — not just the 32 GB Vega II — runs
out-of-the-box without operator hand-tuning.

### VRAM-tier-adaptive sizing

Prior to this release every AMD-discrete profile inherited the Vega II
bench constants (`--ctx-size 8192`, embed `--ubatch-size 1024`)
regardless of card. On a smaller card (8 GB RX 5700, 4 GB MacBook Pro
dGPU) those defaults could oversubscribe VRAM, forcing the operator to
discover and set `QUENCHFORGE_MAX_CONTEXT` / `QUENCHFORGE_EMBED_UBATCH_SIZE`
by hand. `internal/tuning/tuning.go::amdSizing` now derives both from the
detected headline VRAM (`hardware.Info.GPUVRAMGB`, threaded into
`KernelParams`):

| VRAM | `--ctx-size` cap | embed `--ubatch-size` | example cards |
|---|---|---|---|
| ≥ 12 GB | none (keeps `MaxContext`) | 1024 | Vega II/Duo, W6800X, W6900X, Vega 56/64, 5600M |
| 7–11 GB | 4096 | 512 | RX 5700 / 5700 XT, W5700X |
| ≤ 6 GB | 2048 | 256 | 4 GB MacBook Pro dGPUs (5300M/5500M), Polaris 560X |

Design guarantees:

- **Zero regression on the validated path.** The ≥ 12 GB tier (and any
  VRAM probe that returns 0/unknown) keeps the exact Vega-II-benched
  values, so the canonical Mac Pro config is byte-for-byte unchanged.
- **Caps only ever lower.** `buildSlotArgs` applies the context ceiling
  as `min(cfg.MaxContext, cap)`, so an operator who raised
  `QUENCHFORGE_MAX_CONTEXT` on a big card is never clamped.
- **Operator overrides still win.** An explicit
  `QUENCHFORGE_EMBED_UBATCH_SIZE` beats the tier ubatch; the context cap
  is an independent safety knob.
- The fix is family-agnostic: unlisted/future AMD cards fall through
  `classifyProfile` to `vega-pro` and are sized by VRAM like any other.

New coverage: `TestAmdSizing_Tiers`, `TestKernelParams_EmbedLowVRAMScalesDown`,
`TestKernelParams_ContextCapAppliesToAllAMDSlots`,
`TestKernelParams_HighVRAMAndNonAMDHaveNoContextCap`,
`TestKernelParams_UbatchOverrideBeatsTierButCapStands`, and
`TestBuildSlotArgs_LowVRAMAMDCapsContextAndUbatch`.

---

## v0.8.0-rc2 — AMD-discrete GPU mode revival (2026-05-25)

The Mac Pro 7,1 + Radeon Pro Vega II 32 GB configuration now runs all
four production slot types (chat, embed, code-embed, rerank) on GPU
by default. The Xeon W's 16 cores are freed for the gateway and the
OS. v0.7.0 through v0.7.2.x ran AMD-discrete slots on CPU as a safety
fallback after two distinct Metal-on-AMD bugs were discovered; both
are now addressed.

7-day production observation window in progress; tag `v0.8.0` (final)
will land at the end of that window pending zero kernel panics and
AutoRespawn count ≤ 10/week.

### Two-layer fix

**Kernel layer:** new patch `0002-metal-staging-buffer-pool` replaces
the per-call `newBufferWithBytesNoCopy` allocation in
`ggml_metal_buffer_set_tensor` and `_get_tensor` with a bounded
MTLBuffer pool (15 power-of-two size classes from 4 KiB to 64 MiB,
per-class FIFO cap of 4). Eliminates the AMD-discrete IOMMU
registration churn that exhausts the driver's mapping pool under
sustained load and triggers the `GGML_ASSERT(buf)` family-B SIGABRT
documented in `patches/README.md` Section 3.

**Supervisor layer:** `internal/tuning/tuning.go::chatParams /
embedParams / rerankParams` AMD-discrete branches flip `--gpu-layers`
from `0` (CPU route) to `999` (GPU) and set the new
`SlotTuning.MetalConcurrencyDisable: true` field. `slotEnv` in
`cmd/quenchforge/main.go` injects `GGML_METAL_CONCURRENCY_DISABLE=1`
in the slot's environment, disabling the upstream
`MTLDispatchTypeConcurrent` path that produced cross-call
non-determinism in BERT embeddings on non-UMA Macs (llama.cpp
[issue #19563](https://github.com/ggml-org/llama.cpp/issues/19563)).

`embedParams` additionally re-enables the 1024 ubatch cap
(`amdEmbedUbatchDefault`) — CLAUDE.md operational gotcha #2 caps
per-call Metal staging-buffer pressure even with patch 0002's pool
in place. `AutoRespawn: true` retained on all three slot kinds as
defense in depth for unknown crash classes.

### Bench results

Validated on Mac Pro 2019 + Vega II 32 GB HBM2, 2026-05-25:

| Bench | Result | Throughput / latency |
|---|---|---|
| `bench-bert-correctness.py` (nomic) | PASS | cos_sim 1.000000 |
| `bench-bert-correctness.py` (jina) | PASS | cos_sim 1.000000 |
| `bench-bert-correctness.py --rerank` (bge) | PASS | identical scores |
| `bench-llama-correctness.py` (llama3.1-8b) | PASS | 10 identical responses |
| `bench-bert-sustained-load.py --duration 1800` (nomic) | PASS | 2227 reqs / 1.24 req/s / p50=2.66s / p95=4.80s / RSS 1.03× |
| `bench-bert-sustained-load.py --duration 1800` (jina) | PASS | 1571 reqs / 0.87 req/s / p50=3.03s / p95=11.37s / RSS 1.03× |
| `bench-llama-sustained-load.py --duration 1800` (chat) | PASS | 157 reqs / 0.09 req/s / p50=29.4s / p95=36.0s / RSS 1.00× |
| Escape-hatch (`GGML_METAL_DISABLE_STAGING_POOL=1`) | PASS (fails as expected) | Process DIED at ~4 min / 212 reqs with family-B SIGABRT |

Combined 3955 sustained requests across 90 min wall-clock, zero
family-B SIGABRTs, zero kernel panics. Throughput speedup vs CPU
baseline: ~2.5× for nomic embed (1.24 vs ~0.5 req/s), ~1.7× for
jina code-embed (0.87 vs ~0.5 req/s).

The escape-hatch test confirms patch 0002 is empirically what's
preventing the family-B class: with the pool bypassed via the env
var, the SIGABRT returns within minutes.

### Operator escape hatches

- `GGML_METAL_DISABLE_STAGING_POOL=1` — disables the pool, reverts
  to upstream per-call allocation. Family-B SIGABRT returns within
  ~5 min under sustained load.
- Downgrade to v0.7.2 binary — full CPU route restored.

### Apple Silicon

Unaffected. The pool code is short-circuited by `buf->is_shared` on
UMA devices; the env var is only injected when
`SlotTuning.MetalConcurrencyDisable` is true, which is only set on
AMD-discrete branches.

### Bench harness additions

- `scripts/bench-llama-sustained-load.py` (new) — chat counterpart of
  `bench-bert-sustained-load.py`. 30-min hammer with deterministic
  drift probes interleaved.
- `scripts/bench-llama-correctness.py` (new) — short determinism +
  semantic-keyword probes for chat slots. Mirrors
  `bench-bert-correctness.py` exit-code contract.
- `bench-bert-sustained-load.py` `RSS_GROWTH_FACTOR` raised from 2.0
  to 4.0 to accommodate the pool's worst-case ~512 MB working set
  without false leak alarms.

### Scope reduction

Patches `0003-metal-amd-bert-fallback-kernels.patch` and
`0004-metal-amd-bert-matmul-fallback.patch` parked back to
`patches/llama.cpp/drafts/.broken` pending a Metal kernel template
signature fix (`helper_mv_reduce_and_write_fb<NR0=2>` doesn't
compile). The critical path validated above does NOT depend on those
patches; the _fb fallback kernels remain future optimization work,
not a correctness gate.

The bbbe40a commit en route fixed a pre-existing bug in 0003 (the
patch had a duplicate-of-0001 hunk silently rejected at apply-time);
the underlying template bug surfaced once 0003 actually applied,
hence the parking decision.

### Files changed

- `patches/llama.cpp/0002-metal-staging-buffer-pool.patch` (new — promoted from drafts)
- `patches/llama.cpp/drafts/0003-metal-amd-bert-fallback-kernels.patch.broken` (parked from active)
- `patches/llama.cpp/drafts/0004-metal-amd-bert-matmul-fallback.patch.broken` (parked from active)
- `internal/tuning/tuning.go` (new `MetalConcurrencyDisable` field; three AMD branches updated)
- `internal/tuning/tuning_test.go` (three tests renamed + updated; +1 `containsSubslice` helper)
- `cmd/quenchforge/main.go` (`slotEnv` extension)
- `cmd/quenchforge/serve_test.go` (+1 test: `TestSlotEnv_AMDIncludesConcurrencyDisable`)
- `scripts/bench-llama-{correctness,sustained-load}.py` (new chat benches)
- `scripts/bench-bert-sustained-load.py` (RSS threshold raised to 4.0×)
- `patches/README.md` (Section 3 updated to SHIPPED v0.8.0)
- `docs/superpowers/specs/2026-05-25-amd-metal-staging-buffer-pool-revival-design.md` (new — the design spec)
- `docs/superpowers/specs/2026-05-25-amd-metal-acceleration-design.md` (marked superseded)
- `docs/superpowers/plans/2026-05-25-amd-metal-staging-buffer-pool-revival.md` (the implementation plan)

---

## v0.7.2 — stability hardening + Ollama deconfliction (2026-05-25)

Driven by a 2026-05-24 system freeze on the dev Mac Pro requiring two
hard reboots. RCA traced the failure to a multi-day cascade rooted in
quenchforge's gateway crash-spamming when Ollama.app's login agent won
the race for 127.0.0.1:11434, combined with unbounded slot log growth
and continued chat-slot SIGABRT on Vega II Metal.

### Stability fixes

- **Pre-bind port check** (`internal/portcheck`). Before binding 11434,
  identify any existing listener via `lsof` (netstat fallback).
  Verdict-specific exits: Ollama → canonical actionable error + exit 0;
  stale quenchforge → graceful takeover; other → log + exit 0. The
  clean exit pairs with the plist KeepAlive dict (below) so launchd
  does not crash-spam.

- **LaunchAgent plist hardening** (`plist_template.plist`). KeepAlive
  changes from `<true/>` to `<dict><SuccessfulExit><false/></dict>`,
  suppressing respawn on the new clean-exit path. ThrottleInterval=10
  stays as belt-and-suspenders for real crashes.

- **Slot log rotation** (`internal/supervisor/logrotate.go`). 100 MB
  per file, 5 backups by default. Overridable via
  QUENCHFORGE_LOG_MAX_BYTES / QUENCHFORGE_LOG_BACKUPS. Eliminates the
  3.73 GB unbounded embed.log class.

- **Chat slot routes to CPU on AMD-discrete** (`internal/tuning/
  tuning.go::chatParams`). Quantized chat models hit the same
  family-B SIGABRT pattern as embed/rerank pre-v0.7.0 (patch 0003/0004
  cover fp32/fp16 only). Mirror of the existing embed/rerank CPU
  policy. Reversal: planned patch 0005.

### Diagnostics

- **`quenchforge doctor` extended** with four new sections: Ollama
  LaunchAgent state, disk free on /System/Volumes/Data, per-slot log
  file sizes, port 11434 holder. `--explain` flag appends per-finding
  remediation guidance for bug-report triage.

### Documentation

- README: new "Coexistence with Ollama" section under Installation.
- patches/README.md Section 3: extended to cover chat-slot symmetry.

### No upstream patches added or removed

The single llama.cpp patch (0001-metal-correctness-on-non-apple-silicon)
remains the only load-bearing patch; 0003/0004 stay staged but
inactive in production (per v0.8.0-rc1 changelog).

---

## v0.8.0-rc1 — matmul fallback kernels (2026-05-18, patch-staged)

Completes the kernel-level fix series that v0.7.1 began. Adds
fallback Metal kernels for `kernel_mul_mv_t_t` and
`kernel_mul_mv_t_t_4` (fp32/fp16 mat-vec) that bypass
`helper_mv_reduce_and_write`'s two `simd_sum` reductions with a pure
threadgroup tree-reduction over fixed-size function-local memory.

Combined with patch 0003, the four reduction-heavy kernels BERT
forward passes exercise (LayerNorm, softmax, matmul reduce, matmul
reduce vector) now have AMD-safe fallback variants. The dispatcher in
`pipeline_mul_mv` selects the `_fb` suffix when both
`has_simdgroup_reduction == false` AND the tensor type is fp32/fp16
— quantized BERT paths fall through to the upstream (broken-on-AMD)
kernel, so the dispatcher's predicate makes the coverage state
explicit.

### Status — patch staged, NOT yet activated in production

`patches/llama.cpp/0004-metal-amd-bert-matmul-fallback.patch` lands
this release. The patch is applied automatically at build time via
`scripts/apply-patches.sh`. **Production still runs on the v0.7.0
CPU route** (`--gpu-layers 0` in `internal/tuning/tuning.go` for the
AMD-discrete embed/code-embed/rerank slots).

Activation requires:

1. Build with all four patches applied (automatic via apply-patches.sh).
2. `scripts/bench-bert-correctness.py` PASSES (same-batch cos_sim,
   separate-call cos_sim, semantic sanity, L2 norm bounds).
3. `scripts/bench-bert-sustained-load.py --duration 1800` PASSES
   (no SIGABRT, no HTTP 5xx burst, no catastrophic drift, no RSS
   leak, no latency cliff).
4. Operator edits `internal/tuning/tuning.go` to remove the
   `--gpu-layers 0` arg from `embedParams` / `rerankParams` on
   AMD-discrete profiles. Rebuild, restart.

Reversible: add the flag back to `tuning.go`, rebuild, restart.

### What's covered

- `helper_mv_reduce_and_write_fb` — pure-threadgroup tree-reduce helper
- `kernel_mul_mv_t_t_fb` (f32/f32, f16/f32, f16/f16)
- `kernel_mul_mv_t_t_4_fb` (vector variants)
- Dispatcher gating in `ggml_metal_library_get_pipeline_mul_mv`

### What's deferred

- Quantized mat-vec variants (Q4_0, Q5_0, Q8_0, MXFP4, K-quants,
  IQ-quants). Not exercised by the fp32 embed/rerank models we run;
  would be needed for a quantized BERT-family model.
- `_short` variant (ne00 < 32). Not exercised by BERT shapes.
- Other reduction-heavy kernels (`kernel_argmax`, `group_norm`).
  Not exercised by BERT forward pass for embed.

### New benchmarks

Two operator-runnable safety benches that gate the activation step:

- `scripts/bench-bert-correctness.py` — four-probe numeric
  correctness harness (~30s wall on a healthy daemon). Catches the
  specific class of failure that v0.7.0-rc1's staging-buffer-pool
  patch slipped through (passed HTTP-200, broke recall).
- `scripts/bench-bert-sustained-load.py` — 30-min concurrent load
  test that watches for SIGABRT, HTTP 5xx burst, output drift, RSS
  leak, and the latency-cliff IOSurface-exhaustion pattern that
  caused the 2026-05-14 kernel panic.

---

## v0.7.1 — partial kernel-level BERT correctness: norm + softmax fallbacks (2026-05-17)

First foundation slice toward the v0.8.0 kernel-level fix. Ships
fallback Metal kernels for `kernel_norm_fuse_impl` (LayerNorm) and
`kernel_soft_max{,_4}` (bidirectional-attention softmax) that use
fixed-size function-local threadgroup memory + pure threadgroup
tree-reductions instead of `simd_sum`/`simd_max`+dynamic
threadgroup memory parameters.

The dispatchers in `ggml-metal-device.cpp` now select the `_fb`
suffix variant when `has_simdgroup_reduction == false`. Apple
Silicon is unaffected — the existing kernels stay in use there.

### Measured impact (Vega II, 2026-05-17)

| Test | Pre-patch | Post-patch | CPU route |
|---|---|---|---|
| identical "hello" same batch | cos_sim 0.07 | cos_sim **0.29** | cos_sim 1.0000 |
| two separate "hello" calls | cos_sim 0.15 | cos_sim **0.06** | cos_sim 1.0000 |
| chat (llama3.1-8b seeded) | deterministic | deterministic | n/a (GPU) |

**Not sufficient alone.** The `simd_sum` bug also affects matmul
kernels (`kernel_mul_mv_t_t`, `kernel_mul_mv_q*_f32`, etc.) that
compute attention's QKV projections and the FFN forward pass. The
norm/softmax fallback is correct but the broken matmuls keep the
overall BERT output non-deterministic.

**Production stays on the v0.7.0 CPU route** (`--gpu-layers 0` for
AMD-discrete embed/rerank in `internal/tuning/tuning.go`). The
fallback dispatch path is wired but only takes effect when an
operator overrides the CPU-route flag.

### Files

- `patches/llama.cpp/0003-...` (in-tree edit to ggml-metal.metal +
  ggml-metal-device.cpp): adds `kernel_norm_fuse_fb_impl` (with all
  6 host-name template instantiations for f32/f32_4 × fuse=1,2,3)
  and `kernel_soft_max{,_4}_fb` (4 host names: f16/f32 × scalar/_4)
- `docs/METAL_AMD_BERT_CORRECTNESS.md`: "v0.7.1 partial fix"
  section with the measurement table, plus an expanded "v0.8.0
  full scope" section enumerating the matmul + argmax kernels
  that also need fallbacks (40+ `simd_sum` hits across the file)

### v0.8.0 scope

The grep across `ggml-metal.metal` shows 40+ `simd_*` hits across:
- `kernel_mul_mv_*` (quantized mat-vec): q1/q4/q5/q8/iq* — ~10
  variants per quantization scheme
- `kernel_mul_mv_t_t`, `kernel_mul_mv_t_t_4` — fp16/fp32 mat-vec
  (the dominant path for nomic-embed-text-v1.5)
- `kernel_mul_mv_ext_*` — extended quantized paths
- `kernel_argmax_*` — argmax with bool / arg pair
- `kernel_rms_norm_fuse_impl` — RMSNorm (works for chat, may
  surface a regression under sustained throughput; deferred)

Estimated full BERT Metal correctness: ~5-10 days of focused
shader work + bench-validation.

---

## v0.7.0 — Metal-on-AMD BERT correctness: route embed + rerank to CPU (2026-05-17)

**Closes the fourth Metal-on-AMD failure class** — BERT-family
embedding and reranker models produce **non-deterministic garbage
output** on AMD-discrete Metal even with patch 0001's
`has_simdgroup_reduction` gate active. The bug was masked for the
v0.6.x release cycle because cerid's eval harness used CPU ONNX
embeddings; the cerid v0.96.0 quality uplift's switch to GPU embed
exposed it. Identical input "hello" returns cos_sim 0.07 between
two calls through the Metal path; the same model on CPU returns
cos_sim 1.0000.

Root cause (full design in
[`docs/METAL_AMD_BERT_CORRECTNESS.md`](docs/METAL_AMD_BERT_CORRECTNESS.md)):
the BERT forward pass hits `kernel_norm_fuse_impl` (LayerNorm) and
`kernel_soft_max` (bidirectional attention) which unconditionally
call `simd_sum()` / `simd_max()` AND use dynamic threadgroup memory
as an entry-point parameter — combination hits both the AMD
simd-reduction divergence AND the
[documented Metal-compiler threadgroup-memory barrier-ordering bug](https://github.com/gfx-rs/wgpu/issues/4500).
The Llama chat path takes a different route through the kernel
dispatcher and is unaffected. Patch 0001's flag is checked at the
device-capability level but the affected kernel dispatchers ignore
it.

### Operational fix

`internal/tuning/tuning.go::embedParams` and `::rerankParams` now
append `--gpu-layers 0` to per-slot args on AMD-discrete profiles.
The chat slot stays on GPU (Llama path is correct + fast). Effect:

- Embed identical "hello" → cos_sim **1.0000** ✓
  (Metal path: 0.07; corrupt)
- Rerank identical (query, docs) → identical scores across calls ✓
  (Metal path: nondeterministic)
- LongMemEval observed pace: ~1.6 min per 10 items on CPU embed
  (vs ~1.7 min on broken GPU embed)

### `0002-metal-staging-buffer-pool.patch` parked

The hand-written staging-buffer-pool patch that briefly tagged
v0.7.0 (`0b0e7fa`, now retagged) addressed family-B SIGABRTs at
the kernel level. The 3-min HTTP bench validated 1597/0 calls
without crashes, but the bench did not check **numerical
correctness** — and when the BERT-bug investigation forced a hard
revert, we lost the ability to attribute correctness issues to the
patch vs. the underlying BERT bug. The patch lives at
`patches/llama.cpp/drafts/0002-metal-staging-buffer-pool.patch.broken`
as a starting point for a future re-evaluation; for now the
v0.6.x AutoRespawn safety net is the family-B mitigation.

### v0.8.0 candidate — kernel-level BERT fix

`docs/METAL_AMD_BERT_CORRECTNESS.md` lays out the design:
fallback kernels using fixed-size function-local threadgroup memory
+ pure threadgroup tree-reductions (no simd_sum), gated by the
existing `has_simdgroup_reduction` flag at the dispatcher level.
~400 LOC of MSL + ~50 LOC of dispatcher logic + bench acceptance
criteria documented. Estimated 3-5 days of focused work.

### Tests

`internal/tuning/tuning_test.go` adds
`TestKernelParams_EmbedAMDForcesCPU` and
`TestKernelParams_RerankAMDForcesCPU` — assert `--gpu-layers 0`
present on AMD-discrete profiles, absent on every other profile.
Existing tests unchanged.

---

## v0.7.0-rc1 (retagged) — staging-buffer-pool experiment (DROPPED)

Originally tagged 2026-05-17 on commit `0b0e7fa` based on a 3-min
HTTP-status-only bench. Dropped after the per-component ablation
exposed the unrelated BERT correctness bug that v0.7.0 (above)
addresses. Notes preserved in
[`patches/llama.cpp/drafts/README.md`](patches/llama.cpp/drafts/README.md)
for any future re-evaluation.

### `0002-metal-staging-buffer-pool.patch`

Replaces the per-call `newBufferWithBytesNoCopy` in
`ggml_metal_buffer_set_tensor` and `ggml_metal_buffer_get_tensor`
with a bounded MTLBuffer pool keyed on power-of-two size classes
(4 KiB → 64 MiB, capped at 4 buffers per class). One pool buffer =
one IOMMU registration on AMD discrete; reused across calls, so
the AMD driver's ~256-512 active-mapping pool never exhausts. Apple
Silicon is unaffected — `buf->is_shared` short-circuits to the
`memcpy` fast path before either patched function touches the pool.

Operator escape hatch: `GGML_METAL_DISABLE_STAGING_POOL=1` reverts
to the unpatched `newBufferWithBytesNoCopy` path for A/B testing
during rollout.

### Bench validation

Mac Pro 2019 + Radeon Pro Vega II 32 GB HBM2, sustained-embed
against `nomic-embed-text-v1.5`:

| Run | Calls | Duration | Errors | p50 | p99 | ratio |
|---|---|---|---|---|---|---|
| v0.6.2 unpatched (prior session) | ~80 | ~2 min | 1 SIGABRT | ~110 ms | — | — |
| **v0.7.0 patched** | **1597** | **3 min** | **0** | **109 ms** | **147 ms** | **1.34** |

Zero family-B SIGABRTs across 1597 sustained calls. p99/p50 ratio
1.34 is well below the critical-ratio = 5 degradation threshold.
No throughput regression vs v0.6.2 baseline (the added CPU→pool
memcpy is dwarfed by the avoided IOMMU registration cost).

The 30-minute extended validation is in scope for the next
quenchforge sprint cycle; the 3-minute sustained-load smoke alone
already exceeds v0.6.2's mean-time-to-failure by ~20×.

### Files

- `patches/llama.cpp/0002-metal-staging-buffer-pool.patch` —
  the live patch (164 lines)
- `patches/README.md` — updated "3. Sustained-load …" section to
  mark patch #2 as shipped + reference the bench results
- `patches/llama.cpp/drafts/README.md` — supplemental design doc
  + upstream-issue draft (kept for the ggml-org/llama.cpp filing)

---

## v0.6.2 — bench-validated AMD defaults baked into tuning module (UNRELEASED)

Reliability release. Bakes the bench-validated conservative tuning
into `internal/tuning/tuning.go` so AMD-discrete operators no longer
need to set `QUENCHFORGE_EMBED_UBATCH_SIZE=1024` +
`QUENCHFORGE_EMBED_METAL_N_CB=1` in the launchd plist by hand. The
env knobs still exist and still override the baked defaults; this is
purely about the out-of-box behaviour on AMD-discrete hardware.

- **Vega II tested-stable embed defaults** (apply to all AMD-discrete
  profiles): `ubatch=1024`, `MetalNCB=1`. Empirical evidence from
  the cerid LongMemEval canonical run 2026-05-17: ubatch=1024 sustained
  ~0.5 req/s for >70 min with a single auto-respawned family-B crash;
  ubatch=8192 (v0.5.x default) crashed within ~80 calls / ~2 min.
- **AMD-discrete rerank defaults**: `MetalNCB=1` (mirrors embed).
  RerankBatchSize stays operator-opt-in because the right value is
  workload-specific.
- **Other AMD profiles** (W6800X, RDNA1, RDNA2) inherit Vega II's
  values until benched independently — fail-conservative matches the
  CLAUDE.md "fail to slower-but-stable" policy.
- **Apple Silicon / CPU / iGPU / unknown** profiles unchanged
  (family-B is structurally impossible on shared-memory hosts;
  ubatch=MaxContext=8192 stays the default).
- **Operator env overrides win** as before — set
  `QUENCHFORGE_EMBED_UBATCH_SIZE` to override.

Tests: `internal/tuning/tuning_test.go::TestKernelParams_EmbedDefaultsByProfile`
asserts the per-profile defaults. `cmd/quenchforge/serve_test.go`
gets two new cases (AMD vs Apple) and one renamed env test
(`TestSlotEnv_AMDEmbedGetsBakedDefault`).

No new env knobs, no plist changes required to upgrade. Operators
who'd manually set the conservative env in their plist can leave it
in place (no-op now) or remove it (defaults take over).

---

## v0.6.1 — chat slot AutoRespawn on AMD discrete (UNRELEASED)

Caught during cerid v0.96.0 GPU validation: PR-3 wired AutoRespawn
for embed/rerank on AMD but missed chat. cerid's LongMemEval
extraction workload hammered the chat slot through ~30 successful
completions; task 143 hit `GGML_ASSERT(buf_src)` at `set_tensor` and
SIGABRT'd. The process stayed dead because `RestartPolicy` was
`PolicyNone`; subsequent /api/chat returned 502 until manual
`launchctl kickstart`.

The fix is one line in `chatParams`: `AutoRespawn: true` on AMD
profiles. Same supervisor-side restart-on-SIGABRT logic the
embed/rerank slots already use; 2s/4s/8s exp backoff, cap 3 per
60s. Tuning-test `TestKernelParams_ChatAMDGetsCorrectnessFlags`
updated to assert `AutoRespawn=true` on AMD chat.

---

## v0.6.0 — sustained-load Metal hardening for embed/rerank slots (UNRELEASED)

Reliability release. Closes the third Metal-on-AMD failure class
(graph-compute buffer-corruption under sustained embed/rerank load —
the family-B SIGABRT documented in `patches/README.md` section 3). No
behaviour change at defaults; operators on AMD-discrete hardware
running batch eval / bulk-ingest workloads can opt into the new
tuning knobs and observability surface. Follow-up PR will bench
per-profile defaults on the `[amd-gpu]` runner and flip them in
`internal/tuning/`.

- **New package `internal/tuning/`** is the sole owner of `(profile,
  slot-kind) → SlotTuning` mapping. Replaces inline `if spec.Kind ==
  ... && IsAMDDiscrete()` blocks in `buildSlotArgs`. Pure function,
  table-driven tests cover every (profile, kind) pair.
- **Four new env knobs** (defaults preserve current behaviour):
  `QUENCHFORGE_EMBED_UBATCH_SIZE`, `QUENCHFORGE_EMBED_METAL_N_CB`,
  `QUENCHFORGE_RERANK_BATCH_SIZE`, `QUENCHFORGE_RERANK_METAL_N_CB`.
  Set the embed knobs to smaller values (e.g. ubatch=1024, N_CB=1)
  on AMD discrete to bound the per-call Metal staging allocations
  and let the staging-buffer pool drain between calls.
- **Auto-respawn on SIGABRT.** AMD-discrete embed and rerank slots
  set `supervisor.RestartPolicy = PolicyExpBackoff`. The supervisor
  waits 2s / 4s / 8s after a non-zero exit and retries `Start`,
  capped at 3 attempts per 60-second window. Default `PolicyNone`
  preserves prior behaviour for other slot kinds.
- **Gateway latency tracker + `/health`.** The gateway records every
  upstream call's duration + error-flag in a rolling 60-second
  per-kind window. `/health` returns a JSON snapshot with `p50_ms`,
  `p99_ms`, `error_rate`, and a `status` field (`ok` | `degraded` |
  `critical`). Consumers can poll this to back off before the
  family-B SIGABRT.
- **Opt-in auto-backoff.** Setting `QUENCHFORGE_AUTO_BACKOFF=true`
  turns a `critical` snapshot into an automatic `HTTP 503` +
  `Retry-After: 2` on the upstream proxy paths. Default off —
  observability without behaviour change.
- **New binary `cmd/quenchforge-bench`.** Subcommand
  `sustained-embed --gateway <url> --model <name> --duration 10m`
  POST-loops `/v1/embeddings` and reports per-30s progress + final
  p50/p99 + crash-time. Exits 1 on slot crash, 2 on degraded but
  not crashed, 0 on clean completion. Used as the harness for the
  follow-up Vega II default-tuning PR.
- **Per-slot `GGML_METAL_N_CB` env injection.** `startSlot` previously
  passed a single global `MetalNCB` to every slot's env. It now
  consults `tuning.KernelParams` per slot so an operator can set
  `EmbedMetalNCB=1` without affecting the chat slot.
- **Tests:** `internal/tuning/tuning_test.go`,
  `internal/gateway/latency_test.go`,
  `cmd/quenchforge-bench/sustained_embed_test.go`, plus three new
  cases in `cmd/quenchforge/serve_test.go`. `go test ./...` green.
- **Documentation:** new section 3 in `patches/README.md` documents
  the staging-buffer-pool mechanism + supervisor mitigations.
  Operators on cerid workloads see the gotcha update in `CLAUDE.md`.

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
