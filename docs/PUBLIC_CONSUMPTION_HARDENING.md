# Quenchforge — public-consumption hardening plan

> **Audit date:** 2026-05-13
> **Audited version:** v0.3.3 (latest tag), v0.3.4-target
> **Audit scope:** docs accuracy, build provenance, error UX, test coverage,
> first-run experience, feature surface for broader applicability

---

## Executive summary

Quenchforge is a solid niche project that has earned the right to bid for
broader adoption. The patches work, the gateway is well-architected, the
docs are honest about scope. **But the project advertises a "v0.3.1
pre-release" while the latest tag is v0.3.3, the local build doesn't
stamp version metadata, and the model on-ramp still requires manual GGUF
placement.** Those three gaps are the highest-leverage polish items
before pushing for broader pickup.

The current positioning ("AMD-Mac specifically") is correct as the unique
value prop, but the project is **already usable as a generic Ollama
replacement on macOS** — patches runtime-gate to non-Apple-Silicon only,
so Apple Silicon users get Ollama-equivalent behavior. A modest re-framing
of the README opens the audience without giving up the focused message.

This doc lays out:
1. **Doc/state drift** to fix
2. **Build provenance** to fix
3. **First-run UX** to improve (the model on-ramp)
4. **Test coverage** gaps worth closing
5. **Feature additions** worth considering for broader applicability
6. **What NOT to do** (lessons from the 2026-05-13 crash session)

---

## 1. Documentation drift to fix

### High priority

- **README claims v0.3.1 pre-release** (line 24) — actual latest is v0.3.3.
  Bump the status block, list what v0.3.2 and v0.3.3 actually shipped.
- **README's "Configuration" table doesn't list** `QUENCHFORGE_NLI_MODEL`,
  `QUENCHFORGE_SD_MODEL`, `QUENCHFORGE_BARK_MODEL` (image-gen + TTS env
  vars are real config knobs, currently undocumented in README).
- **README's "What's in the box" table** doesn't mention the v0.3.3
  Ollama-wire body translation (the gateway now translates `/api/chat`,
  `/api/generate`, `/api/embeddings`, `/api/embed` into OpenAI-wire calls
  for llama-server). That's a real user-facing improvement that the
  README doesn't surface.
- **`patches/README.md`** v0.3.3 supervisor-level companion fixes
  (`--flash-attn off`, `--cache-ram 0`, `--no-cache-prompt`) section is
  already there ✓. Good.

### Medium priority

- **README "Status" line** says image-gen and TTS slots are "shipped".
  That's literally true (the slot infrastructure exists), but those
  workloads have NOT been verified correct on AMD Mac — the README
  should say "wired, AMD-Mac correctness unverified — likely needs the
  same patch surface that whisper has".
- **Hardware compatibility matrix** doesn't list "Intel Mac Pro 2013
  (D300/D500/D700)" — those were specifically flagged as gibberish-output
  in [llama.cpp#20104](https://github.com/ggml-org/llama.cpp/issues/20104).
  Add a "Known incompatible" row or note.
- **Telemetry / Sentry consent flow** is referenced in README ("v0.4 ships
  this") and CLAUDE.md, but no code is committed and the v0.4 target
  has no plan. Either commit a stub + plan or remove the v0.4 promise.

### Low priority

- README "Why ggml, not just LLMs" framing is good but could be tightened.
- `docs/APPLE_DEVELOPER_ID.md` and `docs/SELF_HOSTED_RUNNER.md` are
  maintainer-facing — fine to keep but should be linked from a "for
  maintainers" section at the bottom of CONTRIBUTING.md.

---

## 2. Build provenance

### The bug

`quenchforge doctor` reports:
```
  version:    0.3.1-dev (unknown)
  build date: unknown
```

Even after we tagged v0.3.3. The reason: local `go build` doesn't pass
the ldflags that goreleaser passes. The `Version`, `Commit`, `BuildDate`
strings in `cmd/quenchforge/main.go` only get stamped on official
release builds.

### Fix

Add a `Makefile` at repo root that wraps the canonical build:

```makefile
.PHONY: build
build:
	bash scripts/apply-patches.sh
	bash scripts/build-llama.sh
	go build \
	  -ldflags "\
	    -X main.Version=$$(git describe --tags --always --dirty) \
	    -X main.Commit=$$(git rev-parse --short HEAD) \
	    -X main.BuildDate=$$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
	  -o /usr/local/bin/quenchforge ./cmd/quenchforge

.PHONY: test
test:
	go test ./...

.PHONY: clean
clean:
	rm -rf llama.cpp/build-* bin/
```

Mirror in CONTRIBUTING.md as `make build` (it's already referenced there).

Also: CLAUDE.md says `make build` is the canonical command, but there's
no Makefile. Currently CLAUDE.md is lying. Adding the Makefile fixes
that contract.

---

## 3. First-run UX — the model on-ramp

### Current state

After install, the user has to:
1. Find a GGUF model on HuggingFace
2. Download it manually (no help from quenchforge)
3. Symlink or place it in `~/.quenchforge/models/`
4. Pick a filename for it
5. Set `QUENCHFORGE_DEFAULT_MODEL` to the filename (minus `.gguf`)
6. Restart `quenchforge serve`

Ollama users get the much smoother:
1. `ollama pull llama3.2:3b`
2. `ollama run llama3.2:3b`

That UX gap is the biggest barrier to adoption for users who aren't
already running Ollama.

### Proposed: `quenchforge pull <hf-repo>:<file>` subcommand

```sh
# Pull a specific quant from a HuggingFace GGUF repo
quenchforge pull bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M

# Or by friendly alias from a curated catalog
quenchforge pull llama3.2:3b

# List installed
quenchforge list

# Remove
quenchforge rm llama3.2:3b
```

Implementation: HuggingFace's HF Hub API is well-documented and the
filename → SHA256 mapping is exposed for atomic, resumable downloads
with Etag integrity checks. The internal/registry package already has a
"GGUF pull (HF Etag integrity)" comment in CLAUDE.md — looks like the
intent was there but the code isn't shipped.

This is a HIGH-VALUE add for broader applicability. Estimated ~200 LOC
across `cmd/quenchforge/main.go` + `internal/registry/`.

### Curated catalog (optional)

A small `catalog.json` of well-tested (HF-repo, file, alias) tuples for
the AMD-Mac VRAM tiers documented in cerid-ai's
[`AMD_GPU_MODEL_RECOMMENDATIONS.md`](https://github.com/Cerid-AI/cerid-ai/blob/main/docs/AMD_GPU_MODEL_RECOMMENDATIONS.md).
Lets users say `quenchforge pull llama3.2:3b` without knowing the HF
repo.

The catalog itself can ship in-binary (small JSON file embedded). No
network call needed to resolve.

---

## 4. Test coverage gaps

| Package | Tests today | Gap |
|---|---|---|
| `cmd/quenchforge` | 5 (serve_test.go) | OK |
| `cmd/quenchforge-preflight` | 1 | Could use more |
| `internal/gateway` | 22 | Strong |
| `internal/gateway/ollama_translate` | 9 | Strong (v0.3.3) |
| `internal/hardware` | 7 | Adequate (Detect mostly platform-specific) |
| `internal/supervisor` | unknown — file exists | **Audit** |
| `internal/scheduler` | unknown — file exists | **Audit** |
| `internal/config` | unknown — file exists | **Audit** |
| `internal/discovery` | unknown — file exists | **Audit** |

`go test ./...` already runs in CI. If the *_test.go files are populated
adequately, this gap closes itself. If they're stubs, add tests for
supervisor lifecycle (Start, Stop, KeepAlive crash recovery,
orphan reaper), scheduler priority order, config load priority, and
mDNS Start/Stop.

---

## 5. Feature additions for broader applicability

Ranked by impact × tractability.

### HIGHEST: HuggingFace model pull (covered above in §3)

### HIGH: VRAM-aware slot startup

Today: if the operator configures chat + embed + rerank slots whose
combined model weights exceed available VRAM, all 3 slots fail at
load time with cryptic Metal errors.

Fix: pre-check via `MTLDevice.recommendedMaxWorkingSetSize` before
starting any slot. Sum the GGUF file sizes (close enough to actual
VRAM usage for FP16 / Q4_K_M). Refuse to start with a helpful error if
the math doesn't fit:

```
quenchforge: VRAM check: 38.4 GB needed (chat 14.2 + embed 0.1 + rerank 0.5 + KV 8.0 + headroom 4.0 + overhead 8.0)
quenchforge: VRAM available: 32 GB
quenchforge: error: configured slots won't fit in VRAM. Either reduce a model size or unset one slot:
  $ unset QUENCHFORGE_RERANK_MODEL
or downsize:
  $ export QUENCHFORGE_DEFAULT_MODEL=qwen2.5:7b-instruct-q4_k_m  # 7b instead of 14b
```

### MEDIUM: Web dashboard at `/dashboard`

Current state observability is `quenchforge doctor` (one-shot) and the
log files. A small HTML+SSE dashboard on the gateway port showing:
- Live slot status (model, port, PID, uptime, request count)
- Recent requests + per-route latency p50/p95
- VRAM usage
- Patch series applied

~150 LOC if we avoid frontend frameworks (vanilla JS + EventSource).
Real value for operators tuning their setup.

### MEDIUM: Streaming Ollama-wire `/api/embed` for cerid-style consumers

Currently embeddings are non-streaming on both wires (correct — they
don't naturally stream). But cerid's batch ingest path could benefit
from a chunked-Ollama-wire response that lets the client process
embeddings as they finish. Out of scope for v0.3.4; flag for v0.4.

### LOW: macOS Status Bar app + Sparkle 2.x auto-updater

CLAUDE.md mentions this. Real value for users who don't live in
terminals. Substantial work (~500 LOC + Sparkle integration). Defer
unless adoption justifies it.

### LOW: Drop-in mode for `OPENAI_BASE_URL=http://localhost:11434/v1`

Anyone using LangChain / LlamaIndex / OpenAI SDK with `base_url` pointed
at a local server already works with Quenchforge — `/v1/chat/completions`
and `/v1/embeddings` are real OpenAI-wire endpoints. README could add
explicit "drop-in for OpenAI clients" section with code snippets but no
code change needed.

### REJECT: `quenchforge bench` subcommand

CLAUDE.md mentions a `cmd/quenchforge-bench/` directory but it's not in
the actual repo layout. Either it was removed or never built. Don't add
it — the bench dashboard at `bench.quenchforge.dev` (planned per
CLAUDE.md) is the right place for shared benchmark data, and "run a
benchmark locally" is `llama-bench` (which the user can run directly).

---

## 6. Re-framing for broader audience

The current README's status block (line 24) reads:

> v0.3.1 pre-release. Chat + embeddings + reranker + whisper transcription
> verified live on Mac Pro 2019 + Radeon Pro Vega II (32 GB HBM2).

Honest but narrow. The "broader applicability" framing would be:

> **Quenchforge is Ollama for Mac users who care about correctness.**
>
> It runs unchanged on Apple Silicon (where Ollama also works fine) and
> on Intel Mac + AMD discrete (where Ollama silently falls back to CPU
> and llama.cpp produces garbage tokens). The same binary, the same
> Ollama-API + OpenAI-API surface, the same model formats.
>
> If you're on Apple Silicon, use it for the Ollama-API + OpenAI-API
> dual-wire gateway and the multi-slot supervisor (chat + embed + rerank
> simultaneously without managing three llama-server processes
> yourself). If you're on Intel Mac + AMD discrete, this is the only
> way to get correct output without falling back to cloud.

That widens the audience without giving up the focused niche.

---

## 7. What NOT to do (2026-05-13 crash lessons)

Three hard crashes in one session from progressively more conservative
Metal experiments on Mac Pro 2019 + Vega II. The driver fault state
persists across attempts in ways that escape process isolation.

**Do not pursue from this codebase / on this hardware:**
- Force-enable `simdgroup_mm` on AMD Metal3
- Force-enable `simdgroup_reduction` via butterfly workarounds
- `StorageModeShared` → `StorageModeManaged` conversion (the PR #21177
  approach)
- Whisper Metal re-enable on AMD without a dedicated test host
- Any experiment that requires repeated `llama-server` restarts during
  development

**Do pursue:**
- Doc / build / UX hardening (this plan)
- HuggingFace model pull
- VRAM-aware startup
- Test coverage in existing patterns
- Web dashboard for observability

These are all SAFE in the sense that they don't touch GPU driver state
or kernel paths.

---

## 8. Recommended execution order

1. **README + doc currency** (1-2 hour edit, this session)
2. **Add Makefile + version stamping** (30 min, this session)
3. **Surface v0.3.3 ollama-wire translation** in README features
4. **Add `quenchforge models list / pull / rm` subcommands** with HF Hub
   integration (~half day, future session)
5. **VRAM-aware slot startup** with helpful error messages (~2 hours,
   future session)
6. **Web dashboard at `/dashboard`** (~half day, future session)

Phases 1-3 land in v0.3.4. Phases 4-6 land in v0.4.0.

---

## 9. Out of scope for this hardening pass

- Telemetry consent screen + benchmark dashboard at
  `bench.quenchforge.dev` — wait for v0.4 plan
- Sparkle 2.x updater + macOS status bar app — defer
- Multi-host / mDNS-discovered cluster of Quenchforge daemons — defer
- Custom GGUF quantization workflow — defer (use upstream
  `llama.cpp/quantize` directly)
