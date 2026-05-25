# 2026-05-25 — System stability fix + quenchforge ⇄ Ollama deconfliction

**Status:** design approved 2026-05-25, implementation pending plan
**Driver:** May 24 ~19:45 EDT system freeze on the dev Mac Pro (MacPro7,1, AMD Radeon Pro Vega II), requiring two hard reboots. Same failure class as the 2026-05-14 panic and the 2026-05-17 `vm_page_wire wire_count overflow @vm_resident.c:6502` panic from `llama-server`.

---

## Problem

A deep audit of macOS system logs, quenchforge logs, and the cerid-ai-internal codebase identified a multi-layer cascade that has now produced three system-level failures in ten days. The cascade has both a code-fixable root cause inside quenchforge and an environmental aggravator (a pre-installed Ollama.app) that any public quenchforge user could hit identically.

### Observed signals (May 17 → May 24 uptime window)

- `~/Library/Logs/quenchforge/quenchforge.err.log`: **257** `slot chat exited (signal: abort trap); respawn in 2s (1/3 in window)` events, **40** `port 127.0.0.1:11434 is already in use` events, **117** `spawned /usr/local/bin/llama-server` events.
- `~/Library/Logs/quenchforge/embed.log`: **3.73 GB** of unbounded growth in 7 days.
- Disk Data volume at **97% full** (32 GB free of 808 GB); vnodes available dropped to 8.5%.
- macOS unified log shows `(AppleSMC) Previous shutdown cause: 1` (user-initiated power reset) as the only shutdown event in 7 days.
- No panic file written for the May 24 freeze — likely because the disk was too full for APFS to allocate the dump.
- `/Applications/Ollama.app` is installed; its `com.ollama.ollama` LaunchAgent runs at every login and repeatedly tries to bind `127.0.0.1:11434`, generating kernel `tcp (in_pcbbind) ... EADDRINUSE caused by process: quenchforge:1196` events continuously.

### Root cause chain

1. **External (environmental).** Ollama.app's LaunchAgent races quenchforge for port 11434 at login. Sometimes Ollama wins the bind. Quenchforge's gateway then fails to bind, the supervisor exits, and macOS launchd respawns it every `ThrottleInterval=10` seconds because `KeepAlive=true`.
2. **Quenchforge (gateway behavior).** The supervisor does not detect a pre-existing listener before attempting to bind. On bind failure it exits with a non-zero status, which the current `KeepAlive=true` interprets as a crash to restart. Result: a sustained crash-spam loop, with each cycle spawning slot processes that load multi-GB GGUFs into Vega II VRAM before being orphaned.
3. **Quenchforge (chat slot Metal).** The chat slot runs `llama3.1-8b` (Q4_K_M quantized) on AMD Vega II. Quantized mat-vec kernels are not covered by patches 0001/0003/0004 — only fp32/fp16 BERT shapes are. The slot SIGABRT-traps periodically under load. Each abort triggers the `PolicyExpBackoff` 2s/4s/8s respawn cycle inside the supervisor, which reloads the 8 GB model onto Vega II again, compounding GPU memory wear.
4. **Quenchforge (operational hygiene).** No log rotation; per-slot logs grow unbounded. No first-run conflict detection.
5. **macOS operational hygiene.** Disk at 97% full (Docker.raw can claim up to 808 GB; embed.log holds 3.73 GB). No scheduled reboot policy. Long uptimes (7+ days) accumulate the cumulative GPU/FS pressure that turns layers 1–4 from latent to fatal.

### Why this is a product issue, not just an operator issue

Any public user installing quenchforge on macOS with a pre-existing Ollama.app install will hit layers 1–2 immediately. They have no way to diagnose it without reading logs that don't yet exist (no `quenchforge doctor`) or understanding launchd internals. Adding deconfliction to the quenchforge codebase removes this footgun for every future user.

---

## Goals

1. End the chat-slot SIGABRT + port-11434 fight that's freezing this Mac every 3–7 days.
2. Make quenchforge resilient to coexistence with a pre-installed Ollama (public-user UX).
3. Re-enable AMD Vega II for embed/rerank via the v0.8.0-rc1 patches (gated on benches).
4. Zero regression to cerid AI features — the listen port stays `127.0.0.1:11434`, all four slots (chat/embed/code-embed/rerank) remain, all model names unchanged.

## Non-goals

- **Patch 0005 (quantized matmul fallback)** that would make chat-slot Vega II safe. Tracked as separate follow-up work; this spec ships chat-slot CPU routing as the interim.
- **Changing the gateway listen port.** 60+ cerid-ai-internal references hardcode `11434`; we honor that contract.
- **Uninstalling Ollama.app.** We disable its LaunchAgent only — reversible with one `launchctl bootstrap`.
- **Adding Ollama as an upstream / proxy fallback inside quenchforge.** Quenchforge already implements the cerid-required API surface natively.

---

## The six layers

### Layer 1 — Quenchforge deconfliction code

All paths relative to `~/Develop/quenchforge`.

#### 1a. Gateway pre-bind check
**Files:** `cmd/quenchforge/main.go`, new helper in `internal/gateway/preflight.go`
- Before `net.Listen("tcp", listenAddr)`, attempt a TCP probe to the configured address.
- If something answers: identify the holder via `lsof -i :PORT -sTCP:LISTEN -F pcn` (graceful degradation if `lsof` unavailable — fall back to `netstat -anv` parse).
- Decision tree based on the identified holder:
  - **`Ollama` / `ollama` / `ollama serve`**: log a clear actionable error (see Error Messages section), `os.Exit(0)`.
  - **stale `quenchforge` (PID matches no live process, or socket is in `TIME_WAIT`)**: attempt graceful takeover — send SIGTERM, wait up to 5 s, then bind. If still held, escalate to SIGKILL with 2 s wait, then bind or exit.
  - **anything else**: log the holder PID + command + path, `os.Exit(0)`.
- Exit code `0` is intentional. Pairs with the LaunchAgent plist change in Layer 2 so launchd does not crash-spam respawn.

**Error Messages (canonical text, used by `quenchforge doctor` too):**
```
quenchforge: port 127.0.0.1:11434 is held by Ollama.app (pid <N>).
  Two ways forward:
    1. Disable Ollama's login agent and use quenchforge:
         launchctl bootout gui/$(id -u)/com.ollama.ollama
    2. Coexist on different ports — set QUENCHFORGE_LISTEN_ADDR=:11435
       in your LaunchAgent plist's <EnvironmentVariables> dict, then:
         launchctl kickstart -k gui/$(id -u)/com.cerid.quenchforge
       (clients pointing at :11434 will hit Ollama; clients pointing at
       :11435 will hit quenchforge.)
  See: quenchforge doctor --explain
```

#### 1b. `quenchforge doctor` subcommand
**Files:** new `cmd/quenchforge/doctor.go`, wire into existing CLI dispatch.
- Print a structured diagnostic report covering:
  - Port 11434 status: listener PID, command, executable path.
  - Conflicting LaunchAgents loaded: `com.ollama.ollama`, anything Homebrew-services-managed binding 11434.
  - Disk free on `/System/Volumes/Data` (warn < 20 GB, critical < 10 GB).
  - Vnode availability (warn < 15%, critical < 10%).
  - Per-slot log file sizes (warn > 50 MB, critical > 500 MB).
  - Last N supervisor restarts from `quenchforge.err.log` (default N=5).
  - GPU presence (`system_profiler SPDisplaysDataType -json` parsing).
  - Active slot model names and CPU/GPU placement.
- Default output: one section per check with PASS/WARN/FAIL status.
- `quenchforge doctor --explain`: for each non-PASS finding, print remediation steps (the same actionable text from 1a's error templates).
- `quenchforge doctor --json`: machine-readable for monitoring / cron integration.

#### 1c. Slot log rotation
**Files:** `internal/supervisor/supervisor.go` (the `startSlot` path; per the explore agent, lines ~656–690 in `cmd/quenchforge/main.go` build the slot, and the log file is opened around lines 660–665).
- Replace direct `os.OpenFile` of slot log path with a size-limited rotating writer.
- Config: 100 MB max per current file, 5 rotation generations (`chat.log` → `chat.log.1` → … → `chat.log.5`, oldest deleted on rotation).
- Implementation: `gopkg.in/natefinch/lumberjack.v2` if dep policy allows (zero transitive deps, ~300 LOC well-tested). Otherwise inline a ~150-LOC equivalent in `internal/supervisor/logrotate.go`.
- Apply uniformly to the four currently-configured slot logs (chat, embed, code-embed, rerank) and to `quenchforge.out.log` / `quenchforge.err.log`. The rotation hook lives in the slot construction path, so future slots (whisper/image-gen/tts) inherit it automatically without further work.
- Make rotation thresholds env-overridable: `QUENCHFORGE_LOG_MAX_MB` (default 100), `QUENCHFORGE_LOG_BACKUPS` (default 5).

#### 1d. Chat-slot CPU routing on AMD-discrete profiles
**File:** `internal/tuning/tuning.go` — `chatParams` function (per explore agent: lines ~73–78).
- For `Profile == AMDDiscrete` only, append `--gpu-layers 0` to the chat slot args, matching the existing v0.7.0 policy already applied to `embedParams` / `rerankParams` for the same profile.
- Apple-Silicon and Intel-iGPU profiles unchanged — the quantized-Metal SIGABRT is AMD-specific.
- Add a comment block referencing CHANGELOG v0.7.0 (precedent) and pending future patch 0005 (the reversal trigger):
  ```go
  // AMDDiscrete chat slot routes to CPU pending patch 0005
  // (quantized matmul fallback kernels). Patches 0001/0003/0004
  // cover fp32/fp16 BERT only; chat-slot Q4_K_M models still
  // SIGABRT on Vega II Metal under sustained load. Mirror of the
  // v0.7.0 embed/rerank policy. Remove this line when patch 0005
  // lands and bench-llama-sustained-load.py passes.
  ```
- Reversal is a one-line revert in this function.

#### 1e. README + patches/README.md updates
- `README.md`: add a new "Coexistence with Ollama" section under Installation. Explains the port conflict, recommends `quenchforge doctor` as the first troubleshooting step, points at the two coexistence options (disable Ollama's LaunchAgent vs. run quenchforge on `:11435`).
- `patches/README.md`: extend Section 3 to note that the chat slot inherits the same Metal-stability concern as embed/rerank and currently mitigates via CPU routing; reference patch 0005 as planned future work.

### Layer 2 — LaunchAgent template

**File:** `plist_template.plist` (per explore agent: lines 77, 82 hold `KeepAlive` and `ThrottleInterval`).

Change:
```xml
<key>KeepAlive</key>
<true/>
```
to:
```xml
<key>KeepAlive</key>
<dict>
  <key>SuccessfulExit</key>
  <false/>
</dict>
```

Effect: launchd restarts quenchforge on crashes (signals, abort) but NOT on `os.Exit(0)`. The new pre-bind check in Layer 1a uses `os.Exit(0)` precisely so launchd will not crash-spam when Ollama holds the port. `ThrottleInterval=10` stays unchanged — still rate-limits any genuine crash loop.

The user-installed `~/Library/LaunchAgents/com.cerid.quenchforge.plist` needs the same edit. Document this in CHANGELOG + README; consider a `quenchforge install` subcommand in a follow-up that re-installs the agent from the current template.

### Layer 3 — Operational fixes (this machine, OS-level)

Out-of-tree commands run by the operator:

1. **Quit Ollama:**
   ```sh
   osascript -e 'tell application "Ollama" to quit'
   ```
2. **Disable Ollama LaunchAgent (reversible):**
   ```sh
   launchctl bootout gui/$(id -u)/com.ollama.ollama
   ```
3. **Verify quenchforge owns the port:**
   ```sh
   lsof -i :11434 -sTCP:LISTEN
   # Expected: only quenchfor (truncated) listening
   ```
4. **Reclaim disk space (target: ≥ 50 GB free on Data volume):**
   ```sh
   docker system prune -a --volumes
   : > ~/Library/Logs/quenchforge/embed.log
   df -h /System/Volumes/Data
   ```
   (Layer 1c handles future growth.)
5. **Add weekly scheduled restart (belt and suspenders until long-uptime stability is proven):**
   ```sh
   sudo pmset repeat restartall MTWRFSU 04:00
   pmset -g sched
   ```
6. **Reload quenchforge with the new plist + binary (after Layers 1–2 ship):**
   ```sh
   launchctl kickstart -k gui/$(id -u)/com.cerid.quenchforge
   ```

### Layer 4 — cerid-ai-internal documentation alignment

Small text changes in cerid-ai-internal to reflect that quenchforge is the preferred local backend on AMD Vega II:

1. **`scripts/detect-gpu.sh` lines 80–95**: change the AMD-Vega-II case from `recommended_backend="ollama"` to `recommended_backend="quenchforge"`. Update the comment referencing `ollama/ollama#1016`.
2. **`scripts/start-cerid.sh:445`** error string: replace `"Install via: brew install ollama && ollama serve"` with `"Install via: quenchforge install (or: brew install ollama && ollama serve)"`.
3. **`scripts/validate-env.sh:183`** warn string: replace `"start with: ollama serve (native)"` with `"start with: launchctl kickstart -k gui/$UID/com.cerid.quenchforge (or: ollama serve)"`.

No production code paths change. This is documentation/UX hygiene aligning recommendations with current production reality (`.env` already sets `EMBEDDINGS_PROVIDER=quenchforge` and `RERANK_PROVIDER=quenchforge`).

### Layer 5 — v0.8.0-rc1 GPU activation (gated on benches)

Per quenchforge `CHANGELOG.md` v0.8.0-rc1 activation procedure. Executed after Layers 1–4 are stable in production.

1. Confirm build has patches 0001–0004 applied (`scripts/apply-patches.sh` is automatic; `quenchforge --version` should report ≥ v0.8.0-rc1 once Layer 1 ships).
2. Run correctness bench:
   ```sh
   python scripts/bench-bert-correctness.py
   ```
   Must PASS (cos_sim same-batch, cos_sim separate-call, semantic sanity, L2 norm bounds).
3. Run sustained-load bench:
   ```sh
   python scripts/bench-bert-sustained-load.py --duration 1800
   ```
   Must PASS (no SIGABRT, no HTTP 5xx burst, no catastrophic drift, no RSS leak, no latency cliff). This bench explicitly tests for the IOSurface-exhaustion pattern that caused the 2026-05-14 panic, so passing it is the regression gate for the v0.7.0 → v0.8.0 transition.
4. **If both pass:** edit `internal/tuning/tuning.go` to remove `--gpu-layers 0` from `embedParams` / `rerankParams` for `AMDDiscrete`. Rebuild. `launchctl kickstart -k gui/$(id -u)/com.cerid.quenchforge`. Verify with `ps aux | grep llama-server`.
5. **If either fails:** leave v0.7.0 CPU route in place. File v0.8.0-rc2 issue against this repo. Layers 1–4 are independently complete and successful.

Chat slot stays on CPU regardless of bench outcome — patch 0004 covers fp32/fp16 BERT only, not quantized chat models. Patch 0005 (quantized matmul fallback) is the chat-slot reversal trigger.

### Layer 6 — Memory + monitoring updates

1. **Update `~/.claude/projects/-Users-sunrunner-Develop/memory/feedback_quenchforge_safety.md`** to reflect post-fix state:
   - Ollama LaunchAgent is now disabled (not removed); reactivation instructions.
   - Quenchforge has built-in deconfliction (`quenchforge doctor`, pre-bind check).
   - Chat slot is on CPU pending patch 0005; reversal one-liner.
2. **New memory `project_cerid_quenchforge_chat_on_cpu.md`**: short note documenting the chat-slot CPU placement, the patch-0005 reversal trigger, the env-var override (`QUENCHFORGE_CHAT_GPU_LAYERS` if we add one), and the bench command to validate.
3. **Optional cron / Hammerspoon hook**: nightly `quenchforge doctor --json` with alerting on any non-PASS. Defer to follow-up; not in scope for this spec.

---

## Order of operations

1. **Layer 3 (OS cleanup)** — immediate, no code needed. Stops the bleeding today.
2. **Layer 1c (log rotation)** — short, eliminates one whole failure mode independently.
3. **Layer 1a + 1b + Layer 2 (gateway pre-check + doctor + plist hardening)** — ship as a single PR; they only make sense together.
4. **Layer 1d (chat slot CPU)** — trivial diff; can be in the same PR as 1a/1b/2.
5. **Layer 1e + Layer 4 (docs)** — small PRs, low priority but should ship before public release.
6. **Layer 5 (v0.8.0 activation, gated)** — last; runs benches and conditionally activates. Independent of all prior layers' success.

## Verification per layer

| Layer | Verification |
|---|---|
| 1a | Manually launch Ollama via `osascript -e 'tell app "Ollama" to launch'`; quenchforge should print the canonical error and exit cleanly; `pmset -g log` should show launchd not respawning quenchforge within `ThrottleInterval`. |
| 1b | `quenchforge doctor` prints a complete report with all sections. `quenchforge doctor --explain` prints remediation for every WARN/FAIL. `quenchforge doctor --json` parses as valid JSON. |
| 1c | Test: write > 100 MB to a slot log path; verify rotation creates `.1` file and primary log resets near zero. |
| 1d | `ps aux \| grep "llama-server.*chat"` shows `--gpu-layers 0`. `curl 127.0.0.1:11500/health` (or the gateway endpoint for chat) returns OK. A `/api/chat` request returns a normal completion. |
| 2 | After Layer 1a triggers (Ollama holding port), `pmset -g log` shows quenchforge did NOT respawn during the `ThrottleInterval` window. |
| 3 | `lsof -i :11434 -sTCP:LISTEN` shows only quenchforge. `df -h /System/Volumes/Data` shows ≥ 10% free. `pmset -g sched` shows the weekly restart entry. |
| 4 | `bash ~/Develop/cerid-ai-internal/scripts/detect-gpu.sh` on this machine emits `CERID_RECOMMENDED_LOCAL_BACKEND=quenchforge`. |
| 5 | Both bench scripts output PASS. Post-activation: `ps aux \| grep llama-server` shows embed/rerank processes without `--gpu-layers 0`. The sustained-load bench can be re-run any time as a regression gate. |

## Rollback

Every layer has a one-step revert:

- **Layer 1a/1b:** revert the new file + the single hook call in `main.go`.
- **Layer 1c:** swap the rotator back to `os.OpenFile`.
- **Layer 1d:** delete the `--gpu-layers 0` append line in `chatParams`. Rebuild. Kickstart.
- **Layer 2:** revert `plist_template.plist` to `<key>KeepAlive</key><true/>`.
- **Layer 3:**
  - Re-enable Ollama: `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ollama.ollama.plist` (or `open -a Ollama`).
  - Cancel reboot schedule: `sudo pmset repeat cancel`.
- **Layer 4:** revert the three text strings in cerid-ai-internal.
- **Layer 5:** re-add `--gpu-layers 0` to `embedParams` / `rerankParams`. Rebuild. Kickstart.

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Chat slot CPU is too slow for interactive use | Medium | Low (cerid uses OpenRouter for chat per `.env`; chat-slot is for ad-hoc local queries only) | Reversible (Layer 1d revert). Optional env-var override `QUENCHFORGE_CHAT_GPU_LAYERS=999` for users who want to opt in to GPU chat at their own risk. |
| `bench-bert-sustained-load.py` fails | Medium | None — Layer 5 simply doesn't activate; Layers 1–4 stand on their own | File v0.8.0-rc2 issue with the bench failure mode; v0.7.0 CPU route stays in place. |
| `lsof` not available (minimal macOS install, hardened-sandbox CI) | Low | Low (fallback path exists) | Fall back to `netstat -anv` parse; if both fail, log "could not identify holder" and still exit cleanly (the pre-bind check still works; we just can't name the holder). |
| Public users have Ollama configured to bind a different port (`OLLAMA_HOST=127.0.0.1:11500` etc.) | Low | None (no conflict) | The pre-bind check only fires if 11434 is contested; if Ollama is on another port, it simply doesn't appear and quenchforge binds normally. |
| Lumberjack dep rejected by maintainer policy | Low | Low | Inline a ~150-LOC equivalent (`internal/supervisor/logrotate.go`). The rotation contract is simple. |
| `quenchforge doctor` output is too noisy | Low | Low | Add per-check `--only=ports,disk,…` filter later; ship default behavior first. |
| Weekly scheduled reboot annoys the user | Low | Low | Local config (Layer 3); easy to disable with `sudo pmset repeat cancel`. Only applies to this machine. |
| Cerid `detect-gpu.sh` change confuses public cerid-ai users who don't have quenchforge installed | Low | Low | Detection logic gates on backend availability; if quenchforge isn't installed it falls back to ollama. (Implementation detail: confirm the script's fallback path is intact during Layer 4 work.) |

---

## Acceptance criteria

This spec is implemented when:

1. Quenchforge running on this Mac Pro has not freed or panicked the OS for ≥ 14 days of continuous uptime.
2. `quenchforge doctor` returns all-PASS with Ollama.app LaunchAgent disabled.
3. Manually re-enabling Ollama LaunchAgent for a coexistence test produces the canonical error from Layer 1a and does NOT trigger a launchd respawn loop (verified in `pmset -g log`).
4. No quenchforge slot log exceeds 100 MB ten days after the rotation lands.
5. cerid-ai-internal embedding and reranking workflows succeed against quenchforge (no regression — verified by running the existing `tests/test_ollama_proxy_quenchforge.py` and `tests/test_e2e_integration.py` suites).
6. (If Layer 5 activates) `scripts/bench-bert-sustained-load.py --duration 1800` passes on a clean run.

## Open questions deferred to implementation plan

- Exact location for the pre-bind helper file (`internal/gateway/preflight.go` vs `internal/preflight/`).
- Whether `quenchforge doctor` should be a top-level subcommand or live under `quenchforge admin doctor`.
- Whether to ship a `quenchforge install` companion that handles plist installation + Ollama agent detection in one shot. (Recommended for a follow-up spec, not this one.)
- The exact form of any env-var override for chat-slot GPU layers if we want to expose it to users.

These are tactical choices for the implementation plan; they don't change the design.
