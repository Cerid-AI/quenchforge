# Patch series

Quenchforge carries **two** patches against `ggml-org/llama.cpp`. Every patch is keyed on a public upstream issue, and a draft upstream PR is filed for each so the diff has a stable reference even when the patch isn't accepted.

| File | Upstream issue | Purpose |
|---|---|---|
| `0001-private-vram-force.patch` | [ggml-org/llama.cpp#15228](https://github.com/ggml-org/llama.cpp/issues/15228) | Force private VRAM allocation (`MTLStorageModePrivate`) for discrete AMD Metal devices. Without this, Metal falls back to shared-storage buffers on Intel-Mac + AMD systems and KV-cache copies become CPU-bound. |
| `0002-disable-simdgroup-mm-non-apple-silicon.patch` | [ggml-org/llama.cpp#19563](https://github.com/ggml-org/llama.cpp/issues/19563) | Set `has_simdgroup_mm=false` for non-Apple-Silicon Metal devices. The simdgroup matrix-multiply path miscompiles on AMD Metal; the scalar fallback is correct and within 10% of the simdgroup path on Vega II. |

## Why only two

The earlier plan mentioned a third tweak (`GGML_METAL_N_CB=2`, the Metal command-buffer count). That isn't a code change — it's a runtime environment variable. Quenchforge sets it as a default in `internal/supervisor/` when launching `llama-server`, so it ships as behavior without code churn.

## Provenance — important

These patches are **re-derived from the public issue descriptions**, not copied from third-party gists. In particular:

- Issue #15228 has an attached gist by user `Basten7` with a candidate patch. That gist carries **no explicit license**, so it cannot be redistributed. Our `0001-private-vram-force.patch` reaches the same outcome (private VRAM allocation) from the public issue's symptom + suggested-fix description, with our own variable names, code structure, and comments. If `Basten7` ever publishes the gist under an OSI license, we'll consider replacing our version with a clean import + attribution.
- Issue #19563 has discussion and proposed approaches in-thread but no canonical patch attached. Our `0002-disable-simdgroup-mm-non-apple-silicon.patch` implements the in-thread guidance: detect non-Apple-Silicon Metal devices via `MTLDevice.architecture.name` and the `appleArchitecture` family bits, then gate `has_simdgroup_mm` accordingly.

Both patches will carry an `Upstream-PR:` git-trailer pointing at the corresponding draft PR on `ggml-org/llama.cpp`, so anyone reading the patch series can find the upstream conversation in one click.

## Re-applying after upstream changes

`scripts/rebase-upstream.sh` runs weekly via GitHub Actions and replays this series onto the latest `ggml-org/llama.cpp` master with `git am -3`. The three-way merge survives benign refactors like `ggml-metal.m` -> `ggml-metal/ggml-metal.m` because both patches are keyed on stable function names, not line numbers.

On a real conflict, the rebase action stops with rejected hunks visible and opens a PR with the `rebase-conflict` label for human resolution.

## Stub status (as of 2026-05-12)

The two `.patch` files in this directory are currently **stubs** carrying the commit message + Upstream-PR trailer only. The actual hunks land during MVP week 2 of Effort 1 (per the plan), once the `llama.cpp` submodule is wired in and `scripts/build-llama.sh` produces a Metal binary we can validate the changes against on the Mac Pro 2019. The stub commits exist so the patch-format pipeline, `scripts/apply-patches.sh --check` plumbing, and the rebase action all have something to operate on while the real patches are being developed.
