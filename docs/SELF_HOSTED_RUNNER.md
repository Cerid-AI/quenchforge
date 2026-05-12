# Self-hosted `[amd-gpu]` runner

The `rebase-upstream.yml` and `ci.yml` workflows reference a self-hosted runner
labelled `[amd-gpu]`. That runner exists to:

1. Validate the patch series against real AMD-Mac hardware on every push
   (via `go test -tags=amd_gpu ./tests/integration/...`).
2. Block any weekly rebase PR that produces a binary that no longer generates
   coherent output.
3. Eventually run the v0.2 multi-device + Vega II Duo tests.

Without this runner, GitHub-hosted macos-latest runners build the patched
`llama-server` but can't validate it against an AMD Metal device (those
runners are all Apple Silicon). The amd_gpu-tagged tests are therefore
skipped on hosted CI and depend on this runner for the merge gate.

## One-time setup on the Mac Pro

```sh
# 1. Pick a directory for the runner (NOT inside the repo — runner writes
#    config + work dirs in here).
mkdir -p ~/quenchforge-runner
cd ~/quenchforge-runner

# 2. Download the GitHub Actions runner tarball — version pinned to match
#    what GitHub Actions currently expects. Look up the latest at:
#    https://github.com/actions/runner/releases
RUNNER_VER=2.321.0
ARCH=x64  # Intel Mac. Use osx-arm64 on Apple Silicon.
curl -fsSLO \
  https://github.com/actions/runner/releases/download/v${RUNNER_VER}/actions-runner-osx-${ARCH}-${RUNNER_VER}.tar.gz
tar xzf actions-runner-osx-${ARCH}-${RUNNER_VER}.tar.gz

# 3. Register the runner with the Cerid-AI/quenchforge repo. Get the
#    registration token from:
#    https://github.com/Cerid-AI/quenchforge/settings/actions/runners/new
#    (don't reuse a token across registrations; they expire after 1h).
./config.sh \
  --url https://github.com/Cerid-AI/quenchforge \
  --token <REGISTRATION_TOKEN_FROM_GITHUB> \
  --name mac-pro-2019-vega-ii \
  --labels self-hosted,macOS,x86_64-macos,amd-gpu \
  --work _work \
  --unattended

# 4. Install as a launchd service so it survives reboots.
./svc.sh install
./svc.sh start

# 5. Verify it's online.
./svc.sh status
gh api repos/Cerid-AI/quenchforge/actions/runners \
  --jq '.runners[] | {name, status, labels: [.labels[].name]}'
```

The two labels that matter to workflows:

- `x86_64-macos` — required by the goreleaser dual-arch build matrix when we
  flip from emulated to native x86_64 macOS builds.
- `amd-gpu` — required by the integration tests + the rebase-upstream PR gate.

Both can live on the same runner (one machine, two labels) per the original
plan.

## Prereqs the runner needs installed

Before the runner can run a job successfully, the host needs:

| Tool       | Source                          | Why |
|------------|---------------------------------|-----|
| Xcode CLT  | `xcode-select --install`        | clang, ld, codesign, dns-sd |
| Homebrew   | <https://brew.sh>               | toolchain manager |
| cmake      | `brew install cmake`            | llama.cpp build driver |
| ninja      | `brew install ninja`            | llama.cpp build backend |
| Go 1.23+   | `brew install go`               | quenchforge build |
| Git LFS    | `brew install git-lfs`          | future model fixtures |
| (optional) GitHub CLI | `brew install gh`    | manual workflow runs |

The runner itself runs as the user who registered it. Don't run as root —
`security set-keychain-settings` and `xcrun notarytool` both expect a
user-owned keychain.

## What runs on this runner

| Job | Workflow | What it does |
|---|---|---|
| `amd-gpu-integration` | `ci.yml` | `go test -tags=amd_gpu ./tests/integration/...`. Reproduces the patch's "coherent output" assertion against the live Vega II. Currently `if: github.event_name == 'workflow_dispatch'` — manual-trigger only until the runner is online. **After registering the runner, flip that `if:` to `always()` (or remove it) so the job runs on every push/PR as the merge gate.** |
| `rebase-upstream` (gate stage) | `rebase-upstream.yml` | After the weekly rebase action picks the latest `ggml-org/llama.cpp` master and applies our patches, this runner builds + runs the regression test on the rebased patch. PR labelled `needs-hw-verify` gets removed when the test passes. |

The `release.yml` workflow does NOT use this runner — it runs on
GitHub-hosted `macos-latest` so the signing keychain stays inside the GitHub
secure environment. The self-hosted runner is for *correctness* validation
(patch still produces coherent output), not for release artifact production.

## Runner-down handling

The `runner-health.yml` workflow (TODO: write) pings the runner daily. If
it's been offline >24h, an issue gets opened with the `runner-down` label
so the maintainer notices. Until that lands, `gh api repos/Cerid-AI/quenchforge/actions/runners`
shows the current status.

## Security notes

- The runner runs jobs with the privileges of the user that registered it.
  Do NOT install this runner on a machine where you handle sensitive work.
- The Mac Pro 2019 is the maintainer's dedicated CI box; if you adopt this
  pattern on a shared machine, isolate via a separate user account.
- Public PRs from external contributors can spawn arbitrary code on this
  runner. The repo's "Require approval for first-time contributors" setting
  (under Settings → Actions → General) is on; review every external PR's
  workflow file before approving the first run.
