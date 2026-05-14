# Apple Developer ID + notarization setup

Quenchforge's [`release.yml`](../.github/workflows/release.yml) workflow
ships unsigned binaries today because the repo doesn't have Apple
Developer ID credentials yet. This doc walks through the **exact steps**
to flip signed + notarized releases on.

## Status for this maintainer

> **As of 2026-05-14:** Apple signing + notarization are LIVE. v0.3.3,
> v0.3.4, and v0.4.0 all shipped signed + notarized binaries via the
> release workflow. The one remaining gap is the Homebrew tap PAT
> (step 5 below) — without it, the tap formula is updated manually
> after each release. Releases otherwise succeed end-to-end.

| Item | Status |
|---|---|
| Apple ID | `sunrunnerfire@mac.com` — confirmed configured |
| Apple Developer Team ID | **`4A5VDRMRB8`** |
| Xcode + `notarytool` | installed at `/Applications/Xcode.app/...` |
| CSR | generated at `~/quenchforge-signing/quenchforge-developer-id.csr` |
| Private key | at `~/quenchforge-signing/quenchforge-developer-id.key` (mode 0600) |
| Developer ID Application certificate | ✅ **minted** (Justin Michaels / 4A5VDRMRB8) |
| App Store Connect API key | ✅ **created + uploaded as repo secret** |
| GitHub repo secrets (5 Apple ones) | ✅ **set** — `APPLE_DEVELOPER_ID_CERT_P12_B64`, `APPLE_DEVELOPER_ID_CERT_PASSWORD`, `APPLE_NOTARY_API_KEY_ID`, `APPLE_NOTARY_API_KEY_ISSUER`, `APPLE_NOTARY_API_KEY_P8_B64` |
| Homebrew tap PAT (`HOMEBREW_TAP_GITHUB_TOKEN`) | ⚠ **not yet set** — see step 5 below. Without this, the tap formula needs a manual update after each release. |

You'll still need:

- Admin access to the repo's GitHub Settings → Secrets and variables → Actions.
- ~10 minutes of focused browser time across developer.apple.com,
  appstoreconnect.apple.com, and github.com.

---

## 1. Mint the Developer ID Application certificate

The CSR is already generated at
`~/quenchforge-signing/quenchforge-developer-id.csr`. Upload it to Apple:

1. Browse to <https://developer.apple.com/account/resources/certificates/add>
2. Make sure your team selector (top-right) shows **`4A5VDRMRB8`**.
3. Choose **Developer ID Application** (NOT "Mac App Distribution"; the
   standalone tarball + Homebrew bottle distribution path needs Developer ID).
4. **Profile Type**: **G2 Sub-CA (Xcode 11.4.1 or later)**.
5. Upload `~/quenchforge-signing/quenchforge-developer-id.csr`.
6. Download the resulting `developerID_application.cer` (it usually lands
   in `~/Downloads/`).

## 2. Combine the cert + key into a `.p12`

This step keeps the signing material entirely outside the macOS keychain —
ideal for CI because there's no "unlock the keychain" dance:

```zsh
mv ~/Downloads/developerID_application.cer ~/quenchforge-signing/
cd ~/quenchforge-signing

# Convert the DER cert Apple gave us to PEM, then bundle with our key:
openssl x509 -inform DER -in developerID_application.cer -out developerID_application.pem

# Pick a strong passphrase — you'll paste it into a GitHub Secret in step 4.
# (NOTE: zsh's `read -p` means "read from coprocess" — different from bash.
#  Using printf + read -s keeps this portable across bash and zsh on macOS.)
printf "Pick a strong P12 passphrase: " >&2
read -s P12_PASS
echo

# CRITICAL: the `-legacy` flag is load-bearing on modern openssl (3.x ships
# with macOS Sonoma+; Homebrew openssl defaults to 3.6+). Without it,
# openssl exports a .p12 with a SHA-256 MAC + AES-256 PBE, which macOS
# Security framework (`security import`, the CI's keychain ingest) cannot
# consume. The failure mode is a misleading "MAC verification failed
# (wrong password?)" — the passphrase is fine; the algorithm is wrong.
# `-legacy` opts into SHA-1 MAC + PBE-SHA1-3DES, which both macOS Security
# framework and Apple's notarytool understand.
openssl pkcs12 -export -legacy \
  -inkey quenchforge-developer-id.key \
  -in developerID_application.pem \
  -name "Quenchforge Developer ID" \
  -password "pass:${P12_PASS}" \
  -out quenchforge-developer-id.p12

# Verify it parses AND verify the MAC algorithm is SHA-1 (NOT SHA-256):
openssl pkcs12 -info -in quenchforge-developer-id.p12 -password "pass:${P12_PASS}" -nokeys | grep -E '^(MAC|PKCS7)' | head -5
# Expected: "MAC: sha1, Iteration 2048" — if you see sha256, re-run the
# export with -legacy and try again. The release.yml workflow's pre-flight
# step will also catch this and fail with a clear "use -legacy" error.

# Base64-encode for the GitHub Secret (unwrapped — pbcopy on macOS wraps at
# 76 chars by default; the workflow tolerates wrapped input but unwrapped
# is unambiguous):
base64 -i quenchforge-developer-id.p12 | tr -d '\n' | pbcopy
echo "P12 base64 now on clipboard. Keep the passphrase in your password manager."
unset P12_PASS
```

> **Don't put your passphrase in the `read -p` prompt argument.** It will
> show up in `~/.zsh_history` (or `~/.bash_history`). If you accidentally
> do, scrub it with:
>
> ```zsh
> sed -i '' '/read -s -p .*P12_PASS/d' ~/.zsh_history
> ```
>
> and pick a different passphrase before re-running.

## 3. Create an App Store Connect API key for notarization

`notarytool` (the only supported notarization path since `altool` was
deprecated) authenticates with an App Store Connect API key, not your
Apple ID + app-specific password.

1. Browse to <https://appstoreconnect.apple.com/access/integrations/api>
2. Click **+** to add a new key.
3. Name it `Quenchforge Notarization`.
4. Access: **Developer** (the minimal scope that lets notarytool submit).
5. Click **Generate**. **Download the `.p8` file immediately** — it's only
   shown once.
6. Note the **Key ID** (10 chars, e.g. `ABC123XYZ9`) and **Issuer ID**
   (a UUID, e.g. `12345678-1234-1234-1234-123456789012`). Both are on
   the same page.

Base64-encode the `.p8`:

```sh
base64 -i ~/Downloads/AuthKey_ABC123XYZ9.p8 | pbcopy
# Paste into APPLE_NOTARY_API_KEY_P8_B64 below.
```

## 4. Drop secrets into the GitHub repo

Browse to <https://github.com/Cerid-AI/quenchforge/settings/secrets/actions>
and add **all five**:

| Secret | Value |
|---|---|
| `APPLE_DEVELOPER_ID_CERT_P12_B64` | base64 of `~/quenchforge-developer-id.p12` (step 2) |
| `APPLE_DEVELOPER_ID_CERT_PASSWORD` | the passphrase you chose for the `.p12` |
| `APPLE_NOTARY_API_KEY_ID` | the 10-char Key ID from step 3 |
| `APPLE_NOTARY_API_KEY_ISSUER` | the UUID Issuer ID from step 3 |
| `APPLE_NOTARY_API_KEY_P8_B64` | base64 of the `.p8` (step 3) |

The release workflow's `if: env.X != ''` guards will then activate the
sign + notarize steps automatically on the next tag push.

## 5. (Optional, recommended) Add the Homebrew tap PAT

Quenchforge's release also pushes an updated formula to
[`Cerid-AI/homebrew-tap`](https://github.com/Cerid-AI/homebrew-tap).
That cross-repo push needs a fine-grained PAT:

1. Browse to <https://github.com/settings/personal-access-tokens/new>
2. Resource owner: **Cerid-AI**
3. Repository access: **Only select repositories** → `homebrew-tap`
4. Permissions: **Contents** → **Read and write**
5. Generate, copy the token.
6. Add as repo secret: `HOMEBREW_TAP_GITHUB_TOKEN` on
   <https://github.com/Cerid-AI/quenchforge/settings/secrets/actions>.

Without this token, the release succeeds but the tap formula isn't
updated automatically — operators have to wait for a manual sync. The
release workflow's brews skip_upload guard handles the missing-token
case gracefully (added in commit `f16267c`).

### Manual tap update recipe (run after each release until the PAT lands)

```sh
# 1. Get the live SHA256s from the GitHub release
cd /tmp && curl -sL "https://github.com/Cerid-AI/quenchforge/releases/download/v${VERSION}/checksums.txt" -o /tmp/qf-checksums.txt
ARM64_SHA=$(grep "darwin_arm64.tar.gz" /tmp/qf-checksums.txt | awk '{print $1}')
AMD64_SHA=$(grep "darwin_amd64.tar.gz" /tmp/qf-checksums.txt | awk '{print $1}')

# 2. Clone the tap, edit version + both SHAs, push
git clone git@github.com:Cerid-AI/homebrew-tap.git /tmp/homebrew-tap-wip
cd /tmp/homebrew-tap-wip
# (edit Formula/quenchforge.rb — bump `version`, replace both `sha256` lines)
git add Formula/quenchforge.rb
git commit -m "chore(formula): bump quenchforge to v${VERSION}"
git push origin main

# 3. Verify
brew untap cerid-ai/tap && brew tap cerid-ai/tap
brew audit --strict --new cerid-ai/tap/quenchforge  # must exit 0
brew info cerid-ai/tap/quenchforge | head -3        # should show new version
```

Once `HOMEBREW_TAP_GITHUB_TOKEN` is set on the quenchforge repo, the
above goes away — goreleaser auto-pushes on each tag.

## 6. Verify the first signed release

```sh
git tag v0.3.2  # or whatever's next
git push origin v0.3.2
# Then watch:
gh run watch --repo Cerid-AI/quenchforge $(gh run list \
  --repo Cerid-AI/quenchforge --workflow=release.yml --limit 1 \
  --json databaseId --jq '.[0].databaseId')
```

When it's green, the GitHub Release should:

- Show signed `quenchforge_X.Y.Z_darwin_{amd64,arm64,all}.tar.gz` archives
- The binaries inside should pass `spctl -a -t exec quenchforge` after
  unpacking (Gatekeeper accepts notarized binaries)
- The Homebrew tap formula should auto-update — `brew install
  cerid-ai/tap/quenchforge` works without `--no-quarantine`

### Verifying notarization out-of-band

The release workflow uses `wait: false` on the notarize block (goreleaser
fire-and-forgets each submission and exits the step without polling Apple).
This avoids Apple's per-API-key hourly rate limit (HTTP 429 `RATE_LIMIT`)
that gets tripped when 4 binaries × goreleaser's ~50s polling interval
cross the ~50/hour ceiling.

To confirm Apple accepted the submissions for a given tag, run locally:

```sh
# Find the API key file you downloaded in step 3:
KEY_FILE=$(ls ~/Downloads/AuthKey_*.p8 ~/quenchforge-signing/AuthKey_*.p8 2>/dev/null | head -1)
KEY_ID=$(basename "$KEY_FILE" | sed -E 's/AuthKey_(.+)\.p8/\1/')
ISSUER_ID="<the Issuer ID UUID you noted in step 3>"

# List recent submissions (should show your tag's 4 binaries):
xcrun notarytool history --key "$KEY_FILE" --key-id "$KEY_ID" --issuer "$ISSUER_ID" | head -20

# Drill into any specific submission ID — status should be "Accepted":
xcrun notarytool info <submission-uuid> \
  --key "$KEY_FILE" --key-id "$KEY_ID" --issuer "$ISSUER_ID"

# If "Invalid", pull the failure log:
xcrun notarytool log <submission-uuid> \
  --key "$KEY_FILE" --key-id "$KEY_ID" --issuer "$ISSUER_ID"
```

The submission IDs are visible in the GitHub Actions log under the
"Run goreleaser/goreleaser-action@v6" step — search for `notarizing`.

End-user impact of `wait: false` is zero: Apple's cloud-based Gatekeeper
check at install time uses the binary's hash regardless of whether
goreleaser polled. Stapling does not apply to raw Mach-O CLI binaries
(only `.app` / `.dmg` / `.pkg` can be stapled), so there's nothing the
wait would have given us beyond a confirmation log line.

## Rotation

- Developer ID Application certificates expire **5 years** after issue.
  Re-do steps 1-2 + 4 when that approaches.
- App Store Connect API keys don't expire automatically, but rotate
  them quarterly as security hygiene. Re-do step 3 + 4 ("API key" secrets).
- The Homebrew tap PAT has whatever expiry you set when minting it.
  Fine-grained PATs default to 30/60/90/365 days — set a calendar reminder.

## Local-machine snapshot (no signing)

You don't need any of the above to develop locally. From the repo:

```sh
goreleaser release --snapshot --clean --skip=sign,publish,notarize
```

That produces `dist/*.tar.gz` artifacts identical to the GitHub Release
ones, just unsigned. Useful for verifying the build matrix before
spending the time on Apple Developer enrollment.
