# Apple Developer ID + notarization setup

Quenchforge's [`release.yml`](../.github/workflows/release.yml) workflow
ships unsigned binaries today because the repo doesn't have Apple
Developer ID credentials yet. This doc walks through the **exact steps**
to flip signed + notarized releases on.

## Status for this maintainer

| Item | Status |
|---|---|
| Apple ID | `sunrunnerfire@mac.com` — confirmed configured |
| Apple Developer Team ID | **`4A5VDRMRB8`** |
| Xcode + `notarytool` | installed at `/Applications/Xcode.app/...` |
| CSR | generated at `~/quenchforge-signing/quenchforge-developer-id.csr` |
| Private key | at `~/quenchforge-signing/quenchforge-developer-id.key` (mode 0600) |
| Developer ID Application certificate | **not yet minted** — see step 1 below |
| App Store Connect API key | **not yet created** — see step 3 below |
| GitHub repo secrets | **not yet set** — see step 4 below |
| Homebrew tap PAT | **not yet set** — see step 5 below |

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
openssl pkcs12 -export \
  -inkey quenchforge-developer-id.key \
  -in developerID_application.pem \
  -name "Quenchforge Developer ID" \
  -password "pass:${P12_PASS}" \
  -out quenchforge-developer-id.p12

# Verify it parses:
openssl pkcs12 -info -in quenchforge-developer-id.p12 -password "pass:${P12_PASS}" -nokeys | head -10

# Base64-encode for the GitHub Secret:
base64 -i quenchforge-developer-id.p12 | pbcopy
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
