# Apple Developer ID + notarization setup

Quenchforge's [`release.yml`](../.github/workflows/release.yml) workflow
ships unsigned binaries today because the repo doesn't have Apple
Developer ID credentials yet. This doc walks through the **exact steps**
to flip signed + notarized releases on.

You'll need:

- An Apple Developer Program membership ($99/year) — sign up at
  <https://developer.apple.com/programs/enroll/> if you don't have one.
- Admin access to the repo's GitHub Settings → Secrets and variables → Actions.
- A Mac (any Mac) to generate the certificate request and export the `.p12`.
  Most steps work from Terminal; a couple need Keychain Access.app.

Total time: about 30 minutes the first time. Each subsequent year's
renewal is ~5 minutes.

---

## 1. Create a Developer ID Application certificate

1. Open **Keychain Access.app** → top menu → **Certificate Assistant** →
   **Request a Certificate From a Certificate Authority…**
2. Fill in:
   - **User Email Address**: your Apple Developer Program email
   - **Common Name**: anything memorable, e.g. `Quenchforge Release Signing`
   - **CA Email Address**: leave blank
   - **Request is**: select **Saved to disk**
3. Save the resulting `.certSigningRequest` (CSR) file somewhere temporary.

4. Browse to <https://developer.apple.com/account/resources/certificates/add>
   - Choose **Developer ID Application** (NOT "Mac App Distribution"; the
     standalone tarball/Homebrew bottle distribution path needs Developer ID).
   - Upload the CSR file from step 3.
   - Download the resulting `developerID_application.cer`.

5. Double-click `developerID_application.cer` to install it into your
   keychain. You should now see a private key + certificate pair under
   the **My Certificates** category in Keychain Access.

### Verify locally

```sh
security find-identity -v -p codesigning | grep "Developer ID Application"
```

Should show one line ending in your team ID and name. If it shows
`0 valid identities found`, repeat step 5.

## 2. Export the cert as a `.p12`

```sh
# In Keychain Access:
# - Right-click your "Developer ID Application: ..." cert
# - Export
# - Format: Personal Information Exchange (.p12)
# - Save as: ~/quenchforge-developer-id.p12
# - Pick a strong passphrase you can paste into GitHub Secrets
```

Or via CLI (also needs a passphrase):

```sh
security export \
  -k login.keychain \
  -t identities \
  -f pkcs12 \
  -P 'PICK-A-STRONG-PASSPHRASE' \
  -o ~/quenchforge-developer-id.p12 \
  "Developer ID Application: Your Team Name (TEAMID)"
```

Base64-encode it for the secret:

```sh
base64 -i ~/quenchforge-developer-id.p12 | pbcopy
# Now paste into the APPLE_DEVELOPER_ID_CERT_P12_B64 secret below.
```

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
