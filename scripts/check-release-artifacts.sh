#!/usr/bin/env bash
# check-release-artifacts.sh — assert every shipped binary is notarized.
#
# The release footer and install.sh both promise Developer-ID-signed,
# Apple-notarized binaries. goreleaser only submits the build ids listed under
# notarize.macos[].ids, and that list once omitted the two universal ids — the
# exact artifacts install.sh downloads (quenchforge_<v>_darwin_all.tar.gz), so
# the default one-line install shipped un-notarized binaries while the
# per-arch Homebrew bottle was covered.
#
# Usage: scripts/check-release-artifacts.sh [path/to/.goreleaser.yaml]
set -euo pipefail

CONFIG="${1:-$(dirname "$0")/../.goreleaser.yaml}"
[ -f "$CONFIG" ] || {
	echo "check-release-artifacts: no config at $CONFIG" >&2
	exit 2
}

# All declared artifact ids: `  - id: x` under builds:, `  - id: x` under
# universal_binaries:. Both use the same two-space list indent.
declared="$(awk '
	/^(builds|universal_binaries):/ { in_section = 1; next }
	/^[a-z_]+:/ { in_section = 0 }
	in_section && $1 == "-" && $2 == "id:" { print $3 }
	in_section && $1 == "id:" { print $2 }
' "$CONFIG" | sort -u)"

# The notarize id list: everything indented under `      ids:` inside notarize:.
notarized="$(awk '
	/^notarize:/ { in_notarize = 1; next }
	/^[a-z_]+:/ { in_notarize = 0 }
	in_notarize && $1 == "ids:" { in_ids = 1; next }
	in_notarize && in_ids && $1 == "-" { print $2; next }
	in_notarize && in_ids && $1 != "-" && $0 !~ /^ *#/ { in_ids = 0 }
' "$CONFIG" | sort -u)"

[ -n "$declared" ] || {
	echo "check-release-artifacts: parsed no build ids from $CONFIG" >&2
	exit 2
}
[ -n "$notarized" ] || {
	echo "check-release-artifacts: parsed no notarize ids from $CONFIG" >&2
	exit 2
}

missing="$(comm -23 <(printf '%s\n' "$declared") <(printf '%s\n' "$notarized"))"
if [ -n "$missing" ]; then
	echo "check-release-artifacts: these artifacts ship but are never notarized:" >&2
	printf '  %s\n' $missing >&2
	echo "Add them to notarize.macos[].ids in $CONFIG." >&2
	exit 1
fi

echo "check-release-artifacts: OK — all $(printf '%s\n' "$declared" | wc -l | tr -d ' ') artifact ids are notarized"
