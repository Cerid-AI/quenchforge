#!/bin/sh
# Quenchforge installer  —  https://github.com/Cerid-AI/quenchforge
#
#   curl -fsSL https://raw.githubusercontent.com/Cerid-AI/quenchforge/main/install.sh | sh
#
# Downloads the latest signed + notarized universal release, verifies its
# SHA-256 against the published checksums, installs the binaries to
# /usr/local/bin, gates on `quenchforge-preflight`, and (unless
# QUENCHFORGE_NO_SERVICE=1) writes the per-user LaunchAgent + prestart port
# guard via `quenchforge install`.
#
# Env knobs:
#   QUENCHFORGE_VERSION       pin a release tag (default: latest)
#   QUENCHFORGE_PREFIX        install prefix (default: /usr/local)
#   QUENCHFORGE_NO_SERVICE=1  install binaries only; skip the LaunchAgent
#
# Why macOS-only, why /usr/local: the prestart guard + LaunchAgent default
# to /usr/local/bin/quenchforge, and the daemon needs a per-user GUI
# (Aqua) session for Metal GPU access — a system LaunchDaemon would be
# denied the GPU. See packaging/macos/README.md.
set -eu

REPO="Cerid-AI/quenchforge"
PREFIX="${QUENCHFORGE_PREFIX:-/usr/local}"
BINDIR="$PREFIX/bin"

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# --- platform + tool gate ------------------------------------------------
[ "$(uname -s)" = "Darwin" ] || die "Quenchforge is macOS-only (detected $(uname -s)). On Linux/Windows use stock Ollama with CUDA/ROCm/DirectML."
command -v curl   >/dev/null 2>&1 || die "curl is required"
command -v shasum >/dev/null 2>&1 || die "shasum is required"
command -v tar    >/dev/null 2>&1 || die "tar is required"

# --- resolve version -----------------------------------------------------
VERSION="${QUENCHFORGE_VERSION:-}"
if [ -z "$VERSION" ]; then
	say "Resolving latest release…"
	VERSION="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" |
		grep -m1 '"tag_name"' | sed -E 's/.*"(v?[^"]+)".*/\1/')"
	[ -n "$VERSION" ] || die "could not resolve the latest release tag (GitHub API rate-limited? set QUENCHFORGE_VERSION)"
fi
VER="${VERSION#v}" # asset filenames carry the bare version
say "Installing quenchforge $VERSION"

# --- download + verify ---------------------------------------------------
TARBALL="quenchforge_${VER}_darwin_all.tar.gz" # universal: both binaries
BASE="https://github.com/$REPO/releases/download/$VERSION"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT INT TERM

say "Downloading $TARBALL"
curl -fSL "$BASE/$TARBALL"        -o "$TMP/$TARBALL"      || die "download failed: $BASE/$TARBALL"
curl -fsSL "$BASE/checksums.txt"  -o "$TMP/checksums.txt" || die "checksums download failed"

say "Verifying SHA-256"
EXPECT="$(grep " ${TARBALL}\$" "$TMP/checksums.txt" | awk '{print $1}')"
[ -n "$EXPECT" ] || die "no checksum entry for $TARBALL in checksums.txt"
ACTUAL="$(shasum -a 256 "$TMP/$TARBALL" | awk '{print $1}')"
[ "$EXPECT" = "$ACTUAL" ] || die "checksum mismatch for $TARBALL (expected $EXPECT, got $ACTUAL)"

tar xzf "$TMP/$TARBALL" -C "$TMP" || die "extract failed"
[ -f "$TMP/quenchforge" ] || die "archive did not contain the quenchforge binary"

# --- install binaries ----------------------------------------------------
SUDO=""
if [ ! -w "$BINDIR" ] && [ ! -w "$PREFIX" ]; then
	SUDO="sudo"
	say "$BINDIR is not writable; using sudo (you may be prompted)"
fi
say "Installing binaries to $BINDIR"
$SUDO mkdir -p "$BINDIR"
$SUDO install -m 0755 "$TMP/quenchforge" "$BINDIR/quenchforge"
[ -f "$TMP/quenchforge-preflight" ] && $SUDO install -m 0755 "$TMP/quenchforge-preflight" "$BINDIR/quenchforge-preflight"

# --- hardware gate -------------------------------------------------------
if [ -x "$BINDIR/quenchforge-preflight" ]; then
	say "Checking hardware support"
	PF="$("$BINDIR/quenchforge-preflight" 2>&1)" || true
	printf '%s\n' "$PF"
	case "$PF" in
	*status=ok*) : ;;
	*) die "this machine is not supported (preflight did not report status=ok). Set QUENCHFORGE_NO_SERVICE=1 to install the binaries anyway." ;;
	esac
fi

# --- service setup -------------------------------------------------------
if [ "${QUENCHFORGE_NO_SERVICE:-0}" = "1" ]; then
	say "Skipping LaunchAgent (QUENCHFORGE_NO_SERVICE=1)."
else
	say "Installing the LaunchAgent + prestart port guard"
	USER="${USER:-$(id -un)}"
	export USER
	"$BINDIR/quenchforge" install --force
fi

# --- done ----------------------------------------------------------------
printf '\n'
say "$("$BINDIR/quenchforge" version 2>/dev/null | head -1) installed to $BINDIR"
cat <<EOF

Next steps:
  1. Pull a model (\`quenchforge pull --list\` shows the curated catalog):
       quenchforge pull llama3.2:3b
  2. Start the service:
       launchctl bootstrap gui/\$(id -u) ~/Library/LaunchAgents/com.cerid.quenchforge.plist
  3. Verify it's serving:
       curl http://127.0.0.1:11434/

Point any Ollama- or OpenAI-compatible client at http://127.0.0.1:11434.
Docs: https://github.com/$REPO#readme
EOF
