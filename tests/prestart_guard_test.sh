#!/bin/bash
# Runs cmd/quenchforge/prestart-guard.sh against a listener it does not
# recognise and asserts the guard refuses (non-zero exit, no hand-off to
# quenchforge) and leaves the listener running. Only the refusal paths are
# exercised: the hand-off path boots out a live Ollama job on the host.
set -euo pipefail

if [ "$(uname)" != Darwin ]; then
	echo "skip: prestart guard is macOS-only"
	exit 0
fi

root="$(cd "$(dirname "$0")/.." && pwd)"
guard="${root}/cmd/quenchforge/prestart-guard.sh"
tmp="$(mktemp -d)"
listener=""
cleanup() {
	if [ -n "$listener" ]; then kill "$listener" 2>/dev/null || true; fi
	rm -rf "$tmp"
}
trap cleanup EXIT

port="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
python3 -m http.server --bind 127.0.0.1 "$port" >/dev/null 2>&1 &
listener=$!
for _ in $(seq 50); do
	/usr/sbin/lsof -ti "tcp:${port}" -sTCP:LISTEN >/dev/null 2>&1 && break
	sleep 0.1
done
/usr/sbin/lsof -ti "tcp:${port}" -sTCP:LISTEN >/dev/null || { echo "FAIL: listener never came up on :${port}"; exit 1; }

stub="${tmp}/quenchforge"
printf '#!/bin/sh\ntouch "%s/handed-off"\n' "$tmp" >"$stub"
chmod +x "$stub"

failures=0
fail() { echo "FAIL: $*"; failures=$((failures + 1)); }

expect_refusal() {
	local name="$1"
	shift
	local before="$failures"
	rm -f "${tmp}/handed-off"
	local rc=0
	env -u QUENCHFORGE_GUARD_PORT -u QUENCHFORGE_LISTEN_ADDR \
		QUENCHFORGE_BIN="$stub" "$@" bash "$guard" serve 2>"${tmp}/stderr" || rc=$?
	[ "$rc" -ne 0 ] || fail "${name}: guard exited 0 with a foreign listener on :${port}"
	kill -0 "$listener" 2>/dev/null || fail "${name}: guard killed the foreign listener"
	[ ! -e "${tmp}/handed-off" ] || fail "${name}: guard handed off to quenchforge"
	grep -q "pid=${listener}" "${tmp}/stderr" || fail "${name}: message does not name pid ${listener}"
	grep -q QUENCHFORGE_LISTEN_ADDR "${tmp}/stderr" || fail "${name}: message does not point to QUENCHFORGE_LISTEN_ADDR"
	if [ "$failures" -eq "$before" ]; then echo "ok: ${name} (exit ${rc})"; fi
}

expect_refusal "port from QUENCHFORGE_GUARD_PORT" QUENCHFORGE_GUARD_PORT="$port"
expect_refusal "port from QUENCHFORGE_LISTEN_ADDR" QUENCHFORGE_LISTEN_ADDR="127.0.0.1:${port}"

[ "$failures" -eq 0 ] || exit 1
echo "PASS: prestart guard leaves foreign listeners alone"
