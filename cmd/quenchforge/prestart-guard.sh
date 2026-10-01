#!/bin/bash
# quenchforge prestart guard
# -------------------------------------------------------------------------
# Reclaims the gateway port (default 11434, the canonical Ollama-API port)
# from Ollama — in practice Ollama.app's auto-launched `ollama serve`
# child — BEFORE handing off to `quenchforge serve`.
#
# Why this exists: quenchforge's own pre-bind check (v0.7.2+) deliberately
# exits 0 when the port is already held, and the LaunchAgent's
# KeepAlive.SuccessfulExit=false then leaves it dead. That makes quenchforge
# yield to a squatter. Wiring this guard as the LaunchAgent's
# ProgramArguments[0] means the port is reclaimed on every (re)start and at
# login, so quenchforge stays authoritative on the canonical port without
# ceding it — and without the operator hand-evicting Ollama during restart
# windows.
#
# The guard only evicts what it recognises (Ollama). Any other listener on
# the port — e.g. a different inference server deliberately bound there —
# is left running, and the guard exits non-zero naming it.
#
# Install: see packaging/macos/README.md. Idempotent and safe to run when
# no squatter is present.
set -u

case "$(uname -m)" in
arm64) BREW_BIN="/opt/homebrew/bin" ;;
*) BREW_BIN="/usr/local/bin" ;;
esac

# launchd hands jobs a minimal PATH; lsof lives in /usr/sbin, launchctl in
# /bin. Use an explicit PATH so the guard works regardless of the plist's.
# quenchforge inherits it via exec, and its pressure probes (ioreg, sysctl)
# also live in /usr/sbin.
export PATH="/usr/sbin:/usr/bin:/bin:${BREW_BIN}"

LISTEN_ADDR="${QUENCHFORGE_LISTEN_ADDR:-127.0.0.1:11434}"
PORT="${QUENCHFORGE_GUARD_PORT:-${LISTEN_ADDR##*:}}"
QF_BIN="${QUENCHFORGE_BIN:-${BREW_BIN}/quenchforge}"
UID_NUM="$(id -u)"

log() { printf '[prestart-guard] %s\n' "$*" >&2; }

# 1. Classify every listener before touching anything. Our own quenchforge /
#    llama-server processes (a concurrent instance or our own slots) are left
#    to quenchforge's pre-bind check; Ollama is evicted; anything else stops
#    the guard.
evict=""
foreign=""
for pid in $(lsof -ti "tcp:${PORT}" -sTCP:LISTEN 2>/dev/null); do
	cmd="$(ps -p "$pid" -o comm= 2>/dev/null)"
	[ -n "$cmd" ] || continue
	case "${cmd##*/}" in
	*quenchforge* | *llama-server*) ;;
	ollama | Ollama) evict="${evict} ${pid}" ;;
	*) foreign="${foreign} pid=${pid} (${cmd})" ;;
	esac
done

# Non-zero so the refusal shows as the job's last exit status; launchd
# retries after ThrottleInterval, so quenchforge starts once the port frees.
if [ -n "$foreign" ]; then
	log "port ${PORT} is held by${foreign}, which is not Ollama or quenchforge; leaving it running"
	log "to run quenchforge alongside it, set QUENCHFORGE_LISTEN_ADDR to a free address (e.g. 127.0.0.1:11435) in the LaunchAgent's EnvironmentVariables"
	exit 1
fi

# 2. Boot out Ollama's launchd job so it cannot immediately respawn the
#    serve child we are about to evict. Best-effort: not-loaded is fine.
if launchctl print "gui/${UID_NUM}/com.ollama.ollama" >/dev/null 2>&1; then
	log "booting out com.ollama.ollama"
	launchctl bootout "gui/${UID_NUM}/com.ollama.ollama" 2>/dev/null || true
fi

for pid in $evict; do
	log "evicting Ollama on :${PORT} — pid=${pid}"
	kill "$pid" 2>/dev/null || true
done

# 3. Brief settle so the kernel releases the port before quenchforge's
#    own pre-bind check runs.
sleep 1

# 4. Hand off. exec so launchd supervises quenchforge directly (PID,
#    signals, KeepAlive, ProcessType all apply to the server, not this
#    wrapper). Args after the guard in ProgramArguments flow through, so
#    the plist provides `serve`.
log "starting: ${QF_BIN} $*"
exec "${QF_BIN}" "$@"
