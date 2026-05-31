#!/bin/bash
# quenchforge prestart guard
# -------------------------------------------------------------------------
# Reclaims the gateway port (default 11434, the canonical Ollama-API port)
# from a squatter — in practice Ollama.app's auto-launched `ollama serve`
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
# Install: see packaging/macos/README.md. Idempotent and safe to run when
# no squatter is present.
set -u

# launchd hands jobs a minimal PATH; lsof lives in /usr/sbin, launchctl in
# /bin. Use an explicit PATH so the guard works regardless of the plist's.
export PATH="/usr/sbin:/usr/bin:/bin:/usr/local/bin"

PORT="${QUENCHFORGE_GUARD_PORT:-11434}"
QF_BIN="${QUENCHFORGE_BIN:-/usr/local/bin/quenchforge}"
UID_NUM="$(id -u)"

log() { printf '[prestart-guard] %s\n' "$*" >&2; }

# 1. Boot out Ollama's launchd job so it cannot immediately respawn the
#    serve child we are about to evict. Best-effort: not-loaded is fine.
if launchctl print "gui/${UID_NUM}/com.ollama.ollama" >/dev/null 2>&1; then
	log "booting out com.ollama.ollama"
	launchctl bootout "gui/${UID_NUM}/com.ollama.ollama" 2>/dev/null || true
fi

# 2. Evict any NON-quenchforge listener still holding the port. We never
#    kill our own quenchforge / llama-server processes (a concurrent
#    instance or our own slots), only a foreign squatter.
for pid in $(lsof -ti "tcp:${PORT}" -sTCP:LISTEN 2>/dev/null); do
	cmd="$(ps -p "$pid" -o comm= 2>/dev/null)"
	case "$cmd" in
	*quenchforge* | *llama-server*)
		: # ours — leave it
		;;
	*)
		log "evicting squatter on :${PORT} — pid=${pid} (${cmd:-unknown})"
		kill "$pid" 2>/dev/null || true
		;;
	esac
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
