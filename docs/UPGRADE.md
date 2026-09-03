# Upgrading Quenchforge

Version-specific notes for operators. Everything not listed here is a
drop-in replacement: stop the LaunchAgent, install the new binary, start it.

---

## v0.11.0 — first restart after upgrading from v0.10.1 or earlier

Two changes are visible on that first restart, both in how the supervisor
decides what it is allowed to talk to.

### Orphans from the old pidfile format are reaped, for this release only

Every build up to v0.10.1 wrote a slot's pidfile as a bare integer. Builds
from v0.11.0 write the child's start time and executable path alongside the
PID, because a PID alone cannot be trusted: macOS recycles PIDs freely, a
pidfile outlives the crash that stranded it, and the reaper's signal goes to
the whole process group.

That leaves one restart where every genuine orphan — a slot left running by a
crash before the upgrade — is described by a pidfile carrying no identity.
Refusing to signal those would leave a stale `llama-server` holding its slot
port and its VRAM, and the slot meant to replace it could never bind.

So for pidfiles in the old format, and only those, the reaper falls back to
one piece of evidence: the running process's executable path must be exactly
one of the executables this build spawns (`llama-server`, `whisper-server`,
`sd-server`, the bark server). That is strictly narrower than the substring
match those same orphans were reaped by before the upgrade, and it expires by
itself — every pidfile written from here on carries a full identity.

The grace is removed in **v0.12.0**. It is marked in the source as
`LEGACY PIDFILE GRACE`; deleting that branch and the `ourExec` argument to
`ReapOrphans` is the whole removal.

What you will see on the restart:

```
quenchforge: reap chat.pid pid=41207 action=killed legacy pidfile (no recorded identity); pid runs "/usr/local/bin/llama-server", which this build spawns
```

and, for a PID that has been recycled by something unrelated:

```
quenchforge: reap embed.pid pid=41208 action=skip legacy pidfile (no recorded identity) and pid runs "/bin/zsh", not one of our executables; refusing to signal
```

A skip is not an error — the pidfile is removed either way. It means that
PID no longer belongs to us and never gets signalled.

If an orphan was started from a path this build no longer uses (a source
build you have since moved, say), it is skipped too. Find and stop it by
hand:

```sh
lsof -nP -iTCP:8081 -sTCP:LISTEN     # 8081 = the slot port named in the log
kill <pid>
```

### A slot upstream is registered only after the slot names its model

Registration used to wait for the slot port to accept a TCP connection.
Accepting proves only that *something* holds the port — and the thing most
likely to hold it after an unclean restart is an orphaned server from the
previous run. Registering it points the lane's live traffic at a stale
process serving the previous model, and `/health` reports the lane healthy.

Each `llama-server`-backed slot (chat, embed, code-embed, rerank) is now
asked over `/v1/models` which model it loaded, and its upstream is registered
only when the answer is the model that slot was started with. The GGUF path
the server echoes back, the configured filename and the `qwen2.5:7b`
shorthand are all treated as one name; only a genuinely different model is
refused.

A refusal is loud and names both models:

```
quenchforge: ERROR: 127.0.0.1:8080 serves "/models/llama3.1-8b-instruct-q4_k_m.gguf", but the chat slot was started with "qwen2.5:7b-instruct-q4_k_m" — refusing to register it as the chat upstream.
```

Until the port is served by the right model the lane returns 503 — visible
and diagnosable, unlike answers from the wrong model. Probing continues, so
the lane comes up on its own once the real slot wins the port.

The image-gen, TTS and whisper slots serve no `/v1/models`; their readiness
is unchanged (TCP accept).
