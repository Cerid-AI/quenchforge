# macOS distribution helpers

## Auto-install (recommended)

`quenchforge install` drops the LaunchAgent plist into
`~/Library/LaunchAgents/` with your `$USER` substituted into the
`REPLACE_ME` placeholders automatically.

```bash
quenchforge install
# Inspect (model env vars may need editing for your GGUFs):
less ~/Library/LaunchAgents/com.cerid.quenchforge.plist
# Bootstrap the daemon:
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.cerid.quenchforge.plist
# Verify:
curl http://127.0.0.1:11434/
```

Flags:

- `--force` — overwrite an existing plist (default: refuse if one exists)
- `--skip-user-substitution` — leave `REPLACE_ME` unchanged so you can
  edit by hand
- `--print-path` — print the target path and exit (no write)

The canonical source for the embedded template is
[`cmd/quenchforge/plist_template.plist`](../../cmd/quenchforge/plist_template.plist) —
inspect it before installing if you want to see what will land.

To uninstall:

```bash
launchctl bootout gui/$(id -u)/com.cerid.quenchforge
rm ~/Library/LaunchAgents/com.cerid.quenchforge.plist
```

## Why not use `brew services`?

Operators who installed via `brew install cerid-ai/tap/quenchforge`
already get a LaunchAgent generated automatically. Use:

```bash
brew services start quenchforge
brew services stop quenchforge
brew services restart quenchforge
```

`quenchforge install` is for the from-source case where the Homebrew
formula's release artifact isn't published yet (development builds,
custom patches, machines without Homebrew).

## Hardware-aware slot args (informational)

When Quenchforge detects AMD discrete hardware (Vega Pro, W6800X,
RDNA1/2), the chat slot is launched with two additional flags:

- `--flash-attn off` — keeps standard attention GPU-resident on AMD.
  The default `flash-attn auto` correctly detects that flash attention
  can't run on AMD discrete, but it ferries the FA tensor to CPU each
  decode step, throttling tok/s by an order of magnitude.

- `--no-cache-prompt` — disables the prompt-cache state-save path
  that triggers a `GGML_ASSERT(buf_dst)` failure in
  `ggml_metal_buffer_get_tensor` on Vega II. Crash is reproducible on
  the second chat request with similar prefix (LCP > 10%).

Embed and rerank slots don't decode autoregressively and don't touch
the prompt-cache state-save path, so they keep the upstream defaults
regardless of profile. Apple Silicon also keeps the upstream defaults
across all slots.

These flags are applied automatically by the supervisor — no manual
override needed.

## Logs

```bash
tail -F ~/Library/Logs/quenchforge/quenchforge.out.log
tail -F ~/Library/Logs/quenchforge/quenchforge.err.log
```
