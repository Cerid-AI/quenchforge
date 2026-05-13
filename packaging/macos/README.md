# macOS distribution helpers

Templates for operators installing Quenchforge outside Homebrew.

## Files

- `com.cerid.quenchforge.plist` — LaunchAgent template for from-source
  installs. Replace `REPLACE_ME` with your username and the
  `QUENCHFORGE_*_MODEL` values with GGUF filenames under
  `~/.quenchforge/models/`. Install with:

  ```bash
  cp packaging/macos/com.cerid.quenchforge.plist \
      ~/Library/LaunchAgents/com.cerid.quenchforge.plist
  # then edit the file:
  #   - replace REPLACE_ME with $USER (3 occurrences)
  #   - point the three model env vars at GGUFs you have locally
  launchctl load -w ~/Library/LaunchAgents/com.cerid.quenchforge.plist
  ```

  Verify:

  ```bash
  launchctl list | grep com.cerid.quenchforge   # expect non-empty
  curl http://127.0.0.1:11434/health             # {"status":"ok"}
  ```

  Tail logs:

  ```bash
  tail -F ~/Library/Logs/quenchforge/quenchforge.out.log
  ```

  Uninstall:

  ```bash
  launchctl unload ~/Library/LaunchAgents/com.cerid.quenchforge.plist
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

This template is for the from-source case where the Homebrew formula's
release artifact isn't published yet (development builds, custom
patches, etc.).

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
