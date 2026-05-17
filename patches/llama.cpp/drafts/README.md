# Quenchforge llama.cpp patches — drafts

Patches under this directory are **not** picked up by `scripts/apply-patches.sh`
(it globs `${sub_patch_dir}/*.patch`, ignoring nested dirs). Move a patch
up to `patches/llama.cpp/` to apply it.

## `0002-metal-staging-buffer-pool.patch`

**Status:** designed, not yet applied. Targets the third Metal-on-AMD
crash family ("family-B"): sustained-load `GGML_ASSERT(buf_src)` SIGABRT
in `ggml_metal_buffer_set_tensor` from PCIe-mapped staging-buffer pool
exhaustion. See [quenchforge `patches/README.md` § 3](../../README.md)
for the cross-reference (added by the same commit that ships this draft).

### Why a draft and not the live patch

The Cerid v0.96.0 Phase 1 LongMemEval re-measurement was in flight at
the time the design was finalised (item 20/60 at 07:17 EDT 2026-05-17).
Applying this patch would have required rebuilding `llama-server`,
which means restarting the `com.cerid.quenchforge` LaunchAgent — and a
restart in the middle of the canonical 60-item run would taint the
result. The patch landed as a draft instead; the bench validation
plan below activates the moment the eval completes.

### Apply + bench protocol

After the Cerid eval at `/tmp/lme_v096_judge.log` reaches "LongMemEval
run complete":

```bash
# 1. Promote draft → live patch.
mv patches/llama.cpp/drafts/0002-metal-staging-buffer-pool.patch \
   patches/llama.cpp/0002-metal-staging-buffer-pool.patch

# 2. Apply on a clean submodule tree, regenerate the diff so file
#    indices match the actual checkout (the draft's indices were
#    hand-written and will not be byte-accurate).
bash scripts/apply-patches.sh --reset
bash scripts/apply-patches.sh
cd llama.cpp
# Re-emit the patch with real indices for the kernel-mailbox-compatible
# upstream submission.
git format-patch HEAD~1 --stdout -1 > \
    ../patches/llama.cpp/0002-metal-staging-buffer-pool.patch
cd ..

# 3. Rebuild + reinstall.
bash scripts/build-llama.sh
make
sudo install -m 0755 bin/quenchforge          /usr/local/bin/quenchforge
sudo install -m 0755 bin/quenchforge-bench    /usr/local/bin/quenchforge-bench
sudo install -m 0755 bin/quenchforge-preflight /usr/local/bin/quenchforge-preflight

# 4. Restart the daemon. Plist env unchanged → kickstart is the right verb.
launchctl kickstart -k gui/$(id -u)/com.cerid.quenchforge

# 5. Smoke check.
curl -s http://127.0.0.1:11434/ | jq '.slots | to_entries | map(select(.value.configured))'
```

### Bench acceptance criteria

The patch is "validated" when all four hold on Mac Pro 2019 + Vega II:

1. **Crash elimination.** `quenchforge-bench sustained-embed --duration 30m
   --model nomic-embed-text-v1.5` completes without any `family-B`
   SIGABRT (compare to the v0.6.2 baseline at ~1 crash per ~70 min).
   Auto-respawn count must be 0.
2. **No throughput regression.** Mean req/sec across the 30 min run is
   within 5% of v0.6.2's 0.5 req/sec. The patch adds one memcpy per
   call; on AMD-discrete with already-poor sub-ms allocation cost the
   memcpy is dwarfed by the avoided IOMMU registration, so this should
   come in slightly faster, not slower.
3. **Apple Silicon zero regression.** On the M-series test machine,
   throughput delta must be ≤ 1%. Apple Silicon never enters the
   buffer-pool codepath (`buf->is_shared` fast path); this check just
   guards against accidental impact from the static initialisation.
4. **Escape hatch works.** `GGML_METAL_DISABLE_STAGING_POOL=1
   quenchforge-bench sustained-embed --duration 5m` exactly reproduces
   v0.6.2 behaviour (crash within ~2 min on Vega II).

### Failure modes to watch

- Per-class memory waste: a 4097-byte tensor uses an 8 KiB pool buffer.
  With `QF_STAGING_PER_CLASS_CAP=4` and 15 size classes, worst-case
  pool footprint is bounded but observable. If `quenchforge-bench`
  reports `max_pool_bytes` over 500 MiB, drop the cap to 2.
- Sub-class fragmentation under bursty workloads: if calls cluster at
  oddly-sized boundaries (e.g. 12 KiB tensors), buckets at adjacent
  classes (16 KiB, 32 KiB) accumulate while the actual hot class
  thrashes. Mitigation: log per-class hit rate from the pool and
  re-tune class boundaries if any single class shows < 20% reuse.
- Pool-buffer release race: `qf_staging_release` runs after
  `[cmd_buf waitUntilCompleted]`, so the GPU is guaranteed to be done
  reading. If we ever switch back to the semaphore async pattern, the
  release must move into the `addCompletedHandler` block. Documented
  inline.

### Upstream issue draft

Target: <https://github.com/ggml-org/llama.cpp/issues/new>

> **Title:** Metal: sustained set_tensor/get_tensor allocations
> exhaust AMD discrete IOMMU pool, cause GGML_ASSERT SIGABRT
>
> **Body:**
>
> `ggml_metal_buffer_set_tensor` and `ggml_metal_buffer_get_tensor`
> call `newBufferWithBytesNoCopy` with `MTLResourceStorageModeShared`
> on every invocation against a non-shared buffer (the `!buf->is_shared`
> branches at `ggml-metal-device.m:1665-1717` and `:1719-1755`).
>
> On Apple Silicon this codepath is never taken: the buffer is shared
> and the function returns through the `memcpy` fast path. On AMD
> discrete (Vega II / W6800X / RDNA1+2), `MTLResourceStorageModeShared`
> goes through IOSurface-backed PCIe-mapped memory, and every
> `newBufferWithBytesNoCopy` registers a new IOMMU page-table entry.
> The AMD driver maintains a bounded pool of active mappings (~256-512
> on Vega II per Apple's published documentation). Sustained per-call
> cadence at sub-millisecond intervals — e.g. an embedding-server
> workload serving 0.5 req/sec over many minutes — exhausts the pool
> faster than reclamation. The next allocation returns nil, and the
> `GGML_ASSERT(buf_src)` SIGABRTs the process.
>
> **Reproduction (Mac Pro 2019, Vega II):**
>
> ```
> # 1. Start a llama-server with an embedding model.
> llama-server --model nomic-embed-text-v1.5.gguf --embedding --pooling cls --port 11501 --host 127.0.0.1
>
> # 2. Hammer /v1/embeddings in a loop.
> for i in $(seq 1 5000); do
>     curl -s -X POST http://127.0.0.1:11501/v1/embeddings \
>         -d "{\"model\": \"nomic\", \"input\": \"sample text $i\"}" > /dev/null
> done
> ```
>
> Within ~80 successful calls the server SIGABRTs with the stack:
>
> ```
> #0 ggml_abort
> #1 ggml_metal_buffer_set_tensor at ggml-metal-device.m:1679
> #2 ...
> ```
>
> **Free VRAM during the run stays at ~24 GB**. The failure is IOMMU /
> staging-pool exhaustion, not memory exhaustion. Apple's "Metal Best
> Practices Guide" § "Manage Memory Allocation Performance" warns
> against repeated short-lived buffer allocation for transient work
> and recommends a buffer pool or `MTLHeap`.
>
> **Proposed fix:** a bounded MTLBuffer pool keyed on power-of-two
> size classes. Each pool buffer registers once; subsequent calls
> reuse. Draft patch attached. Apple Silicon is unaffected (does not
> enter this codepath).
>
> Happy to break this into review-sized commits if maintainers prefer.
