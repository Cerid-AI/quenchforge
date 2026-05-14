# Third-party licenses

Quenchforge vendors / submodules / runtime-depends on the projects
listed below. The full upstream license text for each is reproduced
inline here as required by their respective licenses. Modifications
are documented in `patches/README.md`.

The Go binary in `cmd/quenchforge/` has **zero non-stdlib dependencies**
(verified: `go list -m all` shows only the module itself). The Go side
is a self-contained supervisor + HTTP gateway.

The third-party code Quenchforge ships at runtime lives in the
submodules (built into separate `llama-server` / `whisper-server` /
`sd-server` / `bark-server` binaries that the Go supervisor exec's).
Each submodule is pinned to a known-good commit; submodule SHAs are
in `.gitmodules` (URLs) + the repo's git tree (commit pins).

---

## llama.cpp — MIT

- **Repo:** https://github.com/ggml-org/llama.cpp
- **Vendored as:** git submodule at `llama.cpp/`
- **License:** MIT
- **Modifications:** `patches/llama.cpp/0001-metal-correctness-on-non-apple-silicon.patch`
  applied at build time. Re-derived from public bug report
  [ggml-org/llama.cpp#19563](https://github.com/ggml-org/llama.cpp/issues/19563);
  no copyrighted patch text from third-party gists is incorporated. See
  `patches/README.md` for the full provenance.

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## whisper.cpp — MIT

- **Repo:** https://github.com/ggml-org/whisper.cpp
- **Vendored as:** git submodule at `whisper.cpp/`
- **License:** MIT
- **Modifications:** `patches/whisper.cpp/0001-metal-correctness-on-non-apple-silicon.patch`
  applied at build time. Same Metal-on-AMD correctness story as llama.cpp.

License text identical to llama.cpp above (both projects ship the same
MIT license under "The ggml authors").

---

## stable-diffusion.cpp (sd.cpp) — MIT

- **Repo:** https://github.com/leejet/stable-diffusion.cpp
- **Vendored as:** git submodule at `sd.cpp/`
- **License:** MIT
- **Modifications:** `patches/sd.cpp/0001-metal-correctness-on-non-apple-silicon.patch`
  applied at build time. Same Metal-on-AMD correctness story.

```
MIT License

Copyright (c) 2023-2026 leejet and stable-diffusion.cpp contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
```

---

## bark.cpp — MIT

- **Repo:** https://github.com/PABannier/bark.cpp
- **Vendored as:** git submodule at `bark.cpp/`
- **License:** MIT
- **Modifications:** `patches/bark.cpp/0001-metal-correctness-on-non-apple-silicon.patch`
  applied at build time. The patch targets the nested encodec.cpp
  ggml-metal.m file (older single-file ggml-metal layout) so the
  surface differs from the other three — see `patches/README.md`.

License text reproduces the MIT license; see the upstream
[LICENSE](https://github.com/PABannier/bark.cpp/blob/main/LICENSE)
for the canonical text. The MIT license boilerplate is identical to
the llama.cpp reproduction above modulo the copyright line:

```
Copyright (c) 2023-2026 PABannier and bark.cpp contributors
```

---

## Go standard library

The Go binary uses only the standard library — verifiable by inspecting
`go.mod` (no `require` directives beyond the module declaration) and by
running `go list -m all` on a clean checkout (returns a single line:
the module itself).

Go is licensed under a BSD-3-Clause license. The license accompanies
the Go toolchain, not Quenchforge releases.

---

## What's NOT in here

Quenchforge does NOT bundle any:

- GGUF model weights — `quenchforge pull` downloads from HuggingFace
  at runtime; users are responsible for the model's own license
- Other Go dependencies — see go.mod, currently empty
- Apple frameworks — linked dynamically at runtime via standard macOS
  framework loading
- Olla, vendored HTTP frameworks, or other Go libraries — earlier
  versions of NOTICE referenced an Olla vendoring intent that didn't
  ship; the current `internal/gateway/` is home-grown

---

## Updating this file

When a new patch lands or a new submodule is added:

1. Add an entry above with repo URL + license + modification note
2. Update `NOTICE` with the short-form copyright line
3. Re-run `git submodule status` to confirm the pin point is captured
   in the repo's git tree
