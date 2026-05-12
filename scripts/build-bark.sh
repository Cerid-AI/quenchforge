#!/usr/bin/env bash
# build-bark.sh — build the patched bark.cpp with Metal acceleration.
#
# bark.cpp depends on encodec.cpp, which vendors an older single-file
# ggml-metal.m. The Quenchforge patch hits that older file in
# bark.cpp/encodec.cpp/ggml/src/ggml-metal.m. Same root-cause fix as
# llama / whisper / sd; different file path because the upstream ggml
# refactored ggml-metal.m into ggml-metal/ later in its history.
#
# Produces the bark server example from bark.cpp/examples/server/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BARK_DIR="${REPO_ROOT}/bark.cpp"

ARCH="$(uname -m)"
UNIVERSAL=0
CLEAN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch) ARCH="$2"; shift 2 ;;
    --universal) UNIVERSAL=1; shift ;;
    --clean) CLEAN=1; shift ;;
    -h|--help) sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

if [[ "${OSTYPE}" != darwin* ]]; then
  echo "error: build-bark.sh only runs on macOS (Metal-only stack)" >&2
  exit 1
fi

# bark.cpp depends on encodec.cpp (submodule) which depends on ggml
# (submodule of submodule). Both must be initialized before the cmake
# build can resolve `target_link_libraries(bark PUBLIC ggml encodec)`.
( cd "${BARK_DIR}"
  git submodule update --init --depth 1 encodec.cpp
  cd encodec.cpp && git submodule update --init --depth 1 ggml
)

build_one() {
  local arch="$1"
  local build_dir="${BARK_DIR}/build-${arch}"
  if (( CLEAN )); then rm -rf "${build_dir}"; fi
  echo "==> configuring bark.cpp (${arch})"
  cmake -S "${BARK_DIR}" -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES="${arch}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
    -DGGML_METAL=ON \
    -DGGML_ACCELERATE=ON \
    -DBARK_BUILD_EXAMPLES=ON
  echo "==> building bark.cpp (${arch})"
  # The bark example targets are named per-example. Look up what we have.
  cmake --build "${build_dir}" --parallel
}

if (( UNIVERSAL )); then
  build_one arm64
  build_one x86_64
  echo "Universal binary not built — bark.cpp examples don't all expose a single name; ship per-arch instead."
else
  build_one "${ARCH}"
fi

echo "done. binaries:"
find "${BARK_DIR}" -maxdepth 4 -type f -perm -111 -name 'bark*' 2>/dev/null | head -10
