#!/usr/bin/env bash
# build-llama.sh — build the patched llama.cpp with Metal acceleration for Quenchforge.
#
# Usage:
#   scripts/build-llama.sh                  # native arch (arm64 on Apple Silicon, x86_64 on Intel)
#   scripts/build-llama.sh --arch arm64
#   scripts/build-llama.sh --arch x86_64
#   scripts/build-llama.sh --universal      # lipo arm64 + x86_64 (direct-download bottle only)
#   scripts/build-llama.sh --clean
#
# Produces:
#   build-<arch>/bin/llama-server           # the slot binary Quenchforge supervises
#   build-<arch>/bin/llama-cli              # for smoke tests
#
# Prereqs (Homebrew):
#   brew install cmake ninja
# Xcode Command Line Tools must be installed: `xcode-select -p` should print a path.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${REPO_ROOT}/llama.cpp"

ARCH="$(uname -m)"
UNIVERSAL=0
CLEAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch) ARCH="$2"; shift 2 ;;
    --universal) UNIVERSAL=1; shift ;;
    --clean) CLEAN=1; shift ;;
    -h|--help)
      sed -n '2,18p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "unknown flag: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "${OSTYPE}" != darwin* ]]; then
  echo "error: build-llama.sh only runs on macOS (Metal-only stack)" >&2
  exit 1
fi

build_one() {
  local arch="$1"
  local build_dir="${LLAMA_DIR}/build-${arch}"

  if (( CLEAN )); then
    rm -rf "${build_dir}"
  fi

  echo "==> configuring (${arch})"
  cmake -S "${LLAMA_DIR}" -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES="${arch}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
    -DLLAMA_METAL=ON \
    -DLLAMA_METAL_EMBED_LIBRARY=ON \
    -DLLAMA_ACCELERATE=ON \
    -DLLAMA_NATIVE=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_SERVER=ON

  echo "==> building (${arch})"
  cmake --build "${build_dir}" --parallel --target llama-server llama-cli
}

if (( UNIVERSAL )); then
  build_one arm64
  build_one x86_64
  out="${LLAMA_DIR}/build-universal/bin"
  mkdir -p "${out}"
  for tool in llama-server llama-cli; do
    echo "==> lipo ${tool}"
    lipo -create \
      "${LLAMA_DIR}/build-arm64/bin/${tool}" \
      "${LLAMA_DIR}/build-x86_64/bin/${tool}" \
      -output "${out}/${tool}"
    lipo -info "${out}/${tool}"
  done
else
  build_one "${ARCH}"
fi

echo "done. binaries:"
find "${LLAMA_DIR}" -maxdepth 3 -name 'llama-server' -o -name 'llama-cli'
