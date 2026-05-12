#!/usr/bin/env bash
# build-whisper.sh — build the patched whisper.cpp with Metal acceleration.
#
# Mirrors scripts/build-llama.sh but targets the whisper.cpp submodule.
# Produces whisper-server (HTTP) and whisper-cli (smoke test).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHISPER_DIR="${REPO_ROOT}/whisper.cpp"

ARCH="$(uname -m)"
UNIVERSAL=0
CLEAN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch) ARCH="$2"; shift 2 ;;
    --universal) UNIVERSAL=1; shift ;;
    --clean) CLEAN=1; shift ;;
    -h|--help) sed -n '2,8p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

if [[ "${OSTYPE}" != darwin* ]]; then
  echo "error: build-whisper.sh only runs on macOS (Metal-only stack)" >&2
  exit 1
fi

build_one() {
  local arch="$1"
  local build_dir="${WHISPER_DIR}/build-${arch}"
  if (( CLEAN )); then rm -rf "${build_dir}"; fi
  echo "==> configuring whisper.cpp (${arch})"
  cmake -S "${WHISPER_DIR}" -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES="${arch}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_ACCELERATE=ON \
    -DGGML_NATIVE=OFF \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_EXAMPLES=ON \
    -DWHISPER_BUILD_SERVER=ON
  echo "==> building whisper.cpp (${arch})"
  cmake --build "${build_dir}" --parallel --target whisper-server whisper-cli
}

if (( UNIVERSAL )); then
  build_one arm64
  build_one x86_64
  out="${WHISPER_DIR}/build-universal/bin"
  mkdir -p "${out}"
  for tool in whisper-server whisper-cli; do
    lipo -create \
      "${WHISPER_DIR}/build-arm64/bin/${tool}" \
      "${WHISPER_DIR}/build-x86_64/bin/${tool}" \
      -output "${out}/${tool}"
    lipo -info "${out}/${tool}"
  done
else
  build_one "${ARCH}"
fi

echo "done. binaries:"
find "${WHISPER_DIR}" -maxdepth 3 -name 'whisper-server' -o -name 'whisper-cli'
