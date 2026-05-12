#!/usr/bin/env bash
# build-sd.sh — build the patched stable-diffusion.cpp with Metal acceleration.
#
# Produces sd-server, the HTTP server example that ships with sd.cpp.
# The server exposes OpenAI-compatible /v1/images/generations + sdapi/v1/*
# endpoints; quenchforge supervises it like llama-server.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SD_DIR="${REPO_ROOT}/sd.cpp"

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
  echo "error: build-sd.sh only runs on macOS (Metal-only stack)" >&2
  exit 1
fi

# sd.cpp's `ggml` is a submodule of ggml-org/ggml — we need it checked
# out before the patch can apply or cmake can build.
( cd "${SD_DIR}" && git submodule update --init --depth 1 ggml )

build_one() {
  local arch="$1"
  local build_dir="${SD_DIR}/build-${arch}"
  if (( CLEAN )); then rm -rf "${build_dir}"; fi
  echo "==> configuring sd.cpp (${arch})"
  cmake -S "${SD_DIR}" -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES="${arch}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
    -DSD_METAL=ON \
    -DSD_BUILD_SERVER=ON \
    -DSD_BUILD_EXAMPLES=ON
  echo "==> building sd.cpp (${arch})"
  cmake --build "${build_dir}" --parallel --target sd-server
}

if (( UNIVERSAL )); then
  build_one arm64
  build_one x86_64
  out="${SD_DIR}/build-universal/bin"
  mkdir -p "${out}"
  for tool in sd-server; do
    lipo -create \
      "${SD_DIR}/build-arm64/bin/${tool}" \
      "${SD_DIR}/build-x86_64/bin/${tool}" \
      -output "${out}/${tool}"
    lipo -info "${out}/${tool}"
  done
else
  build_one "${ARCH}"
fi

echo "done. binary:"
find "${SD_DIR}" -maxdepth 3 -name 'sd-server' | head -3
