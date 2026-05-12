#!/usr/bin/env bash
# apply-patches.sh — apply Quenchforge's patch series onto the llama.cpp submodule.
#
# Usage:
#   scripts/apply-patches.sh                # apply onto the pinned llama.cpp/ submodule
#   scripts/apply-patches.sh --check        # dry-run, exit non-zero on conflicts
#   scripts/apply-patches.sh --reset        # reset llama.cpp/ to its pinned SHA first
#
# Patch series lives in patches/00xx-*.patch and is produced by
#   git -C llama.cpp format-patch --zero-commit -N -o ../patches <base>..HEAD
# from a working integration branch. Apply order is lexicographic.
#
# Rebasing on upstream: see scripts/rebase-upstream.sh — it does the three-way
# `git am -3` flow that survives upstream renames like ggml-metal.m -> ggml-metal/.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${REPO_ROOT}/llama.cpp"
PATCH_DIR="${REPO_ROOT}/patches"

MODE="apply"
for arg in "$@"; do
  case "$arg" in
    --check) MODE="check" ;;
    --reset) MODE="reset" ;;
    -h|--help)
      sed -n '2,16p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "unknown flag: $arg" >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "${LLAMA_DIR}/.git" && ! -f "${LLAMA_DIR}/.git" ]]; then
  echo "error: llama.cpp submodule not initialized. Run:" >&2
  echo "  git submodule update --init --recursive" >&2
  exit 1
fi

shopt -s nullglob
patches=( "${PATCH_DIR}"/*.patch )
shopt -u nullglob

if (( ${#patches[@]} == 0 )); then
  echo "no patches found in ${PATCH_DIR}/ — nothing to do" >&2
  exit 0
fi

if [[ "${MODE}" == "reset" ]]; then
  echo "==> resetting llama.cpp/ to pinned submodule SHA"
  git submodule update --force --checkout -- llama.cpp
fi

cd "${LLAMA_DIR}"

# Detect already-applied patches so re-runs are idempotent.
for p in "${patches[@]}"; do
  name="$(basename "$p")"
  if git apply --reverse --check "$p" >/dev/null 2>&1; then
    echo "    [skip] ${name} (already applied)"
    continue
  fi
  if [[ "${MODE}" == "check" ]]; then
    echo "==> ${name} (dry-run)"
    git apply --check "$p"
  else
    echo "==> ${name}"
    git apply --index --whitespace=nowarn "$p"
  fi
done

echo "done."
