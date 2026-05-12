#!/usr/bin/env bash
# apply-patches.sh — apply Quenchforge's patch series to the vendored submodules.
#
# Layout:
#   patches/<submodule-name>/NNNN-*.patch
#
# Walks each subdirectory of patches/ matching an existing submodule and
# applies the .patch files in lexicographic order. Currently:
#   patches/llama.cpp/    -> llama.cpp/
#   patches/whisper.cpp/  -> whisper.cpp/
#
# Usage:
#   scripts/apply-patches.sh                # apply all submodules
#   scripts/apply-patches.sh --check        # dry-run, exit non-zero on conflicts
#   scripts/apply-patches.sh --reset        # reset submodules to their pinned SHAs first
#   scripts/apply-patches.sh --only llama.cpp   # only one submodule

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="${REPO_ROOT}/patches"

MODE="apply"
ONLY=""
for arg in "$@"; do
  case "$arg" in
    --check) MODE="check" ;;
    --reset) MODE="reset" ;;
    --only) shift; ONLY="${1:-}"; shift ;;
    --only=*) ONLY="${arg#--only=}" ;;
    -h|--help)
      sed -n '2,17p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      ;;
  esac
done

apply_one() {
  local submod="$1"
  local sub_patch_dir="${PATCH_DIR}/${submod}"
  local submod_dir="${REPO_ROOT}/${submod}"

  if [[ ! -d "${submod_dir}/.git" && ! -f "${submod_dir}/.git" ]]; then
    echo "    [skip] ${submod} submodule not initialized (run: git submodule update --init --recursive)" >&2
    return 0
  fi

  shopt -s nullglob
  local patches=( "${sub_patch_dir}"/*.patch )
  shopt -u nullglob
  if (( ${#patches[@]} == 0 )); then
    echo "    [skip] no patches in ${sub_patch_dir}/"
    return 0
  fi

  if [[ "${MODE}" == "reset" ]]; then
    echo "==> resetting ${submod} to pinned SHA"
    git submodule update --force --checkout -- "${submod}"
  fi

  ( cd "${submod_dir}"
    for p in "${patches[@]}"; do
      local name="$(basename "$p")"
      if git apply --reverse --check "$p" >/dev/null 2>&1; then
        echo "    [skip] ${submod}/${name} (already applied)"
        continue
      fi
      if [[ "${MODE}" == "check" ]]; then
        echo "==> ${submod}/${name} (dry-run)"
        git apply --check "$p"
      else
        echo "==> ${submod}/${name}"
        # No --index here: when the patch's target lives in a nested
        # submodule (e.g. sd.cpp/ggml/..., bark.cpp/encodec.cpp/ggml/...)
        # the outer submodule's index treats the inner one as a gitlink
        # and `git apply --index` refuses with
        # "does not exist in index". Working-tree-only is what the build
        # actually needs anyway.
        git apply --whitespace=nowarn "$p"
      fi
    done
  )
}

# Discover patch subdirectories — each one whose name matches a submodule path.
shopt -s nullglob
submod_dirs=( "${PATCH_DIR}"/*/ )
shopt -u nullglob
if (( ${#submod_dirs[@]} == 0 )); then
  echo "no patch subdirectories under ${PATCH_DIR}/" >&2
  exit 0
fi

for d in "${submod_dirs[@]}"; do
  submod="$(basename "$d")"
  if [[ -n "${ONLY}" && "${ONLY}" != "${submod}" ]]; then
    continue
  fi
  apply_one "${submod}"
done

echo "done."
