#!/usr/bin/env bash
# rebase-upstream.sh — pull latest upstream for each patched submodule, replay
# our patch series, and regenerate the series in place.
#
# Walks each subdirectory of patches/ that maps to a submodule (mirrors
# apply-patches.sh). For each submodule it fetches `origin` — which points at
# the true upstream (ggml-org/llama.cpp, ggml-org/whisper.cpp,
# leejet/stable-diffusion.cpp, PABannier/bark.cpp) — resets to the upstream
# default branch, and replays patches/<submodule>/*.patch with `git am -3`.
# On conflict it stops with the rejected hunks visible in `git status` so a
# human can resolve them. drafts/*.patch.broken are never applied or rewritten.
#
# Usage:
#   scripts/rebase-upstream.sh                       # rebase every patched submodule onto its origin default branch
#   scripts/rebase-upstream.sh --only llama.cpp      # one submodule
#   scripts/rebase-upstream.sh --ref v0.3.6          # rebase onto a tag/SHA (requires --only)
#   scripts/rebase-upstream.sh --regenerate-only     # don't fetch/reset; just re-run format-patch off current HEAD

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="${REPO_ROOT}/patches"

REF=""              # empty => each submodule's origin default branch
ONLY=""
REGEN_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref) REF="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --regenerate-only) REGEN_ONLY=1; shift ;;
    -h|--help)
      sed -n '2,17p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "unknown flag: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -n "${REF}" && -z "${ONLY}" ]]; then
  echo "--ref is submodule-specific; pass --only <submodule> with it" >&2
  exit 2
fi

# Resolve a submodule's upstream default branch from its local remote refs
# (origin/HEAD is set at clone time; fall back to master then main).
default_branch() {
  local b
  b="$(git symbolic-ref --quiet refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@')"
  if [[ -z "${b}" ]]; then
    for cand in master main; do
      if git show-ref --verify --quiet "refs/remotes/origin/${cand}"; then b="${cand}"; break; fi
    done
  fi
  echo "${b}"
}

rebase_one() {
  local submod="$1"
  local submod_dir="${REPO_ROOT}/${submod}"
  local sub_patch_dir="${PATCH_DIR}/${submod}"

  shopt -s nullglob
  local patches=( "${sub_patch_dir}"/*.patch )   # top-level only — drafts/ is excluded
  shopt -u nullglob
  if (( ${#patches[@]} == 0 )); then
    echo "    [skip] no patches in ${sub_patch_dir}/"
    return 0
  fi
  if [[ ! -d "${submod_dir}/.git" && ! -f "${submod_dir}/.git" ]]; then
    echo "    [skip] ${submod} submodule not initialized (run: git submodule update --init --recursive)" >&2
    return 0
  fi

  cd "${submod_dir}"

  # Resolve the upstream target for this submodule.
  local target="${REF}"
  if [[ -z "${target}" ]]; then
    if (( ! REGEN_ONLY )); then
      echo "==> [${submod}] fetching origin (upstream)"
      git fetch origin --tags --prune
    fi
    local db; db="$(default_branch)"
    if [[ -z "${db}" ]]; then
      echo "    [error] ${submod}: could not resolve origin default branch" >&2
      return 1
    fi
    target="origin/${db}"
  elif (( ! REGEN_ONLY )); then
    echo "==> [${submod}] fetching origin (upstream)"
    git fetch origin --tags --prune
  fi

  if (( ! REGEN_ONLY )); then
    # Apply the committed patch series directly onto the fresh upstream. This
    # does NOT depend on the working tree being pre-patched — the canonical
    # source of truth is patches/<submodule>/*.patch.
    git am --abort >/dev/null 2>&1 || true
    echo "==> [${submod}] resetting to ${target}"
    git reset --hard "${target}"

    echo "==> [${submod}] applying patches/${submod}/*.patch with three-way merge"
    if ! git am -3 "${patches[@]}"; then
      echo
      echo "rebase stopped on conflict in ${submod}. Resolve hunks, then:"
      echo "  git -C ${submod} add <files>"
      echo "  git -C ${submod} am --continue"
      echo "  scripts/rebase-upstream.sh --only ${submod} --regenerate-only"
      return 1
    fi
  fi

  echo "==> [${submod}] regenerating canonical series into patches/${submod}/"
  # After `git am`, HEAD = target + N quenchforge commits; regenerate the
  # series off that base so the on-disk patches are refreshed against upstream
  # (3-way merge may have adjusted context lines).
  local base; base="$(git merge-base HEAD "${target}")"
  local tmp_out; tmp_out="$(mktemp -d)"
  git format-patch --zero-commit -N -o "${tmp_out}" "${base}..HEAD" >/dev/null

  shopt -s nullglob
  local gen=( "${tmp_out}"/*.patch )
  shopt -u nullglob
  # git format-patch names files from the commit subject; preserve the curated
  # filenames (positional — series order is stable) so doc references and the
  # 0001/0002 numbering don't churn on every rebase.
  if (( ${#gen[@]} == ${#patches[@]} )); then
    rm -f "${sub_patch_dir}"/*.patch
    local i
    for i in "${!gen[@]}"; do
      cp "${gen[$i]}" "${sub_patch_dir}/$(basename "${patches[$i]}")"
    done
  else
    echo "    [warn] ${submod}: regenerated ${#gen[@]} patches vs ${#patches[@]} curated — keeping generated names" >&2
    rm -f "${sub_patch_dir}"/*.patch
    cp "${gen[@]}" "${sub_patch_dir}/"
  fi
  rm -rf "${tmp_out}"
  echo "    updated: $(ls -1 "${sub_patch_dir}"/*.patch 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
}

shopt -s nullglob
submod_dirs=( "${PATCH_DIR}"/*/ )
shopt -u nullglob
if (( ${#submod_dirs[@]} == 0 )); then
  echo "no patch subdirectories under ${PATCH_DIR}/" >&2
  exit 0
fi

rc=0
changed=()
for d in "${submod_dirs[@]}"; do
  submod="$(basename "$d")"
  if [[ -n "${ONLY}" && "${ONLY}" != "${submod}" ]]; then
    continue
  fi
  if rebase_one "${submod}"; then
    changed+=( "${submod}" )
  else
    rc=1
  fi
  cd "${REPO_ROOT}"
done

echo
if (( rc == 0 )); then
  echo "Rebase complete. Next: commit the updated submodule pointers + patches:"
  echo "  cd ${REPO_ROOT}"
  echo "  git add ${changed[*]} patches/"
  echo "  git commit -m 'chore(deps): rebase patch series onto upstream'"
fi
exit "${rc}"
