#!/usr/bin/env bash
# rebase-upstream.sh — pull latest llama.cpp master, replay our patches, regenerate the series.
#
# Run weekly (via .github/workflows/rebase-upstream.yml) or manually after an
# upstream API change. On conflict, the action stops with the rejected hunks
# visible in `git status` so a human can resolve.
#
# Usage:
#   scripts/rebase-upstream.sh                      # rebase onto upstream/master
#   scripts/rebase-upstream.sh --ref v0.3.6         # rebase onto a tag or SHA
#   scripts/rebase-upstream.sh --regenerate-only    # don't fetch, just re-run format-patch
#                                                   # off the current llama.cpp HEAD

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${REPO_ROOT}/llama.cpp"
PATCH_DIR="${REPO_ROOT}/patches"

REF="upstream/master"
REGEN_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref) REF="$2"; shift 2 ;;
    --regenerate-only) REGEN_ONLY=1; shift ;;
    -h|--help)
      sed -n '2,12p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "unknown flag: $1" >&2
      exit 2
      ;;
  esac
done

cd "${LLAMA_DIR}"

if (( ! REGEN_ONLY )); then
  echo "==> ensuring upstream remote exists"
  if ! git remote get-url upstream >/dev/null 2>&1; then
    git remote add upstream https://github.com/ggml-org/llama.cpp.git
  fi

  echo "==> fetching upstream"
  git fetch upstream --tags --prune

  echo "==> capturing current patches as commits onto pre-rebase ref"
  PRE_REBASE_REF="$(git rev-parse HEAD)"
  # If the working tree carries our patches (e.g. via apply-patches.sh), they
  # are present but uncommitted. Materialize them as commits so `git am -3`
  # can replay them onto the new base.
  if ! git diff --quiet || ! git diff --cached --quiet; then
    git add -A
    git commit -m "WIP: quenchforge patches (pre-rebase snapshot)" --allow-empty
  fi
  COMMITS_REF="$(git rev-parse HEAD)"

  echo "==> resetting to ${REF}"
  git reset --hard "${REF}"

  echo "==> generating patches from snapshot ${PRE_REBASE_REF}..${COMMITS_REF}"
  rm -f "${PATCH_DIR}"/*.patch
  git format-patch --zero-commit -N -o "${PATCH_DIR}" "${PRE_REBASE_REF}..${COMMITS_REF}"

  echo "==> replaying patches with three-way merge"
  if ! git am -3 "${PATCH_DIR}"/*.patch; then
    echo
    echo "rebase stopped on conflict. Resolve hunks in llama.cpp/, then:"
    echo "  git -C llama.cpp add <files>"
    echo "  git -C llama.cpp am --continue"
    echo "and re-run scripts/rebase-upstream.sh --regenerate-only"
    exit 1
  fi
fi

echo "==> regenerating canonical patch series"
rm -f "${PATCH_DIR}"/*.patch
# Series base = upstream/master before our replay. After `git am`, HEAD is
# upstream/master + N quenchforge commits. Format-patch those N commits.
BASE="$(git merge-base HEAD upstream/master)"
git format-patch --zero-commit -N -o "${PATCH_DIR}" "${BASE}..HEAD"

echo
echo "Updated patches:"
ls -1 "${PATCH_DIR}"
echo
echo "Next: commit the updated submodule pointer and patches in the parent repo:"
echo "  cd ${REPO_ROOT}"
echo "  git add llama.cpp patches/"
echo "  git commit -m 'chore(deps): rebase llama.cpp onto $(date -u +%Y-%m-%d)'"
