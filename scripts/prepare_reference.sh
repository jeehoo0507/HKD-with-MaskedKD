#!/usr/bin/env bash
# Fetch only benchmark preparation code/documents, pinned to the inspected commit.
set -euo pipefail
cd "$(dirname "$0")/.."
destination="external/benchmarks"
revision="e81cc3063dc5f98a2d43366f2e86b08905e80875"
if [[ -e "$destination" ]]; then
  current=$(git -C "$destination" rev-parse HEAD)
  if [[ "$current" != "$revision" ]]; then
    echo "Existing $destination is not the pinned reference; use a separate checkout." >&2
    exit 1
  fi
  echo "$destination already contains the pinned reference."
  exit 0
fi
mkdir -p "$destination"
git -C "$destination" init
git -C "$destination" remote add origin https://github.com/jeehoo0507/kshs-aimlab-benchmarks.git
git -C "$destination" sparse-checkout init --cone
git -C "$destination" sparse-checkout set datasets scripts
git -C "$destination" fetch --depth 1 --filter=blob:none origin "$revision"
git -C "$destination" checkout --detach FETCH_HEAD
echo "Reference ready: $destination. Dataset images have not been downloaded."
