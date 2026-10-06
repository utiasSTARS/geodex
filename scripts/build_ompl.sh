#!/usr/bin/env bash
# Build the pinned OMPL fork as a static library and install it into a prefix of its own.
# A second run with the same pin does nothing. A changed pin or patch replaces the install.
#
# Usage
#   scripts/build_ompl.sh [prefix]
# The prefix defaults to $GEODEX_PREFIX, which pixi sets.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prefix="${1:-${GEODEX_PREFIX:?pass an install prefix or run inside the pixi environment}}"

generator=()
if command -v ninja > /dev/null; then
  generator=(-DCMAKE_GENERATOR=Ninja)
fi

cmake -DGEODEX_OMPL_PREFIX="${prefix}" \
  -DGEODEX_OMPL_WORK_DIR="${repo_root}/.pixi/ompl-fork" \
  -DGEODEX_DOWNLOAD_DIR="${repo_root}/.pixi/downloads" \
  "${generator[@]}" \
  "-DGEODEX_OMPL_CMAKE_ARGS=-DCMAKE_PREFIX_PATH=${CONDA_PREFIX:-}" \
  -P "${repo_root}/cmake/GeodexOmplFork.cmake"
