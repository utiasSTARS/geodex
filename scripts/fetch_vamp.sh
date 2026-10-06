#!/usr/bin/env bash
# Fetch the pinned VAMP source into .pixi/vamp. Pass -DVAMP_DIR=$PWD/.pixi/vamp to the geodex
# build, or leave VAMP_DIR unset and the build fetches VAMP itself.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cmake -DGEODEX_FETCH=vamp -DGEODEX_FETCH_DEST="${repo_root}/.pixi/vamp" \
  -DGEODEX_DOWNLOAD_DIR="${repo_root}/.pixi/downloads" -P "${repo_root}/cmake/GeodexFetch.cmake"
