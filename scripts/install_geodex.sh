#!/usr/bin/env bash
# Build geodex with VAMP and the pinned OMPL fork, and install geodex and the fork into a
# prefix. A C++ project, such as a ROS 2 plugin, adds the prefix to CMAKE_PREFIX_PATH and
# calls find_package(geodex 1.0 CONFIG REQUIRED). Extra arguments go to the configure step.
#
# Usage
#   scripts/install_geodex.sh <prefix> [cmake args...]
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <prefix> [cmake args...]" >&2
  exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prefix="$1"
shift
build="${repo_root}/build/install-$(basename "${prefix}")"

generator=()
if command -v ninja > /dev/null; then
  generator=(-G Ninja)
fi

cmake -S "${repo_root}" -B "${build}" "${generator[@]}" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${prefix}" -DGEODEX_OMPL=ON -DGEODEX_BUILD_OMPL=ON -DGEODEX_VAMP=ON "$@"
cmake --build "${build}"
cmake --install "${build}"
