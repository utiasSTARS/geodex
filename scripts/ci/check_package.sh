#!/usr/bin/env bash
# Build and run examples/cmake_project and scripts/ci/package_check against the geodex
# installed in a prefix. package_check checks the exported targets and components.
#
# Usage
#   scripts/ci/check_package.sh <prefix> [cmake args...]
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <prefix> [cmake args...]" >&2
  exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
prefix="$(cd "$1" && pwd)"
shift
build="${repo_root}/build/package-consumer"
check_build="${repo_root}/build/package-check"

rm -rf "${build}" "${check_build}"
cmake -S "${repo_root}/examples/cmake_project" -B "${build}" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${prefix}" "$@"
cmake --build "${build}"
"${build}/my_planner"

cmake -S "${repo_root}/scripts/ci/package_check" -B "${check_build}" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${prefix}" "$@"
cmake --build "${check_build}"
"${check_build}/package_check"
