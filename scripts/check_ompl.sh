#!/usr/bin/env bash
# Check that the OMPL fork in a prefix is the pinned source with the geodex patch. The
# recorded source must match third_party/, and a probe linked against the library must find
# the patched planner.
#
# Usage
#   scripts/check_ompl.sh [prefix]
# The prefix defaults to $GEODEX_PREFIX.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
prefix="${1:-${GEODEX_PREFIX:?pass an install prefix or run inside the pixi environment}}"
build="${repo_root}/.pixi/ompl-fork/probe"
recorded="${prefix}/share/ompl/geodex-ompl-source.txt"
expected="$(cmake -DGEODEX_FETCH_PRINT_STAMP=ompl -P "${repo_root}/cmake/GeodexFetch.cmake")"

if [[ ! -f "${recorded}" ]]; then
  echo "error: ${prefix} does not have an OMPL fork built by scripts/build_ompl.sh" >&2
  exit 1
fi
if [[ "$(head -n 3 "${recorded}")" != "${expected}" ]]; then
  echo "error: the OMPL fork in ${prefix} was built from a different source or patch" >&2
  echo "installed: $(head -n 3 "${recorded}" | tr '\n' ' ')" >&2
  echo "expected:  $(echo "${expected}" | tr '\n' ' ')" >&2
  exit 1
fi

rm -rf "${build}"
mkdir -p "${build}/src"
cat > "${build}/src/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.20...3.31)
project(geodex_ompl_patch_probe LANGUAGES CXX)
find_package(ompl REQUIRED)
add_executable(ompl_patch_probe "${repo_root}/cmake/ompl_patch_probe.cpp")
target_compile_features(ompl_patch_probe PRIVATE cxx_std_17)
target_link_libraries(ompl_patch_probe PRIVATE ompl::ompl)
EOF
cmake -S "${build}/src" -B "${build}/out" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${prefix};${CONDA_PREFIX:-}" > /dev/null
cmake --build "${build}/out" > /dev/null
"${build}/out/ompl_patch_probe"
