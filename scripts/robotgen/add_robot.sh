#!/usr/bin/env bash
# Add a robot from its URDF. The script writes its CRBA source, registers it and writes its
# Loewner bound. It needs Pinocchio and CppAD::CG.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/robotgen/add_robot.sh --name <robot> --urdf <path> [--build-dir <dir>] [--cmake-arg <arg> ...] [-- <cmake args>]

Example:
  scripts/robotgen/add_robot.sh --name ur5 --urdf /path/to/ur5.urdf \
    -- -Dpinocchio_DIR=/opt/homebrew/lib/cmake/pinocchio
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
build_dir="${repo_root}/build"
robot_name=""
urdf_path=""
cmake_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      robot_name="${2:-}"
      shift 2
      ;;
    --urdf)
      urdf_path="${2:-}"
      shift 2
      ;;
    --build-dir)
      build_dir="${2:-}"
      shift 2
      ;;
    --cmake-arg)
      cmake_args+=("${2:-}")
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      cmake_args+=("$@")
      break
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${robot_name}" || -z "${urdf_path}" ]]; then
  usage >&2
  exit 2
fi

if [[ ! "${robot_name}" =~ ^[a-z][a-z0-9_]*$ ]]; then
  echo "invalid robot name '${robot_name}'; use lowercase [a-z][a-z0-9_]*" >&2
  exit 2
fi

if [[ ! -f "${urdf_path}" ]]; then
  echo "URDF not found: ${urdf_path}" >&2
  exit 1
fi

mkdir -p "${repo_root}/data/robots/${robot_name}"
canonical_urdf="${repo_root}/data/robots/${robot_name}/${robot_name}.urdf"
if [[ "$(cd "$(dirname "${urdf_path}")" && pwd)/$(basename "${urdf_path}")" != "${canonical_urdf}" ]]; then
  cp "${urdf_path}" "${canonical_urdf}"
fi

cmake -S "${repo_root}" -B "${build_dir}" \
  -DGEODEX_ENABLE_ROBOT_REGEN=ON \
  "${cmake_args[@]}"

cmake --build "${build_dir}" --target pinocchio_codegen

codegen_bin=""
while IFS= read -r candidate; do
  if [[ -x "${candidate}" ]]; then
    codegen_bin="${candidate}"
    break
  fi
done < <(find "${build_dir}" -type f -name pinocchio_codegen)

if [[ -z "${codegen_bin}" ]]; then
  echo "could not locate built pinocchio_codegen under ${build_dir}" >&2
  exit 1
fi

generated_dir="${repo_root}/src/robots/generated"
mkdir -p "${generated_dir}"
"${codegen_bin}" "${canonical_urdf}" "${robot_name}" "${generated_dir}"

python3 "${script_dir}/post_process_sincos_simd.py" \
  "${generated_dir}/${robot_name}_crba.cpp" \
  "${generated_dir}/${robot_name}_crba.cpp" \
  "${robot_name}"

python3 "${script_dir}/update_robot_registry.py" \
  --repo-root "${repo_root}" \
  --name "${robot_name}" \
  --urdf "${canonical_urdf}"

# Reconfigure with the new robot, then build and run the bound tool. It writes
# src/robots/generated/<robot>_bound.hpp.
cmake -S "${repo_root}" -B "${build_dir}" \
  -DGEODEX_ENABLE_ROBOT_REGEN=ON \
  "${cmake_args[@]}"

cmake --build "${build_dir}" --target regenerate_robot_bounds_${robot_name}

echo "Added '${robot_name}'. Generated sources are in src/robots/generated/."
