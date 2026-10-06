#!/usr/bin/env bash
# Regenerate the built-in robot data from pinned sources, on Linux x86_64. With no robot
# names, it regenerates all of them. README.md lists the files it writes. For a robot built
# from an upstream description, it also writes a check report to $WORK/work/<robot>/verify.json.
#
#   pixi run --manifest-path scripts/robotgen/pixi.toml scripts/robotgen/generate.sh [robot...]
#
# third_party/*.cmake pins cricket, foam, VAMP and pdqsort by commit and the upstream
# descriptions by sha256. scripts/robotgen/pixi.lock pins the conda and PyPI packages.
#
# Environment
#   WORK     tools, caches and logs (default <repo>/build/robotgen)
#   JOBS     build parallelism (default nproc)
#   SAMPLES  self-collision samples (default 100000)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
WORK="${WORK:-${REPO}/build/robotgen}"
JOBS="${JOBS:-$(nproc)}"
SAMPLES="${SAMPLES:-100000}"
ROBOTS=("$@")
if [[ ${#ROBOTS[@]} -eq 0 ]]; then
  ROBOTS=(stretch3 stretch4 ridgeback_ur5e husky_ur5e fr3_arm_gripper panda ur5 baxter pr2)
fi

# Print the value of set(GEODEX_<name> "...") in third_party/*.cmake.
pin() {
  python "${HERE}/robotgen.py" pin "$1" --repo "${REPO}"
}
CRICKET_COMMIT="$(pin CRICKET_REF)"
FOAM_COMMIT="$(pin FOAM_REF)"
VAMP_COMMIT="$(pin VAMP_REF)"
PDQSORT_COMMIT="$(pin PDQSORT_URL | sed 's#.*/archive/\([0-9a-f]*\)\.tar\.gz#\1#')"

TOOLS="${WORK}/tools"
mkdir -p "${TOOLS}" "${WORK}/logs"
export CPM_SOURCE_CACHE="${WORK}/cpm-cache"

# Clone or update a git repository and check out a commit.
checkout() {  # url dir commit
  if [[ ! -d "$2/.git" ]]; then git clone --quiet "$1" "$2"; fi
  git -C "$2" fetch --quiet origin
  git -C "$2" checkout --quiet --force "$3"
  git -C "$2" submodule update --init --recursive --quiet
}

echo "[tools] cricket ${CRICKET_COMMIT}"
checkout "$(pin CRICKET_GIT)" "${TOOLS}/cricket" "${CRICKET_COMMIT}"
git -C "${TOOLS}/cricket" apply --reverse --check "${HERE}/patches/cricket-base-position-braces.patch" \
  2>/dev/null || git -C "${TOOLS}/cricket" apply "${HERE}/patches/cricket-base-position-braces.patch"
cmake -S "${TOOLS}/cricket" -B "${TOOLS}/cricket-build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCRICKET_BUILD_JIT=OFF -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" > "${WORK}/logs/cricket.log"
cmake --build "${TOOLS}/cricket-build" -j "${JOBS}" >> "${WORK}/logs/cricket.log"

echo "[tools] foam ${FOAM_COMMIT}"
checkout "$(pin FOAM_GIT)" "${TOOLS}/foam" "${FOAM_COMMIT}"
# foam's static libraries use LTO and need the LTO-aware archiver.
cmake -S "${TOOLS}/foam" -B "${TOOLS}/foam/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_AR="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc-ar" \
  -DCMAKE_RANLIB="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc-ranlib" > "${WORK}/logs/foam.log"
cmake --build "${TOOLS}/foam/build" -j "${JOBS}" >> "${WORK}/logs/foam.log"

echo "[tools] vamp ${VAMP_COMMIT} and pdqsort ${PDQSORT_COMMIT} (headers for verification)"
checkout https://github.com/KavrakiLab/vamp.git "${TOOLS}/vamp" "${VAMP_COMMIT}"
checkout https://github.com/orlp/pdqsort.git "${TOOLS}/pdqsort" "${PDQSORT_COMMIT}"

echo "[tools] geodex codegen and bound tools"
# Build the tools without -march=native and fast math. The bound tool links a CRBA built with
# -O2.
cmake -S "${REPO}" -B "${TOOLS}/geodex-build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGEODEX_OMPL=OFF -DGEODEX_PINOCCHIO=ON -DGEODEX_ENABLE_ROBOT_REGEN=ON -DBUILD_TESTING=OFF \
  -DGEODEX_ROBOTS_NATIVE_ARCH=OFF -DGEODEX_ROBOTS_TU_FLAGS=-O2 \
  -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" > "${WORK}/logs/geodex.log"
cmake --build "${TOOLS}/geodex-build" -j "${JOBS}" --target pinocchio_codegen >> "${WORK}/logs/geodex.log"

echo "[sources] fetch and check sha256"
python "${HERE}/robotgen.py" fetch --cache "${WORK}/cache" --repo "${REPO}"

# Print the value of a dotted key in scripts/robotgen/robots/<robot>.json.
recipe() {
  python "${HERE}/robotgen.py" get "$1" "$2"
}

for robot in "${ROBOTS[@]}"; do
  crba_name="$(recipe "${robot}" crba.name)"
  crba_name="${crba_name:-${robot}}"
  if [[ -n "$(recipe "${robot}" base_link)" ]]; then
    echo "[${robot}] model, spheres, self-collision pairs, VAMP kernel, CRBA"
    python "${HERE}/robotgen.py" build "${robot}" --cache "${WORK}/cache" --work "${WORK}/work" \
      --repo "${REPO}" --tools "${TOOLS}" --samples "${SAMPLES}"
  else
    if [[ -n "$(recipe "${robot}" srdf)" ]]; then
      echo "[${robot}] VAMP kernel from the vendored sphere model and SRDF"
      python "${HERE}/robotgen.py" kernel "${robot}" --repo "${REPO}" --tools "${TOOLS}" \
        --work "${WORK}/work"
    fi
    echo "[${robot}] CRBA from the vendored dynamics URDF"
    python "${HERE}/robotgen.py" crba "${robot}" --cache "${WORK}/cache" --work "${WORK}/work" \
      --repo "${REPO}" --tools "${TOOLS}"
  fi

  echo "[${robot}] sphere travel bounds and joint limits"
  python "${HERE}/robotgen.py" sweep "${robot}" --repo "${REPO}" --vamp "${TOOLS}/vamp"

  echo "[${robot}] Loewner bound, proved over the joint box"
  # Configure again. The bound tool then compiles a CRBA that this run wrote.
  cmake "${TOOLS}/geodex-build" >> "${WORK}/logs/geodex.log"
  cmake --build "${TOOLS}/geodex-build" -j "${JOBS}" --target precompute_robot_bound \
    >> "${WORK}/logs/geodex.log"
  "${TOOLS}/geodex-build/precompute_robot_bound" "${crba_name}" "${REPO}/src/robots/generated" \
    > "${WORK}/logs/${crba_name}_bound.log" 2>&1

  if [[ -n "$(recipe "${robot}" base_link)" ]]; then
    echo "[${robot}] verify against pinocchio"
    python "${HERE}/verify.py" "${robot}" --repo "${REPO}" --vamp "${TOOLS}/vamp" \
      --pdqsort "${TOOLS}/pdqsort" --work "${WORK}/work" --cache "${WORK}/cache" --samples 1000
  fi
done
echo "done; logs in ${WORK}/logs"
