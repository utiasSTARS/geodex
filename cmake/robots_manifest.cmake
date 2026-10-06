# Canonical built-in robot registry.
#
# The normal build compiles every robot in `GEODEX_ROBOT_NAMES`, and each listed robot must have
# committed generated sources under `src/robots/generated/`.
#
# `GEODEX_ROBOT_URDFS` is used only by the regeneration tooling.
# Keep both lists in the same alphabetical order.
set(GEODEX_ROBOT_NAMES
  baxter
  fr3_gripper
  husky_ur5e
  panda
  pr2
  ridgeback_ur5e
  stretch3
  stretch4
  ur5
)

set(GEODEX_ROBOT_URDFS
  "${CMAKE_CURRENT_LIST_DIR}/../data/robots/baxter/baxter.urdf"
  "${CMAKE_CURRENT_LIST_DIR}/../data/robots/fr3_gripper/fr3_arm_gripper_dynamics.urdf"
  "${CMAKE_CURRENT_LIST_DIR}/../data/robots/husky_ur5e/husky_ur5e_dynamics.urdf"
  "${CMAKE_CURRENT_LIST_DIR}/../data/robots/panda/urdf/panda.urdf"
  "${CMAKE_CURRENT_LIST_DIR}/../data/robots/pr2/pr2.urdf"
  "${CMAKE_CURRENT_LIST_DIR}/../data/robots/ridgeback_ur5e/ridgeback_ur5e_dynamics.urdf"
  "${CMAKE_CURRENT_LIST_DIR}/../data/robots/stretch3/stretch3_dynamics.urdf"
  "${CMAKE_CURRENT_LIST_DIR}/../data/robots/stretch4/stretch4_dynamics.urdf"
  "${CMAKE_CURRENT_LIST_DIR}/../data/robots/ur5/ur5.urdf"
)

# Public robot names that differ from the file prefix, as `prefix=name`. The public
# name is what geodex::robots::name() returns and what the VAMP registry accepts.
set(GEODEX_ROBOT_PUBLIC_NAMES
  fr3_gripper=fr3_arm_gripper
)

# Robots with a planar mobile base, as `prefix=drive` with drive `holonomic` or
# `differential`. Their generated CRBA covers the arm with the base held still, and
# their VAMP kernel takes the whole-body configuration (x, y, theta, arm...).
set(GEODEX_ROBOT_BASES
  stretch3=differential
  stretch4=holonomic
  ridgeback_ur5e=holonomic
  husky_ur5e=differential
)

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
# Every generated artifact below is reproducible with the pinned tools and
# inputs recorded here, by `scripts/robotgen/generate.sh <robot>` run in the
# environment of scripts/robotgen/pixi.toml and pixi.lock. Upstream descriptions
# are pinned by sha256 in third_party/robot_descriptions.cmake, per-robot choices live in
# scripts/robotgen/robots/<robot>.json, and data/robots/<robot>/README.md records
# the source and license of each robot. VAMP kernels come from CoMMALab/cricket at commit
# 2280986b5b0563f47365be04263ad8a79532790f with
# scripts/robotgen/patches/cricket-base-position-braces.patch, built with
# -DCRICKET_BUILD_JIT=OFF, and compile against KavrakiLab/vamp at
# e3902f1b77e504991b4facec536642f1c522c1cc. CRBA sources come from
# scripts/robotgen/pinocchio_codegen and the Loewner bounds from
# scripts/robotgen/precompute_robot_bound (seed 42). Spheres come from CoMMALab/foam at
# 116928f71aaa7c40356d79c84d3c9ff1f4497d90 (medial, depth 1).
#
# fr3_gripper (public name fr3_arm_gripper), fixed base
#   Upstream  franka_description 2.9.0 with ros2_robotiq_gripper a74d007 (Robotiq 2F-85
#             and coupling), the fingers held 40 mm apart
#   CRBA and bound source  data/robots/fr3_gripper/fr3_arm_gripper_dynamics.urdf
#   VAMP kernel  data/robots/fr3_gripper/fr3_arm_gripper_spherized.urdf and
#     fr3_arm_gripper.srdf
#
# stretch3, stretch4, ridgeback_ur5e, husky_ur5e (planar base, whole-body kernels)
#   Upstream  hello-robot-stretch-urdf 0.1.2, hello-robot-stretch4-urdf 2026.8.21,
#             clearpath_common 2.9.17 with Universal_Robots_ROS2_Description 4.3.1
#   CRBA and bound source  data/robots/<robot>/<robot>_dynamics.urdf
#   VAMP kernel  data/robots/<robot>/<robot>_spherized.urdf and <robot>.srdf
#   The CRBA covers the arm with the base held still. The Stretch arm of nested segments
#   is one coordinate with the segments as four mimic joints, and its CRBA is A^T M(A r) A.
