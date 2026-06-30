# Canonical built-in robot registry.
#
# `GEODEX_ROBOT_NAMES` drives the normal build: each listed robot must have
# committed generated sources under `src/robots/generated/`.
#
# `GEODEX_ROBOT_URDFS` is used only by maintainer regeneration tooling.
# Keep both lists in the same order.
set(GEODEX_ROBOT_NAMES
  panda
  ur5
  fetch
  baxter
  pr2
)

set(GEODEX_ROBOT_URDFS
  "${CMAKE_CURRENT_LIST_DIR}/../data/robots/panda/urdf/panda.urdf"
  "${CMAKE_CURRENT_LIST_DIR}/../data/robots/ur5/ur5.urdf"
  "${CMAKE_CURRENT_LIST_DIR}/../data/robots/fetch/fetch.urdf"
  "${CMAKE_CURRENT_LIST_DIR}/../data/robots/baxter/baxter.urdf"
  "${CMAKE_CURRENT_LIST_DIR}/../data/robots/pr2/pr2.urdf"
)
