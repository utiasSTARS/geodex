# Pinocchio integration (URDF-based primitives).
#
# Defines the GEODEX_PINOCCHIO option. When it is ON, this file adds an INTERFACE target
# `geodex_pinocchio` (alias `geodex::pinocchio`) that propagates the include and link
# requirements of Pinocchio. The integration is header-only under
# `include/geodex/integration/pinocchio/`. The geodex INTERFACE target links
# `geodex_pinocchio` transitively. Consumers link only `geodex` (alias `geodex::geodex`).

option(GEODEX_PINOCCHIO "Build Pinocchio integration" OFF)

if(GEODEX_PINOCCHIO)
  find_package(pinocchio REQUIRED)

  add_library(geodex_pinocchio INTERFACE)
  add_library(geodex::pinocchio ALIAS geodex_pinocchio)
  set_target_properties(geodex_pinocchio PROPERTIES EXPORT_NAME pinocchio)
  target_link_libraries(geodex_pinocchio INTERFACE pinocchio::pinocchio)

  # Link geodex_pinocchio through the geodex target. Consumers do not reference
  # geodex::pinocchio.
  target_link_libraries(geodex INTERFACE geodex_pinocchio)

  # Add geodex_pinocchio to the geodex export set.
  install(TARGETS geodex_pinocchio EXPORT geodexTargets)

  message(STATUS "Pinocchio integration enabled (transitively linked via geodex)")
endif()
