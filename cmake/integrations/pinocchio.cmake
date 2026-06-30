# Pinocchio integration (URDF-driven primitives).
#
# Defines the GEODEX_PINOCCHIO option and, when enabled, an INTERFACE target
# `geodex_pinocchio` (alias `geodex::pinocchio`) that propagates Pinocchio's
# include and link requirements. The integration itself is header-only under
# `include/geodex/integration/pinocchio/`; the geodex INTERFACE target gains
# a transitive INTERFACE link to `geodex_pinocchio`, so consumers only ever
# need to link `geodex` (alias `geodex::geodex`).

option(GEODEX_PINOCCHIO "Build Pinocchio integration" OFF)

if(GEODEX_PINOCCHIO)
  find_package(pinocchio REQUIRED)

  add_library(geodex_pinocchio INTERFACE)
  add_library(geodex::pinocchio ALIAS geodex_pinocchio)
  target_link_libraries(geodex_pinocchio INTERFACE pinocchio::pinocchio)

  # Make the integration follow the geodex target transitively so consumers
  # never need to reference geodex::pinocchio explicitly.
  target_link_libraries(geodex INTERFACE geodex_pinocchio)

  # Add to the geodex export set so install() knows about the transitive dep.
  install(TARGETS geodex_pinocchio EXPORT geodexTargets)

  message(STATUS "Pinocchio integration enabled (transitively linked via geodex)")
endif()
