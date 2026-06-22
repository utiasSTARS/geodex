# VAMP integration (SIMD-accelerated collision checking).
#
# Defines the GEODEX_VAMP option and, when enabled, a STATIC `geodex_vamp`
# target (alias `geodex::vamp`) that hosts the integration's only SIMD
# translation unit (`src/integration/vamp_impl.cpp`). The geodex INTERFACE
# target gains a transitive INTERFACE link to `geodex_vamp`, so consumers
# only ever need to link `geodex` (alias `geodex::geodex`) — the integration
# follows automatically when the project was built with -DGEODEX_VAMP=ON.
#
# All five supported robots (panda, ur5, fetch, baxter, pr2) are compiled into the
# archive; `make_vamp_checker(name, env)` dispatches by name at runtime.
#
# SIMD compile options (-mavx2/-mfma on x86_64; NEON-by-default on aarch64)
# are applied PRIVATELY to the static archive so they never leak to consumer
# translation units, preserving Eigen-alignment ABI parity with non-AVX
# Pinocchio.

option(GEODEX_VAMP "Build VAMP integration" OFF)

if(GEODEX_VAMP)
  if(NOT BUILD_OMPL_EXAMPLES)
    message(FATAL_ERROR
      "GEODEX_VAMP requires BUILD_OMPL_EXAMPLES=ON (depends on OMPL).")
  endif()
  if(NOT DEFINED VAMP_DIR)
    message(FATAL_ERROR "GEODEX_VAMP requires VAMP_DIR to be set to the VAMP source root.")
  endif()
  find_package(yaml-cpp REQUIRED CONFIG)

  # VAMP's CMake injects -march=native (and -mavx2 on x86) into
  # CMAKE_CXX_FLAGS globally. Save/restore around the add_subdirectory so
  # the rest of the project keeps the configured build flags; the SIMD
  # options for our impl TU are applied PRIVATEly to geodex_vamp below.
  set(_geodex_vamp_saved_cxx_flags "${CMAKE_CXX_FLAGS}")
  set(VAMP_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
  set(VAMP_BUILD_CPP_DEMO OFF CACHE BOOL "" FORCE)
  set(VAMP_BUILD_OMPL_DEMO OFF CACHE BOOL "" FORCE)
  add_subdirectory(${VAMP_DIR} ${CMAKE_BINARY_DIR}/vamp EXCLUDE_FROM_ALL)
  set(CMAKE_CXX_FLAGS "${_geodex_vamp_saved_cxx_flags}" CACHE STRING "" FORCE)

  add_library(geodex_vamp STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/integration/vamp_impl.cpp")
  add_library(geodex::vamp ALIAS geodex_vamp)
  set_target_properties(geodex_vamp PROPERTIES POSITION_INDEPENDENT_CODE ON)

  target_include_directories(geodex_vamp
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>)
  target_link_libraries(geodex_vamp
    PUBLIC vamp::vamp yaml-cpp::yaml-cpp ompl::ompl)

  # PRIVATE — SIMD flags apply only to vamp_impl.cpp, never propagate.
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    target_compile_options(geodex_vamp PRIVATE -mavx2 -mfma -Wno-ignored-attributes)
  elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR
         CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    target_compile_options(geodex_vamp PRIVATE -Wno-ignored-attributes)
  else()
    message(WARNING
      "GEODEX_VAMP enabled on unsupported CMAKE_SYSTEM_PROCESSOR "
      "'${CMAKE_SYSTEM_PROCESSOR}' — SIMD compile flags not configured. "
      "VAMP supports x86_64 (AVX2) and aarch64/arm64 (NEON).")
  endif()

  # Make the integration follow the geodex target transitively so consumers
  # never need to reference geodex::vamp explicitly.
  target_link_libraries(geodex INTERFACE geodex_vamp)

  # Add to the geodex export set so install() knows about the transitive dep.
  install(TARGETS geodex_vamp EXPORT geodexTargets)

  message(STATUS "VAMP integration enabled (transitively linked via geodex)")
endif()
