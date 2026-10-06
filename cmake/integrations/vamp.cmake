# VAMP integration (SIMD-accelerated collision checking).
#
# Defines the GEODEX_VAMP option. When it is ON, this file adds a STATIC `geodex_vamp`
# target (alias `geodex::vamp`) with the SIMD translation units of the integration,
# `src/integration/vamp_impl.cpp` and one per robot kernel listed in
# `cmake/vamp_robots.cmake`. The geodex INTERFACE target links `geodex_vamp` transitively.
# Consumers link only `geodex` (alias `geodex::geodex`).
#
# The archive holds every robot in `cmake/vamp_robots.cmake`, and
# `make_vamp_checker(name, env)` dispatches by name at runtime.
#
# The SIMD compile options (-mavx2 -mfma on x86_64, NEON by default on aarch64) are
# PRIVATE to the static archive and do not reach consumer translation units. Consumers
# keep the Eigen alignment ABI of a Pinocchio built without AVX.

option(GEODEX_VAMP "Build VAMP integration" OFF)

if(GEODEX_VAMP)
  if(NOT GEODEX_OMPL)
    message(FATAL_ERROR
      "GEODEX_VAMP requires GEODEX_OMPL=ON (depends on OMPL).")
  endif()
  include(${CMAKE_CURRENT_LIST_DIR}/../GeodexFetch.cmake)
  if(NOT VAMP_DIR)
    geodex_fetch(vamp "${CMAKE_BINARY_DIR}/_geodex_deps/src/vamp")
    set(VAMP_DIR "${CMAKE_BINARY_DIR}/_geodex_deps/src/vamp")
  endif()
  # Hand VAMP's configure step the pinned, hash-checked copies of CPM.cmake and its three
  # header libraries.
  geodex_download("${GEODEX_VAMP_CPM_URL}" "${GEODEX_VAMP_CPM_SHA256}"
                  "${CMAKE_BINARY_DIR}/vamp/cmake/CPM.cmake")
  foreach(_dep nigh pdqsort SIMDxorshift)
    string(TOLOWER "${_dep}" _dep_lower)
    geodex_fetch(${_dep_lower} "${CMAKE_BINARY_DIR}/_geodex_deps/src/${_dep_lower}")
    set(CPM_${_dep}_SOURCE "${CMAKE_BINARY_DIR}/_geodex_deps/src/${_dep_lower}")
  endforeach()
  find_package(yaml-cpp REQUIRED CONFIG)

  # VAMP's CMake adds -march=native (and -mavx2 on x86) to the global CMAKE_CXX_FLAGS.
  # Save and restore CMAKE_CXX_FLAGS around add_subdirectory to keep the configured flags
  # for the rest of the project. geodex_vamp gets its SIMD options as PRIVATE flags below.
  set(_geodex_vamp_saved_cxx_flags "${CMAKE_CXX_FLAGS}")
  set(VAMP_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
  set(VAMP_BUILD_CPP_DEMO OFF CACHE BOOL "" FORCE)
  set(VAMP_BUILD_OMPL_DEMO OFF CACHE BOOL "" FORCE)
  # Do not install VAMP. It is header-only and stays inside geodex_vamp.
  set(VAMP_INSTALL_CPP_LIBRARY OFF CACHE BOOL "" FORCE)
  add_subdirectory(${VAMP_DIR} ${CMAKE_BINARY_DIR}/vamp EXCLUDE_FROM_ALL)
  set(CMAKE_CXX_FLAGS "${_geodex_vamp_saved_cxx_flags}" CACHE STRING "" FORCE)

  include(${CMAKE_CURRENT_LIST_DIR}/../vamp_robots.cmake)
  add_library(geodex_vamp STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/integration/vamp_impl.cpp" ${GEODEX_VAMP_ROBOT_SOURCES})
  add_library(geodex::vamp ALIAS geodex_vamp)
  # Export the target as geodex::vamp, the same name as the alias.
  set_target_properties(geodex_vamp PROPERTIES EXPORT_NAME vamp)
  target_include_directories(geodex_vamp PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/integration/vamp" "${GEODEX_VAMP_GENERATED_DIR}")
  set_target_properties(geodex_vamp PROPERTIES POSITION_INDEPENDENT_CODE ON)

  target_include_directories(geodex_vamp
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>)
  # The public header exposes OMPL types only. VAMP stays inside the archive, and yaml-cpp
  # is a link-only dependency. Neither reaches consumer compiles.
  target_link_libraries(geodex_vamp
    PUBLIC ompl::ompl
    PRIVATE yaml-cpp::yaml-cpp $<BUILD_INTERFACE:vamp::vamp>)

  # The SIMD flags are PRIVATE to the archive's sources and do not propagate.
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    target_compile_options(geodex_vamp PRIVATE -mavx2 -mfma -Wno-ignored-attributes)
    # Keep the 16-byte Eigen fixed-size alignment of the rest of the program. AVX would
    # raise it to 32 bytes and misalign Eigen arguments from other translation units.
    target_compile_definitions(geodex_vamp PRIVATE EIGEN_MAX_STATIC_ALIGN_BYTES=16)
  elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR
         CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    target_compile_options(geodex_vamp PRIVATE -Wno-ignored-attributes)
    # VAMP's NEON code relies on implicit vector conversions that GCC 13 and later reject.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
      target_compile_options(geodex_vamp PRIVATE -flax-vector-conversions)
    endif()
  else()
    message(WARNING
      "GEODEX_VAMP enabled on unsupported CMAKE_SYSTEM_PROCESSOR "
      "'${CMAKE_SYSTEM_PROCESSOR}'. SIMD compile flags are not configured. "
      "VAMP supports x86_64 (AVX2) and aarch64/arm64 (NEON).")
  endif()

  # Link geodex_vamp through the geodex target. Consumers do not reference geodex::vamp.
  target_link_libraries(geodex INTERFACE geodex_vamp)

  # Add geodex_vamp to the geodex export set.
  install(TARGETS geodex_vamp EXPORT geodexTargets)

  message(STATUS "VAMP integration enabled (transitively linked via geodex)")
endif()
