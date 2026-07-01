# Built-in robot dynamics (precompiled CRBA mass matrices).
#
# Defines the always-on `geodex_robots` STATIC target (alias `geodex::robots`)
# hosting the per-robot dispatcher `src/robots/mass_matrix.cpp` plus the
# CppAD::CG-generated sources under `src/robots/generated/`.
#
# The geodex INTERFACE target gains a transitive INTERFACE link to
# `geodex_robots`, so consumers only ever need to link `geodex` (alias
# `geodex::geodex`).
#
# This integration is INDEPENDENT of `GEODEX_PINOCCHIO`: the generated C
# source has no Pinocchio types and no Eigen types crossing the TU boundary;
# only `<math.h>` is included. Pinocchio + CppAD::CG are needed only when
# `GEODEX_ENABLE_ROBOT_REGEN=ON`, which registers maintainer targets for
# refreshing generated sources when a URDF changes.
#
# ---------------------------------------------------------------------------
# Adding a new robot
# ---------------------------------------------------------------------------
# Use `scripts/add_robot.sh --name <robot> --urdf <path>` to copy the URDF
# into `data/robots/`, regenerate sources, certify the Loewner mass-matrix
# lower bound, and update the manifest + public robot registry.

# ---------------------------------------------------------------------------
# Robot list. Normal builds use this manifest to locate committed generated
# sources. Regeneration tooling uses the same names/URDFs only when explicitly
# enabled.
# ---------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/robots_manifest.cmake)

option(GEODEX_ENABLE_ROBOT_REGEN
  "Register maintainer targets for regenerating built-in robot CRBA sources"
  OFF)

list(LENGTH GEODEX_ROBOT_NAMES _n_names)
list(LENGTH GEODEX_ROBOT_URDFS _n_urdfs)
if(_n_names EQUAL 0)
  message(FATAL_ERROR "GEODEX_ROBOT_NAMES must list at least one built-in robot.")
endif()
if(NOT _n_names EQUAL _n_urdfs)
  message(FATAL_ERROR
    "GEODEX_ROBOT_NAMES (${_n_names}) and "
    "GEODEX_ROBOT_URDFS (${_n_urdfs}) must be parallel lists of equal length.")
endif()

# ---------------------------------------------------------------------------
# Collect per-robot generated sources, abort if any is missing.
# ---------------------------------------------------------------------------
set(_robots_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/robots/mass_matrix.cpp)
set(_robots_generated_srcs "")     # generated TUs, for per-source flag-setting
set(_missing_srcs "")

math(EXPR _last "${_n_names} - 1")
foreach(_i RANGE 0 ${_last})
  list(GET GEODEX_ROBOT_NAMES ${_i} _robot)
  set(_src ${CMAKE_CURRENT_SOURCE_DIR}/src/robots/generated/${_robot}_crba.cpp)
  if(EXISTS ${_src})
    list(APPEND _robots_sources ${_src})
    list(APPEND _robots_generated_srcs ${_src})
  else()
    list(APPEND _missing_srcs "${_src}")
  endif()
endforeach()

if(_missing_srcs)
  if(GEODEX_ENABLE_ROBOT_REGEN)
    message(WARNING
      "geodex_robots: missing generated sources while "
      "GEODEX_ENABLE_ROBOT_REGEN=ON:\n"
      "  ${_missing_srcs}\n"
      "Only existing generated sources will be compiled. Build "
      "`pinocchio_codegen` or `regenerate_robots` to refresh the missing files.")
  else()
    message(FATAL_ERROR
      "geodex_robots: missing generated sources:\n"
      "  ${_missing_srcs}\n"
      "These files are committed artifacts for normal builds. To regenerate, "
      "configure with -DGEODEX_ENABLE_ROBOT_REGEN=ON and run "
      "`scripts/add_robot.sh` or the `regenerate_robots` target.")
  endif()
endif()

# ---------------------------------------------------------------------------
# Build the static archive.
# ---------------------------------------------------------------------------
add_library(geodex_robots STATIC ${_robots_sources})
add_library(geodex::robots ALIAS geodex_robots)
set_target_properties(geodex_robots PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(geodex_robots
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>
         # Public header `mass_matrix.hpp` includes the generated per-robot
         # constants from `generated/<robot>_crba.hpp` (constexpr nq, joint
         # limits, extern-C decl), so the generated dir's parent must be on
         # the consumer's include path under the `generated/` prefix.
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/robots>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/robots/generated)
target_compile_features(geodex_robots PUBLIC cxx_std_20)

# ---------------------------------------------------------------------------
# Compile flags (compiler-portable). PRIVATE so they never propagate to
# consumer translation units — preserves Eigen ABI parity with the rest of
# the project (and any system Pinocchio).
# ---------------------------------------------------------------------------
target_compile_options(geodex_robots PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3>
  $<$<CXX_COMPILER_ID:MSVC>:/O2>)

# Per-source aggressive flags applied ONLY to the auto-generated CRBA TUs.
# `-Ofast -ffast-math` enable FP reordering and FMA contraction in the
# straight-line CRBA expressions (worth several × on this TU). `-march=native`
# tunes for the build host; if you cross-compile or want a portable binary,
# override these via -DGEODEX_ROBOTS_TU_FLAGS.
option(GEODEX_ROBOTS_NATIVE_ARCH
  "Add -march=native to the generated robot CRBA translation units." ON)

set(GEODEX_ROBOTS_TU_FLAGS ""
    CACHE STRING "Override compile flags for the per-robot generated TUs (default: arch-tuned aggressive math)")

if(GEODEX_ROBOTS_TU_FLAGS)
  set(_tu_flags ${GEODEX_ROBOTS_TU_FLAGS})
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  set(_tu_flags
    -Wno-unused-variable -Wno-unused-but-set-variable -Wno-cast-function-type
    -Ofast -ffast-math -funroll-loops)
  if(NOT CMAKE_CROSSCOMPILING AND GEODEX_ROBOTS_NATIVE_ARCH)
    list(APPEND _tu_flags -march=native)
  endif()
elseif(MSVC)
  set(_tu_flags /O2 /fp:fast)
else()
  set(_tu_flags "")
  message(WARNING "Unknown compiler '${CMAKE_CXX_COMPILER_ID}' — falling back to "
                  "default -O3 only on the geodex_robots TUs.")
endif()

# Convert list to ;-separated string for COMPILE_OPTIONS property.
string(REPLACE ";" "$<SEMICOLON>" _tu_flags_prop "${_tu_flags}")
foreach(_src IN LISTS _robots_generated_srcs)
  set_source_files_properties(${_src} PROPERTIES COMPILE_OPTIONS "${_tu_flags_prop}")
endforeach()

# ---------------------------------------------------------------------------
# SIMD trig path selection.
#
# The generated source's vectorized-trig prelude has two implementations
# gated on `__APPLE__`:
#   * Apple (any arch): a single `vvsincos(sin_buf, cos_buf, in_buf, &n)`
#     call from Accelerate's vMathLib — NEON-vectorized on Apple Silicon,
#     SSE/AVX on Intel Macs. Linked PRIVATELY so the framework dependency
#     does not propagate to consumers.
#   * Everywhere else: two `for (...) sin/cos` loops that GCC/Clang
#     auto-vectorize into AVX2 calls to GLIBC's libmvec
#     (`_ZGVdN4v_sin`, `_ZGVdN4v_cos`, 4-wide double) on Linux x86_64.
#     On other Unix-likes the loops resolve to scalar `<math.h>` calls.
#
# TODO (Linux aarch64): glibc 2.37+ ships `_ZGVnN2v_sin` / `_ZGVnN2v_cos`
# for aarch64. When ready, add an `elseif(... PROCESSOR ... aarch64 ...)`
# arm that links `mvec` PUBLIC and verifies auto-vectorization picks up
# the 2-wide NEON variants. Sleef (https://sleef.org) is the cross-vendor
# alternative if libmvec coverage is insufficient.
# ---------------------------------------------------------------------------
if(APPLE)
  target_link_libraries(geodex_robots PRIVATE "-framework Accelerate")
  set(_robots_simd_status "Apple Accelerate vvsincos (NEON via vMathLib)")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux"
       AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$"
       AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_libraries(geodex_robots PUBLIC mvec)
  set(_robots_simd_status "libmvec auto-vectorized (Linux x86_64 GCC/Clang)")
else()
  set(_robots_simd_status "scalar sin/cos (no SIMD trig path on this platform)")
endif()

# ---------------------------------------------------------------------------
# Transitive link from geodex INTERFACE so consumers don't reference us
# explicitly.
# ---------------------------------------------------------------------------
target_link_libraries(geodex INTERFACE geodex_robots)
install(TARGETS geodex_robots EXPORT geodexTargets)

list(JOIN GEODEX_ROBOT_NAMES " " _robot_list_str)
message(STATUS "geodex_robots enabled (robots: ${_robot_list_str}; trig: ${_robots_simd_status})")

# ---------------------------------------------------------------------------
# Certify per-robot Loewner lower bounds for the CRBA
# kinetic-energy metric, written to src/robots/generated/<robot>_bound.hpp.
#
# Unlike the CRBA codegen below, this needs NO Pinocchio/CppAD: it links the
# already-compiled generated CRBA (geodex_robots) and runs the header-only
# precompute against the exact M(q) the planner evaluates.
# Gated on GEODEX_ENABLE_ROBOT_REGEN
#
# Usage:
#   cmake --build build --target regenerate_robot_bounds
#       Recompute every robot's bound.
#   cmake --build build --target regenerate_robot_bounds_<robot>
#       Recompute just one robot's bound.
# ---------------------------------------------------------------------------
if(GEODEX_ENABLE_ROBOT_REGEN)
  add_executable(precompute_robot_bound
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/precompute_robot_bound.cpp)
  target_link_libraries(precompute_robot_bound PRIVATE geodex geodex_robots)
  target_compile_features(precompute_robot_bound PRIVATE cxx_std_20)

  set(_robot_bound_targets "")
  foreach(_robot IN LISTS GEODEX_ROBOT_NAMES)
    add_custom_target(regenerate_robot_bounds_${_robot}
      COMMAND $<TARGET_FILE:precompute_robot_bound>
              ${_robot} ${CMAKE_CURRENT_SOURCE_DIR}/src/robots/generated
      DEPENDS precompute_robot_bound
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Certifying Loewner lower bound for '${_robot}'")
    list(APPEND _robot_bound_targets regenerate_robot_bounds_${_robot})
  endforeach()
  add_custom_target(regenerate_robot_bounds DEPENDS ${_robot_bound_targets})
endif()

# ---------------------------------------------------------------------------
# Maintainer regen targets (gated on Pinocchio + CppAD::CG availability).
# ---------------------------------------------------------------------------
if(GEODEX_ENABLE_ROBOT_REGEN AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/pinocchio_codegen.cmake)
  include(${CMAKE_CURRENT_SOURCE_DIR}/scripts/pinocchio_codegen.cmake)
endif()
