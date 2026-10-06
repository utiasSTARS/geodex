# Built-in robot dynamics (precompiled CRBA mass matrices).
#
# Defines the `geodex_robots` STATIC target (alias `geodex::robots`) with the per-robot
# dispatcher `src/robots/mass_matrix.cpp` and the CppAD::CG-generated sources under
# `src/robots/generated/`.
#
# The geodex INTERFACE target links `geodex_robots` transitively. Consumers link only
# `geodex` (alias `geodex::geodex`).
#
# This integration does not depend on `GEODEX_PINOCCHIO`. Pinocchio and Eigen types do not
# cross the translation unit boundary of the generated C source, which includes only
# `<math.h>`. Pinocchio and CppAD::CG are needed only with `GEODEX_ENABLE_ROBOT_REGEN=ON`,
# which registers targets that refresh the generated sources after a URDF changes.
#
# ---------------------------------------------------------------------------
# Adding a new robot
# ---------------------------------------------------------------------------
# `scripts/robotgen/add_robot.sh --name <robot> --urdf <path>` copies the URDF into
# `data/robots/`, regenerates the sources, certifies the Loewner mass-matrix lower bound,
# and updates the manifest and the public robot registry.

# ---------------------------------------------------------------------------
# Robot list. Normal builds find the committed generated sources through this manifest.
# The regeneration tooling, when enabled, uses the same names and URDFs.
# ---------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/robots_manifest.cmake)

option(GEODEX_ENABLE_ROBOT_REGEN
  "Register targets for regenerating built-in robot CRBA sources"
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
# Collect the per-robot generated sources and report any missing one.
# ---------------------------------------------------------------------------
set(_robots_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/robots/mass_matrix.cpp)
set(_robots_generated_srcs "")     # generated TUs that get per-source flags
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
      "`scripts/robotgen/add_robot.sh` or the `regenerate_robots` target.")
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
         # The public header `mass_matrix.hpp` includes the per-robot constants
         # (constexpr nq, joint limits, extern-C declaration) as
         # `generated/<robot>_crba.hpp`. The parent of the generated directory
         # must be on the consumer's include path.
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/robots>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/robots/generated)
target_compile_features(geodex_robots PUBLIC cxx_std_20)

# ---------------------------------------------------------------------------
# Compile flags (compiler-portable). They are PRIVATE and do not reach consumer
# translation units, which keep the Eigen ABI of the rest of the project and of any
# system Pinocchio.
# ---------------------------------------------------------------------------
target_compile_options(geodex_robots PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3>
  $<$<CXX_COMPILER_ID:MSVC>:/O2>)

# Aggressive flags for the generated CRBA TUs only. `-Ofast -ffast-math` allow FP
# reordering and FMA contraction in the straight-line CRBA expressions, which run several
# times faster with them. `-march=native` tunes for the build host. For a cross-compile or
# a portable binary, override the flags with -DGEODEX_ROBOTS_TU_FLAGS.
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
  message(WARNING "Unknown compiler '${CMAKE_CXX_COMPILER_ID}'. The geodex_robots TUs "
                  "use the default -O3 only.")
endif()

# Escape the list separators for the COMPILE_OPTIONS property.
string(REPLACE ";" "$<SEMICOLON>" _tu_flags_prop "${_tu_flags}")
foreach(_src IN LISTS _robots_generated_srcs)
  set_source_files_properties(${_src} PROPERTIES COMPILE_OPTIONS "${_tu_flags_prop}")
endforeach()

# ---------------------------------------------------------------------------
# SIMD trig path selection.
#
# The vectorized-trig prelude of the generated source has two implementations,
# selected by `__APPLE__`.
#   * Apple (any arch) makes one `vvsincos(sin_buf, cos_buf, in_buf, &n)` call into
#     Accelerate's vMathLib, which uses NEON on Apple Silicon and SSE/AVX on Intel
#     Macs. The framework links PRIVATE and does not propagate to consumers.
#   * Elsewhere, GCC and Clang auto-vectorize two `for (...) sin/cos` loops into AVX2
#     calls to glibc's libmvec (`_ZGVdN4v_sin`, `_ZGVdN4v_cos`, 4-wide double) on
#     Linux x86_64. Other Unix-likes, aarch64 Linux among them, call the scalar
#     `<math.h>` functions.
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
# Link geodex_robots through the geodex INTERFACE target. Consumers do not reference it.
# ---------------------------------------------------------------------------
target_link_libraries(geodex INTERFACE geodex_robots)
install(TARGETS geodex_robots EXPORT geodexTargets)
# The top-level CMakeLists.txt installs the generated headers next to the public ones.

list(JOIN GEODEX_ROBOT_NAMES " " _robot_list_str)
message(STATUS "geodex_robots enabled (robots: ${_robot_list_str}; trig: ${_robots_simd_status})")

# ---------------------------------------------------------------------------
# Certify per-robot Loewner lower bounds for the CRBA kinetic-energy metric and write
# them to src/robots/generated/<robot>_bound.hpp. Requires GEODEX_ENABLE_ROBOT_REGEN.
#
# This step does not need Pinocchio or CppAD. It links the compiled generated CRBA
# (geodex_robots) and runs the header-only precompute against the exact M(q) the
# planner evaluates.
#
# Usage:
#   cmake --build build --target regenerate_robot_bounds
#       Recompute every robot's bound.
#   cmake --build build --target regenerate_robot_bounds_<robot>
#       Recompute just one robot's bound.
# ---------------------------------------------------------------------------
if(GEODEX_ENABLE_ROBOT_REGEN)
  # Compile every generated source again as a template over its scalar type. The certifier
  # evaluates each shipped CRBA in interval arithmetic.
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(_crba_template_dir ${CMAKE_CURRENT_BINARY_DIR}/crba_templates)
  set(_crba_templates "")
  set(_crba_includes "// Generated by cmake/robots.cmake. DO NOT EDIT.\n#pragma once\n")
  foreach(_robot IN LISTS GEODEX_ROBOT_NAMES)
    set(_src ${CMAKE_CURRENT_SOURCE_DIR}/src/robots/generated/${_robot}_crba.cpp)
    set(_out ${_crba_template_dir}/${_robot}_crba_template.hpp)
    add_custom_command(OUTPUT ${_out}
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/robotgen/crba_template.py
              ${_src} ${_out} ${_robot}
      DEPENDS ${_src} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/robotgen/crba_template.py
      COMMENT "CRBA template for '${_robot}'")
    list(APPEND _crba_templates ${_out})
    string(APPEND _crba_includes "#include \"${_robot}_crba_template.hpp\"\n")
  endforeach()
  file(CONFIGURE OUTPUT ${_crba_template_dir}/crba_templates.hpp CONTENT "${_crba_includes}")

  add_executable(precompute_robot_bound
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/robotgen/precompute_robot_bound.cpp ${_crba_templates})
  target_include_directories(precompute_robot_bound PRIVATE
    ${_crba_template_dir} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/robotgen)
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
# Maintainer regeneration targets, registered when Pinocchio and CppAD::CG are available.
# ---------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/scripts/robotgen/pinocchio_codegen.cmake)
