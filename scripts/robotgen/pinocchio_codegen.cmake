# Targets that regenerate the CRBA sources in src/robots/generated/ with Pinocchio and
# CppAD::CG and post-process them. They need GEODEX_ENABLE_ROBOT_REGEN=ON, with Pinocchio and
# CppAD::CG on the host. cmake/robots.cmake includes this file after it loads the robot list
# from cmake/robots_manifest.cmake.
#
#   cmake --build build --target regenerate_robots            # every robot
#   cmake --build build --target regenerate_robots_<robot>    # one robot

if(NOT GEODEX_ENABLE_ROBOT_REGEN)
  message(STATUS
    "regenerate_robots targets not registered "
    "(set GEODEX_ENABLE_ROBOT_REGEN=ON for robot codegen).")
  return()
endif()

# Only the codegen tool needs Pinocchio.
find_package(pinocchio QUIET)

# cppadcg does not install a CMake config. Find its headers on the prefix path or under
# CPPADCG_HOME.
find_path(CPPADCG_INCLUDE_DIR
  NAMES cppad/cg/cg.hpp
  HINTS
    $ENV{CPPADCG_HOME}/include
    $ENV{HOME}/.local/include
    /usr/local/include
    /usr/include
    /opt/homebrew/include
    /opt/local/include)

# Find CppAD, which cppadcg needs.
find_path(CPPAD_INCLUDE_DIR
  NAMES cppad/cppad.hpp
  HINTS
    $ENV{CPPAD_HOME}/include
    $ENV{HOME}/.local/include
    /usr/local/include
    /usr/include
    /opt/homebrew/include
    /opt/local/include)

find_library(CPPAD_LIB
  NAMES cppad_lib
  HINTS
    $ENV{CPPAD_HOME}/lib
    $ENV{HOME}/.local/lib
    /usr/local/lib
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /opt/homebrew/lib
    /opt/local/lib)

if(NOT pinocchio_FOUND OR NOT CPPADCG_INCLUDE_DIR OR NOT CPPAD_INCLUDE_DIR OR NOT CPPAD_LIB)
  message(STATUS
    "regenerate_robots targets not registered "
    "(pinocchio_FOUND=${pinocchio_FOUND} cppadcg=${CPPADCG_INCLUDE_DIR} "
    "cppad=${CPPAD_INCLUDE_DIR} cppad_lib=${CPPAD_LIB}). "
    "This is fine if you only need to compile the generated sources.")
  return()
endif()

add_executable(pinocchio_codegen
  ${CMAKE_CURRENT_SOURCE_DIR}/scripts/robotgen/pinocchio_codegen.cpp)
target_link_libraries(pinocchio_codegen
  PRIVATE pinocchio::pinocchio ${CPPAD_LIB} ${CMAKE_DL_LIBS})
target_include_directories(pinocchio_codegen
  PRIVATE ${CPPADCG_INCLUDE_DIR} ${CPPAD_INCLUDE_DIR})
target_compile_features(pinocchio_codegen PRIVATE cxx_std_20)
# Silence the warnings of the CppAD::CG headers.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(pinocchio_codegen PRIVATE
    -Wno-unused-parameter -Wno-unused-variable -Wno-deprecated-declarations
    -Wno-cast-function-type)
endif()
# Add the cppad library directory to the build RPATH.
get_filename_component(_cppad_libdir ${CPPAD_LIB} DIRECTORY)
set_target_properties(pinocchio_codegen PROPERTIES BUILD_RPATH "${_cppad_libdir}")

set(_robots_generated_dir ${CMAKE_CURRENT_SOURCE_DIR}/src/robots/generated)

# Python runs the post-processing step.
find_package(Python3 QUIET COMPONENTS Interpreter)

set(_post_simd_script
  ${CMAKE_CURRENT_SOURCE_DIR}/scripts/robotgen/post_process_sincos_simd.py)

if(NOT Python3_Interpreter_FOUND)
  message(STATUS
    "regenerate_robots: Python3 interpreter not found. The post-processing "
    "step (trig vectorization and noise drop) is skipped, and the raw CppAD::CG "
    "output stays as generated.")
endif()

# One target per robot and `regenerate_robots` for all of them.
set(_per_robot_targets "")
math(EXPR _last "${_n_names} - 1")
foreach(_i RANGE 0 ${_last})
  list(GET GEODEX_ROBOT_NAMES ${_i} _robot)
  list(GET GEODEX_ROBOT_URDFS ${_i} _urdf)

  set(_target_name regenerate_robots_${_robot})
  set(_generated_cpp ${_robots_generated_dir}/${_robot}_crba.cpp)

  set(_commands
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_robots_generated_dir}
    COMMAND $<TARGET_FILE:pinocchio_codegen>
            ${_urdf} ${_robot} ${_robots_generated_dir})

  # Post-process the source in place.
  if(Python3_Interpreter_FOUND)
    list(APPEND _commands
      COMMAND ${Python3_EXECUTABLE} ${_post_simd_script}
              ${_generated_cpp} ${_generated_cpp} ${_robot})
  endif()

  add_custom_target(${_target_name}
    ${_commands}
    DEPENDS pinocchio_codegen ${_urdf}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Regenerating CRBA C source for '${_robot}'")
  list(APPEND _per_robot_targets ${_target_name})
endforeach()

add_custom_target(regenerate_robots DEPENDS ${_per_robot_targets})
