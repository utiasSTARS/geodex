# Checks the exported package of an installed geodex (component targets, Eigen modules, the
# linked OMPL fork, VAMP checkers and robot dynamics).
cmake_minimum_required(VERSION 3.22)
project(geodex_package_check LANGUAGES CXX)

find_package(geodex 1.0 CONFIG REQUIRED COMPONENTS ompl)
message(STATUS "geodex from ${geodex_DIR}, OMPL from ${ompl_DIR}")

# The installed targets have the same names as the build tree's aliases.
foreach(_part robots vamp)
  if(geodex_${_part}_FOUND AND NOT TARGET geodex::${_part})
    message(FATAL_ERROR "the installed geodex lacks the target geodex::${_part}")
  endif()
endforeach()

add_executable(package_check main.cpp)
target_compile_features(package_check PRIVATE cxx_std_20)
target_link_libraries(package_check PRIVATE geodex::geodex ompl::ompl)
if(geodex_vamp_FOUND)
  target_compile_definitions(package_check PRIVATE CHECK_HAS_VAMP)
endif()
if(geodex_robots_FOUND)
  target_compile_definitions(package_check PRIVATE CHECK_HAS_ROBOTS)
endif()
