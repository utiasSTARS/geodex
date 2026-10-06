cmake_minimum_required(VERSION 3.20...3.31)
project(geodex VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Turn off the C++20 module dependency scan. geodex does not use C++ modules.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FetchContent)

# Find Threads once with the -pthread preference that GoogleTest sets. Every configure of a
# build tree then produces the same flags.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Find Python before the integrations when building bindings. VAMP calls
# find_package(Python) with a different component set, and Python::Module must exist first.
option(BUILD_PYTHON_BINDINGS "Build Python bindings via nanobind" OFF)
if(BUILD_PYTHON_BINDINGS)
  find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development Development.SABIModule)
endif()

# Pinned downloads from third_party/dependencies.cmake share one cache per build tree.
include(cmake/GeodexFetch.cmake)
if(NOT GEODEX_DOWNLOAD_DIR)
  set(GEODEX_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/_geodex_deps/downloads")
endif()

# Fetch the pinned Eigen 5.0.0. geodex installs it with its own headers.
set(eigen_SOURCE_DIR "${CMAKE_BINARY_DIR}/_geodex_deps/src/eigen")
geodex_fetch(eigen "${eigen_SOURCE_DIR}")

# Build third-party dependencies from pinned sources for GEODEX_BUILD_OMPL or
# GEODEX_BUNDLE_DEPS.
include(cmake/GeodexDependencies.cmake)

# Header-only interface library
add_library(geodex INTERFACE)
add_library(geodex::geodex ALIAS geodex)
target_include_directories(geodex INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(geodex INTERFACE cxx_std_20)

# OMPL integration with the state space, the planner facade and their tests.
# BUILD_OMPL_EXAMPLES turns it on together with the examples.
option(BUILD_OMPL_EXAMPLES "Build the examples with the planning ones (requires OMPL)" OFF)
option(GEODEX_OMPL "Build OMPL integration and its tests (requires OMPL)" OFF)
if(BUILD_OMPL_EXAMPLES)
  set(GEODEX_OMPL ON CACHE BOOL "" FORCE)
endif()
if(GEODEX_OMPL)
  find_package(ompl REQUIRED)
  geodex_check_ompl_fork()
endif()

# Robot dynamics (precompiled CRBA mass matrices)
option(GEODEX_ROBOTS "Build the built-in robot dynamics library" ON)
if(GEODEX_ROBOTS)
  include(cmake/robots.cmake)
  # Export the target as geodex::robots, the same name as the build-tree alias.
  set_target_properties(geodex_robots PROPERTIES EXPORT_NAME robots)
endif()

# Optional integrations
include(cmake/integrations/pinocchio.cmake)
include(cmake/integrations/vamp.cmake)

# Tests
option(BUILD_TESTING "Build tests" OFF)
option(GEODEX_BUILD_TESTING "Build geodex tests" ${BUILD_TESTING})
if(GEODEX_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

# Examples. Those that plan need GEODEX_OMPL.
option(BUILD_EXAMPLES "Build examples" OFF)
if(BUILD_EXAMPLES OR BUILD_OMPL_EXAMPLES)
  add_subdirectory(examples)
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
  add_subdirectory(python)
endif()

# Documentation
option(BUILD_DOCS "Build documentation" OFF)
if(BUILD_DOCS)
  find_package(Doxygen REQUIRED)
  configure_file(docs/Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/docs/doxygen)
  add_custom_target(docs
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)
endif()

# Install. Downstream CMake projects use the exported package through
# find_package(geodex CONFIG REQUIRED [COMPONENTS ompl vamp robots pinocchio]).
install(TARGETS geodex
  EXPORT geodexTargets
)

install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${eigen_SOURCE_DIR}/Eigen
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install the unsupported Eigen modules too. Consumers and an OMPL built against this prefix
# see one complete Eigen 5.0.0.
install(DIRECTORY ${eigen_SOURCE_DIR}/unsupported/Eigen
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/unsupported
)

# Install the generated per-robot headers next to the public robot headers. The public
# headers include them by a relative path.
if(TARGET geodex_robots)
  install(DIRECTORY src/robots/generated/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/geodex/robots/generated
    FILES_MATCHING PATTERN "*.hpp"
  )
endif()

install(EXPORT geodexTargets
  FILE geodexTargets.cmake
  NAMESPACE geodex::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/geodex
)

configure_package_config_file(
  cmake/geodexConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/geodexConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/geodex
)

# Releases with the same major version are compatible.
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/geodexConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/geodexConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/geodexConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/geodex
)
