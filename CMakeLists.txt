cmake_minimum_required(VERSION 3.20)
project(geodex VERSION 0.1.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FetchContent)

# Eigen3
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 5.0.0
  GIT_SHALLOW TRUE
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()

# Header-only interface library
add_library(geodex INTERFACE)
add_library(geodex::geodex ALIAS geodex)
target_include_directories(geodex INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(geodex INTERFACE cxx_std_20)

# Tests
option(BUILD_TESTING "Build tests" OFF)
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

# Examples
option(BUILD_EXAMPLES "Build examples" OFF)
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings via nanobind" OFF)
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

  find_program(SPHINX_EXECUTABLE sphinx-build
    HINTS ${CMAKE_SOURCE_DIR}/.venv/bin)
  if(SPHINX_EXECUTABLE)
    add_custom_target(sphinx
      COMMAND ${SPHINX_EXECUTABLE} -b html ${CMAKE_SOURCE_DIR}/docs ${CMAKE_BINARY_DIR}/docs/sphinx
      DEPENDS docs
      COMMENT "Building Sphinx documentation"
      VERBATIM)
  endif()
endif()

# Install
install(TARGETS geodex
  EXPORT geodexTargets
)

install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${eigen_SOURCE_DIR}/Eigen
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

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
