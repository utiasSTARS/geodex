cmake_minimum_required(VERSION 3.20)
project(geodex VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# Eigen3
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 5.0.0
  GIT_SHALLOW TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING_SAVE ${BUILD_TESTING})
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
  add_subdirectory(${eigen_SOURCE_DIR} ${eigen_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
set(BUILD_TESTING ${BUILD_TESTING_SAVE} CACHE BOOL "" FORCE)

# Header-only interface library
add_library(geodex INTERFACE)
target_include_directories(geodex INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(geodex INTERFACE Eigen3::Eigen)
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
