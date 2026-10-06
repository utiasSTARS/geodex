# Build third-party dependencies from the pins in third_party/dependencies.cmake.
#
# GEODEX_BUILD_OMPL   Build the OMPL fork (static, position independent) against the Boost on
#                     the prefix path, link geodex against it, and install it in the geodex
#                     prefix. Consumers of the installed package find the fork there.
# GEODEX_BUNDLE_DEPS  Build Boost, yaml-cpp and the OMPL fork as static libraries with hidden
#                     visibility, and use portable instruction sets. The wheel and sdist
#                     builds use it for a self-contained Python module.
#
# Without either option geodex uses the OMPL fork found on CMAKE_PREFIX_PATH, which the
# pixi workspace installs with scripts/build_ompl.sh. The VAMP source is fetched from its
# pin whenever GEODEX_VAMP is ON and VAMP_DIR is unset (cmake/integrations/vamp.cmake).

option(GEODEX_BUILD_OMPL "Build the pinned OMPL fork and install it with geodex" OFF)
option(GEODEX_BUNDLE_DEPS
  "Build Boost, yaml-cpp and the OMPL fork for a self-contained, portable Python module" OFF)
option(GEODEX_CHECK_OMPL_FORK "Check at configure time that OMPL carries the geodex OMPL patch" ON)

include(${CMAKE_CURRENT_LIST_DIR}/GeodexOmplFork.cmake)

# Fail with a clear message when the OMPL found is not the patched fork. Call after
# find_package(ompl).
function(geodex_check_ompl_fork)
  if(NOT GEODEX_CHECK_OMPL_FORK OR CMAKE_CROSSCOMPILING)
    return()
  endif()
  set(_record "${ompl_DIR}/../geodex-ompl-source.txt")
  set(_key "${ompl_DIR}")
  if(EXISTS "${_record}")
    file(READ "${_record}" _content)
    string(APPEND _key "${_content}")
  endif()
  # Cache the hash of the record. A cache value cannot span several lines.
  string(SHA256 _key "${_key}")
  if(GEODEX_OMPL_FORK_CHECKED STREQUAL _key)
    return()
  endif()
  try_run(_run _compiled "${CMAKE_BINARY_DIR}/_geodex_ompl_probe"
    SOURCES "${PROJECT_SOURCE_DIR}/cmake/ompl_patch_probe.cpp"
    LINK_LIBRARIES ompl::ompl
    CXX_STANDARD 17
    COMPILE_OUTPUT_VARIABLE _compile_out
    RUN_OUTPUT_VARIABLE _run_out)
  if(NOT _compiled OR NOT _run EQUAL 0)
    message(FATAL_ERROR
      "The OMPL at ${ompl_DIR} is not the patched geodex fork.\n${_run_out}\n"
      "Build the pinned fork with scripts/build_ompl.sh <prefix>, or configure geodex with "
      "-DGEODEX_BUILD_OMPL=ON.")
  endif()
  set(GEODEX_OMPL_FORK_CHECKED "${_key}" CACHE INTERNAL "")
  message(STATUS "OMPL fork carries the geodex OMPL patch (${ompl_DIR})")
endfunction()

if(NOT GEODEX_BUILD_OMPL AND NOT GEODEX_BUNDLE_DEPS)
  return()
endif()

set(_geodex_deps "${CMAKE_BINARY_DIR}/_geodex_deps")
set(_geodex_deps_prefix "${_geodex_deps}/prefix")
if(NOT GEODEX_DOWNLOAD_DIR)
  set(GEODEX_DOWNLOAD_DIR "${_geodex_deps}/downloads")
endif()
cmake_host_system_information(RESULT _geodex_jobs QUERY NUMBER_OF_LOGICAL_CORES)

# Install the vendored Eigen. OMPL and VAMP compile against it.
set(_geodex_eigen_prefix "${_geodex_deps}/eigen")
geodex_install_eigen("${eigen_SOURCE_DIR}" "${_geodex_eigen_prefix}" _geodex_eigen_config)
if(NOT DEFINED Eigen3_DIR OR NOT Eigen3_DIR)
  set(Eigen3_DIR "${_geodex_eigen_prefix}/share/eigen3/cmake" CACHE PATH "" FORCE)
endif()

# Pass the geodex compiler, Eigen and macOS target settings to every dependency build.
set(_geodex_dep_args
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DEigen3_DIR=${_geodex_eigen_prefix}/share/eigen3/cmake
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5)
if(CMAKE_OSX_ARCHITECTURES)
  string(REPLACE ";" "\\;" _archs "${CMAKE_OSX_ARCHITECTURES}")
  list(APPEND _geodex_dep_args "-DCMAKE_OSX_ARCHITECTURES=${_archs}")
endif()
if(CMAKE_OSX_DEPLOYMENT_TARGET)
  list(APPEND _geodex_dep_args "-DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()

if(GEODEX_BUNDLE_DEPS)
  list(APPEND _geodex_dep_args
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_PREFIX_PATH=${_geodex_deps_prefix}
    -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
    -DBUILD_SHARED_LIBS=OFF)

  # Build static Boost serialization and program_options, the compiled components OMPL requires.
  geodex_fetch(boost "${_geodex_deps}/src/boost")
  file(READ "${_geodex_deps}/src/boost/.geodex-fetch-stamp" _boost_key)
  string(APPEND _boost_key
    "${CMAKE_CXX_COMPILER};${CMAKE_OSX_ARCHITECTURES};${CMAKE_OSX_DEPLOYMENT_TARGET}")
  set(_boost_done "${_geodex_deps_prefix}/share/geodex-boost-source.txt")
  set(_boost_old "")
  if(EXISTS "${_boost_done}")
    file(READ "${_boost_done}" _boost_old)
  endif()
  if(NOT _boost_old STREQUAL _boost_key)
    message(STATUS "geodex: building Boost into ${_geodex_deps_prefix}")
    set(_b2_work "${_geodex_deps}/build/boost")
    file(MAKE_DIRECTORY "${_b2_work}")
    if(APPLE OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      set(_b2_toolset clang)
    else()
      set(_b2_toolset gcc)
    endif()
    file(WRITE "${_b2_work}/user-config.jam" "using ${_b2_toolset} : : \"${CMAKE_CXX_COMPILER}\" ;\n")
    # Target flags for C, C++ and the link with the architectures, deployment target and SDK.
    set(_b2_target "")
    foreach(_arch IN LISTS CMAKE_OSX_ARCHITECTURES)
      string(APPEND _b2_target " -arch ${_arch}")
    endforeach()
    if(CMAKE_OSX_DEPLOYMENT_TARGET)
      string(APPEND _b2_target " -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
    endif()
    if(CMAKE_OSX_SYSROOT)
      string(APPEND _b2_target " -isysroot ${CMAKE_OSX_SYSROOT}")
    endif()
    set(_b2_cflags "-fPIC -fvisibility=hidden${_b2_target}")
    set(_b2_cxxflags "-fPIC -fvisibility=hidden -fvisibility-inlines-hidden -std=c++20${_b2_target}")
    set(_b2_linkflags "${_b2_target}")
    execute_process(
      COMMAND ./bootstrap.sh --with-toolset=${_b2_toolset}
              --with-libraries=serialization,program_options
      WORKING_DIRECTORY "${_geodex_deps}/src/boost"
      OUTPUT_FILE "${_b2_work}/bootstrap.log" ERROR_FILE "${_b2_work}/bootstrap.log"
      RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      file(READ "${_b2_work}/bootstrap.log" _log)
      message(FATAL_ERROR "geodex: Boost bootstrap failed\n${_log}")
    endif()
    execute_process(
      COMMAND ./b2 install -q -j${_geodex_jobs} --prefix=${_geodex_deps_prefix}
              --build-dir=${_b2_work}/b2 --user-config=${_b2_work}/user-config.jam
              toolset=${_b2_toolset} variant=release link=static runtime-link=shared
              threading=multi visibility=hidden "cflags=${_b2_cflags}"
              "cxxflags=${_b2_cxxflags}" "linkflags=${_b2_linkflags}"
              --with-serialization --with-program_options
      WORKING_DIRECTORY "${_geodex_deps}/src/boost"
      OUTPUT_FILE "${_b2_work}/b2.log" ERROR_FILE "${_b2_work}/b2.log"
      RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      file(READ "${_b2_work}/b2.log" _log)
      string(LENGTH "${_log}" _len)
      if(_len GREATER 20000)
        math(EXPR _from "${_len} - 20000")
        string(SUBSTRING "${_log}" ${_from} -1 _log)
      endif()
      message(FATAL_ERROR "geodex: Boost build failed\n${_log}")
    endif()
    file(WRITE "${_boost_done}" "${_boost_key}")
  endif()

  # Build yaml-cpp for VAMP scene loading.
  geodex_fetch(yaml_cpp "${_geodex_deps}/src/yaml-cpp")
  file(READ "${_geodex_deps}/src/yaml-cpp/.geodex-fetch-stamp" _yaml_key)
  string(SHA256 _yaml_key "${_yaml_key}${_geodex_dep_args}")
  set(_yaml_done "${_geodex_deps_prefix}/share/geodex-yaml-cpp-source.txt")
  set(_yaml_old "")
  if(EXISTS "${_yaml_done}")
    file(READ "${_yaml_done}" _yaml_old)
  endif()
  if(NOT _yaml_old STREQUAL _yaml_key)
    message(STATUS "geodex: building yaml-cpp into ${_geodex_deps_prefix}")
    set(_yaml_build "${_geodex_deps}/build/yaml-cpp")
    file(REMOVE_RECURSE "${_yaml_build}")
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -S "${_geodex_deps}/src/yaml-cpp" -B "${_yaml_build}"
              -G "${CMAKE_GENERATOR}" -DCMAKE_INSTALL_PREFIX=${_geodex_deps_prefix}
              -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_POSITION_INDEPENDENT_CODE=ON
              ${_geodex_dep_args} -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF
              -DYAML_CPP_BUILD_CONTRIB=OFF -DYAML_BUILD_SHARED_LIBS=OFF -DYAML_CPP_INSTALL=ON
              -DYAML_CPP_FORMAT_SOURCE=OFF
      OUTPUT_QUIET ERROR_VARIABLE _err RESULT_VARIABLE _rc)
    if(_rc EQUAL 0)
      execute_process(COMMAND "${CMAKE_COMMAND}" --build "${_yaml_build}" --parallel ${_geodex_jobs}
                      OUTPUT_QUIET ERROR_VARIABLE _err RESULT_VARIABLE _rc)
    endif()
    if(_rc EQUAL 0)
      execute_process(COMMAND "${CMAKE_COMMAND}" --install "${_yaml_build}"
                      OUTPUT_QUIET ERROR_VARIABLE _err RESULT_VARIABLE _rc)
    endif()
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "geodex: building yaml-cpp failed\n${_err}")
    endif()
    file(WRITE "${_yaml_done}" "${_yaml_key}")
  endif()

  # The prefix map keeps the build directory out of OMPL's log messages.
  geodex_build_ompl_fork("${_geodex_deps_prefix}" WORK_DIR "${_geodex_deps}/ompl"
    CMAKE_ARGS ${_geodex_dep_args}
      -DBoost_USE_STATIC_LIBS=ON -DBoost_NO_SYSTEM_PATHS=ON -DBOOST_ROOT=${_geodex_deps_prefix}
      "-DCMAKE_CXX_FLAGS=-ffile-prefix-map=${_geodex_deps}/ompl/src/=ompl/")

  set(Boost_USE_STATIC_LIBS ON)
  set(Boost_NO_SYSTEM_PATHS ON)
  set(BOOST_ROOT "${_geodex_deps_prefix}")
  set(Boost_DIR "${_geodex_deps_prefix}/lib/cmake/Boost-${GEODEX_BOOST_REF}" CACHE PATH "" FORCE)
  set(yaml-cpp_DIR "${_geodex_deps_prefix}/lib/cmake/yaml-cpp" CACHE PATH "" FORCE)

  # Use portable instruction sets. By default, VAMP and the generated robot dynamics tune
  # for the build machine.
  if(NOT DEFINED VAMP_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
      set(VAMP_ARCH "-mavx2 -mfma")
    elseif(APPLE)
      set(VAMP_ARCH "-mcpu=apple-m1")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
      set(VAMP_ARCH "-march=armv8-a -flax-vector-conversions")
    else()
      set(VAMP_ARCH "-march=armv8-a")
    endif()
  endif()
  set(VAMP_LTO OFF CACHE BOOL "" FORCE)
  set(GEODEX_ROBOTS_NATIVE_ARCH OFF CACHE BOOL "Add -march=native to the generated robot CRBA translation units.")
else()
  # GEODEX_BUILD_OMPL builds the fork against the Boost on the caller's prefix path.
  string(REPLACE ";" "\\;" _prefix_path "${CMAKE_PREFIX_PATH}")
  geodex_build_ompl_fork("${_geodex_deps_prefix}" WORK_DIR "${_geodex_deps}/ompl"
    CMAKE_ARGS ${_geodex_dep_args} -DCMAKE_PREFIX_PATH=${_prefix_path})
  install(DIRECTORY "${_geodex_deps_prefix}/"
    DESTINATION .
    USE_SOURCE_PERMISSIONS
    PATTERN "pkgconfig" EXCLUDE
    PATTERN "geodex-install-manifest.txt" EXCLUDE)
endif()

list(PREPEND CMAKE_PREFIX_PATH "${_geodex_deps_prefix}")
set(ompl_DIR "${_geodex_deps_prefix}/share/ompl/cmake" CACHE PATH "" FORCE)
