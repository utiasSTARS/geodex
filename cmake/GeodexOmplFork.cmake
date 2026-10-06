# Build the pinned OMPL fork as a static, position-independent library with hidden
# visibility into a prefix.
#
# Project mode:
#   include(cmake/GeodexOmplFork.cmake)
#   geodex_build_ompl_fork(<prefix> [CMAKE_ARGS <arg>...])
#
# Script mode (used by scripts/build_ompl.sh):
#   cmake -DGEODEX_OMPL_PREFIX=<prefix> [-DGEODEX_DOWNLOAD_DIR=<dir>] -P cmake/GeodexOmplFork.cmake
#
# The prefix records the source and options it was built from in
# share/ompl/geodex-ompl-source.txt. A matching record makes the call a no-op. A different
# one removes the files of the previous installation before the new one is installed.

include_guard(GLOBAL)
include(${CMAKE_CURRENT_LIST_DIR}/GeodexFetch.cmake)

# Build with hidden visibility. The fork's symbols stay out of the dynamic symbol table of a
# shared library that links it and cannot clash with a stock OMPL in the same process.
set(GEODEX_OMPL_FORK_ARGS
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_LIBDIR=lib
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DCMAKE_C_VISIBILITY_PRESET=hidden
  -DCMAKE_CXX_VISIBILITY_PRESET=hidden
  -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
  -DOMPL_BUILD_SHARED=OFF
  -DOMPL_BUILD_DEMOS=OFF
  -DOMPL_BUILD_TESTS=OFF
  -DOMPL_BUILD_PYTHON_BINDINGS=OFF
  -DOMPL_BUILD_VAMP=OFF
  -DOMPL_SKIP_RPATH=ON
  -DCMAKE_DISABLE_FIND_PACKAGE_flann=ON
  -DCMAKE_DISABLE_FIND_PACKAGE_spot=ON
  -DCMAKE_DISABLE_FIND_PACKAGE_Triangle=ON
  -DCMAKE_DISABLE_FIND_PACKAGE_Doxygen=ON
  -DCMAKE_DISABLE_FIND_PACKAGE_Python=ON)

# Install the Eigen headers and CMake config from `source` into `prefix`, once per source,
# and return the config directory in `out_var`. The fork and VAMP builds compile against
# this Eigen.
function(geodex_install_eigen source prefix out_var)
  set(_config "${prefix}/share/eigen3/cmake")
  set(_record "${prefix}/share/eigen3/geodex-eigen-source.txt")
  file(READ "${source}/.geodex-fetch-stamp" _wanted)
  set(_have "")
  if(EXISTS "${_record}")
    file(READ "${_record}" _have)
  endif()
  if(NOT _have STREQUAL _wanted OR NOT EXISTS "${_config}/Eigen3Config.cmake")
    file(REMOVE_RECURSE "${prefix}" "${prefix}-build")
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -S "${source}" -B "${prefix}-build"
              -DCMAKE_INSTALL_PREFIX=${prefix} -DBUILD_TESTING=OFF
              -DEIGEN_BUILD_TESTING=OFF -DEIGEN_BUILD_DOC=OFF -DEIGEN_BUILD_BLAS=OFF
              -DEIGEN_BUILD_LAPACK=OFF -DEIGEN_BUILD_PKGCONFIG=OFF
      OUTPUT_QUIET ERROR_VARIABLE _err RESULT_VARIABLE _rc)
    if(_rc EQUAL 0)
      execute_process(COMMAND "${CMAKE_COMMAND}" --install "${prefix}-build"
                      OUTPUT_QUIET ERROR_VARIABLE _err RESULT_VARIABLE _rc)
    endif()
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "geodex: installing the vendored Eigen into ${prefix} failed\n${_err}")
    endif()
    file(REMOVE_RECURSE "${prefix}-build")
    file(WRITE "${_record}" "${_wanted}")
  endif()
  set(${out_var} "${_config}" PARENT_SCOPE)
endfunction()

function(geodex_build_ompl_fork prefix)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "" "WORK_DIR" "CMAKE_ARGS")
  if(ARG_WORK_DIR)
    set(_work "${ARG_WORK_DIR}")
  else()
    set(_work "${prefix}/../ompl-fork-work")
  endif()
  get_filename_component(_work "${_work}" ABSOLUTE)
  set(_src "${_work}/src")
  set(_build "${_work}/build")
  set(_record "${prefix}/share/ompl/geodex-ompl-source.txt")
  set(_manifest "${prefix}/share/ompl/geodex-install-manifest.txt")
  if(NOT GEODEX_DOWNLOAD_DIR)
    set(GEODEX_DOWNLOAD_DIR "${_work}/downloads")
  endif()

  geodex_fetch(ompl "${_src}")
  file(READ "${_src}/.geodex-fetch-stamp" _stamp)
  set(_args ${GEODEX_OMPL_FORK_ARGS} ${ARG_CMAKE_ARGS})
  list(JOIN _args "\n" _args_text)
  set(_wanted "${_stamp}${_args_text}\n")

  if(EXISTS "${_record}")
    file(READ "${_record}" _have)
    if(_have STREQUAL _wanted)
      message(STATUS "geodex: OMPL fork up to date in ${prefix}")
      return()
    endif()
  endif()

  if(EXISTS "${_manifest}")
    message(STATUS "geodex: removing the previous OMPL fork installation from ${prefix}")
    file(STRINGS "${_manifest}" _old_files)
    file(REMOVE ${_old_files} "${_manifest}" "${_record}")
  endif()

  message(STATUS "geodex: building the OMPL fork into ${prefix}")
  file(REMOVE_RECURSE "${_build}")
  set(_generator)
  if(CMAKE_GENERATOR)
    set(_generator -G "${CMAKE_GENERATOR}")
  endif()
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${_src}" -B "${_build}" ${_generator}
            -DCMAKE_INSTALL_PREFIX=${prefix} ${_args}
    OUTPUT_FILE "${_work}/configure.log" ERROR_FILE "${_work}/configure.log"
    RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    file(READ "${_work}/configure.log" _log)
    message(FATAL_ERROR "geodex: configuring the OMPL fork failed\n${_log}")
  endif()
  cmake_host_system_information(RESULT _jobs QUERY NUMBER_OF_LOGICAL_CORES)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${_build}" --parallel ${_jobs}
    OUTPUT_FILE "${_work}/build.log" ERROR_FILE "${_work}/build.log"
    RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    file(READ "${_work}/build.log" _log)
    string(LENGTH "${_log}" _len)
    if(_len GREATER 20000)
      math(EXPR _from "${_len} - 20000")
      string(SUBSTRING "${_log}" ${_from} -1 _log)
    endif()
    message(FATAL_ERROR "geodex: building the OMPL fork failed\n${_log}")
  endif()
  execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${_build}"
    OUTPUT_QUIET ERROR_VARIABLE _err RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "geodex: installing the OMPL fork failed\n${_err}")
  endif()

  # Touch the installed files. CMake installs them with their source timestamps, and every
  # target compiled against the previous fork must rebuild.
  file(STRINGS "${_build}/install_manifest.txt" _installed_files)
  file(TOUCH_NOCREATE ${_installed_files})
  file(READ "${_build}/install_manifest.txt" _installed)
  file(WRITE "${_manifest}" "${_installed}")
  file(WRITE "${_record}" "${_wanted}")
  message(STATUS "geodex: OMPL fork installed in ${prefix}")
endfunction()

if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
  if(NOT GEODEX_OMPL_PREFIX)
    message(FATAL_ERROR
      "usage: cmake -DGEODEX_OMPL_PREFIX=<prefix> -P cmake/GeodexOmplFork.cmake")
  endif()
  get_filename_component(_prefix "${GEODEX_OMPL_PREFIX}" ABSOLUTE)
  set(_extra)
  if(GEODEX_OMPL_CMAKE_ARGS)
    set(_extra CMAKE_ARGS ${GEODEX_OMPL_CMAKE_ARGS})
  endif()
  if(GEODEX_OMPL_WORK_DIR)
    get_filename_component(_work "${GEODEX_OMPL_WORK_DIR}" ABSOLUTE)
  else()
    get_filename_component(_work "${_prefix}/../ompl-fork-work" ABSOLUTE)
  endif()
  # The fork compiles against the Eigen geodex vendors, not one found on the prefix path.
  geodex_fetch(eigen "${_work}/eigen-src")
  geodex_install_eigen("${_work}/eigen-src" "${_work}/eigen" _eigen_config)
  if(NOT _extra)
    set(_extra CMAKE_ARGS)
  endif()
  list(APPEND _extra "-DEigen3_DIR=${_eigen_config}")
  geodex_build_ompl_fork("${_prefix}" WORK_DIR "${_work}" ${_extra})
endif()
