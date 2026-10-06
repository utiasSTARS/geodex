# Download, verify, extract and patch a pinned third-party source.
#
# Project mode:
#   include(cmake/GeodexFetch.cmake)
#   geodex_fetch(<name> <dest>)          # name is a pin such as eigen, ompl or vamp
#
# Script mode:
#   cmake -DGEODEX_FETCH=<name> -DGEODEX_FETCH_DEST=<dest> [-DGEODEX_DOWNLOAD_DIR=<dir>]
#         -P cmake/GeodexFetch.cmake
#   cmake -DGEODEX_FETCH_PRINT_STAMP=<name> -P cmake/GeodexFetch.cmake   # prints the stamp
#
# Pins come from third_party/dependencies.cmake. Every archive and patch is checked against
# its SHA-256 before use. <dest> holds a stamp that names the exact inputs. A changed pin
# replaces the tree, and a matching stamp keeps it as is.

include_guard(GLOBAL)

set(_GEODEX_FETCH_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
include("${_GEODEX_FETCH_ROOT}/third_party/dependencies.cmake")

function(_geodex_fetch_check_sha256 file expected)
  file(SHA256 "${file}" _actual)
  if(NOT _actual STREQUAL expected)
    message(FATAL_ERROR
      "SHA-256 mismatch for ${file}\n  expected ${expected}\n  actual   ${_actual}")
  endif()
endfunction()

# Download one file to `dest` unless it is already there with the expected SHA-256. Try up
# to four times when the network fails or a transfer stalls for a minute.
function(geodex_download url sha256 dest)
  if(EXISTS "${dest}")
    file(SHA256 "${dest}" _existing)
    if(_existing STREQUAL sha256)
      return()
    endif()
  endif()
  message(STATUS "geodex_fetch: downloading ${url}")
  foreach(_attempt 1 2 3 4)
    file(DOWNLOAD "${url}" "${dest}.part" STATUS _status TLS_VERIFY ON
         INACTIVITY_TIMEOUT 60 TIMEOUT 1800)
    list(GET _status 0 _code)
    if(_code EQUAL 0)
      break()
    endif()
    list(GET _status 1 _msg)
    file(REMOVE "${dest}.part")
    if(_attempt EQUAL 4)
      message(FATAL_ERROR "geodex_fetch: download of ${url} failed: ${_msg}")
    endif()
    math(EXPR _wait "5 * ${_attempt}")
    message(STATUS "geodex_fetch: download failed (${_msg}), retrying in ${_wait} s")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep ${_wait})
  endforeach()
  _geodex_fetch_check_sha256("${dest}.part" "${sha256}")
  file(RENAME "${dest}.part" "${dest}")
endfunction()

# Set out_var to the stamp text of a fetched tree. It lists the archive URL, the archive hash
# and the patch hashes.
function(geodex_fetch_stamp name out_var)
  string(TOUPPER "${name}" _up)
  set(_url "${GEODEX_${_up}_URL}")
  set(_sha "${GEODEX_${_up}_SHA256}")
  if(NOT _url OR NOT _sha)
    message(FATAL_ERROR "geodex_fetch: no pinned source named '${name}'")
  endif()
  set(${out_var} "${_url}\n${_sha}\n${GEODEX_${_up}_PATCHES_SHA256}\n" PARENT_SCOPE)
endfunction()

function(geodex_fetch name dest)
  string(TOUPPER "${name}" _up)
  set(_url "${GEODEX_${_up}_URL}")
  set(_sha "${GEODEX_${_up}_SHA256}")
  set(_patches "${GEODEX_${_up}_PATCHES}")
  set(_patches_sha "${GEODEX_${_up}_PATCHES_SHA256}")
  geodex_fetch_stamp(${name} _stamp_text)
  list(LENGTH _patches _n_patches)
  list(LENGTH _patches_sha _n_patches_sha)
  if(NOT _n_patches EQUAL _n_patches_sha)
    message(FATAL_ERROR "geodex_fetch: ${name} lists ${_n_patches} patches "
                        "but ${_n_patches_sha} patch hashes")
  endif()

  set(_stamp "${dest}/.geodex-fetch-stamp")
  if(EXISTS "${_stamp}")
    file(READ "${_stamp}" _old)
    if(_old STREQUAL _stamp_text)
      message(STATUS "geodex_fetch: ${name} up to date in ${dest}")
      return()
    endif()
  endif()

  if(GEODEX_DOWNLOAD_DIR)
    set(_dl_dir "${GEODEX_DOWNLOAD_DIR}")
  elseif(CMAKE_BINARY_DIR AND NOT CMAKE_SCRIPT_MODE_FILE)
    set(_dl_dir "${CMAKE_BINARY_DIR}/_downloads")
  else()
    get_filename_component(_dl_dir "${dest}/../_downloads" ABSOLUTE)
  endif()
  string(REGEX REPLACE ".*/" "" _file "${_url}")
  set(_archive "${_dl_dir}/${name}-${_file}")

  geodex_download("${_url}" "${_sha}" "${_archive}")

  set(_tmp "${dest}.extract")
  file(REMOVE_RECURSE "${_tmp}" "${dest}")
  get_filename_component(_parent "${dest}" DIRECTORY)
  file(MAKE_DIRECTORY "${_parent}")
  file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${_tmp}")
  file(GLOB _top LIST_DIRECTORIES true "${_tmp}/*")
  list(LENGTH _top _n_top)
  if(NOT _n_top EQUAL 1)
    message(FATAL_ERROR "geodex_fetch: expected one top-level directory in ${_archive}")
  endif()
  file(RENAME "${_top}" "${dest}")
  file(REMOVE_RECURSE "${_tmp}")

  if(_n_patches GREATER 0)
    find_program(GEODEX_PATCH_EXECUTABLE patch REQUIRED)
    math(EXPR _last "${_n_patches} - 1")
    foreach(_i RANGE ${_last})
      list(GET _patches ${_i} _patch)
      list(GET _patches_sha ${_i} _patch_sha)
      _geodex_fetch_check_sha256("${_patch}" "${_patch_sha}")
      message(STATUS "geodex_fetch: applying ${_patch}")
      execute_process(
        COMMAND "${GEODEX_PATCH_EXECUTABLE}" -p1 --forward --batch -i "${_patch}"
        WORKING_DIRECTORY "${dest}"
        RESULT_VARIABLE _rc OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
      if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "geodex_fetch: ${_patch} does not apply to ${name}\n${_out}${_err}")
      endif()
    endforeach()
  endif()

  file(WRITE "${_stamp}" "${_stamp_text}")
  message(STATUS "geodex_fetch: ${name} ready in ${dest}")
endfunction()

if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE AND GEODEX_FETCH_PRINT_STAMP)
  geodex_fetch_stamp("${GEODEX_FETCH_PRINT_STAMP}" _text)
  execute_process(COMMAND "${CMAKE_COMMAND}" -E echo_append "${_text}")
elseif(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
  if(NOT GEODEX_FETCH OR NOT GEODEX_FETCH_DEST)
    message(FATAL_ERROR "usage: cmake -DGEODEX_FETCH=<name> -DGEODEX_FETCH_DEST=<dir> "
                        "-P cmake/GeodexFetch.cmake")
  endif()
  get_filename_component(_dest "${GEODEX_FETCH_DEST}" ABSOLUTE)
  geodex_fetch("${GEODEX_FETCH}" "${_dest}")
endif()
