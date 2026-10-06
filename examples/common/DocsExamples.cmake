include_guard(GLOBAL)

# geodex_docs_example builds a documentation example under the name of its source file. With
# testing on, it also registers a ctest that runs the example and writes its JSON output.
#
#   geodex_docs_example(<target> <source> [OMPL] [ROBOTS] [VAMP] [ARGS <arg>...])
#
# OMPL, ROBOTS and VAMP link that part of geodex. The function skips the example when that
# part is not configured.
function(geodex_docs_example target source)
  cmake_parse_arguments(ARG "OMPL;ROBOTS;VAMP" "" "ARGS" ${ARGN})
  if(ARG_OMPL AND NOT GEODEX_OMPL)
    return()
  endif()
  if(ARG_ROBOTS AND NOT TARGET geodex_robots)
    return()
  endif()
  if(ARG_VAMP AND NOT TARGET geodex_vamp)
    return()
  endif()
  get_filename_component(stem "${source}" NAME_WE)
  add_executable(${target} "${source}")
  set_target_properties(${target} PROPERTIES OUTPUT_NAME "${stem}")
  target_link_libraries(${target} PRIVATE geodex)
  if(ARG_OMPL)
    target_link_libraries(${target} PRIVATE ompl::ompl)
  endif()
  if(ARG_ROBOTS)
    target_link_libraries(${target} PRIVATE geodex_robots)
  endif()
  if(ARG_VAMP)
    target_link_libraries(${target} PRIVATE geodex_vamp)
  endif()
  if(GEODEX_BUILD_TESTING)
    add_test(NAME docs_example.${target}
             COMMAND ${target} ${ARG_ARGS} --json "${CMAKE_CURRENT_BINARY_DIR}/${stem}.json"
             WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
endfunction()
