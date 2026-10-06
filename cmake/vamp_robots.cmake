# Robot kernels compiled into the geodex_vamp archive, one translation unit each.
#
# Every name has a header include/geodex/integration/vamp/robots/<name>.hpp that includes
# the kernel and its sweep data and defines <name>_model. For each name, this
# file generates a source from src/integration/vamp/robot.cpp.in. It also writes the robot
# list that src/integration/vamp_impl.cpp builds its registry from. It sets
#   GEODEX_VAMP_ROBOT_SOURCES   the generated sources
#   GEODEX_VAMP_GENERATED_DIR   the directory holding them and the list

set(GEODEX_VAMP_ROBOTS
  baxter
  fr3_arm_gripper
  husky_ur5e
  panda
  pr2
  ridgeback_ur5e
  stretch3
  stretch4
  ur5
)

set(GEODEX_VAMP_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/geodex_vamp")
set(GEODEX_VAMP_ROBOT_SOURCES "")
set(_geodex_vamp_robot_list "")
foreach(GEODEX_VAMP_ROBOT IN LISTS GEODEX_VAMP_ROBOTS)
  set(_source "${GEODEX_VAMP_GENERATED_DIR}/robot_${GEODEX_VAMP_ROBOT}.cpp")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/integration/vamp/robot.cpp.in" "${_source}"
                 @ONLY)
  list(APPEND GEODEX_VAMP_ROBOT_SOURCES "${_source}")
  string(APPEND _geodex_vamp_robot_list "GEODEX_VAMP_ROBOT(${GEODEX_VAMP_ROBOT})\n")
endforeach()
file(CONFIGURE OUTPUT "${GEODEX_VAMP_GENERATED_DIR}/geodex_vamp_robots.inc"
     CONTENT "${_geodex_vamp_robot_list}")
