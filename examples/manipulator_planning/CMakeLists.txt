# Manipulator planning example (7-DoF Panda, MotionBenchMaker table-pick).
if(GEODEX_VAMP AND GEODEX_PINOCCHIO)
  add_executable(manipulator_planning manipulator_planning.cpp)
  target_link_libraries(manipulator_planning PRIVATE geodex ompl::ompl yaml-cpp::yaml-cpp)
  target_compile_definitions(manipulator_planning PRIVATE
    GEODEX_ROBOT_DATA_DIR="${CMAKE_SOURCE_DIR}/data/robots"
    GEODEX_DATASET_DATA_DIR="${CMAKE_SOURCE_DIR}/data/datasets")
  install(TARGETS manipulator_planning
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
