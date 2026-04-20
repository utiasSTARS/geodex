add_executable(euclidean_planning euclidean_planning.cpp)
target_link_libraries(euclidean_planning PRIVATE geodex ompl::ompl)

add_executable(se2_planning se2_planning.cpp)
target_link_libraries(se2_planning PRIVATE geodex ompl::ompl)

add_executable(se2_willow_planning se2_willow_planning.cpp)
target_link_libraries(se2_willow_planning PRIVATE geodex ompl::ompl)

add_executable(cspace_rrt_star cspace_rrt_star.cpp)
target_link_libraries(cspace_rrt_star PRIVATE geodex ompl::ompl)

add_executable(minimum_energy_planning minimum_energy_planning.cpp)
target_link_libraries(minimum_energy_planning PRIVATE geodex ompl::ompl)

install(TARGETS
  euclidean_planning
  se2_planning
  se2_willow_planning
  cspace_rrt_star
  minimum_energy_planning
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
