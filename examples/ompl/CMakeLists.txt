add_executable(minimum_energy_planning minimum_energy_planning.cpp)
target_link_libraries(minimum_energy_planning PRIVATE geodex ompl::ompl)

add_executable(se2_tutorial se2_tutorial.cpp)
target_link_libraries(se2_tutorial PRIVATE geodex ompl::ompl)

install(TARGETS
  minimum_energy_planning
  se2_tutorial
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
