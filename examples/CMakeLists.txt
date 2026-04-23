add_executable(sphere_basics sphere_basics.cpp)
target_link_libraries(sphere_basics PRIVATE geodex)

add_executable(sphere_distance sphere_distance.cpp)
target_link_libraries(sphere_distance PRIVATE geodex)

add_executable(sphere_interpolation sphere_interpolation.cpp)
target_link_libraries(sphere_interpolation PRIVATE geodex)

add_executable(minimum_energy_grid minimum_energy_grid.cpp)
target_link_libraries(minimum_energy_grid PRIVATE geodex)

install(TARGETS
  sphere_basics
  sphere_distance
  sphere_interpolation
  minimum_energy_grid
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
