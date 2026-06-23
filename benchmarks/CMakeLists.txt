include(FetchContent)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.1
  GIT_SHALLOW TRUE
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_INSTALL_DOCS   OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(bench_manifold_ops bench_manifold_ops.cpp)
target_link_libraries(bench_manifold_ops PRIVATE geodex benchmark::benchmark_main)

add_executable(bench_algorithms bench_algorithms.cpp)
target_link_libraries(bench_algorithms PRIVATE geodex benchmark::benchmark_main)
target_include_directories(bench_algorithms PRIVATE ${CMAKE_SOURCE_DIR}/tests/fixtures)

add_executable(bench_metrics bench_metrics.cpp)
target_link_libraries(bench_metrics PRIVATE geodex benchmark::benchmark_main)
target_include_directories(bench_metrics PRIVATE ${CMAKE_SOURCE_DIR}/tests/fixtures)

add_executable(bench_retractions bench_retractions.cpp)
target_link_libraries(bench_retractions PRIVATE geodex benchmark::benchmark_main)

add_executable(bench_collision bench_collision.cpp)
target_link_libraries(bench_collision PRIVATE geodex benchmark::benchmark_main)

if(TARGET ompl::ompl)
  add_executable(bench_ompl_interpolation bench_ompl_interpolation.cpp)
  target_link_libraries(bench_ompl_interpolation PRIVATE geodex benchmark::benchmark_main ompl::ompl)
endif()

if(GEODEX_PINOCCHIO)
  # Comparison bench: geodex::robots vs pinocchio::crba on the Panda profile.
  # Pinocchio is needed only as the comparison reference; geodex_robots itself
  # is always built.
  add_executable(bench_robots bench_robots.cpp)
  target_link_libraries(bench_robots
    PRIVATE geodex geodex_pinocchio benchmark::benchmark)
  target_compile_definitions(bench_robots PRIVATE
    GEODEX_PANDA_URDF="${CMAKE_SOURCE_DIR}/data/robots/panda/urdf/panda.urdf")
endif()
