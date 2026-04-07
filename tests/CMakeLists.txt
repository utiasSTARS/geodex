include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(googletest)

add_executable(test_concepts test_concepts.cpp)
target_link_libraries(test_concepts PRIVATE geodex GTest::gtest_main)

add_executable(test_sphere test_sphere.cpp)
target_link_libraries(test_sphere PRIVATE geodex GTest::gtest_main)

add_executable(test_sphere_ndim test_sphere_ndim.cpp)
target_link_libraries(test_sphere_ndim PRIVATE geodex GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_concepts)
gtest_discover_tests(test_sphere)
gtest_discover_tests(test_sphere_ndim)

add_executable(test_interpolation test_interpolation.cpp)
target_link_libraries(test_interpolation PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_interpolation)

add_executable(test_euclidean test_euclidean.cpp)
target_link_libraries(test_euclidean PRIVATE geodex GTest::gtest_main)
target_include_directories(test_euclidean PRIVATE ${CMAKE_SOURCE_DIR}/tests/fixtures)
gtest_discover_tests(test_euclidean)

add_executable(test_torus test_torus.cpp)
target_link_libraries(test_torus PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_torus)

add_executable(test_se2 test_se2.cpp)
target_link_libraries(test_se2 PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_se2)

add_executable(test_metrics test_metrics.cpp)
target_link_libraries(test_metrics PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_metrics)

add_executable(test_configuration_space test_configuration_space.cpp)
target_link_libraries(test_configuration_space PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_configuration_space)

add_executable(test_sampler test_sampler.cpp)
target_link_libraries(test_sampler PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_sampler)

if(BUILD_OMPL_EXAMPLES)
  add_executable(test_geodex_state_space test_geodex_state_space.cpp)
  # GTest must come before ompl::ompl so FetchContent headers take priority
  # over any system-installed GTest (e.g. /opt/homebrew/include/gtest).
  target_link_libraries(test_geodex_state_space PRIVATE geodex GTest::gtest_main ompl::ompl)
  gtest_discover_tests(test_geodex_state_space)

  add_executable(test_path_resampling test_path_resampling.cpp)
  target_link_libraries(test_path_resampling PRIVATE geodex GTest::gtest_main ompl::ompl)
  gtest_discover_tests(test_path_resampling)
endif()
