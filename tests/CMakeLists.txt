include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
  GIT_SHALLOW TRUE
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Canonical Panda / PR2 URDFs consumed by the Pinocchio and robots tests.
set(GEODEX_PANDA_URDF_PATH
    "${CMAKE_SOURCE_DIR}/data/robots/panda/urdf/panda.urdf")
set(GEODEX_PR2_URDF_PATH
    "${CMAKE_SOURCE_DIR}/data/robots/pr2/pr2.urdf")

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

add_executable(test_clearance_metric test_clearance_metric.cpp)
target_link_libraries(test_clearance_metric PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_clearance_metric)

add_executable(test_sampler test_sampler.cpp)
target_link_libraries(test_sampler PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_sampler)

add_executable(test_collision test_collision.cpp)
target_link_libraries(test_collision PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_collision)

add_executable(test_heuristics test_heuristics.cpp)
target_link_libraries(test_heuristics PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_heuristics)

add_executable(test_affine_combined_metric test_affine_combined_metric.cpp)
target_link_libraries(test_affine_combined_metric PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_affine_combined_metric)

add_executable(test_simplify_path test_simplify_path.cpp)
target_link_libraries(test_simplify_path PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_simplify_path)

add_executable(test_precompute_matrix_lower_bound test_precompute_matrix_lower_bound.cpp)
target_link_libraries(test_precompute_matrix_lower_bound PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_precompute_matrix_lower_bound)

# Admissibility of the precompiled per-robot Loewner bounds vs the generated CRBA.
# Links only geodex (geodex_robots comes transitively); no Pinocchio needed.
add_executable(test_robots_mass_lower_bound test_robots_mass_lower_bound.cpp)
target_link_libraries(test_robots_mass_lower_bound PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_robots_mass_lower_bound)

if(BUILD_OMPL_EXAMPLES)
  add_executable(test_geodex_state_space test_geodex_state_space.cpp)
  # GTest must come before ompl::ompl so FetchContent headers take priority
  # over any system-installed GTest (e.g. /opt/homebrew/include/gtest).
  target_link_libraries(test_geodex_state_space PRIVATE geodex GTest::gtest_main ompl::ompl)
  gtest_discover_tests(test_geodex_state_space)

  add_executable(test_path_resampling test_path_resampling.cpp)
  target_link_libraries(test_path_resampling PRIVATE geodex GTest::gtest_main ompl::ompl)
  gtest_discover_tests(test_path_resampling)

  add_executable(test_geodex_informed_sampling test_geodex_informed_sampling.cpp)
  target_link_libraries(test_geodex_informed_sampling PRIVATE geodex GTest::gtest_main ompl::ompl)
  gtest_discover_tests(test_geodex_informed_sampling)
endif()

if(GEODEX_PINOCCHIO)
  add_executable(test_pinocchio_mass_matrix test_pinocchio_mass_matrix.cpp)
  target_link_libraries(test_pinocchio_mass_matrix
    PRIVATE geodex geodex_pinocchio GTest::gtest_main)
  target_compile_definitions(test_pinocchio_mass_matrix PRIVATE
    GEODEX_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures"
    GEODEX_PANDA_URDF="${GEODEX_PANDA_URDF_PATH}")
  gtest_discover_tests(test_pinocchio_mass_matrix)

  add_executable(test_pinocchio_jacobian test_pinocchio_jacobian.cpp)
  target_link_libraries(test_pinocchio_jacobian
    PRIVATE geodex geodex_pinocchio GTest::gtest_main)
  target_compile_definitions(test_pinocchio_jacobian PRIVATE
    GEODEX_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures"
    GEODEX_PANDA_URDF="${GEODEX_PANDA_URDF_PATH}")
  gtest_discover_tests(test_pinocchio_jacobian)

  add_executable(test_pinocchio_pullback test_pinocchio_pullback.cpp)
  target_link_libraries(test_pinocchio_pullback
    PRIVATE geodex geodex_pinocchio GTest::gtest_main)
  target_compile_definitions(test_pinocchio_pullback PRIVATE
    GEODEX_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures"
    GEODEX_PANDA_URDF="${GEODEX_PANDA_URDF_PATH}")
  gtest_discover_tests(test_pinocchio_pullback)
endif()

if(GEODEX_PINOCCHIO)
  # Parity test for geodex::robots::MassMatrix uses pinocchio::crba as oracle.
  add_executable(test_robots_panda test_robots_panda.cpp)
  target_link_libraries(test_robots_panda
    PRIVATE geodex geodex_pinocchio GTest::gtest_main)
  target_compile_definitions(test_robots_panda PRIVATE
    GEODEX_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures"
    GEODEX_PANDA_URDF="${GEODEX_PANDA_URDF_PATH}"
    GEODEX_PR2_URDF="${GEODEX_PR2_URDF_PATH}")
  gtest_discover_tests(test_robots_panda)
endif()

if(GEODEX_VAMP)
  foreach(_robot panda ur5 baxter fetch pr2)
    add_executable(test_vamp_${_robot} test_vamp_${_robot}.cpp)
    target_link_libraries(test_vamp_${_robot} PRIVATE geodex GTest::gtest_main)
    target_compile_definitions(test_vamp_${_robot} PRIVATE
      GEODEX_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
    gtest_discover_tests(test_vamp_${_robot})
  endforeach()

  # Only test_vamp_panda needs the canonical Panda URDF.
  target_compile_definitions(test_vamp_panda PRIVATE
    GEODEX_PANDA_URDF="${GEODEX_PANDA_URDF_PATH}")

  if(GEODEX_PINOCCHIO)
    target_compile_definitions(test_vamp_panda PRIVATE GEODEX_TEST_HAS_PINOCCHIO)
  endif()
endif()
