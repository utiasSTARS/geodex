include(FetchContent)

FetchContent_Declare(
  googletest
  URL "${GEODEX_GOOGLETEST_URL}"
  URL_HASH "SHA256=${GEODEX_GOOGLETEST_SHA256}"
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# The Panda URDF that the manipulator planning example, the regenerate_robots target
# and these tests share.
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

# Discover tests at ctest time. POST_BUILD discovery runs each binary with a 5 s
# timeout, and a busy parallel build can exceed it.
set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)

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

# Lie groups and product manifolds
add_executable(test_lie test_lie.cpp)
target_link_libraries(test_lie PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_lie)

add_executable(test_so2 test_so2.cpp)
target_link_libraries(test_so2 PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_so2)

add_executable(test_so3 test_so3.cpp)
target_link_libraries(test_so3 PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_so3)

add_executable(test_se3 test_se3.cpp)
target_link_libraries(test_se3 PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_se3)

add_executable(test_se2_right test_se2_right.cpp)
target_link_libraries(test_se2_right PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_se2_right)

add_executable(test_product test_product.cpp)
target_link_libraries(test_product PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_product)

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

add_executable(test_path_smoothing test_path_smoothing.cpp)
target_link_libraries(test_path_smoothing PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_path_smoothing)

# Compares the type-erased manifolds of the Python bindings with the typed manifolds they
# mirror. The wrappers under python/src/wrappers are plain C++.
add_executable(test_dynamic_agreement test_dynamic_agreement.cpp)
target_link_libraries(test_dynamic_agreement PRIVATE geodex GTest::gtest_main)
target_include_directories(test_dynamic_agreement PRIVATE ${CMAKE_SOURCE_DIR}/python/src)
if(GEODEX_ROBOTS)
  target_compile_definitions(test_dynamic_agreement PRIVATE GEODEX_TEST_ROBOTS)
endif()
if(GEODEX_OMPL)
  target_link_libraries(test_dynamic_agreement PRIVATE ompl::ompl)
  target_compile_definitions(test_dynamic_agreement PRIVATE GEODEX_TEST_OMPL)
endif()
gtest_discover_tests(test_dynamic_agreement)

add_executable(test_affine_combined_metric test_affine_combined_metric.cpp)
target_link_libraries(test_affine_combined_metric PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_affine_combined_metric)

add_executable(test_precompute_matrix_lower_bound test_precompute_matrix_lower_bound.cpp)
target_link_libraries(test_precompute_matrix_lower_bound PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_precompute_matrix_lower_bound)

# Checks the precompiled per-robot Loewner bounds against the generated CRBA. The
# test links only geodex and does not need Pinocchio.
add_executable(test_robots_mass_lower_bound test_robots_mass_lower_bound.cpp)
target_link_libraries(test_robots_mass_lower_bound PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_robots_mass_lower_bound)

add_executable(test_robots_registry test_robots_registry.cpp)
target_link_libraries(test_robots_registry PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_robots_registry)

add_executable(test_robots_whole_body test_robots_whole_body.cpp)
target_link_libraries(test_robots_whole_body PRIVATE geodex GTest::gtest_main)
gtest_discover_tests(test_robots_whole_body)

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
  # Checks geodex_robots::MassMatrix against pinocchio::crba. Only this oracle needs
  # Pinocchio.
  add_executable(test_robots_panda test_robots_panda.cpp)
  target_link_libraries(test_robots_panda
    PRIVATE geodex geodex_pinocchio GTest::gtest_main)
  target_compile_definitions(test_robots_panda PRIVATE
    GEODEX_PANDA_URDF="${GEODEX_PANDA_URDF_PATH}"
    GEODEX_PR2_URDF="${GEODEX_PR2_URDF_PATH}")
  gtest_discover_tests(test_robots_panda)
endif()

if(GEODEX_OMPL)
  add_executable(test_geodex_state_space test_geodex_state_space.cpp)
  # Link GTest before ompl::ompl. Its headers must take priority over a system GTest
  # such as /opt/homebrew/include/gtest.
  target_link_libraries(test_geodex_state_space PRIVATE geodex GTest::gtest_main ompl::ompl)
  gtest_discover_tests(test_geodex_state_space)

  add_executable(test_segment_count test_segment_count.cpp)
  # Link GTest before ompl::ompl. Its headers must take priority over a system GTest
  # such as /opt/homebrew/include/gtest.
  target_link_libraries(test_segment_count PRIVATE geodex GTest::gtest_main ompl::ompl)
  gtest_discover_tests(test_segment_count)

  add_executable(test_geodex_informed_sampling test_geodex_informed_sampling.cpp)
  target_link_libraries(test_geodex_informed_sampling PRIVATE geodex GTest::gtest_main ompl::ompl)
  gtest_discover_tests(test_geodex_informed_sampling)

  add_executable(test_planning test_planning.cpp)
  target_link_libraries(test_planning PRIVATE geodex GTest::gtest_main ompl::ompl)
  gtest_discover_tests(test_planning)
endif()

if(GEODEX_VAMP)
  foreach(_robot panda ur5 baxter pr2 fr3_arm_gripper scene mobile registry sweep)
    add_executable(test_vamp_${_robot} test_vamp_${_robot}.cpp)
    target_link_libraries(test_vamp_${_robot} PRIVATE geodex GTest::gtest_main)
    target_compile_definitions(test_vamp_${_robot} PRIVATE
      GEODEX_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
    gtest_discover_tests(test_vamp_${_robot})
  endforeach()

  add_executable(test_robots_planning test_robots_planning.cpp)
  target_link_libraries(test_robots_planning PRIVATE geodex GTest::gtest_main)
  target_compile_definitions(test_robots_planning PRIVATE
    GEODEX_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
  gtest_discover_tests(test_robots_planning DISCOVERY_TIMEOUT 60)

  # Only test_vamp_panda needs the Panda URDF. The other robots use their own scene
  # fixtures.
  target_compile_definitions(test_vamp_panda PRIVATE
    GEODEX_PANDA_URDF="${GEODEX_PANDA_URDF_PATH}")

  if(GEODEX_PINOCCHIO)
    target_compile_definitions(test_vamp_panda PRIVATE GEODEX_TEST_HAS_PINOCCHIO)
  endif()

  if(GEODEX_OMPL)
    # Plans a seeded robot query under many heap layouts.
    add_executable(test_planning_heap test_planning_heap.cpp)
    target_link_libraries(test_planning_heap PRIVATE geodex GTest::gtest_main)
    gtest_discover_tests(test_planning_heap DISCOVERY_TIMEOUT 60)

    add_executable(test_smoothing_vamp test_smoothing_vamp.cpp)
    target_link_libraries(test_smoothing_vamp PRIVATE geodex GTest::gtest_main ompl::ompl)
    target_compile_definitions(test_smoothing_vamp PRIVATE
      GEODEX_TEST_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures")
    gtest_discover_tests(test_smoothing_vamp)
  endif()
endif()
