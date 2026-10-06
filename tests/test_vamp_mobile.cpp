/// @file test_vamp_mobile.cpp
/// @brief VAMP kernels of the mobile manipulators, with whole-body configurations, the
/// base joint box, invariance under planar motion and the heading wrap of the validator.

#include <array>
#include <cmath>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>
#include <ompl/base/SpaceInformation.h>

#include "geodex/integration/ompl/geodex_state_space.hpp"
#include "geodex/integration/vamp/registry.hpp"
#include "geodex/manifold/euclidean.hpp"
#include "geodex/manifold/product.hpp"
#include "geodex/manifold/se2.hpp"

namespace {

namespace vamp_int = geodex::integration::vamp;

constexpr const char* kFixturesDir = GEODEX_TEST_FIXTURES_DIR;

auto fixture_path(const std::string& relative) -> std::string {
  return std::string(kFixturesDir) + "/" + relative;
}

struct MobileCase {
  std::string name;
  int dim;
  std::vector<double> free_pose;  // self-collision-free at the origin
};

std::vector<MobileCase> cases() {
  return {{"stretch3", 8, {0.0, 0.0, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0}},
          {"stretch4", 8, {0.0, 0.0, 0.0, 0.6, 0.1, 0.0, 0.0, 0.0}},
          {"ridgeback_ur5e", 9, {0.0, 0.0, 0.0, 0.0, -1.57, 1.57, -1.57, -1.57, 0.0}},
          {"husky_ur5e", 9, {0.0, 0.0, 0.0, 0.0, -1.57, 1.57, -1.57, -1.57, 0.0}}};
}

std::vector<double> with_base(std::vector<double> q, double x, double y, double theta) {
  q[0] = x;
  q[1] = y;
  q[2] = theta;
  return q;
}

class VampMobile : public ::testing::TestWithParam<MobileCase> {};

TEST_P(VampMobile, RegisteredWithWholeBodyDimension) {
  const auto& c = GetParam();
  EXPECT_EQ(vamp_int::robot_dimension(c.name), c.dim);
  auto checker = vamp_int::make_vamp_checker(c.name, vamp_int::load_scene(fixture_path(
                                                         "vamp/mobile/empty.yaml")));
  EXPECT_FALSE(checker->is_valid(c.free_pose.data(), c.dim - 1));
}

TEST_P(VampMobile, FreePoseValidAnywhereInTheEmptyScene) {
  const auto& c = GetParam();
  auto checker =
      vamp_int::make_vamp_checker(c.name, vamp_int::load_scene(fixture_path("vamp/mobile/empty.yaml")));
  for (const double theta : {0.0, 1.0, -2.5, std::numbers::pi, -std::numbers::pi}) {
    const auto q = with_base(c.free_pose, 3.0, -7.0, theta);
    EXPECT_TRUE(checker->is_valid(q.data(), c.dim)) << "theta " << theta;
  }
}

TEST_P(VampMobile, BaseBoxAndJointLimitsAreEnforced) {
  const auto& c = GetParam();
  auto checker =
      vamp_int::make_vamp_checker(c.name, vamp_int::load_scene(fixture_path("vamp/mobile/empty.yaml")));
  const auto far = with_base(c.free_pose, 2000.0, 0.0, 0.0);
  EXPECT_FALSE(checker->is_valid(far.data(), c.dim));
  auto beyond = c.free_pose;
  beyond[3] = 50.0;  // first arm coordinate far past its limit
  EXPECT_FALSE(checker->is_valid(beyond.data(), c.dim));
}

TEST_P(VampMobile, WallCollidesOnlyWhereTheBaseStands) {
  const auto& c = GetParam();
  auto checker =
      vamp_int::make_vamp_checker(c.name, vamp_int::load_scene(fixture_path("vamp/mobile/wall.yaml")));
  const auto on_wall = with_base(c.free_pose, 0.0, 0.0, 0.3);
  const auto clear = with_base(c.free_pose, -2.0, 0.0, 0.3);
  EXPECT_FALSE(checker->is_valid(on_wall.data(), c.dim));
  EXPECT_TRUE(checker->is_valid(clear.data(), c.dim));
}

TEST_P(VampMobile, ValidityIsInvariantUnderPlanarMotion) {
  // Moving the robot and a sphere obstacle together by a planar rigid motion keeps
  // every verdict.
  const auto& c = GetParam();
  const Eigen::Vector3d obstacle(0.35, -0.4, 0.8);
  const double dx = 1.3, dy = -0.7, dtheta = 0.9;
  const Eigen::Vector3d moved = Eigen::AngleAxisd(dtheta, Eigen::Vector3d::UnitZ()) * obstacle +
                                Eigen::Vector3d(dx, dy, 0.0);
  int agree = 0, total = 0;
  for (const double r : {0.05, 0.15, 0.3}) {
    auto s0 = vamp_int::make_scene_builder();
    vamp_int::scene_add_sphere(s0, obstacle, r);
    auto s1 = vamp_int::make_scene_builder();
    vamp_int::scene_add_sphere(s1, moved, r);
    auto a = vamp_int::make_vamp_checker(c.name, vamp_int::build_scene(s0));
    auto b = vamp_int::make_vamp_checker(c.name, vamp_int::build_scene(s1));
    for (const double theta : {-2.0, 0.0, 1.2}) {
      const auto q0 = with_base(c.free_pose, 0.0, 0.0, theta);
      const auto q1 = with_base(c.free_pose, dx, dy, theta + dtheta);
      agree += a->is_valid(q0.data(), c.dim) == b->is_valid(q1.data(), c.dim);
      ++total;
    }
  }
  EXPECT_EQ(agree, total);
}

TEST_P(VampMobile, MotionValidatorFollowsTheHeadingWrap) {
  // Find a small sphere the robot hits at heading 0 but misses for headings within
  // 0.3 rad of pi. The SE(2) edge from 3.0 to -3.0 rad turns through pi and must be
  // clear. The edge from -1.0 to 1.0 rad turns through 0 and must be blocked.
  const auto& c = GetParam();
  const int n = c.dim - 3;
  using Arm = geodex::Euclidean<Eigen::Dynamic>;
  Arm arm(n);
  auto space_manifold = geodex::make_product(geodex::SE2<>(), arm);
  using Space = geodex::integration::ompl::GeodexStateSpace<decltype(space_manifold)>;
  ompl::base::RealVectorBounds bounds(c.dim);
  bounds.setLow(-10.0);
  bounds.setHigh(10.0);
  auto space = std::make_shared<Space>(space_manifold, bounds);
  space->setInterpolationMode(geodex::integration::ompl::InterpolationMode::BaseGeodesic);
  space->setCollisionResolution(0.01);

  vamp_int::EnvHandle env;
  bool have = false;
  for (double phi = 0.0; phi < 2.0 * std::numbers::pi && !have; phi += std::numbers::pi / 6.0) {
    for (double radius = 0.3; radius <= 0.9 && !have; radius += 0.1) {
      for (double z = 0.2; z <= 1.2 && !have; z += 0.2) {
        auto s = vamp_int::make_scene_builder();
        vamp_int::scene_add_sphere(
            s, Eigen::Vector3d(radius * std::cos(phi), radius * std::sin(phi), z), 0.08);
        env = vamp_int::build_scene(s);
        auto checker = vamp_int::make_vamp_checker(c.name, env);
        const auto at = [&](double theta) {
          const auto q = with_base(c.free_pose, 0.0, 0.0, theta);
          return checker->is_valid(q.data(), c.dim);
        };
        bool clear_near_pi = true;
        for (double t = 2.8; t <= std::numbers::pi + 1e-9; t += 0.02) {
          clear_near_pi = clear_near_pi && at(t) && at(-t);
        }
        have = !at(0.0) && clear_near_pi;
      }
    }
  }
  ASSERT_TRUE(have) << "no probe sphere separates heading 0 from heading pi";

  for (const bool certified : {false, true}) {
    SCOPED_TRACE(certified ? "certified validator" : "sampling validator");
    auto si = std::make_shared<ompl::base::SpaceInformation>(space);
    si->setMotionValidator(
        certified ? vamp_int::make_vamp_certified_motion_validator(c.name, si, env)
                  : vamp_int::make_vamp_motion_validator(c.name, si, env));
    si->setup();
    auto state = [&](double theta) {
      auto* s = si->allocState();
      const auto q = with_base(c.free_pose, 0.0, 0.0, theta);
      for (int i = 0; i < c.dim; ++i) *space->getValueAddressAtIndex(s, i) = q[i];
      return s;
    };
    auto* a = state(3.0);
    auto* b = state(-3.0);
    auto* d = state(-1.0);
    auto* e = state(1.0);
    EXPECT_TRUE(si->checkMotion(a, b));
    EXPECT_FALSE(si->checkMotion(d, e));
    for (auto* s : {a, b, d, e}) si->freeState(s);
  }
}

TEST_P(VampMobile, CertifiedValidatorKeepsTheBaseArcInsideTheBounds) {
  // Both ends of the edge lie inside the box, and its SE(2) arc reaches x = x_arc.
  const auto& c = GetParam();
  const int n = c.dim - 3;
  geodex::Euclidean<Eigen::Dynamic> arm(n);
  auto manifold = geodex::make_product(geodex::SE2<>(), arm);
  using Space = geodex::integration::ompl::GeodexStateSpace<decltype(manifold)>;
  const auto qa = with_base(c.free_pose, 0.0, -0.5, 0.5 * std::numbers::pi - 1.2);
  const auto qb = with_base(c.free_pose, 0.0, 0.5, 0.5 * std::numbers::pi + 1.2);
  const Eigen::Vector3d ba(qa[0], qa[1], qa[2]), bb(qb[0], qb[1], qb[2]);
  const Eigen::Vector3d twist = geodex::SE2LeftExponentialMap{}.inverse_retract(ba, bb);
  double x_arc = 0.0;
  for (int k = 0; k <= 1000; ++k) {
    x_arc = std::max(x_arc, geodex::SE2LeftExponentialMap{}.retract(ba, k / 1000.0 * twist)[0]);
  }
  ASSERT_GT(x_arc, 0.05);
  const auto env = vamp_int::build_scene(vamp_int::make_scene_builder());
  for (const double x_hi : {0.5 * x_arc, 2.0 * x_arc}) {
    ompl::base::RealVectorBounds bounds(c.dim);
    bounds.setLow(-10.0);
    bounds.setHigh(10.0);
    bounds.setLow(0, -1.0);
    bounds.setHigh(0, x_hi);
    bounds.setLow(2, -std::numbers::pi);
    bounds.setHigh(2, std::numbers::pi);
    auto space = std::make_shared<Space>(manifold, bounds);
    space->setInterpolationMode(geodex::integration::ompl::InterpolationMode::BaseGeodesic);
    auto si = std::make_shared<ompl::base::SpaceInformation>(space);
    si->setMotionValidator(vamp_int::make_vamp_certified_motion_validator(c.name, si, env));
    si->setup();
    auto* a = si->allocState();
    auto* b = si->allocState();
    for (int i = 0; i < c.dim; ++i) {
      *space->getValueAddressAtIndex(a, i) = qa[i];
      *space->getValueAddressAtIndex(b, i) = qb[i];
    }
    ASSERT_TRUE(si->satisfiesBounds(a) && si->satisfiesBounds(b));
    EXPECT_EQ(si->checkMotion(a, b), x_hi > x_arc) << "x_hi " << x_hi << ", arc reaches " << x_arc;
    si->freeState(a);
    si->freeState(b);
  }
}

INSTANTIATE_TEST_SUITE_P(Robots, VampMobile, ::testing::ValuesIn(cases()),
                         [](const auto& info) { return info.param.name; });

}  // namespace
