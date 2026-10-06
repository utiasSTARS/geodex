/// @file test_vamp_scene.cpp
/// @brief Scene loading and the programmatic scene builder reach the lists VAMP checks.

#include <array>
#include <cstdint>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "geodex/integration/vamp/registry.hpp"
#include "geodex/utils/random.hpp"

namespace {

constexpr const char* kFixturesDir = GEODEX_TEST_FIXTURES_DIR;

auto fixture_path(const std::string& relative) -> std::string {
  return std::string(kFixturesDir) + "/" + relative;
}

constexpr std::array<double, 7> kReady{0.0, -0.7853981633974483, 0.0, -2.356194490192345,
                                       0.0, 1.5707963267948966,  0.7853981633974483};

auto panda_valid(const geodex::integration::vamp::EnvHandle& env) -> bool {
  return geodex::integration::vamp::make_vamp_checker("panda", env)->is_valid(kReady.data(), 7);
}

}  // namespace

namespace vamp_int = geodex::integration::vamp;

TEST(VampScene, YamlCylinderIsChecked) {
  EXPECT_TRUE(panda_valid(vamp_int::load_scene(fixture_path("vamp/panda/empty.yaml"))));
  EXPECT_FALSE(panda_valid(vamp_int::load_scene(fixture_path("vamp/panda/cylinder.yaml"))));
}

TEST(VampScene, BuilderUprightCylinderIsChecked) {
  auto s = vamp_int::make_scene_builder();
  vamp_int::scene_add_cylinder(s, Eigen::Vector3d(0.0, 0.0, 0.5), 1.5, 2.0,
                               Eigen::Quaterniond::Identity());
  EXPECT_FALSE(panda_valid(vamp_int::build_scene(s)));
}

TEST(VampScene, BuilderTiltedCylinderIsChecked) {
  // A horizontal bar through the arm's workspace, axis along world x.
  auto s = vamp_int::make_scene_builder();
  const Eigen::Quaterniond about_y(Eigen::AngleAxisd(0.5 * M_PI, Eigen::Vector3d::UnitY()));
  vamp_int::scene_add_cylinder(s, Eigen::Vector3d(0.0, 0.0, 0.5), 0.6, 3.0, about_y);
  EXPECT_FALSE(panda_valid(vamp_int::build_scene(s)));
}

TEST(VampScene, DistantCylinderLeavesPoseValid) {
  auto s = vamp_int::make_scene_builder();
  vamp_int::scene_add_cylinder(s, Eigen::Vector3d(3.0, 3.0, 0.5), 0.1, 1.0,
                               Eigen::Quaterniond::Identity());
  vamp_int::scene_add_sphere(s, Eigen::Vector3d(-3.0, 0.0, 0.5), 0.2);
  EXPECT_TRUE(panda_valid(vamp_int::build_scene(s)));
}

TEST(VampScene, BuilderSphereAndBoxAreChecked) {
  auto sphere = vamp_int::make_scene_builder();
  vamp_int::scene_add_sphere(sphere, Eigen::Vector3d(0.0, 0.0, 0.5), 1.5);
  EXPECT_FALSE(panda_valid(vamp_int::build_scene(sphere)));

  auto box = vamp_int::make_scene_builder();
  vamp_int::scene_add_box(box, Eigen::Vector3d(0.0, 0.0, 0.5), Eigen::Vector3d(4.0, 4.0, 2.0),
                          Eigen::Quaterniond::Identity());
  EXPECT_FALSE(panda_valid(vamp_int::build_scene(box)));
}

TEST(VampScene, QuaternionAtSixteenByteOffsetIsSafe) {
  // A default-compiled caller can place a quaternion on a 16-byte boundary that is not
  // 32-byte aligned. The archive must read it without 32-byte aligned loads.
  alignas(64) unsigned char storage[sizeof(Eigen::Quaterniond) + 16];
  const auto* q = new (storage + 16) Eigen::Quaterniond(
      Eigen::AngleAxisd(0.5 * M_PI, Eigen::Vector3d::UnitY()));
  auto s = vamp_int::make_scene_builder();
  vamp_int::scene_add_box(s, Eigen::Vector3d(0.0, 0.0, 0.5), Eigen::Vector3d(4.0, 4.0, 2.0), *q);
  vamp_int::scene_add_cylinder(s, Eigen::Vector3d(0.0, 0.0, 0.5), 0.6, 3.0, *q);
  EXPECT_FALSE(panda_valid(vamp_int::build_scene(s)));
}

TEST(VampScene, BuilderBoxMatchesYamlBox) {
  // Both take full extents. The same box gives the same verdicts.
  const auto yaml = vamp_int::load_scene(fixture_path("vamp/panda/box.yaml"));
  auto s = vamp_int::make_scene_builder();
  vamp_int::scene_add_box(s, Eigen::Vector3d(0.45, 0.0, 0.4), Eigen::Vector3d(0.2, 0.3, 0.4),
                          Eigen::Quaterniond::Identity());
  const auto built = vamp_int::build_scene(s);
  auto a = vamp_int::make_vamp_checker("panda", yaml);
  auto b = vamp_int::make_vamp_checker("panda", built);
  constexpr std::array<double, 7> lo{-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973};
  constexpr std::array<double, 7> hi{2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973};
  std::uint64_t state = 12345;
  int disagreements = 0, collisions = 0;
  for (int k = 0; k < 400; ++k) {
    std::array<double, 7> q{};
    for (int i = 0; i < 7; ++i) {
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      const double u = static_cast<double>(state >> 11) / 9007199254740992.0;
      q[i] = lo[i] + u * (hi[i] - lo[i]);
    }
    const bool va = a->is_valid(q.data(), 7);
    disagreements += va != b->is_valid(q.data(), 7);
    collisions += !va;
  }
  EXPECT_EQ(disagreements, 0);
  EXPECT_GT(collisions, 0);
}

TEST(VampScene, CylinderThroughTheOriginIsChecked) {
  // An upright cylinder whose axis passes through the world origin, and one beside it.
  for (const double x : {0.0, 0.05}) {
    auto s = vamp_int::make_scene_builder();
    vamp_int::scene_add_cylinder(s, Eigen::Vector3d(x, 0.0, 0.5), 1.5, 2.0,
                                 Eigen::Quaterniond::Identity());
    EXPECT_FALSE(panda_valid(vamp_int::build_scene(s))) << "x " << x;
  }
}

TEST(VampScene, UnknownPrimitiveTypeIsRejected) {
  EXPECT_THROW(vamp_int::load_scene(fixture_path("vamp/panda/cone.yaml")), std::invalid_argument);
}

namespace {

/// Panda configurations inside its joint box, from a fixed seed.
auto panda_configurations(int n) -> std::vector<std::array<double, 7>> {
  constexpr std::array<double, 7> lo{-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973};
  constexpr std::array<double, 7> hi{2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973};
  std::mt19937 rng(11);
  std::vector<std::array<double, 7>> qs(static_cast<std::size_t>(n));
  for (auto& q : qs) {
    for (int j = 0; j < 7; ++j) q[j] = geodex::utils::uniform_real(rng, lo[j], hi[j]);
  }
  return qs;
}

/// A table and a shelf with a held body of three spheres at the end effector.
auto scene_with_attachment() -> vamp_int::EnvHandle {
  auto b = vamp_int::make_scene_builder();
  vamp_int::scene_add_box(b, Eigen::Vector3d(0.55, 0.0, 0.25), Eigen::Vector3d(0.4, 1.0, 0.05),
                          Eigen::Quaterniond::Identity());
  vamp_int::scene_add_box(b, Eigen::Vector3d(0.0, 0.6, 0.6), Eigen::Vector3d(1.0, 0.3, 0.05),
                          Eigen::Quaterniond::Identity());
  auto env = vamp_int::build_scene(b);
  vamp_int::attach_spheres(env, {{0.0, 0.0, 0.08, 0.06}, {0.0, 0.08, 0.12, 0.05},
                                 {0.0, -0.08, 0.12, 0.05}});
  return env;
}

}  // namespace

TEST(VampScene, SharedEnvironmentWithAttachmentGivesSingleThreadedAnswers) {
  const auto env = scene_with_attachment();
  const auto qs = panda_configurations(4000);
  std::vector<char> reference(qs.size());
  {
    const auto checker = vamp_int::make_vamp_checker("panda", env);
    for (std::size_t i = 0; i < qs.size(); ++i) reference[i] = checker->is_valid(qs[i].data(), 7);
  }
  int valid = 0;
  for (const char r : reference) valid += r;
  ASSERT_GT(valid, 400);  // Many answers are valid and many are invalid.
  ASSERT_LT(valid, 3600);

  // Four threads, each with its own checker on the shared environment, then four threads
  // on one shared checker.
  const auto shared = vamp_int::make_vamp_checker("panda", env);
  for (const bool one_checker : {false, true}) {
    std::vector<int> mismatches(4, 0);
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
      threads.emplace_back([&, t] {
        const auto own = one_checker ? nullptr : vamp_int::make_vamp_checker("panda", env);
        const auto& checker = one_checker ? *shared : *own;
        for (int round = 0; round < 5; ++round) {
          for (std::size_t i = 0; i < qs.size(); ++i) {
            mismatches[static_cast<std::size_t>(t)] +=
                static_cast<char>(checker.is_valid(qs[i].data(), 7)) != reference[i];
          }
        }
      });
    }
    for (auto& th : threads) th.join();
    for (int t = 0; t < 4; ++t) EXPECT_EQ(mismatches[static_cast<std::size_t>(t)], 0) << one_checker;
  }
}
