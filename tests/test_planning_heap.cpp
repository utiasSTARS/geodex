// Tests that a seeded plan under an iteration budget does not depend on where the allocator
// places the planner's objects.

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <array>
#include <random>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include "geodex/integration/vamp/registry.hpp"
#include "geodex/robots/planning.hpp"

namespace gr = geodex::robots;
namespace gv = geodex::integration::vamp;
namespace gp = geodex::planning;

namespace {

// The seven boards of a shelf in front of the FR3, as center and size in meters, and a
// 0.04 x 0.24 x 0.16 m box held between the gripper's fingers as 112 spheres. The rows
// give the spheres with x, y, z >= 0 in the box's frame, and their mirror images give
// the rest.
gv::EnvHandle shelf_with_box() {
  const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> boards = {
      {{0.793492, 0.618718, 0.53}, {0.018, 0.28, 1.06}},
      {{0.011492, 0.618718, 0.53}, {0.018, 0.28, 1.06}},
      {{0.402492, 0.618718, 1.051}, {0.8, 0.28, 0.018}},
      {{0.402492, 0.618718, 0.04}, {0.764, 0.28, 0.08}},
      {{0.402492, 0.608718, 0.3977}, {0.764, 0.26, 0.018}},
      {{0.402492, 0.608718, 0.7243}, {0.764, 0.26, 0.018}},
      {{0.402492, 0.756218, 0.53}, {0.78, 0.005, 1.06}},
  };
  auto builder = gv::make_scene_builder();
  for (const auto& [center, size] : boards) {
    gv::scene_add_box(builder, center, size, Eigen::Quaterniond::Identity());
  }
  auto env = gv::build_scene(builder);
  const std::vector<std::array<double, 4>> octant = {
      {0.012, 0.112, 0.072, 0.018}, {0.01, 0.0, 0.07, 0.02},      {0.01, 0.052, 0.07, 0.02},
      {0.01, 0.072, 0.07, 0.02},    {0.01, 0.092, 0.07, 0.02},    {0.01, 0.11, 0.032, 0.02},
      {0.01, 0.11, 0.052, 0.02},    {0.008, 0.018, 0.068, 0.022}, {0.008, 0.034, 0.068, 0.022},
      {0.008, 0.108, 0.0, 0.022},   {0.008, 0.108, 0.014, 0.022}, {0.0, 0.106, 0.066, 0.024},
      {0.0, 0.0, 0.014, 0.03},      {0.0, 0.0, 0.046, 0.03},      {0.0, 0.026, 0.0, 0.03},
      {0.0, 0.026, 0.036, 0.03},    {0.0, 0.05, 0.046, 0.03},     {0.0, 0.052, 0.014, 0.03},
      {0.0, 0.08, 0.0, 0.03},       {0.0, 0.08, 0.026, 0.03},     {0.0, 0.082, 0.046, 0.03},
  };
  std::vector<std::array<double, 4>> box;
  for (const auto& [x, y, z, r] : octant) {
    for (const double sx : {1.0, -1.0}) {
      for (const double sy : {1.0, -1.0}) {
        for (const double sz : {1.0, -1.0}) {
          if ((sx < 0 && x == 0.0) || (sy < 0 && y == 0.0) || (sz < 0 && z == 0.0)) continue;
          box.push_back({sx * x, sy * y, 0.055 + sz * z, r});
        }
      }
    }
  }
  gv::attach_spheres(env, box);
  return env;
}

// A hash of every coordinate of a path.
std::uint64_t path_hash(const std::vector<Eigen::VectorXd>& path) {
  std::uint64_t h = 1469598103934665603ULL;
  for (const auto& q : path) {
    for (Eigen::Index i = 0; i < q.size(); ++i) {
      std::uint64_t bits;
      const double x = q[i];
      std::memcpy(&bits, &x, sizeof bits);
      h = (h ^ bits) * 1099511628211ULL;
    }
  }
  return h;
}

// Allocates blocks of seeded sizes and frees a seeded share of the blocks kept so far.
void shake(std::mt19937_64& rng, std::vector<void*>& keep) {
  std::uniform_int_distribution<int> count(0, 400), small(1, 3000), large(8, 40000), coin(0, 3);
  for (int i = count(rng); i > 0; --i) {
    const int n = coin(rng) == 0 ? large(rng) : small(rng);
    keep.push_back(std::malloc(static_cast<std::size_t>(n)));
  }
  for (void*& p : keep) {
    if (p != nullptr && coin(rng) == 0) {
      std::free(p);
      p = nullptr;
    }
  }
}

}  // namespace

TEST(PlanningDeterminism, SeededPlanDoesNotDependOnTheHeapLayout) {
  // The FR3 moves the box between two bays of the shelf. Before every plan the test
  // allocates and frees a seeded pattern of blocks, which changes the freed addresses the
  // planner's objects reuse. Every plan must return the raw path of the first. The layouts
  // include one where a new motion reuses a pruned motion's address.
  const auto env = shelf_with_box();
  Eigen::VectorXd start(7), goal(7);
  start << 0.1758, -0.1952, 0.4451, -2.1536, 1.9130, 2.0966, 1.0981;
  goal << 0.1591, -0.0746, 0.4710, -1.3579, 1.4042, 2.1810, 1.9137;
  gp::PlanSettings settings;
  settings.iterations = 1500;
  settings.seed = 12;
  settings.smooth = false;
  gr::RobotPlanOptions options;
  options.metric = gr::ArmMetric::Euclidean;
  const auto plan_hash = [&] {
    const auto r = gr::plan<gr::Robot::Fr3Gripper>(start, goal, env, settings, options);
    EXPECT_TRUE(r.solved);
    return path_hash(r.raw_path);
  };
  const std::uint64_t reference = plan_hash();
  for (std::uint64_t layout_seed = 1; layout_seed <= 16; ++layout_seed) {
    std::mt19937_64 rng(layout_seed);
    std::vector<void*> keep;
    for (int layout = 0; layout < 3; ++layout) {
      shake(rng, keep);
      EXPECT_EQ(plan_hash(), reference) << "layout seed " << layout_seed << ", layout " << layout;
    }
    for (void* p : keep) std::free(p);
  }
}
