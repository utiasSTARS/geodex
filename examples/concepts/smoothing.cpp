// Examples of the Path Smoothing concept page, the C++ version of smoothing.py.
//
// The program runs the smoother on a hand-made path in the plane, then with an
// edge proof, then inside plan() for a differential-drive robot on SE(2).
//
// Usage: smoothing [--json out.json]

// [docs-start:standalone]
#include <cmath>
#include <cstdio>

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

#include <geodex/geodex.hpp>
#include <geodex/planning/plan.hpp>
// [docs-end:standalone]

#include "../common/json_output.hpp"

using geodex_examples::Json;

// [docs-start:standalone]
namespace ga = geodex::algorithm;
namespace gp = geodex::planning;

// Two discs (center x, center y, radius) in the plane.
constexpr std::array<std::array<double, 3>, 2> kDiscs{
    {{1.0, 0.2, 0.5}, {2.5, -0.3, 0.6}}};

/// True when q lies outside both discs.
bool is_valid(const Eigen::Vector2d& q) {
  return std::all_of(kDiscs.begin(), kDiscs.end(), [&](const auto& c) {
    return std::hypot(q[0] - c[0], q[1] - c[1]) > c[2];
  });
}
// [docs-end:standalone]

// [docs-start:proof]
/// True when the straight segment from a to b stays outside both discs.
bool segment_clear(const Eigen::Ref<const Eigen::VectorXd>& a,
                   const Eigen::Ref<const Eigen::VectorXd>& b) {
  for (const auto& [cx, cy, r] : kDiscs) {
    const Eigen::Vector2d d = b - a;
    const double t = std::clamp(
        ((cx - a[0]) * d[0] + (cy - a[1]) * d[1]) / (d[0] * d[0] + d[1] * d[1]),
        0.0, 1.0);
    if (std::hypot(a[0] + t * d[0] - cx, a[1] + t * d[1] - cy) <= r) return false;
  }
  return true;
}
// [docs-end:proof]

template <typename Result>
Json summary(const Result& result) {
  return Json::object({{"collision_free", result.collision_free},
                       {"length", result.length},
                       {"path", Json::path(result.path)},
                       {"fallback", result.profile.fallback}});
}

int main(int argc, char** argv) {
  std::vector<std::pair<std::string, Json>> out;

  // [docs-start:standalone]
  geodex::Euclidean<2> plane;

  // A feasible but suboptimal path, as a planner might return it.
  const std::vector<Eigen::Vector2d> path = {{0.0, 0.0},  {0.2, 1.0},  {1.8, 1.0},
                                             {1.8, -1.2}, {3.4, -1.1}, {4.0, 0.0}};

  ga::PathSmoothingSettings settings;
  settings.collision_check_resolution = 0.01;
  const auto result = ga::smooth_path(plane, is_valid, path, settings);

  std::printf("collision_free=%d length=%.4f waypoints=%zu\n",
              result.collision_free, result.length, result.path.size());
  // [docs-end:standalone]
  out.emplace_back("standalone", summary(result));

  // [docs-start:profile]
  const auto& p = result.profile;
  std::printf(
      "%d -> %d waypoints, %ld shortcuts, %ld waypoint moves, %ld validity checks, "
      "fallback stage %d\n",
      p.input_waypoints, p.output_waypoints, p.shortcuts, p.relax_moves,
      p.point_checks, p.fallback);
  // [docs-end:profile]
  out.emplace_back("profile", Json::object({{"input", p.input_waypoints},
                                            {"output", p.output_waypoints}}));

  {
    // [docs-start:corners]
    // The same path without corner rounding keeps its corners.
    ga::PathSmoothingSettings sharp_settings;
    sharp_settings.collision_check_resolution = 0.01;
    sharp_settings.round_corners = false;
    const auto sharp = ga::smooth_path(plane, is_valid, path, sharp_settings);
    std::printf(
        "rounded=%ld kept=%ld cusps=%ld points=%zu, without rounding %zu\n",
        p.rounded_corners, p.kept_corners, p.cusps, result.path.size(), sharp.path.size());
    // [docs-end:corners]
    out.emplace_back("corners", Json::object({{"rounded", p.rounded_corners},
                                            {"kept", p.kept_corners},
                                            {"sharp", summary(sharp)}}));
  }


  {
    // [docs-start:proof]
    // A sound proof that a whole edge is clear lets the smoother skip its samples.
    ga::PathSmoothingSettings settings;
    settings.collision_check_resolution = 0.01;
    settings.edge_provably_clear = segment_clear;
    const auto proved = ga::smooth_path(plane, is_valid, path, settings);
    std::printf("same path=%d edges settled by the proof=%ld\n",
                proved.path == result.path, proved.profile.edge_proofs);
    // [docs-end:proof]
    out.emplace_back("proof", summary(proved));
  }

  {
    // [docs-start:spacing]
    // A looser corner tolerance allows a longer step, and output_spacing limits the step.
    std::vector<ga::PathSmoothingResult<Eigen::Vector2d>> spaced;
    for (const auto& [tolerance, limit] : {std::pair{1e-3, 0.0}, std::pair{1e-4, 0.01}}) {
      ga::PathSmoothingSettings settings;
      settings.collision_check_resolution = 0.01;
      settings.corner_tolerance = tolerance;
      settings.output_spacing = limit;
      const auto even = ga::smooth_path(plane, is_valid, path, settings);
      double step = 0.0;
      for (std::size_t i = 0; i + 1 < even.path.size(); ++i) {
        step = std::max(step, plane.distance(even.path[i], even.path[i + 1]));
      }
      std::printf("corner_tolerance=%g output_spacing=%g: waypoints=%zu step=%.4f\n", tolerance,
                  limit, even.path.size(), step);
      spaced.push_back(even);
    }
    // [docs-end:spacing]
    out.emplace_back("spacing", Json::object({{"coarse", summary(spaced[0])},
                                             {"limited", summary(spaced[1])}}));
  }

  // [docs-start:in-plan]
  // A differential-drive robot on SE(2). The weight 10 on the squared sideways
  // speed makes a meter of sliding cost as much as sqrt(10), about 3.2, meters of
  // driving.
  geodex::SE2<> se2{geodex::SE2LeftInvariantMetric{1.0, 10.0, 1.0},
                    geodex::SE2LeftExponentialMap{},
                    Eigen::Vector3d(-1.0, -2.0, -M_PI),
                    Eigen::Vector3d(5.0, 2.0, M_PI)};

  auto pose_is_valid = [](const Eigen::Vector3d& q) {
    return is_valid(q.head<2>());
  };

  gp::PlanSettings plan_settings;
  plan_settings.iterations = 3000;
  plan_settings.seed = 5;
  plan_settings.collision_check_resolution = 0.01;
  plan_settings.smoothing.output_spacing = 0.2;
  const Eigen::Vector3d start(0.0, 0.0, 0.0), goal(4.0, 0.0, 0.0);
  const auto planned = gp::plan(se2, start, goal, pose_is_valid, plan_settings);

  auto length = [&](const std::vector<Eigen::Vector3d>& waypoints) {
    double sum = 0.0;
    for (std::size_t i = 0; i + 1 < waypoints.size(); ++i) {
      sum += se2.distance(waypoints[i], waypoints[i + 1]);
    }
    return sum;
  };
  std::printf("smoothed=%d raw length=%.3f smoothed length=%.3f\n",
              planned.smoothed, length(planned.raw_path), length(planned.path));
  // [docs-end:in-plan]
  out.emplace_back("in_plan",
                   Json::object({{"solved", planned.solved},
                                 {"smoothed", planned.smoothed},
                                 {"raw_path", Json::path(planned.raw_path)},
                                 {"path", Json::path(planned.path)},
                                 {"raw_length", length(planned.raw_path)},
                                 {"length", length(planned.path)}}));

  geodex_examples::write_json_arg(argc, argv, Json::object(out));
  return 0;
}
