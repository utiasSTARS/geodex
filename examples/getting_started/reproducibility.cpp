// Seeds and budgets, the C++ version of reproducibility.py.
//
// The program runs the plan of the installation page twice with one seed, then
// with other seeds, with a time budget and without a seed. It also seeds the
// samplers outside the planner. Usage: reproducibility [--json out.json]

#include <cmath>
#include <cstdio>

#include <string>
#include <vector>

// [docs-start:same-seed]
#include <geodex/geodex.hpp>
#include <geodex/planning/plan.hpp>
// [docs-end:same-seed]

#include "../common/json_output.hpp"

using geodex_examples::Json;

// [docs-start:same-seed]
bool is_valid(const Eigen::Vector3d& q) {
  return std::hypot(q[0] - 2.0, q[1] - 2.0) > 0.8;
}
// [docs-end:same-seed]

int main(int argc, char** argv) {
  namespace gp = geodex::planning;
  std::vector<std::pair<std::string, Json>> out;

  // [docs-start:same-seed]
  // The plan of the installation page, a differential-drive robot around a disc.
  geodex::SE2<> space{geodex::SE2LeftInvariantMetric{1.0, 100.0, 1.0},
                      geodex::SE2LeftExponentialMap{},
                      Eigen::Vector3d(0.0, 0.0, -M_PI),
                      Eigen::Vector3d(4.0, 4.0, M_PI)};
  const Eigen::Vector3d start(0.5, 2.0, 0.0), goal(3.5, 2.0, 0.0);
  // [docs-end:same-seed]

  {
    // [docs-start:same-seed]
    gp::PlanSettings settings;
    settings.iterations = 2000;
    settings.seed = 1;
    const auto a = gp::plan(space, start, goal, is_valid, settings);
    const auto b = gp::plan(space, start, goal, is_valid, settings);
    std::printf("same path: %d same cost: %d\n", a.path == b.path,
                a.cost == b.cost);
    // [docs-end:same-seed]
    out.emplace_back(
        "same_seed",
        Json::object({{"identical", a.path == b.path && a.cost == b.cost},
                      {"cost", a.cost},
                      {"path", Json::path(a.path)}}));
  }

  // [docs-start:other-seeds]
  for (std::uint64_t seed : {1, 2, 3}) {
    gp::PlanSettings settings;
    settings.iterations = 2000;
    settings.seed = seed;
    const auto result = gp::plan(space, start, goal, is_valid, settings);
    std::printf("seed %llu: cost %.4f\n", static_cast<unsigned long long>(seed),
                result.cost);
    // [docs-end:other-seeds]
    out.emplace_back("seed_" + std::to_string(seed), Json(result.cost));
    // [docs-start:other-seeds]
  }
  // [docs-end:other-seeds]

  {
    // [docs-start:time-budget]
    // With iterations = 0 (the default) the planner stops after `time` seconds.
    gp::PlanSettings settings;
    settings.time = 0.2;
    settings.seed = 1;
    const auto timed = gp::plan(space, start, goal, is_valid, settings);
    std::printf("time budget: cost %.4f\n", timed.cost);
    // [docs-end:time-budget]
  }

  // [docs-start:sampling]
  geodex::set_default_seed(7);  // reseed the source of default samplers
  geodex::Sphere<> sphere;      // constructed afterwards, it samples from seed 7
  const Eigen::Vector3d first = sphere.random_point();

  sphere.seed(7);  // reseed this manifold's own sampler
  const Eigen::Vector3d again = sphere.random_point();
  std::printf("first sample: %.6f %.6f %.6f after sphere.seed(7): %.6f %.6f %.6f\n",
              first[0], first[1], first[2], again[0], again[1], again[2]);
  // [docs-end:sampling]
  out.emplace_back("sampling",
                   Json::object({{"first", Json(first)}, {"again", Json(again)}}));

  // [docs-start:try-it]
  gp::PlanSettings unseeded;  // seed 0, the default
  unseeded.iterations = 2000;
  space.seed(3);
  const auto first_plan = gp::plan(space, start, goal, is_valid, unseeded);
  const auto second_plan = gp::plan(space, start, goal, is_valid, unseeded);
  space.seed(3);
  const auto again_plan = gp::plan(space, start, goal, is_valid, unseeded);
  std::printf("costs %.4f and %.4f, after space.seed(3) %.4f\n", first_plan.cost,
              second_plan.cost, again_plan.cost);
  // [docs-end:try-it]
  out.emplace_back(
      "try_it", Json::array({first_plan.cost, second_plan.cost, again_plan.cost}));

  geodex_examples::write_json_arg(argc, argv, Json::object(out));
  return 0;
}
