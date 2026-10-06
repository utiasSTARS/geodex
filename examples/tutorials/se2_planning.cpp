// SE(2) planning for a disc robot, a differential-drive robot and a car, the C++
// version of se2_planning.py.
//
// Run it from this directory, which holds the corridor distance grid.
//
// Usage: se2_planning [--json out.json]

#include <cmath>
#include <cstdio>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

// [docs-start:pose]
#include <geodex/collision/collision.hpp>
#include <geodex/geodex.hpp>
#include <geodex/planning/plan.hpp>
#include <ompl/base/DiscreteMotionValidator.h>
// [docs-end:pose]

#include "../common/json_output.hpp"

using geodex_examples::Json;

int main(int argc, char** argv) {
  namespace gc = geodex::collision;
  namespace gp = geodex::planning;
  std::vector<std::pair<std::string, Json>> out;

  // [docs-start:pose]
  geodex::SE2<> se2;
  Eigen::Vector3d pose{5.0, 3.0, M_PI / 4.0};  // (x, y, theta)
  // [docs-end:pose]
  out.emplace_back("pose", Json(se2.distance(Eigen::Vector3d::Zero(), pose)));

  {
    // [docs-start:footprints]
    // A circular footprint is a single radius. Collision reduces to a point query
    // against an obstacle distance field inflated by this value.
    constexpr double robot_radius = 0.3;

    // Build a rectangular footprint with the factory.
    auto rect_fp =
        gc::PolygonFootprint::rectangle(/*half_length=*/0.35, /*half_width=*/0.25,
                                        /*samples_per_edge=*/6);

    // Build any convex polygon from body-frame vertices in counter-clockwise order.
    std::vector<Eigen::Vector2d> verts = {
        {-0.35, -0.30},  // rear right
        {0.35, -0.20},   // front right
        {0.35, 0.20},    // front left
        {-0.35, 0.30},   // rear left
    };
    gc::PolygonFootprint poly_fp(verts, /*samples_per_edge=*/4);
    // [docs-end:footprints]
    (void)robot_radius;
  }

  {
    // [docs-start:metrics]
    // Holonomic. Every direction and turning cost the same.
    geodex::SE2LeftInvariantMetric metric_holo{1.0, 1.0, 1.0};

    // Differential drive. A meter of sliding costs sqrt(10), about 3.2 meters of
    // driving.
    geodex::SE2LeftInvariantMetric metric_diff{1.0, 10.0, 1.0};

    // Car-like. Turning trades against driving at a radius of 1.5 m.
    auto metric_car =
        geodex::SE2LeftInvariantMetric::car_like(/*turning_radius=*/1.5,
                                                 /*lateral_penalty=*/20.0);
    // [docs-end:metrics]
    const Eigen::Vector3d v(0.3, 0.4, 0.5);
    out.emplace_back("metrics", Json::array({metric_holo.norm(pose, v),
                                             metric_diff.norm(pose, v),
                                             metric_car.norm(pose, v)}));
  }

  // [docs-start:grid]
  gc::DistanceGrid grid;
  grid.load("willow_corridor_dist.txt");

  // World dimensions in meters.
  const double world_w = grid.width() * grid.resolution();
  const double world_h = grid.height() * grid.resolution();
  // [docs-end:grid]
  out.emplace_back("grid", Json::array({world_w, world_h}));

  // [docs-start:parking-lot]
  using RectObstacle = gc::RectObstacle;
  constexpr double car_hl = 2.25, car_hw = 0.9;  // a 4.5 m x 1.8 m car

  std::vector<RectObstacle> obstacles = {
      {5.0, 1.35, 0.0, car_hl, car_hw},   // parked car 1
      {10.0, 1.35, 0.0, car_hl, car_hw},  // parked car 2
      {21.0, 1.35, 0.0, car_hl, car_hw},  // parked car 3
      {26.0, 1.35, 0.0, car_hl, car_hw},  // parked car 4
      {15.0, -0.05, 0.0, 15.0, 0.05},     // curb
      {15.0, 10.05, 0.0, 15.0, 0.05},     // sidewalk
  };
  // [docs-end:parking-lot]

  const Eigen::Vector3d start{2.0, 5.0, 0.0};
  const Eigen::Vector3d goal{12.0, 6.0, -M_PI / 2.0};
  {
    // [docs-start:disc-validity]
    constexpr double robot_radius = 0.3;
    constexpr double safety_margin = 0.10;  // extra buffer beyond the robot radius

    auto is_valid = [&grid](const Eigen::Vector3d& q) {
      return grid.distance_at(q[0], q[1]) > robot_radius + safety_margin;
    };
    // [docs-end:disc-validity]

    // [docs-start:holonomic-plan]
    // A manifold whose workspace bounds span the corridor map.
    geodex::SE2<> se2{geodex::SE2LeftInvariantMetric{1.0, 1.0, 1.0},
                      geodex::SE2LeftExponentialMap{},
                      Eigen::Vector3d(0.0, 0.0, -M_PI),
                      Eigen::Vector3d(world_w, world_h, M_PI)};

    const Eigen::Vector3d start{2.0, 5.0, 0.0};
    const Eigen::Vector3d goal{12.0, 6.0, -M_PI / 2.0};

    gp::PlanSettings settings;
    settings.iterations = 6000;
    settings.planner = gp::planners::GreedyRRTstar{
        .range = 4.5, .greedy_ratio = 0.9, .rewire_factor = 0.5};
    settings.collision_check_resolution = grid.resolution();
    settings.seed = 1;
    auto result = gp::plan(se2, start, goal, is_valid, settings);
    std::printf("holonomic: solved=%d cost=%.3f\n", result.solved, result.cost);
    // [docs-end:holonomic-plan]
    out.emplace_back("holonomic",
                     Json::object({{"solved", result.solved},
                                   {"cost", result.cost},
                                   {"path", Json::path(result.path)}}));
  }

  {
    // [docs-start:time-budget]
    gp::PlanSettings settings;
    settings.time = 1.0;  // seconds, used when iterations is 0
    // [docs-end:time-budget]
  }

  // [docs-start:footprint-checker]
  auto footprint =
      gc::PolygonFootprint::rectangle(/*half_length=*/0.35, /*half_width=*/0.25,
                                      /*samples_per_edge=*/6);
  gc::FootprintGridChecker checker{&grid, footprint, /*safety_margin=*/0.10};

  auto is_valid = [&checker](const Eigen::Vector3d& q) {
    return checker.is_valid(q);
  };
  // [docs-end:footprint-checker]
  out.emplace_back("footprint_checker",
                   Json::array({checker.is_valid(start), checker(start)}));

  {
    // [docs-start:diff-manifold]
    geodex::SE2<> se2{geodex::SE2LeftInvariantMetric{1.0, 10.0, 1.0},
                      geodex::SE2LeftExponentialMap{},
                      Eigen::Vector3d(0.0, 0.0, -M_PI),
                      Eigen::Vector3d(world_w, world_h, M_PI)};
    // [docs-end:diff-manifold]

    // [docs-start:directional]
    using geodex::integration::ompl::DirectionalMotionValidator;
    auto directional = [&se2](const ompl::base::SpaceInformationPtr& si) {
      return std::make_shared<DirectionalMotionValidator<geodex::SE2<>>>(
          si.get(), se2,
          std::make_shared<ompl::base::DiscreteMotionValidator>(si.get()),
          /*max_reverse_length=*/0.5);
    };

    gp::PlanSettings settings;
    settings.iterations = 6000;
    settings.planner = gp::planners::GreedyRRTstar{
        .range = 7.0, .greedy_ratio = 0.9, .rewire_factor = 1.1};
    settings.collision_check_resolution = grid.resolution();
    settings.seed = 1;
    auto result =
        gp::plan(se2, start, goal, is_valid, settings,
                 /*heuristic=*/geodex::heuristics::Euclidean{}, directional);
    std::printf("differential drive: solved=%d cost=%.3f\n", result.solved,
                result.cost);
    // [docs-end:directional]
    out.emplace_back("directional",
                     Json::object({{"solved", result.solved},
                                   {"cost", result.cost},
                                   {"path", Json::path(result.path)}}));

    // [docs-start:try-it]
    // Distance the robot drives backward, from the SE(2) logarithm of each edge.
    auto reverse_travel = [&se2](const std::vector<Eigen::Vector3d>& path) {
      double reverse = 0.0;
      for (std::size_t i = 0; i + 1 < path.size(); ++i) {
        reverse += std::max(0.0, -se2.log(path[i], path[i + 1])[0]);
      }
      return reverse;
    };

    const auto free = gp::plan(se2, start, goal, is_valid, settings);
    std::printf("backward driving: %.3f m with the validator, %.3f m without\n",
                reverse_travel(result.path), reverse_travel(free.path));
    // [docs-end:try-it]
    out.emplace_back(
        "try_it",
        Json::object({{"solved", free.solved},
                      {"cost", free.cost},
                      {"reverse", Json::array({reverse_travel(result.path),
                                               reverse_travel(free.path)})}}));
  }

  {
    // [docs-start:clearance]
    // Base metric, and an SDF inflated by the robot radius for a circular robot.
    geodex::SE2LeftInvariantMetric base_metric{1.0, 1.0, 1.0};
    gc::InflatedSDF inflated_sdf{gc::GridSDF{&grid}, 0.3};

    // Conformal clearance metric with kappa = 1.5 and beta = 3.0.
    geodex::SDFConformalMetric clearance_metric{base_metric, inflated_sdf, 1.5,
                                                3.0};

    // SE(2) topology with the clearance geometry.
    geodex::SE2<> se2{base_metric, geodex::SE2LeftExponentialMap{},
                      Eigen::Vector3d(0.0, 0.0, -M_PI),
                      Eigen::Vector3d(world_w, world_h, M_PI)};
    geodex::ConfigurationSpace cspace{se2, clearance_metric};
    // [docs-end:clearance]
    out.emplace_back("clearance",
                     Json(cspace.norm(start, Eigen::Vector3d(1.0, 0.0, 0.0))));
  }

  {
    // [docs-start:diff-clearance]
    auto footprint = gc::PolygonFootprint::rectangle(0.35, 0.25, 6);
    gc::FootprintGridChecker checker{&grid, footprint, 0.01};

    // Binary validity for the planner.
    auto is_valid = [&checker](const Eigen::Vector3d& q) {
      return checker.is_valid(q);
    };

    // Continuous signed distance for the clearance metric, from the same object.
    geodex::SE2LeftInvariantMetric base_metric{1.0, 10.0, 1.0};
    geodex::SDFConformalMetric clearance_metric{base_metric, checker, 1.5, 3.0};

    geodex::SE2<> se2{base_metric, geodex::SE2LeftExponentialMap{},
                      Eigen::Vector3d(0.0, 0.0, -M_PI),
                      Eigen::Vector3d(world_w, world_h, M_PI)};
    geodex::ConfigurationSpace cspace{se2, clearance_metric};
    // [docs-end:diff-clearance]
    (void)is_valid;
    out.emplace_back("diff_clearance",
                     Json(cspace.norm(start, Eigen::Vector3d(1.0, 0.0, 0.0))));
  }

  {
    // [docs-start:car-metric]
    auto metric = geodex::SE2LeftInvariantMetric::car_like(1.5, 20.0);
    // Weights wx = 1.0, wy = 20.0, w_theta = 2.25 (= 1.5^2)
    // [docs-end:car-metric]
    (void)metric;
  }

  {
    // [docs-start:rect-sdf]
    gc::RectSmoothSDF sdf{obstacles, 20.0, car_hw};
    // beta = 20 (smoothness), inflation = car_hw (half-width of the ego vehicle)
    // [docs-end:rect-sdf]
    out.emplace_back("rect_sdf", Json(sdf(Eigen::Vector3d(15.0, 5.0, 0.0))));
  }

  {
    // [docs-start:sat-validity]
    auto is_valid = [&](const Eigen::Vector3d& q) {
      RectObstacle ego{q[0], q[1], q[2], car_hl, car_hw};
      for (const auto& obs : obstacles) {
        if (gc::rects_overlap(ego, obs)) return false;
      }
      return true;
    };
    // [docs-end:sat-validity]
    out.emplace_back("sat_validity",
                     Json::array({is_valid(Eigen::Vector3d(15.0, 5.0, 0.0)),
                                  is_valid(Eigen::Vector3d(5.0, 1.35, 0.0))}));
  }

  {
    // [docs-start:parking]
    auto base_metric = geodex::SE2LeftInvariantMetric::car_like(1.5, 20.0);
    gc::RectSmoothSDF sdf{obstacles, 20.0, car_hw};
    geodex::SDFConformalMetric clearance_metric{base_metric, sdf, 8.0, 3.0};

    geodex::SE2<> se2{geodex::SE2LeftInvariantMetric::car_like(1.5, 20.0),
                      geodex::SE2LeftExponentialMap{},
                      Eigen::Vector3d(0.0, 0.0, -M_PI),
                      Eigen::Vector3d(30.0, 12.0, M_PI)};
    geodex::ConfigurationSpace cspace{se2, clearance_metric};
    // [docs-end:parking]
    out.emplace_back("parking", Json(cspace.norm(Eigen::Vector3d(15.0, 5.0, 0.0),
                                                 Eigen::Vector3d(1.0, 0.0, 0.0))));
  }

  geodex_examples::write_json_arg(argc, argv, Json::object(out));
  return 0;
}
