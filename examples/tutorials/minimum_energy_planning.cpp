// Minimum-energy planning for a two-link planar arm, the C++ version of
// minimum_energy_planning.py.
//
// Plans one start and goal on [-pi, pi]^2 under the Euclidean, the kinetic-energy
// and the Jacobi metric, then under the Jacobi metric at a higher energy. G-RRT*
// runs as an uninformed RRT*.
//
// Usage: minimum_energy_planning [--json out.json]

// [docs-start:setup]
#include <cmath>
#include <cstdio>

#include <algorithm>
#include <utility>
#include <vector>

#include <geodex/geodex.hpp>
#include <geodex/planning/plan.hpp>

namespace gp = geodex::planning;
// [docs-end:setup]

#include "../common/json_output.hpp"

using geodex_examples::Json;

// [docs-start:mass-matrix]
/// Mass matrix M(q) of a two-link planar arm with uniform rods.
struct PlanarArmMassMatrix {
  double l1 = 1.0, l2 = 1.0, m1 = 1.0, m2 = 1.0;
  double lc1 = 0.5, lc2 = 0.5;
  double I1 = 1.0 / 12.0, I2 = 1.0 / 12.0;

  Eigen::Matrix2d operator()(const Eigen::Vector2d& q) const {
    const double c2 = std::cos(q[1]);  // cos(q2), the elbow coupling term
    const double h = l1 * lc2 * c2;    // inertial coupling coefficient
    Eigen::Matrix2d M;
    M(0, 0) = I1 + I2 + m1 * lc1 * lc1 + m2 * (l1 * l1 + lc2 * lc2 + 2.0 * h);
    M(0, 1) = I2 + m2 * (lc2 * lc2 + h);
    M(1, 0) = M(0, 1);
    M(1, 1) = I2 + m2 * lc2 * lc2;
    return M;
  }
};
// [docs-end:mass-matrix]

// [docs-start:potential]
/// Gravitational potential P(q), the mass-weighted heights of the two link centers.
double potential(const Eigen::Vector2d& q) {
  constexpr double g = 9.81, m1 = 1.0, m2 = 1.0, l1 = 1.0, lc1 = 0.5, lc2 = 0.5;
  return m1 * g * lc1 * std::sin(q[0]) +
         m2 * g * (l1 * std::sin(q[0]) + lc2 * std::sin(q[0] + q[1]));
}
// [docs-end:potential]

int main(int argc, char** argv) {
  std::vector<Json> runs;

  // [docs-start:ke-space]
  PlanarArmMassMatrix mass_fn;

  // Joint space [-pi, pi]^2. plan() searches the sampling bounds of the base
  // manifold.
  geodex::Euclidean<2> base;
  base.set_sampling_bounds(Eigen::Vector2d(-M_PI, -M_PI),
                           Eigen::Vector2d(M_PI, M_PI));

  geodex::KineticEnergyMetric ke_metric{mass_fn};
  geodex::ConfigurationSpace cspace_ke{base, ke_metric};
  // [docs-end:ke-space]

  // [docs-start:jacobi-space]
  const double pmax =
      9.81 * (1.0 * 0.5 + 1.0 * (1.0 + 0.5));  // arm straight up, about 19.62 J
  const double H =
      1.2 * pmax;  // total energy, 20 percent above the largest potential

  geodex::JacobiMetric jacobi_metric{mass_fn, potential, H};
  geodex::ConfigurationSpace cspace_jacobi{base, jacobi_metric};
  // [docs-end:jacobi-space]

  // [docs-start:plan]
  const Eigen::Vector2d start(-M_PI / 4, -M_PI / 4);
  const Eigen::Vector2d goal(3 * M_PI / 4, 3 * M_PI / 4);

  // G-RRT* with the greedy bias off and the zero heuristic runs as an uninformed
  // RRT*.
  gp::PlanSettings settings;
  settings.iterations = 3000;
  settings.seed = 1;
  settings.planner = gp::planners::GreedyRRTstar{.greedy_ratio = 0.0};

  auto run = [&](const char* label, const auto& space) {
    auto result = gp::plan(space, start, goal, /*is_valid=*/{}, settings,
                           geodex::heuristics::Zero{});
    std::printf("%-15s solved=%d cost=%.4f waypoints=%zu\n", label, result.solved,
                result.cost, result.path.size());
    return result;
  };
  const auto flat = run("Euclidean", base);
  const auto ke = run("Kinetic energy", cspace_ke);
  const auto jacobi = run("Jacobi", cspace_jacobi);
  // [docs-end:plan]

  // [docs-start:try-it]
  geodex::JacobiMetric high_metric{mass_fn, potential, 5.0 * pmax};
  geodex::ConfigurationSpace high{base, high_metric};
  const auto high_result = gp::plan(high, start, goal, /*is_valid=*/{}, settings,
                                    geodex::heuristics::Zero{});
  for (const auto& [label, result] :
       {std::pair{"Kinetic energy", &ke},
        std::pair{"Jacobi, H = 1.2 Pmax", &jacobi},
        std::pair{"Jacobi, H = 5 Pmax", &high_result}}) {
    double elbow = 0.0;
    for (const auto& q : result->path) elbow = std::max(elbow, std::abs(q[1]));
    std::printf("%-21s largest elbow angle %.3f rad\n", label, elbow);
  }
  // [docs-end:try-it]

  for (const auto& [label, result] :
       {std::pair{"Euclidean", &flat}, std::pair{"Kinetic energy", &ke},
        std::pair{"Jacobi", &jacobi}}) {
    runs.push_back(Json::object({{"label", label},
                                 {"solved", result->solved},
                                 {"cost", result->cost},
                                 {"raw_path", Json::path(result->raw_path)},
                                 {"path", Json::path(result->path)}}));
  }

  geodex_examples::write_json_arg(
      argc, argv,
      Json::object({{"runs", Json::array(runs)},
                    {"start", Json(start)},
                    {"goal", Json(goal)},
                    {"H", H},
                    {"pmax", pmax},
                    {"try_it", Json::object({
                                   {"solved", high_result.solved},
                                   {"cost", high_result.cost},
                                   {"path", Json::path(high_result.path)},
                               })}}));
  return 0;
}
