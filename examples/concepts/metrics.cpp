// Examples of the Metrics concept page, the C++ version of metrics.py.
//
// Each snippet builds one kind of metric and prints the cost of the same kinds of
// motion under it.
//
// Usage: metrics [--json out.json]

#include <cmath>
#include <cstdio>

#include <string>
#include <utility>
#include <vector>

// [docs-start:se2-weights]
#include <geodex/collision/collision.hpp>
#include <geodex/geodex.hpp>
// [docs-end:se2-weights]

#include "../common/json_output.hpp"

using geodex_examples::Json;

/// Mass matrix of a two-link planar arm with unit links and masses.
Eigen::Matrix2d arm_mass_matrix(const Eigen::Vector2d& q) {
  const double c2 = std::cos(q[1]);
  const double m00 = 1.0 / 12 + 1.0 / 12 + 0.25 + (1.0 + 0.25 + 2.0 * 0.5 * c2);
  const double m01 = 1.0 / 12 + (0.25 + 0.5 * c2);
  const double m11 = 1.0 / 12 + 0.25;
  Eigen::Matrix2d M;
  M << m00, m01, m01, m11;
  return M;
}

int main(int argc, char** argv) {
  std::vector<std::pair<std::string, Json>> out;

  // [docs-start:se2-weights]
  const Eigen::Vector3d pose(0.0, 0.0, 0.0);  // at the origin, heading along +x
  const Eigen::Vector3d forward(1.0, 0.0,
                                0.0);  // body-frame velocity (vx, vy, omega)
  const Eigen::Vector3d sideways(0.0, 1.0, 0.0);
  const Eigen::Vector3d turn(0.0, 0.0, 1.0);

  const std::vector<std::pair<std::string, geodex::SE2<>>> bases = {
      {"holonomic", geodex::SE2<>{geodex::SE2LeftInvariantMetric{1.0, 1.0, 0.5}}},
      {"differential drive",
       geodex::SE2<>{geodex::SE2LeftInvariantMetric{1.0, 100.0, 1.0}}},
      {"car-like",
       geodex::SE2<>{geodex::SE2LeftInvariantMetric::car_like(1.5, 20.0)}},
  };
  for (const auto& [name, se2] : bases) {
    std::printf("%s forward %.3f  sideways %.3f  turn %.3f\n", name.c_str(),
                se2.norm(pose, forward), se2.norm(pose, sideways),
                se2.norm(pose, turn));
  }
  // [docs-end:se2-weights]
  {
    std::vector<std::pair<std::string, Json>> weights;
    for (const auto& [name, se2] : bases) {
      weights.emplace_back(
          name, Json::array({se2.norm(pose, forward), se2.norm(pose, sideways),
                             se2.norm(pose, turn)}));
    }
    out.emplace_back("se2_weights", Json::object(weights));
  }

  // [docs-start:kinetic-energy]
  geodex::ConfigurationSpace arm{geodex::Euclidean<2>{},
                                 geodex::KineticEnergyMetric{arm_mass_matrix}};
  const Eigen::Vector2d shoulder(1.0, 0.0);  // turn the shoulder at 1 rad/s
  const Eigen::Vector2d stretched(0.0, 0.0), folded(0.0, 2.8);
  std::printf("stretched: shoulder speed costs %.3f\n",
              arm.norm(stretched, shoulder));
  std::printf("folded: shoulder speed costs %.3f\n", arm.norm(folded, shoulder));
  // [docs-end:kinetic-energy]
  out.emplace_back("kinetic_energy", Json::array({arm.norm(stretched, shoulder),
                                                  arm.norm(folded, shoulder)}));

  {
    // [docs-start:robot]
    geodex::robots::MassMatrix<geodex::robots::Robot::Panda> panda;
    Eigen::Matrix<double, 7, 1> q;
    q << 0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785;
    const auto& M = panda(q);  // 7 x 7, from the robot's precompiled CRBA
    std::printf("Panda mass matrix diagonal: %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                M(0, 0), M(1, 1), M(2, 2), M(3, 3), M(4, 4), M(5, 5), M(6, 6));
    // [docs-end:robot]
    out.emplace_back("robot", Json(Eigen::VectorXd(M.diagonal())));
  }

  // [docs-start:jacobi]
  auto potential = [](const Eigen::Vector2d& q) {
    constexpr double g = 9.81;  // height of the two link centers times their weight
    return g *
           (0.5 * std::sin(q[0]) + (std::sin(q[0]) + 0.5 * std::sin(q[0] + q[1])));
  };

  const double H =
      1.2 * 9.81 * (0.5 + 1.5);  // 20 percent above the largest potential
  geodex::ConfigurationSpace jacobi{
      geodex::Euclidean<2>{}, geodex::JacobiMetric{arm_mass_matrix, potential, H}};
  const Eigen::Vector2d hanging(-1.5, 0.0), raised(1.5, 0.0);
  std::printf("hanging: shoulder speed costs %.3f\n",
              jacobi.norm(hanging, shoulder));
  std::printf("raised: shoulder speed costs %.3f\n", jacobi.norm(raised, shoulder));
  // [docs-end:jacobi]
  out.emplace_back("jacobi", Json::array({jacobi.norm(hanging, shoulder),
                                          jacobi.norm(raised, shoulder)}));

  // [docs-start:clearance]
  geodex::collision::CircleSDF obstacle{2.0, 0.0,
                                        0.5};  // a disc of radius 0.5 at (2, 0)
  geodex::SE2LeftInvariantMetric base_metric{1.0, 10.0, 1.0};
  geodex::SDFConformalMetric clearance{base_metric, obstacle, 1.5,
                                       3.0};  // kappa, beta
  geodex::ConfigurationSpace room{
      geodex::SE2<>{base_metric, geodex::SE2LeftExponentialMap{},
                    Eigen::Vector3d(0.0, -2.0, -M_PI),
                    Eigen::Vector3d(4.0, 2.0, M_PI)},
      clearance};
  const Eigen::Vector3d near(1.3, 0.0, 0.0), far(0.2, 1.8, 0.0);
  std::printf("next to the disc: forward speed costs %.3f\n",
              room.norm(near, forward));
  std::printf("far from it: forward speed costs %.3f\n", room.norm(far, forward));
  // [docs-end:clearance]
  out.emplace_back("clearance", Json::array({room.norm(near, forward),
                                             room.norm(far, forward)}));

  // [docs-start:pullback]
  auto jacobian = [](const Eigen::Vector2d& q) {
    // End-effector velocity of the planar arm per joint velocity.
    const double s1 = std::sin(q[0]), c1 = std::cos(q[0]);
    const double s12 = std::sin(q[0] + q[1]), c12 = std::cos(q[0] + q[1]);
    Eigen::Matrix2d J;
    J << -s1 - s12, -s12, c1 + c12, c12;
    return J;
  };
  auto task_metric = [](const Eigen::Vector2d&) {
    return Eigen::Matrix2d::Identity()
        .eval();  // plain Euclidean speed of the end effector
  };

  geodex::ConfigurationSpace hand{
      geodex::Euclidean<2>{}, geodex::PullbackMetric{jacobian, task_metric, 1e-3}};
  std::printf("stretched: shoulder speed moves the hand at %.3f\n",
              hand.norm(stretched, shoulder));
  std::printf("folded: shoulder speed moves the hand at %.3f\n",
              hand.norm(folded, shoulder));
  // [docs-end:pullback]
  out.emplace_back("pullback", Json::array({hand.norm(stretched, shoulder),
                                            hand.norm(folded, shoulder)}));

  geodex_examples::write_json_arg(argc, argv, Json::object(out));
  return 0;
}
