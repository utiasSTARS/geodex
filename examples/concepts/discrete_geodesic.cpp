// Examples of the Discrete Geodesic Interpolation page, the C++ version of
// discrete_geodesic.py. Usage: discrete_geodesic [--json out.json]

// [docs-start:sphere]
#include <cmath>
#include <cstdio>

#include <geodex/geodex.hpp>
// [docs-end:sphere]

#include "../common/json_output.hpp"

// [docs-start:sphere]
int main(int argc, char** argv) {
  const Eigen::Vector3d start(0.0, 0.0, 1.0);
  const Eigen::Vector3d target(std::sin(1.3) * std::cos(0.5),
                               std::sin(1.3) * std::sin(0.5), std::cos(1.3));

  geodex::InterpolationSettings settings;
  settings.step_size = 0.05;
  settings.max_steps = 500;

  // 1. The round sphere takes the fast path and traces the great circle.
  geodex::Sphere<> round_sphere;
  auto great = geodex::discrete_geodesic(round_sphere, start, target, settings);

  // 2. An anisotropic constant SPD metric takes the finite-difference path.
  const Eigen::Matrix3d A = Eigen::Vector3d(25.0, 1.0, 1.0).asDiagonal();
  geodex::ConfigurationSpace stretched{round_sphere,
                                       geodex::ConstantSPDMetric<3>{A}};
  auto bent = geodex::discrete_geodesic(stretched, start, target, settings);

  std::printf("%s %zu %s %zu\n", geodex::to_string(great.status), great.path.size(),
              geodex::to_string(bent.status), bent.path.size());
  // [docs-end:sphere]

  using geodex_examples::Json;
  geodex_examples::write_json_arg(
      argc, argv,
      Json::object({{"great", Json::path(great.path)},
                    {"bent", Json::path(bent.path)},
                    {"status", Json::array({geodex::to_string(great.status),
                                            geodex::to_string(bent.status)})}}));
  // [docs-start:sphere]
  return 0;
}
// [docs-end:sphere]
