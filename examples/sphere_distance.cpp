/// @file sphere_distance.cpp
/// @brief Distance computation on the sphere with different metrics and retractions.
///
/// Compares geodesic distance (exact) with the midpoint distance approximation
/// across three setups:
///   1. Round metric + exponential map
///   2. Anisotropic metric (A=diag(4,1,1)) + exponential map
///   3. Round metric + projection retraction

#include <Eigen/Core>
#include <cmath>
#include <geodex/geodex.hpp>
#include <iomanip>
#include <iostream>
#include <numbers>

using namespace geodex;

static Eigen::Vector3d point_at_theta(double theta) {
  return Eigen::Vector3d(std::sin(theta), 0.0, std::cos(theta));
}

template <typename SphereT>
void print_distance_table(const std::string& label, SphereT& sphere, const Eigen::Vector3d& p,
                          const std::vector<double>& thetas) {
  std::cout << "\n=== " << label << " ===\n";
  std::cout << std::setw(10) << "theta" << std::setw(15) << "exact" << std::setw(15) << "midpoint"
            << std::setw(15) << "error\n";
  std::cout << std::string(55, '-') << "\n";

  for (double theta : thetas) {
    auto q = point_at_theta(theta);
    double exact = sphere.distance(p, q);
    double midpoint = distance_midpoint(sphere, p, q);
    double error = std::abs(midpoint - exact);

    std::cout << std::fixed << std::setprecision(4) << std::setw(10) << theta << std::setw(15)
              << exact << std::setw(15) << midpoint << std::scientific << std::setprecision(2)
              << std::setw(15) << error << "\n";
  }
}

int main() {
  Eigen::Vector3d north(0.0, 0.0, 1.0);
  std::vector<double> thetas = {0.1, 0.5, 1.0, std::numbers::pi / 2.0, 2.0, 3.0, std::numbers::pi};

  // 1. Round metric + true exp/log
  Sphere<> round_sphere;
  print_distance_table("Round metric + Exponential map", round_sphere, north, thetas);

  // 2. Anisotropic metric + true exp/log
  Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
  A(0, 0) = 4.0;
  A(1, 1) = 1.0;
  Sphere<2, ConstantSPDMetric<3>> aniso_sphere{ConstantSPDMetric<3>{A}};
  print_distance_table("Anisotropic metric (A=diag(4,1,1)) + Exponential map", aniso_sphere, north,
                       thetas);

  // 3. Round metric + projection retraction
  Sphere<2, SphereRoundMetric, SphereProjectionRetraction> proj_sphere;
  print_distance_table("Round metric + Projection retraction", proj_sphere, north, thetas);

  return 0;
}
