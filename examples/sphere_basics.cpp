/// @file sphere_basics.cpp
/// @brief Basic operations on the 2-sphere: exp, log, distance, geodesic, metrics, retractions.
///
/// This example demonstrates the core geodex API on S^2, including:
///   - Creating manifolds with default and custom policies
///   - Exponential and logarithmic maps
///   - Geodesic distance and interpolation
///   - Swapping metrics (ConstantSPDMetric) and retractions (projection retraction)

#include <Eigen/Core>
#include <cmath>
#include <geodex/geodex.hpp>
#include <iomanip>
#include <iostream>
#include <numbers>

using namespace geodex;

int main() {
  std::cout << std::fixed << std::setprecision(4);
  Eigen::IOFormat fmt(4, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");

  // --- 1. Default sphere: round metric + exponential map ---
  Sphere<> sphere;
  std::cout << "=== Sphere (round metric, exponential map) ===\n";
  std::cout << "dim = " << sphere.dim() << "\n\n";

  Eigen::Vector3d p{0.0, 0.0, 1.0};  // north pole
  Eigen::Vector3d q{1.0, 0.0, 0.0};  // equator

  // Logarithmic map: tangent vector at p pointing toward q
  Eigen::Vector3d v = sphere.log(p, q);
  std::cout << "p = " << p.transpose().format(fmt) << "\n";
  std::cout << "q = " << q.transpose().format(fmt) << "\n";
  std::cout << "log(p, q) = " << v.transpose().format(fmt) << "\n";

  // Exponential map: recover q from p and the tangent vector
  Eigen::Vector3d q_recovered = sphere.exp(p, v);
  std::cout << "exp(p, v) = " << q_recovered.transpose().format(fmt) << "\n\n";

  // Geodesic distance: should be pi/2
  double d = sphere.distance(p, q);
  std::cout << "distance(p, q) = " << d << "  (expected: " << std::numbers::pi / 2.0 << ")\n\n";

  // Geodesic interpolation: trace the great circle
  std::cout << "Geodesic interpolation:\n";
  for (int i = 0; i <= 5; ++i) {
    double t = i / 5.0;
    Eigen::Vector3d pt = sphere.geodesic(p, q, t);
    std::cout << "  t=" << std::setprecision(1) << t << ": " << std::setprecision(4)
              << pt.transpose().format(fmt) << "\n";
  }

  // --- 2. Anisotropic metric: ConstantSPDMetric ---
  std::cout << "\n=== Sphere (anisotropic metric A=diag(4,1,1)) ===\n";

  Eigen::Matrix3d A = Eigen::Vector3d(4.0, 1.0, 1.0).asDiagonal();
  Sphere<2, ConstantSPDMetric<3>> aniso_sphere{ConstantSPDMetric<3>{A}};

  // The anisotropic metric changes norms and distances
  Eigen::Vector3d u{1.0, 0.0, 0.0};
  double norm_round = sphere.norm(p, u);
  double norm_aniso = aniso_sphere.norm(p, u);
  std::cout << "norm_round(p, [1,0,0]) = " << norm_round << "\n";
  std::cout << "norm_aniso(p, [1,0,0]) = " << norm_aniso << "  (scaled by sqrt(4))\n";

  double d_aniso = aniso_sphere.distance(p, q);
  std::cout << "distance_round(p, q) = " << d << "\n";
  std::cout << "distance_aniso(p, q) = " << d_aniso << "\n";

  // --- 3. Projection retraction ---
  std::cout << "\n=== Sphere (round metric, projection retraction) ===\n";

  Sphere<2, SphereRoundMetric, SphereProjectionRetraction> proj_sphere;

  // Projection retraction: cheaper but approximate
  Eigen::Vector3d v_proj = proj_sphere.log(p, q);
  Eigen::Vector3d q_proj = proj_sphere.exp(p, v_proj);
  std::cout << "log(p, q) = " << v_proj.transpose().format(fmt) << "\n";
  std::cout << "exp(p, log(p, q)) = " << q_proj.transpose().format(fmt)
            << "  (approximate round-trip)\n";

  double d_proj = proj_sphere.distance(p, q);
  std::cout << "distance_proj(p, q) = " << d_proj << "\n";
  std::cout << "distance_exact(p, q) = " << d << "\n";

  // --- 4. Random sampling ---
  std::cout << "\n=== Random sampling ===\n";
  for (int i = 0; i < 3; ++i) {
    auto rp = sphere.random_point();
    std::cout << "  random_point " << i << ": " << rp.transpose().format(fmt)
              << "  (norm=" << rp.norm() << ")\n";
  }

  return 0;
}
