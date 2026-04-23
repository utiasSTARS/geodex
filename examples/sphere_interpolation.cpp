/// @file sphere_interpolation.cpp
/// @brief Demonstrates `discrete_geodesic` on the sphere across several
/// metric / retraction combinations and writes a JSON file for visualisation
/// via `scripts/visualize.py`.
///
/// Run the example, then visualise:
/// ```
/// ./build/examples/sphere_interpolation sphere_interpolation.json
/// python scripts/visualize.py sphere_interpolation.json -o sphere.html
/// open sphere.html
/// ```

#include <cmath>
#include <cstdio>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "geodex/geodex.hpp"

using namespace geodex;

static Eigen::Vector3d point_at_theta(double theta) {
  return Eigen::Vector3d(std::sin(theta), 0.0, std::cos(theta));
}

/// Point on the unit sphere at polar angle `theta` from the north pole and
/// azimuth `phi` around the z axis (0 → x-axis, π/2 → y-axis).
static Eigen::Vector3d spherical_point(double theta, double phi) {
  return Eigen::Vector3d(std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi),
                         std::cos(theta));
}

/// Write a JSON array of 3D points: [[x,y,z], ...]
static void write_points(std::ofstream& out, const std::vector<Eigen::Vector3d>& pts) {
  out << "[";
  for (size_t i = 0; i < pts.size(); ++i) {
    if (i > 0) out << ",";
    out << "\n        [" << std::setprecision(8) << pts[i][0] << "," << pts[i][1] << ","
        << pts[i][2] << "]";
  }
  out << "\n      ]";
}

static void write_vec3(std::ofstream& out, const Eigen::Vector3d& v) {
  out << "[" << std::setprecision(8) << v[0] << "," << v[1] << "," << v[2] << "]";
}

/// A single path entry for the visualization JSON.
struct PathEntry {
  std::string label;
  Eigen::Vector3d start;
  Eigen::Vector3d target;
  std::vector<Eigen::Vector3d> points;
};

/// Run `discrete_geodesic` on the given sphere and return a labelled path entry
/// plus a short console summary.
template <typename SphereT>
static PathEntry run_scenario(const std::string& label, const SphereT& sphere,
                              const Eigen::Vector3d& start, const Eigen::Vector3d& target,
                              const InterpolationSettings& settings) {
  auto result = discrete_geodesic(sphere, start, target, settings);

  std::printf(
      "  [%-48s] status=%-17s iters=%4d halvings=%3d initial_d=%.4f final_d=%.2e points=%zu\n",
      label.c_str(), to_string(result.status), result.iterations, result.distortion_halvings,
      result.initial_distance, result.final_distance, result.path.size());

  return PathEntry{label, start, target, std::move(result.path)};
}

static void write_json(const std::string& output_file, const std::vector<PathEntry>& entries) {
  std::ofstream out(output_file);
  if (!out) {
    std::cerr << "Error: cannot open " << output_file << "\n";
    std::exit(1);
  }

  out << std::fixed;
  out << "{\n";
  out << "  \"manifold\": { \"type\": \"S2\" },\n";
  out << "  \"paths\": [\n";
  for (size_t i = 0; i < entries.size(); ++i) {
    const auto& e = entries[i];
    out << "    {\n";
    out << "      \"label\": \"" << e.label << "\",\n";
    out << "      \"start\": ";
    write_vec3(out, e.start);
    out << ",\n";
    out << "      \"target\": ";
    write_vec3(out, e.target);
    out << ",\n";
    out << "      \"points\": ";
    write_points(out, e.points);
    out << "\n    }" << (i + 1 == entries.size() ? "\n" : ",\n");
  }
  out << "  ]\n";
  out << "}\n";

  std::cout << "Wrote " << output_file << "\n";
}

int main(int argc, char* argv[]) {
  std::string output_file = "sphere_interpolation.json";
  if (argc > 1) output_file = argv[1];

  // Common start point at the north pole.
  const Eigen::Vector3d north(0.0, 0.0, 1.0);

  std::vector<PathEntry> entries;

  std::cout << "Running discrete_geodesic scenarios on the sphere:\n";

  // All anisotropic scenarios use a target with non-zero x AND y components so
  // the metric's directional preference is actually visible. A target on the
  // y=0 plane makes the problem one-dimensional along the great circle, and
  // any metric yields the same (straight great-circle) path.
  const double theta = 1.3;  // polar angle from north
  const double phi = 0.9;    // azimuth in x-y plane (~52 deg)
  const Eigen::Vector3d shared_target = spherical_point(theta, phi);

  // ---------------------------------------------------------------------
  // 1. Round metric + exponential map — the textbook great-circle geodesic.
  //    `log` is the Riemannian logarithm, `-log` is the natural gradient,
  //    and the path lies exactly on the great-circle arc.
  // ---------------------------------------------------------------------
  {
    Sphere<> sphere;  // default: SphereRoundMetric + SphereExponentialMap
    InterpolationSettings s;
    s.step_size = 0.1;
    entries.push_back(
        run_scenario("1. Round metric, true exp/log", sphere, north, shared_target, s));
  }

  // ---------------------------------------------------------------------
  // 2. Round metric + projection retraction — the retraction is first-order,
  //    but the metric is still the round one so the walk still follows the
  //    great circle. Provides a reference for "projection retraction alone"
  //    vs "anisotropic metric" effects.
  // ---------------------------------------------------------------------
  {
    Sphere<2, SphereRoundMetric, SphereProjectionRetraction> sphere;
    InterpolationSettings s;
    s.step_size = 0.1;
    entries.push_back(
        run_scenario("2. Round metric, projection retraction", sphere, north, shared_target, s));
  }

  // ---------------------------------------------------------------------
  // 3. Moderate anisotropic metric — A = diag(4, 1, 1) penalises motion in
  //    the x direction. The optimal path trades y-motion for less x-motion
  //    early on, producing a visible bend away from the great circle.
  //    `has_riemannian_log` is false for ConstantSPDMetric, so
  //    `discrete_geodesic` always uses the FD natural gradient here.
  // ---------------------------------------------------------------------
  {
    Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
    A(0, 0) = 4.0;
    Sphere<2, ConstantSPDMetric<3>> sphere{ConstantSPDMetric<3>{A}};
    InterpolationSettings s;
    s.step_size = 0.1;
    entries.push_back(
        run_scenario("3. Anisotropic metric A=diag(4,1,1)", sphere, north, shared_target, s));
  }

  // ---------------------------------------------------------------------
  // 4. Heavy anisotropic metric — A = diag(25, 1, 1). The bend toward
  //    y-motion should be pronounced. Smaller step_size keeps the walk
  //    stable under the stretched metric.
  // ---------------------------------------------------------------------
  {
    Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
    A(0, 0) = 25.0;
    Sphere<2, ConstantSPDMetric<3>> sphere{ConstantSPDMetric<3>{A}};
    InterpolationSettings s;
    s.step_size = 0.05;
    s.max_steps = 500;
    entries.push_back(
        run_scenario("4. Anisotropic metric A=diag(25,1,1)", sphere, north, shared_target, s));
  }

  // ---------------------------------------------------------------------
  // 5. Near-antipodal — shows graceful degradation near the cut locus. The
  //    log is still well-defined (distance < pi), so this should converge.
  // ---------------------------------------------------------------------
  {
    Sphere<> sphere;
    InterpolationSettings s;
    s.step_size = 0.2;
    const Eigen::Vector3d near_antipodal =
        spherical_point(std::numbers::pi - 0.1, 0.9);  // just inside cut
    entries.push_back(
        run_scenario("5. Round metric, near-antipodal", sphere, north, near_antipodal, s));
  }

  std::cout << "\n";
  write_json(output_file, entries);
  std::cout << "\nVisualise with:\n";
  std::cout << "  python scripts/visualize.py " << output_file << " -o sphere.html\n";
  return 0;
}
