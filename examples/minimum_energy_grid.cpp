/// @file minimum_energy_grid.cpp
/// @brief Evaluate KE and Jacobi metrics on a grid over T^2 (2-link planar arm).
///
/// Outputs a JSON file with:
///   - fine grid (50x50) of potential and det(M) values for background heatmaps
///   - coarse grid (12x12) of inverse metric tensors for ellipse visualization
///
/// Usage:
///   ./minimum_energy_grid [output.json]

#include <Eigen/Core>
#include <Eigen/LU>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

/// 2-link planar arm parameters.
struct ArmParams {
  double l1 = 1.0, l2 = 1.0;
  double m1 = 1.0, m2 = 1.0;
  double lc1 = 0.5, lc2 = 0.5;
  double I1 = 1.0 / 12.0, I2 = 1.0 / 12.0;
  double g = 9.81;
};

/// Compute the 2x2 mass matrix M(q) for a 2-link planar arm.
///
///   M11 = I1 + I2 + m1*lc1^2 + m2*(l1^2 + lc2^2 + 2*l1*lc2*cos(q2))
///   M12 = M21 = I2 + m2*(lc2^2 + l1*lc2*cos(q2))
///   M22 = I2 + m2*lc2^2
Eigen::Matrix2d mass_matrix(const Eigen::Vector2d& q, const ArmParams& p) {
  double c2 = std::cos(q[1]);
  double h = p.l1 * p.lc2 * c2;
  Eigen::Matrix2d M;
  M(0, 0) = p.I1 + p.I2 + p.m1 * p.lc1 * p.lc1 + p.m2 * (p.l1 * p.l1 + p.lc2 * p.lc2 + 2.0 * h);
  M(0, 1) = p.I2 + p.m2 * (p.lc2 * p.lc2 + h);
  M(1, 0) = M(0, 1);
  M(1, 1) = p.I2 + p.m2 * p.lc2 * p.lc2;
  return M;
}

/// Gravity potential energy:
///   P(q) = m1*g*lc1*sin(q1) + m2*g*(l1*sin(q1) + lc2*sin(q1+q2))
double potential(const Eigen::Vector2d& q, const ArmParams& p) {
  return p.m1 * p.g * p.lc1 * std::sin(q[0]) +
         p.m2 * p.g * (p.l1 * std::sin(q[0]) + p.lc2 * std::sin(q[0] + q[1]));
}

/// Analytical upper bound on potential energy.
///   Pmax = g * (m1*lc1 + m2*(l1 + lc2))
double pmax_analytical(const ArmParams& p) {
  return p.g * (p.m1 * p.lc1 + p.m2 * (p.l1 + p.lc2));
}

/// Write a 2x2 matrix as [[a,b],[c,d]].
void write_matrix2d(std::ostream& out, const Eigen::Matrix2d& M) {
  out << "[[" << M(0, 0) << "," << M(0, 1) << "],[" << M(1, 0) << "," << M(1, 1) << "]]";
}

}  // namespace

int main(int argc, char* argv[]) {
  std::string output_file = "minimum_energy_grid.json";
  if (argc > 1) output_file = argv[1];

  ArmParams arm;
  const double pmax = pmax_analytical(arm);

  // Grid parameters
  const int N_fine = 50;
  const int N_ellipse = 12;
  const double lo = -M_PI;
  const double hi = M_PI;

  // Build fine grid coordinates
  std::vector<double> q1_fine(N_fine), q2_fine(N_fine);
  for (int i = 0; i < N_fine; ++i) {
    q1_fine[i] = lo + (hi - lo) * i / (N_fine - 1);
    q2_fine[i] = lo + (hi - lo) * i / (N_fine - 1);
  }

  // Build coarse grid coordinates (ellipse centers)
  std::vector<double> q1_coarse(N_ellipse), q2_coarse(N_ellipse);
  for (int i = 0; i < N_ellipse; ++i) {
    // Place ellipses slightly inset from the edges
    q1_coarse[i] = lo + (hi - lo) * (i + 0.5) / N_ellipse;
    q2_coarse[i] = lo + (hi - lo) * (i + 0.5) / N_ellipse;
  }

  // Evaluate potential and det(M) on fine grid
  // potential_grid[i][j] = P(q1_fine[i], q2_fine[j])
  // det_grid[i][j] = det(M(q1_fine[i], q2_fine[j]))
  std::vector<std::vector<double>> pot_grid(N_fine, std::vector<double>(N_fine));
  std::vector<std::vector<double>> det_grid(N_fine, std::vector<double>(N_fine));
  for (int i = 0; i < N_fine; ++i) {
    for (int j = 0; j < N_fine; ++j) {
      Eigen::Vector2d q(q1_fine[i], q2_fine[j]);
      pot_grid[i][j] = potential(q, arm);
      det_grid[i][j] = mass_matrix(q, arm).determinant();
    }
  }

  // Jacobi H values
  const std::vector<double> h_factors = {1.2, 2.0, 5.0};

  // Open output file
  std::ofstream out(output_file);
  if (!out) {
    std::cerr << "Error: cannot open " << output_file << "\n";
    return 1;
  }
  out << std::fixed << std::setprecision(8);

  // --- Write JSON ---
  out << "{\n";

  // Arm parameters
  out << "  \"arm\": {\n";
  out << "    \"l1\": " << arm.l1 << ", \"l2\": " << arm.l2 << ",\n";
  out << "    \"m1\": " << arm.m1 << ", \"m2\": " << arm.m2 << ",\n";
  out << "    \"lc1\": " << arm.lc1 << ", \"lc2\": " << arm.lc2 << ",\n";
  out << "    \"I1\": " << arm.I1 << ", \"I2\": " << arm.I2 << ",\n";
  out << "    \"g\": " << arm.g << "\n";
  out << "  },\n";
  out << "  \"pmax\": " << pmax << ",\n";

  // Fine grid
  out << "  \"grid_fine\": {\n";
  out << "    \"n\": " << N_fine << ",\n";

  out << "    \"q1\": [";
  for (int i = 0; i < N_fine; ++i) {
    out << q1_fine[i];
    if (i + 1 < N_fine) out << ", ";
  }
  out << "],\n";

  out << "    \"q2\": [";
  for (int j = 0; j < N_fine; ++j) {
    out << q2_fine[j];
    if (j + 1 < N_fine) out << ", ";
  }
  out << "],\n";

  // potential[i][j]: row = q1 index, col = q2 index
  out << "    \"potential\": [\n";
  for (int i = 0; i < N_fine; ++i) {
    out << "      [";
    for (int j = 0; j < N_fine; ++j) {
      out << pot_grid[i][j];
      if (j + 1 < N_fine) out << ", ";
    }
    out << "]";
    if (i + 1 < N_fine) out << ",";
    out << "\n";
  }
  out << "    ],\n";

  // det_M[i][j]: determinant of mass matrix at (q1_fine[i], q2_fine[j])
  out << "    \"det_M\": [\n";
  for (int i = 0; i < N_fine; ++i) {
    out << "      [";
    for (int j = 0; j < N_fine; ++j) {
      out << det_grid[i][j];
      if (j + 1 < N_fine) out << ", ";
    }
    out << "]";
    if (i + 1 < N_fine) out << ",";
    out << "\n";
  }
  out << "    ]\n";
  out << "  },\n";

  // Ellipse grid
  out << "  \"grid_ellipse\": {\n";
  out << "    \"n\": " << N_ellipse << ",\n";

  out << "    \"q1\": [";
  for (int i = 0; i < N_ellipse; ++i) {
    out << q1_coarse[i];
    if (i + 1 < N_ellipse) out << ", ";
  }
  out << "],\n";

  out << "    \"q2\": [";
  for (int j = 0; j < N_ellipse; ++j) {
    out << q2_coarse[j];
    if (j + 1 < N_ellipse) out << ", ";
  }
  out << "],\n";

  // KE metric inverse tensors
  out << "    \"ke\": [\n";
  bool first_ke = true;
  for (int i = 0; i < N_ellipse; ++i) {
    for (int j = 0; j < N_ellipse; ++j) {
      Eigen::Vector2d q(q1_coarse[i], q2_coarse[j]);
      Eigen::Matrix2d M = mass_matrix(q, arm);
      Eigen::Matrix2d M_inv = M.inverse();

      if (!first_ke) out << ",\n";
      first_ke = false;
      out << "      { \"q\": [" << q[0] << ", " << q[1] << "], \"M_inv\": ";
      write_matrix2d(out, M_inv);
      out << " }";
    }
  }
  out << "\n    ],\n";

  // Jacobi metric inverse tensors for each H factor
  out << "    \"jacobi\": [\n";
  for (size_t hi_idx = 0; hi_idx < h_factors.size(); ++hi_idx) {
    double alpha = h_factors[hi_idx];
    double H = alpha * pmax;

    out << "      {\n";
    out << "        \"H_over_Pmax\": " << alpha << ",\n";
    out << "        \"H\": " << H << ",\n";
    out << "        \"ellipses\": [\n";

    bool first_j = true;
    for (int i = 0; i < N_ellipse; ++i) {
      for (int j = 0; j < N_ellipse; ++j) {
        Eigen::Vector2d q(q1_coarse[i], q2_coarse[j]);
        Eigen::Matrix2d M = mass_matrix(q, arm);
        Eigen::Matrix2d M_inv = M.inverse();
        double P_q = potential(q, arm);
        double scale = 2.0 * (H - P_q);
        // J_inv = M_inv / scale  (inverse of Jacobi metric tensor)
        Eigen::Matrix2d J_inv = M_inv / scale;

        if (!first_j) out << ",\n";
        first_j = false;
        out << "          { \"q\": [" << q[0] << ", " << q[1] << "], \"J_inv\": ";
        write_matrix2d(out, J_inv);
        out << " }";
      }
    }
    out << "\n        ]\n";
    out << "      }";
    if (hi_idx + 1 < h_factors.size()) out << ",";
    out << "\n";
  }
  out << "    ]\n";

  out << "  }\n";
  out << "}\n";

  std::cout << "Wrote " << output_file << "\n";
  std::cout << "  Pmax (analytical) = " << pmax << " J\n";
  std::cout << "  Fine grid:   " << N_fine << "x" << N_fine << "\n";
  std::cout << "  Coarse grid: " << N_ellipse << "x" << N_ellipse << "\n";
  return 0;
}
