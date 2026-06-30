/// @file bind_heuristics.cpp
/// @brief Python bindings for the geodex::heuristics admissible-heuristics suite.
///
/// @details Exposes `Zero`, `Euclidean`, `EigenvalueLowerBound` (with default
/// `Euclidean` base), and the dynamic-dimension `MatrixLowerBound<Dynamic>` as
/// classes inside the `geodex.heuristics` submodule.

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "geodex/heuristics/heuristics.hpp"

namespace nb = nanobind;

void bind_heuristics(nb::module_& m) {
  auto h = m.def_submodule("heuristics", "Admissible heuristics for motion planning.");

  // --- Zero ---
  nb::class_<geodex::heuristics::Zero>(
      h, "Zero",
      "Zero heuristic — returns h(a, b) = 0 for every pair.\n\n"
      "The weakest possible admissible heuristic: admissible for any non-negative\n"
      "distance, but carries no information. With an informed planner the informed\n"
      "set degenerates to the full configuration space (uniform sampling) and no\n"
      "vertices are pruned.")
      .def(nb::init<>(), "Create a Zero heuristic.")
      .def(
          "__call__",
          [](const geodex::heuristics::Zero& self, const Eigen::VectorXd& a,
             const Eigen::VectorXd& b) { return self(a, b); },
          nb::arg("a"), nb::arg("b"), "Compute h(a, b) = 0.");

  // --- Euclidean ---
  nb::class_<geodex::heuristics::Euclidean>(
      h, "Euclidean",
      "Euclidean (L2) chord-distance heuristic.\n\n"
      "Computes ||a - b||_2. Admissible whenever the geodesic distance is bounded\n"
      "below by the chord distance — equivalently, when lambda_min(M(q)) >= 1\n"
      "everywhere. Inadmissible (over-estimating) when lambda_min < 1 in some\n"
      "direction.")
      .def(nb::init<>(), "Create a Euclidean heuristic.")
      .def(
          "__call__",
          [](const geodex::heuristics::Euclidean& self, const Eigen::VectorXd& a,
             const Eigen::VectorXd& b) { return self(a, b); },
          nb::arg("a"), nb::arg("b"), "Compute ||a - b||_2.");

  // --- EigenvalueLowerBound (default base = Euclidean) ---
  using Eig = geodex::heuristics::EigenvalueLowerBound<geodex::heuristics::Euclidean>;
  nb::class_<Eig>(
      h, "EigenvalueLowerBound",
      "Eigenvalue lower-bound heuristic for configuration-dependent metrics.\n\n"
      "For a Riemannian metric M(q), the geodesic distance satisfies\n"
      "    d_M(a, b) >= sqrt(lambda_min) * ||a - b||_2,\n"
      "where lambda_min is a global lower bound on the eigenvalues of M(q).\n"
      "Tighter than `Zero`, looser than `MatrixLowerBound`.")
      .def(nb::init<double>(), nb::arg("lambda_min"),
           "Construct from the global minimum eigenvalue lambda_min of M(q).")
      .def(
          "__call__",
          [](const Eig& self, const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
            return self(a, b);
          },
          nb::arg("a"), nb::arg("b"), "Compute sqrt(lambda_min) * ||a - b||_2.")
      .def_prop_ro("sqrt_lambda_min", &Eig::sqrt_lambda_min, "Cached sqrt(lambda_min).");

  // --- MatrixLowerBound<Dynamic> ---
  using MLB = geodex::heuristics::MatrixLowerBound<Eigen::Dynamic>;
  nb::class_<MLB>(
      h, "MatrixLowerBound",
      "Matrix lower-bound heuristic via a constant SPD Loewner lower bound.\n\n"
      "For a metric M(q) with M(q) >= M_lower in the Loewner order, the geodesic\n"
      "distance satisfies\n"
      "    d_M(a, b) >= sqrt((a - b)^T M_lower (a - b)).\n"
      "Tighter than the scalar eigenvalue bound because directional information is\n"
      "preserved. The Cholesky factor L is cached and the heuristic evaluates\n"
      "||L^T (a - b)||_2. An optional eigenvalue floor lambda_min guarantees\n"
      "dominance over the scalar bound in every direction.")
      .def(nb::init<const Eigen::MatrixXd&>(), nb::arg("M_lower"),
           "Construct from an SPD matrix M_lower satisfying M(q) >= M_lower.")
      .def(nb::init<const Eigen::MatrixXd&, double>(), nb::arg("M_lower"), nb::arg("lambda_min"),
           "Construct from an SPD matrix M_lower with an eigenvalue floor lambda_min.")
      .def(
          "__call__",
          [](const MLB& self, const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
            return self(a, b);
          },
          nb::arg("a"), nb::arg("b"),
          "Compute the admissible lower bound on geodesic distance.")
      .def("update", &MLB::update, nb::arg("M_new"),
           "Incremental Loewner-meet update with a new SPD observation.\n\n"
           "Returns True if the bound was loosened, False if the new observation\n"
           "is already dominated by the current M_lower.")
      .def("matrix", &MLB::matrix, "Reconstruct the current M_lower from its Cholesky factor.")
      .def("det", &MLB::det, "Determinant of the current M_lower.")
      .def("eigenvalues", &MLB::eigenvalues,
           "Eigenvalues of the current M_lower in ascending order.")
      .def_prop_ro("update_count", &MLB::update_count,
                   "Number of times update() actually loosened the bound.")
      .def_prop_ro("has_eigenvalue_floor", &MLB::has_eigenvalue_floor,
                   "Whether an eigenvalue floor is set.");
}
