/// @file bind_heuristics.cpp
/// @brief Python bindings for the geodex::heuristics admissible-heuristics suite.
///
/// @details Exposes `Zero`, `Euclidean`, `EigenvalueLowerBound` (with default
/// `Euclidean` base), and the dynamic-dimension `MatrixLowerBound<Dynamic>` as
/// classes inside the `geodex.heuristics` submodule, and `product_lower_bound` as a
/// function.

#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "geodex/heuristics/heuristics.hpp"

#include "wrappers/sizes.hpp"

namespace nb = nanobind;
using geodex::python::require_shape;
using geodex::python::require_size;

namespace {

using MLB = geodex::heuristics::MatrixLowerBound<Eigen::Dynamic>;

// A factor of product_lower_bound, its bound matrix or a (matrix, periods) pair.
using Factor =
    std::variant<Eigen::MatrixXd, std::pair<Eigen::MatrixXd, std::optional<Eigen::VectorXd>>>;

// Checks that M_lower of a MatrixLowerBound is square before the Cholesky factorization.
const Eigen::MatrixXd& square(const Eigen::MatrixXd& m) {
  require_shape(m, m.rows(), m.rows(), "MatrixLowerBound", "M_lower");
  return m;
}

}  // namespace

void bind_heuristics(nb::module_& m) {
  auto h = m.def_submodule("heuristics", "Admissible heuristics for motion planning.");

  // --- Zero ---
  nb::class_<geodex::heuristics::Zero>(
      h, "Zero",
      "Zero heuristic, h(a, b) = 0 for every pair.\n\n"
      "The weakest admissible heuristic. It is admissible for any non-negative distance\n"
      "and does not add information. With an informed planner, the informed set is the full\n"
      "configuration space, sampling stays uniform and the planner does not prune any\n"
      "vertex.")
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
      "Computes ||a - b||_2. It is admissible when the chord distance bounds the geodesic\n"
      "distance from below, for example when lambda_min(M(q)) >= 1 everywhere. It\n"
      "overestimates when lambda_min < 1 in some direction.")
      .def(nb::init<>(), "Create a Euclidean heuristic.")
      .def(
          "__call__",
          [](const geodex::heuristics::Euclidean& self, const Eigen::VectorXd& a,
             const Eigen::VectorXd& b) {
            require_size(b, a.size(), "Euclidean", "b");
            return self(a, b);
          },
          nb::arg("a"), nb::arg("b"), "Compute ||a - b||_2.");

  // --- EigenvalueLowerBound (default base = Euclidean) ---
  using Eig = geodex::heuristics::EigenvalueLowerBound<geodex::heuristics::Euclidean>;
  nb::class_<Eig>(h, "EigenvalueLowerBound",
                  "Eigenvalue lower-bound heuristic for configuration-dependent metrics.\n\n"
                  "For a Riemannian metric M(q), the geodesic distance satisfies\n\n"
                  "    d_M(a, b) >= sqrt(lambda_min) * ||a - b||_2,\n\n"
                  "where lambda_min is a global lower bound on the eigenvalues of M(q).\n"
                  "It is tighter than `Zero` and looser than `MatrixLowerBound`.")
      .def(nb::init<double>(), nb::arg("lambda_min"),
           "Construct from the global minimum eigenvalue lambda_min of M(q).")
      .def(
          "__call__",
          [](const Eig& self, const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
            require_size(b, a.size(), "EigenvalueLowerBound", "b");
            return self(a, b);
          },
          nb::arg("a"), nb::arg("b"), "Compute sqrt(lambda_min) * ||a - b||_2.")
      .def_prop_ro("sqrt_lambda_min", &Eig::sqrt_lambda_min, "Cached sqrt(lambda_min).");

  // --- MatrixLowerBound<Dynamic> ---
  nb::class_<MLB>(h, "MatrixLowerBound",
                  "Matrix lower-bound heuristic via a constant SPD Loewner lower bound.\n\n"
                  "For a metric M(q) with M(q) >= M_lower in the Loewner order, the geodesic\n"
                  "distance satisfies\n\n"
                  "    d_M(a, b) >= sqrt((a - b)^T M_lower (a - b)).\n\n"
                  "It keeps directional information and is tighter than the scalar eigenvalue\n"
                  "bound. The heuristic caches the Cholesky factor L and evaluates\n"
                  "||L^T (a - b)||_2. An optional eigenvalue floor lambda_min makes it dominate\n"
                  "the scalar bound in every direction.\n\n"
                  "On a flat quotient (SE2, SO2, Torus), pass the manifold's periods. The\n"
                  "difference then wraps to its nearest representative first. Without periods,\n"
                  "the bound overshoots near a branch cut and is inadmissible there.")
      .def(
          "__init__",
          [](MLB* self, const Eigen::MatrixXd& M_lower) { new (self) MLB(square(M_lower)); },
          nb::arg("M_lower"), "Construct from an SPD matrix M_lower satisfying M(q) >= M_lower.")
      .def(
          "__init__",
          [](MLB* self, const Eigen::MatrixXd& M_lower, const double lambda_min) {
            new (self) MLB(square(M_lower), lambda_min);
          },
          nb::arg("M_lower"), nb::arg("lambda_min"),
          "Construct from an SPD matrix M_lower with an eigenvalue floor lambda_min.")
      .def(
          "__init__",
          [](MLB* self, const Eigen::MatrixXd& M_lower, const Eigen::VectorXd& periods) {
            new (self) MLB(square(M_lower), periods);
          },
          nb::arg("M_lower"), nb::arg("periods"),
          "Construct from an SPD matrix M_lower and per-axis coordinate periods.\n\n"
          "periods holds one entry per coordinate, 0 where the axis is not periodic\n"
          "(typically manifold.periods()). Raises ValueError on a size mismatch, or\n"
          "when a periodic axis is coupled to another axis in M_lower.")
      .def(
          "__call__",
          [](const MLB& self, const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
            const Eigen::Index n = self.size();
            require_size(a, n, "MatrixLowerBound", "a");
            require_size(b, n, "MatrixLowerBound", "b");
            return self(a, b);
          },
          nb::arg("a"), nb::arg("b"),
          "Compute the admissible lower bound on geodesic distance. Raises ValueError\n"
          "unless a and b have the size of M_lower.")
      .def(
          "update",
          [](MLB& self, const Eigen::MatrixXd& M_new) {
            const Eigen::Index n = self.size();
            require_shape(M_new, n, n, "MatrixLowerBound.update", "M_new");
            return self.update(M_new);
          },
          nb::arg("M_new"),
          "Incremental Loewner-meet update with a new SPD observation.\n\n"
          "Returns True if the update loosened the bound, and False if the current M_lower\n"
          "already dominates the new observation.")
      .def("matrix", &MLB::matrix, "Reconstruct the current M_lower from its Cholesky factor.")
      .def("det", &MLB::det, "Determinant of the current M_lower.")
      .def("eigenvalues", &MLB::eigenvalues,
           "Eigenvalues of the current M_lower in ascending order.")
      .def_prop_ro("has_eigenvalue_floor", &MLB::has_eigenvalue_floor,
                   "Whether an eigenvalue floor is set.")
      .def_prop_ro("periods", &MLB::periods, nb::rv_policy::copy,
                   "Per-axis coordinate periods, empty when no axis is periodic.");

  // --- product_lower_bound ---
  h.def(
      "product_lower_bound",
      [](const std::vector<Factor>& factors) {
        std::vector<geodex::heuristics::FactorBound> bounds;
        bounds.reserve(factors.size());
        for (const auto& f : factors) {
          if (const auto* m = std::get_if<Eigen::MatrixXd>(&f)) {
            bounds.push_back({*m, {}});
          } else {
            const auto& [m_lower, periods] = std::get<1>(f);
            bounds.push_back({m_lower, periods.value_or(Eigen::VectorXd())});
          }
        }
        return geodex::heuristics::product_lower_bound(bounds);
      },
      nb::arg("factors"),
      "Matrix lower-bound heuristic of a product metric from the bounds of its factors.\n\n"
      "The metric of a product manifold is the direct sum of the factor metrics. If M_i\n"
      "bounds factor i from below, the block-diagonal matrix of the M_i bounds the product\n"
      "metric from below. The heuristic takes that matrix and the factors' periods in the\n"
      "same order.\n\n"
      "Args:\n"
      "    factors: One entry per factor, in the order of the product's coordinates. An\n"
      "        entry is the factor's bound matrix, or a tuple (matrix, periods) for a factor\n"
      "        with periodic coordinates, such as (metric.coordinate_lower_bound(),\n"
      "        se2.periods()).\n\n"
      "Returns:\n"
      "    A MatrixLowerBound of the block-diagonal bound, with periods when a factor has\n"
      "    them.\n\n"
      "Raises:\n"
      "    ValueError: factors is empty, a matrix is not square, or a factor's periods do\n"
      "        not match its matrix.");
}
