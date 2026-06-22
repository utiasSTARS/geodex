/// @file bind_pinocchio.cpp
/// @brief Python bindings for the geodex::integration::pinocchio submodule.
///
/// @details Exposes URDF-driven primitives (`MassMatrix`, `FrameJacobian`,
/// joint-limit and model-nq utilities) plus three `make_pullback_metric`
/// overloads (unregularized, isotropic-regularization, kinetic-energy
/// regularization) under the `geodex.pinocchio` Python submodule. The
/// regularized overloads land as separately named functions
/// (`make_pullback_metric_iso`, `make_pullback_metric_ke`) since their C++
/// counterparts return distinct concrete types selected by tag dispatch.

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "geodex/integration/pinocchio/jacobian.hpp"
#include "geodex/integration/pinocchio/mass_matrix.hpp"
#include "geodex/integration/pinocchio/pullback.hpp"

#include "wrappers/py_metrics.hpp"

namespace nb = nanobind;
namespace gpin = geodex::integration::pinocchio;
using namespace geodex::python;

namespace {

JacobianFn make_jacobian_fn(const std::string& urdf_path,
                            const std::vector<std::string>& ee_frames, bool position_only) {
  auto impl = std::make_shared<gpin::detail::FrameJacobianImpl>(urdf_path, ee_frames,
                                                                position_only);
  return [impl](const Eigen::VectorXd& q) -> Eigen::MatrixXd { return (*impl)(q); };
}

MassMatrixFn make_mass_fn(const std::string& urdf_path) {
  auto mm = std::make_shared<gpin::MassMatrix>(urdf_path);
  return [mm](const Eigen::VectorXd& q) -> Eigen::MatrixXd { return (*mm)(q); };
}

PyPullbackMetric make_pullback_unregularized(const std::string& urdf_path,
                                             const gpin::PullbackOptions& opts) {
  gpin::detail::FrameJacobianImpl probe(urdf_path, opts.ee_frames, /*position_only=*/false);
  const int task_d = gpin::detail::task_dim(probe);
  const Eigen::MatrixXd W = gpin::detail::build_task_weight(task_d, opts.task_weights);
  JacobianFn jac_fn = make_jacobian_fn(urdf_path, opts.ee_frames, /*position_only=*/false);
  TaskMetricFn task_fn = [W](const Eigen::VectorXd&) -> Eigen::MatrixXd { return W; };
  return PyPullbackMetric(std::move(jac_fn), std::move(task_fn), 0.0);
}

}  // namespace

void bind_pinocchio(nb::module_& m) {
  auto p = m.def_submodule("pinocchio", "URDF-driven primitives (CRBA, frame Jacobians, "
                                        "pullback-metric factory).");

  // --- MassMatrix ---
  using MassMatrix = gpin::MassMatrix;
  nb::class_<MassMatrix>(
      p, "MassMatrix",
      "Joint-space mass matrix evaluator backed by Pinocchio's CRBA.\n\n"
      "Loads a URDF once at construction and evaluates M(q) in place via a\n"
      "mutable cached buffer. Not thread-safe; use one instance per thread.")
      .def(nb::init<const std::string&>(), nb::arg("urdf_path"),
           "Load the URDF and allocate the CRBA data buffer.")
      .def(
          "__call__",
          [](const MassMatrix& self, const Eigen::VectorXd& q) -> Eigen::MatrixXd {
            return self(q);  // copy out of the internal buffer
          },
          nb::arg("q"), "Compute M(q) via CRBA.");

  p.def(
      "mass_matrix",
      [](const std::string& urdf_path) { return gpin::MassMatrix(urdf_path); },
      nb::arg("urdf_path"), "Construct a MassMatrix from a URDF path.");

  p.def(
      "model_nq",
      [](const std::string& urdf_path) { return gpin::model_nq(urdf_path); },
      nb::arg("urdf_path"), "Number of generalized coordinates (nq) in the URDF.");

  p.def(
      "joint_limits",
      [](const std::string& urdf_path) { return gpin::joint_limits(urdf_path); },
      nb::arg("urdf_path"),
      "Return (lower, upper) joint position limits as numpy arrays of size nq.");

  // --- FrameJacobian ---
  using FrameJacobian = gpin::detail::FrameJacobianImpl;
  nb::class_<FrameJacobian>(
      p, "FrameJacobian",
      "Stacked frame-Jacobian evaluator in LOCAL_WORLD_ALIGNED coordinates.\n\n"
      "Construct via `frame_jacobian`, `stacked_jacobian`, "
      "`frame_position_jacobian`, or `stacked_position_jacobian`. Returns the\n"
      "(stacked) Jacobian as a dense matrix; not thread-safe.")
      .def(
          "__call__",
          [](const FrameJacobian& self, const Eigen::VectorXd& q) -> Eigen::MatrixXd {
            return self(q);
          },
          nb::arg("q"), "Evaluate the (stacked) Jacobian at q.");

  p.def(
      "frame_jacobian",
      [](const std::string& urdf, const std::string& ee_frame) {
        return gpin::frame_jacobian(urdf, ee_frame);
      },
      nb::arg("urdf_path"), nb::arg("ee_frame") = std::string{},
      "Single-frame full Jacobian (6 x nv) callable. Empty `ee_frame` triggers "
      "auto-detect of the last BODY frame attached to the final movable joint.");

  p.def(
      "stacked_jacobian",
      [](const std::string& urdf, const std::vector<std::string>& ee_frames) {
        return gpin::stacked_jacobian(urdf, ee_frames);
      },
      nb::arg("urdf_path"), nb::arg("ee_frames"),
      "Multi-frame stacked full Jacobian (6K x nv) callable.");

  p.def(
      "frame_position_jacobian",
      [](const std::string& urdf, const std::string& ee_frame) {
        return gpin::frame_position_jacobian(urdf, ee_frame);
      },
      nb::arg("urdf_path"), nb::arg("ee_frame") = std::string{},
      "Single-frame position-only Jacobian (3 x nv) callable.");

  p.def(
      "stacked_position_jacobian",
      [](const std::string& urdf, const std::vector<std::string>& ee_frames) {
        return gpin::stacked_position_jacobian(urdf, ee_frames);
      },
      nb::arg("urdf_path"), nb::arg("ee_frames"),
      "Multi-frame stacked position-only Jacobian (3K x nv) callable.");

  // --- PullbackOptions ---
  using PullbackOptions = gpin::PullbackOptions;
  nb::class_<PullbackOptions>(
      p, "PullbackOptions",
      "End-effector selection + per-axis task weights for `make_pullback_metric`.\n\n"
      "task_weights = (x, y, z, r, p, y); zero an entry to drop that axis.")
      .def(nb::init<>(),
           "Default options: empty ee_frames (auto-detect), weights (1,1,1,0.1,0.1,0.1).")
      .def_rw("ee_frames", &PullbackOptions::ee_frames,
              "End-effector frame names. Empty -> auto-detect last BODY frame.")
      .def_rw("task_weights", &PullbackOptions::task_weights,
              "Per-axis task-space weights (x, y, z, r, p, y).");

  // --- make_pullback_metric: unregularized ---
  p.def(
      "make_pullback_metric",
      [](const std::string& urdf, const PullbackOptions& opts) {
        return make_pullback_unregularized(urdf, opts);
      },
      nb::arg("urdf_path"), nb::arg("options"),
      "Build an unregularized pullback metric (PullbackMetric) from a URDF.\n\n"
      "Returns a geodex.PullbackMetric whose Jacobian is a Pinocchio frame\n"
      "Jacobian and whose task metric is the constant SPD matrix shaped by\n"
      "`options.task_weights`.");

  // --- make_pullback_metric_iso: + lambda * I ---
  p.def(
      "make_pullback_metric_iso",
      [](const std::string& urdf, const PullbackOptions& opts, double lam) {
        if (!(lam > 0.0)) {
          throw std::invalid_argument(
              "make_pullback_metric_iso requires lam > 0; use make_pullback_metric "
              "for the unregularized case.");
        }
        PyPullbackMetric pullback = make_pullback_unregularized(urdf, opts);
        const int nq = gpin::model_nq(urdf);
        PyConstantSPDMetric identity(Eigen::MatrixXd::Identity(nq, nq));

        std::vector<DynamicMetric> bases;
        bases.push_back(pullback.to_dynamic_metric());
        bases.push_back(identity.to_dynamic_metric());
        return PyAffineCombinedMetric(std::move(bases), std::vector<double>{1.0, lam});
      },
      nb::arg("urdf_path"), nb::arg("options"), nb::arg("lam"),
      "Build a pullback metric with isotropic regularization + lam * I.\n\n"
      "Returns a geodex.AffineCombinedMetric of arity 2 with coefficients\n"
      "(1.0, lam) over (PullbackMetric, IdentityMetric).");

  // --- make_pullback_metric_ke: + beta * M_CRBA(q) ---
  p.def(
      "make_pullback_metric_ke",
      [](const std::string& urdf, const PullbackOptions& opts, double beta) {
        if (!(beta > 0.0)) {
          throw std::invalid_argument(
              "make_pullback_metric_ke requires beta > 0; use make_pullback_metric "
              "for the unregularized case.");
        }
        PyPullbackMetric pullback = make_pullback_unregularized(urdf, opts);
        PyKineticEnergyMetric ke(make_mass_fn(urdf));

        std::vector<DynamicMetric> bases;
        bases.push_back(pullback.to_dynamic_metric());
        bases.push_back(ke.to_dynamic_metric());
        return PyAffineCombinedMetric(std::move(bases), std::vector<double>{1.0, beta});
      },
      nb::arg("urdf_path"), nb::arg("options"), nb::arg("beta"),
      "Build a pullback metric with kinetic-energy regularization + beta * M_CRBA(q).\n\n"
      "Returns a geodex.AffineCombinedMetric of arity 2 with coefficients\n"
      "(1.0, beta) over (PullbackMetric, KineticEnergyMetric).");
}
