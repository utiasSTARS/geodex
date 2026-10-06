/// @file extract_metric.hpp
/// @brief Convert any known Python metric object into a type-erased DynamicMetric.

#pragma once

#include <memory>
#include <stdexcept>

#include <nanobind/nanobind.h>

#include "dynamic_manifold.hpp"
#include "py_callable.hpp"
#include "py_metrics.hpp"

namespace geodex::python {

namespace detail {

/// @brief A DynamicMetric that calls `metric` in place for as long as `ref` holds it.
template <typename MetricT>
DynamicMetric borrow_metric(const MetricT& metric, std::shared_ptr<const PyOwnedRef> ref) {
  const MetricT* m = &metric;
  const auto held = [](const PyOwnedRef& r) {
    if (!r.get()) throw std::runtime_error("metric was released by the garbage collector");
  };
  return DynamicMetric{
      [m, ref, held](const Eigen::VectorXd& p, const Eigen::VectorXd& u, const Eigen::VectorXd& v) {
        held(*ref);
        return m->inner(p, u, v);
      },
      [m, ref, held](const Eigen::VectorXd& p, const Eigen::VectorXd& v) {
        held(*ref);
        return m->norm(p, v);
      }};
}

inline constexpr const char* kUnknownMetric =
    "Unknown metric type. Expected KineticEnergyMetric, JacobiMetric, PullbackMetric, "
    "ConstantSPDMetric, SE2LeftInvariantMetric, WeightedMetric, AffineCombinedMetric, or "
    "ClearanceMetric.";

}  // namespace detail

/// @brief Extract a DynamicMetric holding its own copy of any known Python metric.
/// @throws std::invalid_argument if `obj` is not a recognized metric.
inline DynamicMetric extract_dynamic_metric(nanobind::handle obj) {
  namespace nb = nanobind;
  if (nb::isinstance<PyKineticEnergyMetric>(obj))
    return nb::cast<const PyKineticEnergyMetric&>(obj).to_dynamic_metric();
  if (nb::isinstance<PyJacobiMetric>(obj))
    return nb::cast<const PyJacobiMetric&>(obj).to_dynamic_metric();
  if (nb::isinstance<PyPullbackMetric>(obj))
    return nb::cast<const PyPullbackMetric&>(obj).to_dynamic_metric();
  if (nb::isinstance<PyConstantSPDMetric>(obj))
    return nb::cast<const PyConstantSPDMetric&>(obj).to_dynamic_metric();
  if (nb::isinstance<PySE2LeftInvariantMetric>(obj))
    return nb::cast<const PySE2LeftInvariantMetric&>(obj).to_dynamic_metric();
  if (nb::isinstance<PyWeightedMetric>(obj))
    return nb::cast<const PyWeightedMetric&>(obj).to_dynamic_metric();
  if (nb::isinstance<PyAffineCombinedMetric>(obj))
    return nb::cast<const PyAffineCombinedMetric&>(obj).to_dynamic_metric();
  if (nb::isinstance<PyClearanceMetric>(obj))
    return nb::cast<const PyClearanceMetric&>(obj).to_dynamic_metric();
  throw std::invalid_argument(detail::kUnknownMetric);
}

/// @brief A DynamicMetric that calls the Python metric object in place.
///
/// @details `ref` keeps the object alive. After its owner's `tp_clear` releases it, every
/// call throws. Unlike `extract_dynamic_metric`, it does not copy any callable, and `ref`
/// holds the only reference to the metric.
/// @throws std::invalid_argument if the referenced object is not a recognized metric.
inline DynamicMetric borrow_dynamic_metric(std::shared_ptr<const PyOwnedRef> ref) {
  namespace nb = nanobind;
  const nb::handle obj(ref->get());
  if (nb::isinstance<PyKineticEnergyMetric>(obj))
    return detail::borrow_metric(nb::cast<const PyKineticEnergyMetric&>(obj), ref);
  if (nb::isinstance<PyJacobiMetric>(obj))
    return detail::borrow_metric(nb::cast<const PyJacobiMetric&>(obj), ref);
  if (nb::isinstance<PyPullbackMetric>(obj))
    return detail::borrow_metric(nb::cast<const PyPullbackMetric&>(obj), ref);
  if (nb::isinstance<PyConstantSPDMetric>(obj))
    return detail::borrow_metric(nb::cast<const PyConstantSPDMetric&>(obj), ref);
  if (nb::isinstance<PySE2LeftInvariantMetric>(obj))
    return detail::borrow_metric(nb::cast<const PySE2LeftInvariantMetric&>(obj), ref);
  if (nb::isinstance<PyWeightedMetric>(obj))
    return detail::borrow_metric(nb::cast<const PyWeightedMetric&>(obj), ref);
  if (nb::isinstance<PyAffineCombinedMetric>(obj))
    return detail::borrow_metric(nb::cast<const PyAffineCombinedMetric&>(obj), ref);
  if (nb::isinstance<PyClearanceMetric>(obj))
    return detail::borrow_metric(nb::cast<const PyClearanceMetric&>(obj), ref);
  throw std::invalid_argument(detail::kUnknownMetric);
}

}  // namespace geodex::python
