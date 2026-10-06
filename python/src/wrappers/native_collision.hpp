/// @file native_collision.hpp
/// @brief Bound geodex collision objects that the bindings call in C++, without Python.
///
/// @details ClearanceMetric, InflatedSDF and MemoizedSDF take an SDF callable. When the
/// callable is a bound collision object, such as a FootprintGridChecker, the bindings call
/// the C++ object in place. The value is the one the Python call returns. A Python subclass
/// that overrides `__call__`, and any other callable, runs through Python.

#pragma once

#include <optional>
#include <utility>

#include <Eigen/Core>
#include <nanobind/nanobind.h>

#include "native_sdf.hpp"
#include "py_callable.hpp"

namespace geodex::python {

namespace detail {

/// @brief Whether `obj` is a `T` and the attribute `name` of its type is the one `T` binds.
/// A Python subclass that overrides the attribute does not count.
template <typename T>
bool binds_own(nanobind::handle obj, const char* name) {
  namespace nb = nanobind;
  const nb::handle cls = nb::type<T>();
  if (!cls.is_valid() || !nb::isinstance<T>(obj)) return false;
  return nb::getattr(obj.type(), name).is(nb::getattr(cls, name));
}

/// @brief The `__self__` and `__func__` of a bound method, or nothing for another callable.
inline std::optional<std::pair<nanobind::object, nanobind::object>> bound_method(
    nanobind::handle fn) {
  namespace nb = nanobind;
  if (!nb::hasattr(fn.type(), "__self__") || !nb::hasattr(fn.type(), "__func__")) {
    return std::nullopt;
  }
  return std::make_pair(nb::getattr(fn, "__self__"), nb::getattr(fn, "__func__"));
}

/// @brief Whether `func` is the method `name` that `T` binds and `self` is a `T`.
template <typename T>
bool is_bound_method(nanobind::handle self, nanobind::handle func, const char* name) {
  namespace nb = nanobind;
  const nb::handle cls = nb::type<T>();
  return cls.is_valid() && nb::isinstance<T>(self) && func.is(nb::getattr(cls, name));
}

}  // namespace detail

/// @brief The bound SDF that `obj(q)` evaluates, or nothing for any other callable.
inline std::optional<NativeSdf> find_native_sdf(nanobind::handle obj) {
  std::optional<NativeSdf> out;
  const auto match = [&]<typename T>() {
    if (!out && detail::binds_own<T>(obj, "__call__")) {
      out = NativeSdf(nanobind::cast<const T*>(obj));
    }
  };
  match.template operator()<collision::FootprintGridChecker>();
  match.template operator()<collision::GridSDF>();
  match.template operator()<collision::CircleSDF>();
  match.template operator()<collision::CircleSmoothSDF>();
  match.template operator()<collision::RectSmoothSDF>();
  match.template operator()<InflatedSDFCallable>();
  match.template operator()<MemoizedSDFCallable>();
  return out;
}

/// @brief An SDF callable from Python. A bound geodex SDF runs in C++, and any other
/// callable runs through Python.
///
/// @details It holds its reference like `PyCallable`. A point with fewer entries than the
/// bound SDF takes goes through Python and raises the bound call's ValueError.
class SdfFunction {
 public:
  /// @brief Take `fn` and register it under `owner`.
  SdfFunction(nanobind::callable fn, const void* owner)
      : native_(find_native_sdf(fn)), py_(std::move(fn), owner) {}

  /// @brief The SDF at q.
  double operator()(const Eigen::VectorXd& q) const {
    if (native_ && py_.held() && q.size() >= native_->min_size()) return (*native_)(q);
    return py_(q);
  }

  /// @brief The bound SDF, while the reference to it is held.
  std::optional<NativeSdf> native() const { return py_.held() ? native_ : std::nullopt; }

 private:
  std::optional<NativeSdf> native_;
  PyCallable<double> py_;
};

}  // namespace geodex::python
