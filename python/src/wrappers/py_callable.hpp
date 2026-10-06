/// @file py_callable.hpp
/// @brief Python references held inside C++ objects, made visible to Python's
/// garbage collector.
///
/// @details A bound object that stores Python references in C++ members hides them from
/// the collector, and the collector cannot free a reference cycle through such an object.
/// Each reference registers under the C++ instance that owns it. The owner's type reports
/// and drops the registered references through `tp_traverse` and `tp_clear`
/// (`callable_owner_slots`).

#pragma once

#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <nanobind/nanobind.h>

namespace geodex::python {

/// @brief Python references registered under the C++ instance that owns them.
class CallableRegistry {
 public:
  /// @brief Register the reference in `slot` under `owner`.
  void add(const void* owner, PyObject** slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    slots_[owner].push_back(slot);
  }

  /// @brief Unregister `slot` from `owner`.
  void remove(const void* owner, PyObject** slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = slots_.find(owner);
    if (it == slots_.end()) return;
    std::erase(it->second, slot);
    if (it->second.empty()) slots_.erase(it);
  }

  /// @brief Visit every reference `owner` holds.
  int traverse(const void* owner, visitproc visit, void* arg) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = slots_.find(owner);
    if (it == slots_.end()) return 0;
    for (PyObject** slot : it->second) Py_VISIT(*slot);
    return 0;
  }

  /// @brief Release every reference `owner` holds. It drops the references after releasing
  /// the lock. Freeing one can free other owners that use the registry.
  void clear(const void* owner) {
    std::vector<PyObject*> released;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = slots_.find(owner);
      if (it == slots_.end()) return;
      for (PyObject** slot : it->second) {
        if (*slot) released.push_back(std::exchange(*slot, nullptr));
      }
    }
    for (PyObject* obj : released) Py_DECREF(obj);
  }

 private:
  std::mutex mutex_;
  std::unordered_map<const void*, std::vector<PyObject**>> slots_;
};

/// @brief The process-wide registry, shared by every binding translation unit.
inline CallableRegistry& callable_registry() {
  static auto* registry = new CallableRegistry();
  return *registry;
}

/// @brief Function object that calls a Python callable with a configuration.
///
/// @details The instance built for an owner's constructor registers under that owner, and
/// the owner's `tp_traverse` reports it. Copies, such as those a type-erased metric takes,
/// hold their own untracked reference. The copy constructor is not noexcept. std::function
/// then stores the object on the heap and moves it by pointer.
template <typename R>
class PyCallable {
 public:
  /// @brief Take `fn` and register it under `owner`.
  PyCallable(nanobind::callable fn, const void* owner) : fn_(fn.release().ptr()), owner_(owner) {
    callable_registry().add(owner_, &fn_);
  }

  PyCallable(const PyCallable& other) : fn_(other.fn_), owner_(nullptr) {
    if (fn_) {
      nanobind::gil_scoped_acquire gil;
      Py_INCREF(fn_);
    }
  }

  PyCallable(PyCallable&& other) noexcept
      : fn_(std::exchange(other.fn_, nullptr)), owner_(other.owner_) {
    if (owner_) callable_registry().add(owner_, &fn_);
  }

  PyCallable& operator=(const PyCallable&) = delete;
  PyCallable& operator=(PyCallable&&) = delete;

  ~PyCallable() {
    if (owner_) callable_registry().remove(owner_, &fn_);
    if (fn_) {
      nanobind::gil_scoped_acquire gil;
      Py_DECREF(fn_);
    }
  }

  /// @brief Call the Python callable with `q` and convert the result.
  R operator()(const Eigen::VectorXd& q) const {
    nanobind::gil_scoped_acquire gil;
    if (!fn_) throw std::runtime_error("callable was released by the garbage collector");
    return nanobind::cast<R>(nanobind::handle(fn_)(q));
  }

  /// @brief Call the Python callable with `args` and return the Python result.
  /// The caller holds the GIL.
  template <typename... Args>
  nanobind::object call(const Args&... args) const {
    if (!fn_) throw std::runtime_error("callable was released by the garbage collector");
    return nanobind::handle(fn_)(args...);
  }

  /// @brief Whether the reference is still held. The owner's `tp_clear` releases it.
  bool held() const { return fn_ != nullptr; }

 private:
  PyObject* fn_;
  const void* owner_;
};

/// @brief A strong reference to a Python object, registered under an owner.
///
/// @details Not copyable. The reference belongs to one owner for its lifetime.
/// A null owner holds the reference untracked. `get()` returns null once the
/// owner's `tp_clear` has released the reference.
class PyOwnedRef {
 public:
  /// @brief Take a new reference to `obj` and register it under `owner`, if any.
  PyOwnedRef(nanobind::handle obj, const void* owner) : obj_(obj.inc_ref().ptr()), owner_(owner) {
    if (owner_) callable_registry().add(owner_, &obj_);
  }

  PyOwnedRef(const PyOwnedRef&) = delete;
  PyOwnedRef& operator=(const PyOwnedRef&) = delete;

  ~PyOwnedRef() {
    if (owner_) callable_registry().remove(owner_, &obj_);
    if (obj_) {
      nanobind::gil_scoped_acquire gil;
      Py_DECREF(obj_);
    }
  }

  /// @brief The referenced object, or null once the owner is cleared.
  PyObject* get() const { return obj_; }

 private:
  PyObject* obj_;
  const void* owner_;
};

/// @brief `tp_traverse` for a type whose instances own registered references.
inline int callable_owner_tp_traverse(PyObject* self, visitproc visit, void* arg) {
  Py_VISIT(Py_TYPE(self));
  if (!nanobind::inst_ready(self)) return 0;
  return callable_registry().traverse(nanobind::inst_ptr<void>(self), visit, arg);
}

/// @brief `tp_clear` for a type whose instances own registered references.
inline int callable_owner_tp_clear(PyObject* self) {
  if (nanobind::inst_ready(self)) callable_registry().clear(nanobind::inst_ptr<void>(self));
  return 0;
}

/// @brief Type slots to pass as `nanobind::type_slots(...)` for such a type.
inline PyType_Slot callable_owner_slots[] = {
    {Py_tp_traverse, reinterpret_cast<void*>(callable_owner_tp_traverse)},
    {Py_tp_clear, reinterpret_cast<void*>(callable_owner_tp_clear)},
    {0, nullptr}};

}  // namespace geodex::python
