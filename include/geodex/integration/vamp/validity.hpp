/// @file validity.hpp
/// @brief Point validity backed by a VAMP collision checker, with a batched form.
///
/// @details `VampValidity` is a callable `bool(const Point&)` for planners and the
/// smoother. It also provides `batch(points, n)` and `batch_size()`, the batched form
/// that `algorithm::smooth_path` detects, and edge samples reach VAMP's SIMD rakes in
/// blocks. Every check includes the robot's joint box. The header does not use VAMP or
/// SIMD types.
///
/// @code
/// namespace gv = geodex::integration::vamp;
/// const auto env = gv::load_scene("scene.yaml");
/// const gv::VampValidity<Eigen::VectorXd> valid(gv::make_vamp_checker("stretch3", env));
/// const bool ok = valid(q);
/// @endcode

#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "geodex/integration/vamp/registry.hpp"

namespace geodex::integration::vamp {

/// @brief Validity of configurations of type @p Point by a shared VAMP collision checker.
///
/// Not thread-safe. The batch buffer is reused across calls. Copies share the checker
/// and own their buffer.
template <typename Point>
class VampValidity {
 public:
  /// @brief Wrap @p checker, typically from `make_vamp_checker`.
  explicit VampValidity(std::shared_ptr<const CollisionChecker> checker)
      : checker_(std::move(checker)) {
    if (!checker_) throw std::invalid_argument("VampValidity: null collision checker");
  }

  /// @brief True when @p q is inside the joint box and collision-free.
  bool operator()(const Point& q) const {
    return checker_->is_valid(q.data(), static_cast<int>(q.size()));
  }

  /// @brief True when all @p n configurations starting at @p q are valid.
  bool batch(const Point* q, std::size_t n) const {
    if (n == 0) return true;
    const std::size_t dim = static_cast<std::size_t>(q[0].size());
    buffer_.resize(n * dim);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < dim; ++j) buffer_[i * dim + j] = q[i][j];
    }
    return checker_->all_valid(buffer_.data(), static_cast<int>(dim), static_cast<int>(n));
  }

  /// @brief Configurations the checker evaluates in one SIMD pass.
  std::size_t batch_size() const { return static_cast<std::size_t>(checker_->batch_width()); }

  /// @brief The wrapped checker.
  const CollisionChecker& checker() const { return *checker_; }

 private:
  std::shared_ptr<const CollisionChecker> checker_;
  mutable std::vector<double> buffer_;
};

}  // namespace geodex::integration::vamp
