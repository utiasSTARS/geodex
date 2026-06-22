/// @file integration/pinocchio/mass_matrix.hpp
/// @brief Joint-space mass matrix via Pinocchio CRBA.
///
/// @details The MassMatrix class wraps Pinocchio's Composite Rigid Body
/// Algorithm. Each instance loads a URDF once and evaluates \f$ M(q) \f$
/// in place via a mutable cached data buffer. The returned reference is
/// invalidated by the next call to operator().
///
/// Not thread-safe: the internal pinocchio::Data is mutated in place. Use
/// one instance per thread or run independent evaluations in separate
/// processes.

#pragma once

#include <memory>
#include <string>
#include <utility>

#include <Eigen/Core>

#include <pinocchio/fwd.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace geodex::integration::pinocchio {

/// @brief Joint-space mass matrix evaluator driven by a URDF.
///
/// @details Satisfies the mass-matrix callable contract expected by
/// `KineticEnergyMetric`: `operator()(q)` returns an SPD \f$ nq \times nq \f$
/// matrix.
class MassMatrix {
 public:
  /// @brief Load the URDF at @p urdf_path and allocate the CRBA data buffer.
  explicit MassMatrix(const std::string& urdf_path)
      : model_(build_model(urdf_path)), data_(*model_) {}

  /// @brief Compute \f$ M(q) \f$ via CRBA.
  /// @param q Joint configuration of size `model_nq()`.
  /// @return Reference to the internal SPD mass matrix. Valid until the next
  ///         call to operator().
  auto operator()(const Eigen::VectorXd& q) const -> const Eigen::MatrixXd& {
    ::pinocchio::crba(*model_, data_, q);
    data_.M.template triangularView<Eigen::StrictlyLower>() =
        data_.M.transpose().template triangularView<Eigen::StrictlyLower>();
    return data_.M;
  }

  /// @brief Access the underlying Pinocchio model.
  auto model() const -> const ::pinocchio::Model& { return *model_; }

 private:
  static auto build_model(const std::string& urdf_path)
      -> std::shared_ptr<::pinocchio::Model> {
    auto model = std::make_shared<::pinocchio::Model>();
    ::pinocchio::urdf::buildModel(urdf_path, *model);
    return model;
  }

  std::shared_ptr<::pinocchio::Model> model_;
  mutable ::pinocchio::Data data_;
};

/// @brief Construct a `MassMatrix` from a URDF file.
inline auto mass_function(const std::string& urdf_path) -> MassMatrix {
  return MassMatrix(urdf_path);
}

/// @brief Read the number of generalized coordinates from a URDF.
inline auto model_nq(const std::string& urdf_path) -> int {
  ::pinocchio::Model model;
  ::pinocchio::urdf::buildModel(urdf_path, model);
  return model.nq;
}

/// @brief Read joint position limits from a URDF.
/// @return `(lower, upper)` vectors of size `nq`.
inline auto joint_limits(const std::string& urdf_path)
    -> std::pair<Eigen::VectorXd, Eigen::VectorXd> {
  ::pinocchio::Model model;
  ::pinocchio::urdf::buildModel(urdf_path, model);
  return {model.lowerPositionLimit, model.upperPositionLimit};
}

}  // namespace geodex::integration::pinocchio
