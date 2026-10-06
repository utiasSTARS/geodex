/// @file integration/pinocchio/mass_matrix.hpp
/// @brief Joint-space mass matrix via Pinocchio CRBA.
///
/// @details The MassMatrix class wraps Pinocchio's Composite Rigid Body
/// Algorithm. Each instance loads a URDF once and evaluates \f$ M(q) \f$
/// in place via a mutable cached data buffer. The next call to operator()
/// invalidates the returned reference.
///
/// Not thread-safe. The internal pinocchio::Data is mutated in place. Use one
/// instance per thread or run independent evaluations in separate processes.

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include <pinocchio/fwd.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace geodex::integration::pinocchio {

/// @brief Joint-space mass matrix evaluator of a URDF.
///
/// @details Satisfies the mass-matrix callable contract of `KineticEnergyMetric`,
/// where `operator()(q)` returns an SPD \f$ nq \times nq \f$ matrix.
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

  /// @brief Build from URDF XML held in memory.
  ///
  /// A named factory, not a constructor overload. The path constructor also takes a
  /// `std::string`.
  static auto from_xml(const std::string& urdf_xml) -> MassMatrix {
    return MassMatrix(build_model_from_xml(urdf_xml));
  }

  /// @brief Build from URDF XML, reduced to the joints named in @p active_joints.
  ///
  /// Every other joint is locked at the neutral configuration of the full model, and
  /// the metric dimension matches the planning group.
  /// @warning Locking freezes the inertial contribution of the locked joints at the
  /// neutral pose. This is exact for joints that never move and approximate otherwise.
  static auto from_xml(const std::string& urdf_xml,
                       const std::vector<std::string>& active_joints) -> MassMatrix {
    return MassMatrix(reduce(build_model_from_xml(urdf_xml), active_joints));
  }

 private:
  explicit MassMatrix(std::shared_ptr<::pinocchio::Model> model)
      : model_(std::move(model)), data_(*model_) {}

  static auto build_model_from_xml(const std::string& urdf_xml)
      -> std::shared_ptr<::pinocchio::Model> {
    auto model = std::make_shared<::pinocchio::Model>();
    ::pinocchio::urdf::buildModelFromXML(urdf_xml, *model);
    return model;
  }

  static auto reduce(std::shared_ptr<::pinocchio::Model> full,
                     const std::vector<std::string>& active_joints)
      -> std::shared_ptr<::pinocchio::Model> {
    std::vector<::pinocchio::JointIndex> locked;
    for (::pinocchio::JointIndex j = 1; j < full->joints.size(); ++j) {
      if (std::find(active_joints.begin(), active_joints.end(), full->names[j]) ==
          active_joints.end()) {
        locked.push_back(j);
      }
    }
    if (locked.empty()) return full;
    auto reduced = std::make_shared<::pinocchio::Model>();
    ::pinocchio::buildReducedModel(*full, locked, ::pinocchio::neutral(*full), *reduced);
    return reduced;
  }

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

/// @brief Number of generalized coordinates of the full model in URDF XML.
///
/// @warning This counts every joint. A metric for a planning group needs the reduced
/// count from `model_nq_from_xml(xml, active_joints)`.
inline auto model_nq_from_xml(const std::string& urdf_xml) -> int {
  return MassMatrix::from_xml(urdf_xml).model().nq;
}

/// @brief Number of generalized coordinates after reducing URDF XML to @p active_joints.
inline auto model_nq_from_xml(const std::string& urdf_xml,
                              const std::vector<std::string>& active_joints) -> int {
  return MassMatrix::from_xml(urdf_xml, active_joints).model().nq;
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
