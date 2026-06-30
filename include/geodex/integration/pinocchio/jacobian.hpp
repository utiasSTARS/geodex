/// @file integration/pinocchio/jacobian.hpp
/// @brief Frame Jacobian primitives backed by Pinocchio.
///
/// @details Four free-function factories build a callable that, given a
/// configuration \f$q\f$, returns the frame Jacobian (or stacked frame
/// Jacobians) in `LOCAL_WORLD_ALIGNED` coordinates:
///
/// - `frame_jacobian(urdf, frame)` returns \f$(6 \times n_v)\f$.
/// - `stacked_jacobian(urdf, {f_1, \dots, f_K})` returns \f$(6K \times n_v)\f$.
/// - `frame_position_jacobian(urdf, frame)` returns \f$(3 \times n_v)\f$.
/// - `stacked_position_jacobian(urdf, {f_1, \dots, f_K})` returns
///   \f$(3K \times n_v)\f$.
///
/// Passing an empty frame name auto-detects the last BODY frame attached to
/// the final movable joint — the common case for single-arm manipulators.
///
/// Not thread-safe: each callable owns a mutable pinocchio::Data buffer and
/// returns a reference into that buffer.

#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <pinocchio/fwd.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace geodex::integration::pinocchio {

namespace detail {

/// @brief Callable that computes stacked frame Jacobians in
///        `LOCAL_WORLD_ALIGNED` coordinates.
///
/// @details Constructed via the public factories (`frame_jacobian`,
/// `stacked_jacobian`, `frame_position_jacobian`, `stacked_position_jacobian`).
/// Users typically only interact with the functor through `auto`:
/// ```cpp
/// auto J = geodex::integration::pinocchio::frame_jacobian(urdf, "panda_link7");
/// const Eigen::MatrixXd& Jq = J(q);
/// ```
class FrameJacobianImpl {
 public:
  /// @brief Load the URDF, resolve/auto-detect frame IDs, and allocate
  ///        stacked-Jacobian buffers.
  /// @param urdf_path Path to the robot URDF.
  /// @param ee_frames Frame names to stack. If empty or containing a single
  ///        empty string, auto-detects the last BODY frame attached to the
  ///        final movable joint.
  /// @param position_only If true, keep only the 3 translational rows.
  FrameJacobianImpl(const std::string& urdf_path,
                    const std::vector<std::string>& ee_frames,
                    bool position_only)
      : model_(build_model(urdf_path)),
        data_(*model_),
        frame_ids_(resolve_frames(*model_, ee_frames, urdf_path)),
        position_only_(position_only) {
    const int rows_per_frame = position_only_ ? 3 : 6;
    J_single_.resize(6, model_->nv);
    J_stacked_.resize(rows_per_frame * static_cast<int>(frame_ids_.size()), model_->nv);
  }

  /// @brief Evaluate the stacked Jacobian at configuration @p q.
  /// @return Reference to an internal buffer of shape (rows_per_frame * K, nv).
  ///         Valid until the next call to operator().
  auto operator()(const Eigen::VectorXd& q) const -> const Eigen::MatrixXd& {
    ::pinocchio::computeJointJacobians(*model_, data_, q);
    ::pinocchio::updateFramePlacements(*model_, data_);

    const int rows_per_frame = position_only_ ? 3 : 6;
    for (std::size_t k = 0; k < frame_ids_.size(); ++k) {
      J_single_.setZero();
      ::pinocchio::getFrameJacobian(*model_, data_, frame_ids_[k],
                                    ::pinocchio::LOCAL_WORLD_ALIGNED, J_single_);
      const int row = static_cast<int>(k) * rows_per_frame;
      if (position_only_) {
        J_stacked_.middleRows(row, rows_per_frame) = J_single_.topRows(3);
      } else {
        J_stacked_.middleRows(row, rows_per_frame) = J_single_;
      }
    }
    return J_stacked_;
  }

  /// @brief Access the underlying Pinocchio model.
  auto model() const -> const ::pinocchio::Model& { return *model_; }

  /// @brief Return the resolved frame IDs (in stacking order).
  auto frame_ids() const -> const std::vector<::pinocchio::FrameIndex>& { return frame_ids_; }

 private:
  static auto build_model(const std::string& urdf_path)
      -> std::shared_ptr<::pinocchio::Model> {
    auto model = std::make_shared<::pinocchio::Model>();
    ::pinocchio::urdf::buildModel(urdf_path, *model);
    return model;
  }

  static auto resolve_frames(const ::pinocchio::Model& model,
                             const std::vector<std::string>& ee_frames,
                             const std::string& urdf_path)
      -> std::vector<::pinocchio::FrameIndex> {
    const bool use_auto = ee_frames.empty() ||
                          (ee_frames.size() == 1 && ee_frames[0].empty());
    if (use_auto) {
      return {detect_last_body_frame(model)};
    }
    std::vector<::pinocchio::FrameIndex> ids;
    ids.reserve(ee_frames.size());
    for (const auto& name : ee_frames) {
      const auto fid = model.getFrameId(name);
      if (fid >= model.frames.size()) {
        throw std::runtime_error("Frame '" + name + "' not found in URDF: " + urdf_path);
      }
      ids.push_back(fid);
    }
    return ids;
  }

  /// @brief Pick the first BODY frame whose parent joint is the final movable
  ///        joint. Falls back to the last BODY frame attached to a non-root
  ///        joint if no exact match is found.
  static auto detect_last_body_frame(const ::pinocchio::Model& model)
      -> ::pinocchio::FrameIndex {
    const auto last_joint_id = static_cast<::pinocchio::JointIndex>(model.njoints - 1);
    for (::pinocchio::FrameIndex i = 0; i < model.frames.size(); ++i) {
      const auto& f = model.frames[i];
      if (f.type == ::pinocchio::BODY && f.parentJoint == last_joint_id) {
        return i;
      }
    }
    for (::pinocchio::FrameIndex i = model.frames.size(); i > 0; --i) {
      const auto& f = model.frames[i - 1];
      if (f.type == ::pinocchio::BODY && f.parentJoint > 0) {
        return i - 1;
      }
    }
    throw std::runtime_error("URDF has no BODY frame attached to a non-root joint");
  }

  std::shared_ptr<::pinocchio::Model> model_;
  mutable ::pinocchio::Data data_;
  std::vector<::pinocchio::FrameIndex> frame_ids_;
  bool position_only_;
  mutable Eigen::MatrixXd J_single_;
  mutable Eigen::MatrixXd J_stacked_;
};

}  // namespace detail

/// @brief Build a single-frame Jacobian callable: \f$q \mapsto J(q) \in \mathbb{R}^{6 \times n_v}\f$.
/// @param urdf_path Path to the robot URDF.
/// @param ee_frame Name of the end-effector frame. Empty triggers auto-detect.
inline auto frame_jacobian(const std::string& urdf_path, const std::string& ee_frame = {})
    -> detail::FrameJacobianImpl {
  return detail::FrameJacobianImpl(urdf_path, {ee_frame}, /*position_only=*/false);
}

/// @brief Build a multi-frame stacked Jacobian callable:
///        \f$q \mapsto [J_1(q); \dots; J_K(q)] \in \mathbb{R}^{6K \times n_v}\f$.
inline auto stacked_jacobian(const std::string& urdf_path,
                             const std::vector<std::string>& ee_frames)
    -> detail::FrameJacobianImpl {
  return detail::FrameJacobianImpl(urdf_path, ee_frames, /*position_only=*/false);
}

/// @brief Build a single-frame position-only Jacobian callable
///        (\f$q \mapsto J_p(q) \in \mathbb{R}^{3 \times n_v}\f$).
inline auto frame_position_jacobian(const std::string& urdf_path,
                                    const std::string& ee_frame = {})
    -> detail::FrameJacobianImpl {
  return detail::FrameJacobianImpl(urdf_path, {ee_frame}, /*position_only=*/true);
}

/// @brief Build a multi-frame position-only stacked Jacobian callable.
inline auto stacked_position_jacobian(const std::string& urdf_path,
                                      const std::vector<std::string>& ee_frames)
    -> detail::FrameJacobianImpl {
  return detail::FrameJacobianImpl(urdf_path, ee_frames, /*position_only=*/true);
}

}  // namespace geodex::integration::pinocchio
