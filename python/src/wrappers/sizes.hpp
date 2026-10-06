/// @file sizes.hpp
/// @brief Size checks for the arrays Python hands to the bindings.
///
/// @details Eigen does not check sizes in a release build. An array of the wrong length
/// would read or write past a buffer. The bindings check first and throw
/// `std::invalid_argument`, which Python raises as `ValueError`.

#pragma once

#include <stdexcept>
#include <string>

#include <Eigen/Core>

namespace geodex::python {

/// @brief Throw unless `v` holds `n` entries.
/// @param where The call, such as "Torus.exp".
/// @param name The argument, such as "p".
inline void require_size(const Eigen::Ref<const Eigen::VectorXd>& v, const Eigen::Index n,
                         const char* where, const char* name) {
  if (v.size() != n) {
    throw std::invalid_argument(std::string(where) + ": " + name + " has " +
                                std::to_string(v.size()) + " entries, expected " +
                                std::to_string(n));
  }
}

/// @brief Throw unless `v` holds at least `n` entries.
inline void require_min_size(const Eigen::Ref<const Eigen::VectorXd>& v, const Eigen::Index n,
                             const char* where, const char* name) {
  if (v.size() < n) {
    throw std::invalid_argument(std::string(where) + ": " + name + " has " +
                                std::to_string(v.size()) + " entries, expected at least " +
                                std::to_string(n));
  }
}

/// @brief Throw unless `m` is `rows` by `cols`.
inline void require_shape(const Eigen::Ref<const Eigen::MatrixXd>& m, const Eigen::Index rows,
                          const Eigen::Index cols, const char* where, const char* name) {
  if (m.rows() != rows || m.cols() != cols) {
    throw std::invalid_argument(std::string(where) + ": " + name + " is " +
                                std::to_string(m.rows()) + " by " + std::to_string(m.cols()) +
                                ", expected " + std::to_string(rows) + " by " +
                                std::to_string(cols));
  }
}

}  // namespace geodex::python
