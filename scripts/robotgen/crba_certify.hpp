/// @file crba_certify.hpp
/// @brief Proves a Loewner lower bound of a generated CRBA over its joint box.
///
/// `certify<R>(L)` finds a scale `s` for each independent block of the mass matrix and
/// proves `M(q) >= s L` on that block for every `q` in the joint box. The proof is a branch
/// and bound over sub-boxes. On each box, the shipped CRBA (`CrbaTemplate<R>`, made by
/// `crba_template.py`) runs in interval arithmetic with outward rounding. A Taylor expansion
/// at the box center with an interval bound on the second-order remainder gives a lower
/// bound of the smallest eigenvalue of `C^-1 M(q) C^-T` over the box, where `L = C C^T`. A
/// box whose bound reaches the target is proved, and any other box is split.
///
/// Coordinates that `M` does not depend on, such as the first joint of a fixed-base arm, stay
/// at their midpoint. The tool finds them on random samples. A coordinate that the expression
/// reads only through sine and cosine is proved over at most one period.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "geodex/robots/mass_matrix.hpp"
#include "geodex/utils/random.hpp"

namespace geodex::robots::certify {

/// @brief The shipped CRBA of robot @p R as a template over the scalar type.
template <Robot R>
struct CrbaTemplate;

// ---------------------------------------------------------------------------
// Interval arithmetic with outward rounding
// ---------------------------------------------------------------------------

/// Move @p x down by at least four units in the last place.
inline double down(double x) { return x - (std::abs(x) * 0x1p-50 + 0x1p-1000); }

/// Move @p x up by at least four units in the last place.
inline double up(double x) { return x + (std::abs(x) * 0x1p-50 + 0x1p-1000); }

/// @brief Closed interval `[lo, hi]` of reals.
struct Interval {
  double lo = 0.0;
  double hi = 0.0;
  Interval() = default;
  Interval(double v) : lo(v), hi(v) {}  // NOLINT(google-explicit-constructor)
  Interval(double l, double h) : lo(l), hi(h) {}
  double mid() const { return 0.5 * lo + 0.5 * hi; }
  double mag() const { return std::max(std::abs(lo), std::abs(hi)); }
};

inline Interval operator+(const Interval& a, const Interval& b) {
  return {down(a.lo + b.lo), up(a.hi + b.hi)};
}
inline Interval operator-(const Interval& a, const Interval& b) {
  return {down(a.lo - b.hi), up(a.hi - b.lo)};
}
inline Interval operator-(const Interval& a) { return {-a.hi, -a.lo}; }
inline Interval operator*(const Interval& a, const Interval& b) {
  const double p1 = a.lo * b.lo, p2 = a.lo * b.hi, p3 = a.hi * b.lo, p4 = a.hi * b.hi;
  return {down(std::min({p1, p2, p3, p4})), up(std::max({p1, p2, p3, p4}))};
}

namespace detail {

/// True when the interval holds `c + 2 pi k` for some integer `k`, with a small slack.
inline bool holds_periodic(const Interval& a, double c) {
  constexpr double two_pi = 2.0 * std::numbers::pi;
  const double k = std::ceil((a.lo - c) / two_pi - 1e-9);
  return c + two_pi * k <= a.hi + 1e-9;
}

/// Widen an enclosure of a library sine or cosine by its error and clamp it to [-1, 1].
inline Interval trig_result(double lo, double hi) {
  return {std::max(-1.0, down(down(lo))), std::min(1.0, up(up(hi)))};
}

}  // namespace detail

inline Interval sin(const Interval& a) {
  if (a.hi - a.lo >= 2.0 * std::numbers::pi) return {-1.0, 1.0};
  const double s1 = std::sin(a.lo), s2 = std::sin(a.hi);
  double lo = std::min(s1, s2), hi = std::max(s1, s2);
  if (detail::holds_periodic(a, 0.5 * std::numbers::pi)) hi = 1.0;
  if (detail::holds_periodic(a, -0.5 * std::numbers::pi)) lo = -1.0;
  return detail::trig_result(lo, hi);
}

inline Interval cos(const Interval& a) {
  if (a.hi - a.lo >= 2.0 * std::numbers::pi) return {-1.0, 1.0};
  const double c1 = std::cos(a.lo), c2 = std::cos(a.hi);
  double lo = std::min(c1, c2), hi = std::max(c1, c2);
  if (detail::holds_periodic(a, 0.0)) hi = 1.0;
  if (detail::holds_periodic(a, std::numbers::pi)) lo = -1.0;
  return detail::trig_result(lo, hi);
}

// ---------------------------------------------------------------------------
// Intervals with interval derivatives (forward mode)
// ---------------------------------------------------------------------------

constexpr int kMaxVars = 16;

/// Number of variables the current evaluation differentiates with respect to.
inline thread_local int active_vars = 0;

/// @brief An interval value and interval enclosures of its partial derivatives.
struct Dual {
  Interval v;
  std::array<Interval, kMaxVars> g;
  Dual() = default;
  Dual(double c) : v(c) {  // NOLINT(google-explicit-constructor)
    for (int k = 0; k < active_vars; ++k) g[k] = Interval(0.0);
  }
};

inline Dual operator+(const Dual& a, const Dual& b) {
  Dual r;
  r.v = a.v + b.v;
  for (int k = 0; k < active_vars; ++k) r.g[k] = a.g[k] + b.g[k];
  return r;
}
inline Dual operator-(const Dual& a, const Dual& b) {
  Dual r;
  r.v = a.v - b.v;
  for (int k = 0; k < active_vars; ++k) r.g[k] = a.g[k] - b.g[k];
  return r;
}
inline Dual operator-(const Dual& a) {
  Dual r;
  r.v = -a.v;
  for (int k = 0; k < active_vars; ++k) r.g[k] = -a.g[k];
  return r;
}
inline Dual operator*(const Dual& a, const Dual& b) {
  Dual r;
  r.v = a.v * b.v;
  for (int k = 0; k < active_vars; ++k) r.g[k] = a.g[k] * b.v + b.g[k] * a.v;
  return r;
}
inline Dual operator*(double c, const Dual& b) {
  Dual r;
  r.v = Interval(c) * b.v;
  for (int k = 0; k < active_vars; ++k) r.g[k] = Interval(c) * b.g[k];
  return r;
}
inline Dual operator*(const Dual& a, double c) { return c * a; }
inline Dual operator+(double c, const Dual& b) {
  Dual r = b;
  r.v = Interval(c) + b.v;
  return r;
}
inline Dual operator+(const Dual& a, double c) { return c + a; }
inline Dual operator-(double c, const Dual& b) { return c + (-b); }
inline Dual operator-(const Dual& a, double c) { return a + (-c); }

inline Dual sin(const Dual& a) {
  Dual r;
  r.v = sin(a.v);
  const Interval d = cos(a.v);
  for (int k = 0; k < active_vars; ++k) r.g[k] = d * a.g[k];
  return r;
}
inline Dual cos(const Dual& a) {
  Dual r;
  r.v = cos(a.v);
  const Interval d = -sin(a.v);
  for (int k = 0; k < active_vars; ++k) r.g[k] = d * a.g[k];
  return r;
}

// ---------------------------------------------------------------------------
// Intervals with interval first and second derivatives (forward mode)
// ---------------------------------------------------------------------------

constexpr int kMaxVars2 = 8;
constexpr int kMaxPairs2 = kMaxVars2 * (kMaxVars2 + 1) / 2;

/// Index of the second derivative with respect to variables j <= k.
constexpr int pair_index(int j, int k) { return k * (k + 1) / 2 + j; }

/// @brief An interval value with interval gradient and interval Hessian (upper triangle).
struct Dual2 {
  Interval v;
  std::array<Interval, kMaxVars2> g;
  std::array<Interval, kMaxPairs2> h;
  Dual2() = default;
  Dual2(double c) : v(c) {  // NOLINT(google-explicit-constructor)
    for (int k = 0; k < active_vars; ++k) g[k] = Interval(0.0);
    for (int k = 0; k < active_vars * (active_vars + 1) / 2; ++k) h[k] = Interval(0.0);
  }
};

inline Dual2 operator+(const Dual2& a, const Dual2& b) {
  Dual2 r;
  r.v = a.v + b.v;
  const int n = active_vars, np = n * (n + 1) / 2;
  for (int k = 0; k < n; ++k) r.g[k] = a.g[k] + b.g[k];
  for (int k = 0; k < np; ++k) r.h[k] = a.h[k] + b.h[k];
  return r;
}
inline Dual2 operator-(const Dual2& a) {
  Dual2 r;
  r.v = -a.v;
  const int n = active_vars, np = n * (n + 1) / 2;
  for (int k = 0; k < n; ++k) r.g[k] = -a.g[k];
  for (int k = 0; k < np; ++k) r.h[k] = -a.h[k];
  return r;
}
inline Dual2 operator-(const Dual2& a, const Dual2& b) { return a + (-b); }
inline Dual2 operator*(const Dual2& a, const Dual2& b) {
  Dual2 r;
  r.v = a.v * b.v;
  const int n = active_vars;
  for (int k = 0; k < n; ++k) r.g[k] = a.g[k] * b.v + b.g[k] * a.v;
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j <= k; ++j) {
      const int i = pair_index(j, k);
      r.h[i] = a.h[i] * b.v + b.h[i] * a.v + a.g[j] * b.g[k] + a.g[k] * b.g[j];
    }
  }
  return r;
}
inline Dual2 operator*(double c, const Dual2& b) {
  Dual2 r;
  const Interval ci(c);
  r.v = ci * b.v;
  const int n = active_vars, np = n * (n + 1) / 2;
  for (int k = 0; k < n; ++k) r.g[k] = ci * b.g[k];
  for (int k = 0; k < np; ++k) r.h[k] = ci * b.h[k];
  return r;
}
inline Dual2 operator*(const Dual2& a, double c) { return c * a; }
inline Dual2 operator+(double c, const Dual2& b) {
  Dual2 r = b;
  r.v = Interval(c) + b.v;
  return r;
}
inline Dual2 operator+(const Dual2& a, double c) { return c + a; }
inline Dual2 operator-(double c, const Dual2& b) { return c + (-b); }
inline Dual2 operator-(const Dual2& a, double c) { return a + (-c); }

/// f(a) for f = sin or cos, given f(a.v), f'(a.v) and f''(a.v) enclosures.
inline Dual2 chain2(const Dual2& a, const Interval& f, const Interval& d1, const Interval& d2) {
  Dual2 r;
  r.v = f;
  const int n = active_vars;
  for (int k = 0; k < n; ++k) r.g[k] = d1 * a.g[k];
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j <= k; ++j) {
      const int i = pair_index(j, k);
      r.h[i] = d1 * a.h[i] + d2 * (a.g[j] * a.g[k]);
    }
  }
  return r;
}
inline Dual2 sin(const Dual2& a) {
  const Interval s = sin(a.v), c = cos(a.v);
  return chain2(a, s, c, -s);
}
inline Dual2 cos(const Dual2& a) {
  const Interval s = sin(a.v), c = cos(a.v);
  return chain2(a, c, -s, -c);
}

// ---------------------------------------------------------------------------
// The certification
// ---------------------------------------------------------------------------

/// @brief Settings of `certify`.
struct CertifySettings {
  std::uint64_t seed = 42;           ///< seed of the sampled search
  int random_samples = 200000;       ///< uniform samples per block
  int face_samples = 50000;          ///< samples with one coordinate on a limit, per block
  int refine_starts = 64;            ///< worst samples refined by pattern search
  /// Margins below the sampled minimum to prove, tried in order, each with its budget of
  /// boxes per block.
  std::vector<std::pair<double, long>> attempts{{1e-2, 50000000}};
};

/// @brief One independent block of the mass matrix and its proof.
struct BlockResult {
  std::vector<int> coordinates;  ///< rows and columns of the block
  std::vector<int> split;        ///< coordinates the branch and bound splits
  double sampled_min = 0.0;      ///< smallest lambda found for the input L on this block
  double scale = 0.0;            ///< proved scale, M >= scale * L on this block
  double proved_min = 0.0;       ///< smallest proved lower bound over the boxes
  long boxes = 0;                ///< boxes of the successful branch and bound
};

/// @brief Result of `certify`.
struct CertifyResult {
  bool proved = false;          ///< every block reached a proof. If not, `blocks` holds
                                ///< only each block's sampled minimum
  double margin = 0.0;          ///< margin below the sampled minimum that was proved
  std::vector<BlockResult> blocks;
  std::vector<int> held;        ///< coordinates M does not depend on, held at the midpoint
  long samples = 0;             ///< CRBA evaluations of the sampled search
  double seconds = 0.0;
  Eigen::MatrixXd M_lower;      ///< scale * L on each block
};

namespace detail {

template <Robot R>
Eigen::MatrixXd unpack(const double* upper) {
  constexpr int n = MassMatrix<R>::Nq;
  Eigen::MatrixXd M(n, n);
  int k = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) M(i, j) = M(j, i) = upper[k++];
  }
  return M;
}

template <Robot R>
Eigen::MatrixXd mass(const Eigen::VectorXd& q) {
  std::array<double, MassMatrix<R>::Nq*(MassMatrix<R>::Nq + 1) / 2> upper{};
  CrbaTemplate<R>::template eval<double>(q.data(), upper.data());
  return unpack<R>(upper.data());
}

/// Smallest eigenvalue of the block @p idx of `Ci M Ci^T`.
inline double block_lambda(const Eigen::MatrixXd& Ci, const Eigen::MatrixXd& M,
                           const std::vector<int>& idx) {
  const Eigen::MatrixXd A = Ci * M * Ci.transpose();
  Eigen::MatrixXd B(idx.size(), idx.size());
  for (std::size_t a = 0; a < idx.size(); ++a) {
    for (std::size_t b = 0; b < idx.size(); ++b) B(a, b) = A(idx[a], idx[b]);
  }
  return Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(B, Eigen::EigenvaluesOnly)
      .eigenvalues()
      .minCoeff();
}

/// Index of the upper-triangle entry (i, j), i <= j, in the generated output.
inline int upper_index(int n, int i, int j) {
  if (i > j) std::swap(i, j);
  return i * n - i * (i - 1) / 2 + (j - i);
}

}  // namespace detail

/// @brief Prove `M(q) >= s L` on the joint box for each independent block of `M`.
///
/// @param L Lower bound to scale, symmetric positive definite, `Nq x Nq`.
template <Robot R>
CertifyResult certify(const Eigen::MatrixXd& L, const CertifySettings& settings = {}) {
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  constexpr int n = MassMatrix<R>::Nq;
  constexpr int U = n * (n + 1) / 2;
  static_assert(n <= kMaxVars, "raise kMaxVars");
  const auto [lo_f, hi_f] = MassMatrix<R>::joint_limits();
  const Eigen::VectorXd lo = lo_f, hi = hi_f;
  const Eigen::VectorXd mid = 0.5 * (lo + hi);
  // Prove a periodic coordinate over one period. The expression reads it only through its
  // sine and cosine.
  Eigen::VectorXd proof_hi = hi;
  for (int j = 0; j < n; ++j) {
    if (CrbaTemplate<R>::periodic[j]) proof_hi[j] = std::min(hi[j], lo[j] + 2.0 * std::numbers::pi);
  }

  CertifyResult result;
  const Eigen::MatrixXd C = Eigen::LLT<Eigen::MatrixXd>(L).matrixL();
  const Eigen::MatrixXd Ci = C.inverse();
  const Eigen::MatrixXd Ci_abs = Ci.cwiseAbs();

  std::mt19937_64 rng(settings.seed);
  auto random_q = [&] {
    Eigen::VectorXd q(n);
    for (int i = 0; i < n; ++i) q[i] = lo[i] + (hi[i] - lo[i]) * geodex::utils::uniform01(rng);
    return q;
  };

  // Find the coordinates that M does not depend on.
  std::vector<bool> held(n, false);
  for (int j = 0; j < n; ++j) {
    bool independent = true;
    for (int s = 0; s < 64 && independent; ++s) {
      Eigen::VectorXd q = random_q();
      const Eigen::MatrixXd M0 = detail::mass<R>(q);
      for (int v = 0; v < 4 && independent; ++v) {
        q[j] = lo[j] + (hi[j] - lo[j]) * (v + 0.5) / 4.0;
        const Eigen::MatrixXd M1 = detail::mass<R>(q);
        independent = (M1 - M0).cwiseAbs().maxCoeff() <= 1e-12 * M0.cwiseAbs().maxCoeff();
      }
    }
    held[j] = independent;
    if (independent) result.held.push_back(j);
  }

  // A block groups the coordinates coupled through L or through an entry of M.
  std::vector<std::vector<bool>> coupled(n, std::vector<bool>(n, false));
  for (int s = 0; s < 16; ++s) {
    const Eigen::MatrixXd M = detail::mass<R>(random_q());
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) coupled[i][j] = coupled[i][j] || M(i, j) != 0.0;
    }
  }
  std::vector<int> block_of(n, -1);
  std::vector<std::vector<int>> blocks;
  for (int i = 0; i < n; ++i) {
    if (block_of[i] >= 0) continue;
    std::vector<int> members{i}, todo{i};
    block_of[i] = static_cast<int>(blocks.size());
    while (!todo.empty()) {
      const int a = todo.back();
      todo.pop_back();
      for (int b = 0; b < n; ++b) {
        if (block_of[b] < 0 && (coupled[a][b] || L(a, b) != 0.0)) {
          block_of[b] = block_of[i];
          members.push_back(b);
          todo.push_back(b);
        }
      }
    }
    std::sort(members.begin(), members.end());
    blocks.push_back(members);
  }

  // Coordinates each block depends on, its own non-held ones and any other coordinate
  // whose change moves an entry of the block.
  std::vector<std::vector<int>> split(blocks.size());
  for (std::size_t c = 0; c < blocks.size(); ++c) {
    for (int j = 0; j < n; ++j) {
      if (held[j]) continue;
      bool depends = block_of[j] == static_cast<int>(c);
      for (int s = 0; s < 32 && !depends; ++s) {
        Eigen::VectorXd q = random_q();
        const Eigen::MatrixXd M0 = detail::mass<R>(q);
        q[j] = lo[j] + (hi[j] - lo[j]) * geodex::utils::uniform01(rng);
        const Eigen::MatrixXd M1 = detail::mass<R>(q);
        for (int a : blocks[c]) {
          for (int b : blocks[c]) depends = depends || M1(a, b) != M0(a, b);
        }
      }
      if (depends) split[c].push_back(j);
    }
  }

  for (const auto& dims : split) {
    if (dims.size() > static_cast<std::size_t>(kMaxVars2)) {
      throw std::runtime_error("certify: a block depends on more than kMaxVars2 coordinates");
    }
  }

  // Search samples for each block's smallest lambda.
  auto lambda_at = [&](const Eigen::VectorXd& q, std::size_t c) {
    return detail::block_lambda(Ci, detail::mass<R>(q), blocks[c]);
  };
  std::vector<double> sampled(blocks.size(), std::numeric_limits<double>::infinity());
  for (std::size_t c = 0; c < blocks.size(); ++c) {
    const auto& dims = split[c];
    std::vector<std::pair<double, Eigen::VectorXd>> worst;
    auto consider = [&](const Eigen::VectorXd& q) {
      const double f = lambda_at(q, c);
      ++result.samples;
      worst.emplace_back(f, q);
      if (worst.size() > static_cast<std::size_t>(4 * settings.refine_starts)) {
        std::nth_element(worst.begin(), worst.begin() + settings.refine_starts, worst.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
        worst.resize(static_cast<std::size_t>(settings.refine_starts));
      }
    };
    if (dims.size() <= 14) {
      for (long v = 0; v < (1L << dims.size()); ++v) {
        Eigen::VectorXd q = mid;
        for (std::size_t d = 0; d < dims.size(); ++d) {
          q[dims[d]] = (v >> d) & 1 ? hi[dims[d]] : lo[dims[d]];
        }
        consider(q);
      }
    }
    for (int s = 0; s < settings.face_samples; ++s) {
      Eigen::VectorXd q = random_q();
      const int d = dims[static_cast<std::size_t>(s) % dims.size()];
      q[d] = (s / static_cast<int>(dims.size())) % 2 ? hi[d] : lo[d];
      consider(q);
    }
    for (int s = 0; s < settings.random_samples; ++s) consider(random_q());
    std::sort(worst.begin(), worst.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    worst.resize(std::min(worst.size(), static_cast<std::size_t>(settings.refine_starts)));
    for (auto& [f, q] : worst) {
      // Refine the sample with a projected coordinate pattern search.
      Eigen::VectorXd step = 0.1 * (hi - lo);
      while (step.maxCoeff() > 1e-7 * (hi - lo).maxCoeff()) {
        bool improved = false;
        for (int d : dims) {
          for (const double sign : {1.0, -1.0}) {
            Eigen::VectorXd trial = q;
            trial[d] = std::clamp(q[d] + sign * step[d], lo[d], hi[d]);
            const double ft = lambda_at(trial, c);
            ++result.samples;
            if (ft < f) {
              f = ft;
              q = trial;
              improved = true;
            }
          }
        }
        if (!improved) step *= 0.5;
      }
      sampled[c] = std::min(sampled[c], f);
    }
  }

  // Lower bound of the block's lambda over a box, and the coordinate to split. Taylor's
  // theorem at the box center c with the Lagrange remainder gives, for every q in the box,
  // M(q) = M(c) + sum_j dM/dq_j(c) (q_j - c_j) + E with |E| entrywise at most
  // 1/2 sum_jk |d2M/dq_j dq_k|(box) r_j r_k. The smallest eigenvalue of the affine part is
  // concave in q and takes its minimum over the box at a vertex.
  // Matrices and vectors have a fixed capacity. The per-box work does not allocate.
  constexpr int kCap = 16;
  using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, kCap, kCap>;
  using Vec = Eigen::Matrix<double, Eigen::Dynamic, 1, 0, kCap, 1>;

  // A box with the remainder norms `||C^-1 H_jk C^-T||` proved over an enclosing box.
  // They hold on this box too.
  struct Box {
    Vec lo, hi;
    bool has_h = false;
    std::array<double, kMaxPairs2> hn{};
  };

  auto sym_norm = [](const Mat& X) {
    return std::min(X.norm(), X.cwiseAbs().rowwise().sum().maxCoeff()) * (1.0 + 1e-12);
  };

  // Lower bound of the block's lambda over @p box. Tries the remainder norms inherited
  // from an enclosing box first and computes fresh ones, stored in @p box, only when
  // those do not reach the target.
  auto bound = [&](Box& box, std::size_t c, double target, int* split_dim) {
    const auto& dims = split[c];
    const auto& idx = blocks[c];
    const int m = static_cast<int>(idx.size());
    const int d = static_cast<int>(dims.size());
    const int pairs = d * (d + 1) / 2;
    active_vars = d;
    Vec r(d);
    for (int j = 0; j < d; ++j) r[j] = 0.5 * (box.hi[dims[j]] - box.lo[dims[j]]);

    // Value and gradient at the center, in intervals that cover rounding.
    std::array<Dual, n> xc;
    std::array<Dual, U> yc;
    for (int i = 0; i < n; ++i) xc[i] = Dual(held[i] ? mid[i] : box.lo[i] * 0.5 + box.hi[i] * 0.5);
    for (int j = 0; j < d; ++j) xc[dims[j]].g[j] = Interval(1.0);
    CrbaTemplate<R>::template eval<Dual>(xc.data(), yc.data());

    Mat Cb(m, m), Cb_abs(m, m);
    for (int a = 0; a < m; ++a) {
      for (int b = 0; b < m; ++b) {
        Cb(a, b) = Ci(idx[a], idx[b]);
        Cb_abs(a, b) = Ci_abs(idx[a], idx[b]);
      }
    }
    // Affine part at the center (midpoints) and the rounding radius of its coefficients.
    Mat P0(m, m), Rad(m, m);
    std::array<Mat, kMaxVars2> Pj;
    for (int j = 0; j < d; ++j) Pj[j].resize(m, m);
    for (int a = 0; a < m; ++a) {
      for (int b = a; b < m; ++b) {
        const int k = detail::upper_index(n, idx[a], idx[b]);
        const Interval& v0 = yc[k].v;
        double rad = up(0.5 * (v0.hi - v0.lo));
        P0(a, b) = P0(b, a) = v0.mid();
        for (int j = 0; j < d; ++j) {
          const Interval& gj = yc[k].g[j];
          Pj[j](a, b) = Pj[j](b, a) = gj.mid();
          rad = up(rad + up(up(0.5 * (gj.hi - gj.lo)) * r[j]));
        }
        Rad(a, b) = Rad(b, a) = rad;
      }
    }
    const double w_centre = sym_norm(Cb_abs * Rad * Cb_abs.transpose());
    const Mat A0 = Cb * P0 * Cb.transpose();
    std::array<Mat, kMaxVars2> Aj;
    double linear = 0.0;
    for (int j = 0; j < d; ++j) {
      Aj[j] = Cb * Pj[j] * Cb.transpose();
      linear += r[j] * sym_norm(Aj[j]);
    }
    // Allowance for eigensolver and matrix-product rounding, far above their actual size.
    const double safety = 1e-12 * (A0.norm() + 1.0);
    Eigen::SelfAdjointEigenSolver<Mat> es(m);
    es.compute(A0, Eigen::EigenvaluesOnly);
    const double lam0 = es.eigenvalues().minCoeff();

    // Remainder 1/2 sum over ordered pairs r_j r_k ||C^-1 H_jk C^-T||, and the split choice.
    auto remainder = [&](Vec* contrib) {
      double w = 0.0;
      contrib->setZero(d);
      for (int kk = 0; kk < d; ++kk) {
        for (int j = 0; j <= kk; ++j) {
          const double term =
              (j == kk ? 0.5 : 1.0) * box.hn[pair_index(j, kk)] * r[j] * r[kk] * (1.0 + 1e-12);
          w += term;
          (*contrib)[j] += 0.5 * term;
          (*contrib)[kk] += 0.5 * term;
        }
      }
      return w;
    };
    // The smallest eigenvalue of the affine part over the box, a cheap bound first and the
    // exact vertex minimum when that is short of the target.
    auto affine_min = [&](double slack) {
      const double cheap = lam0 - linear;
      if (cheap - slack >= target) return cheap;
      double lowest = std::numeric_limits<double>::infinity();
      for (long v = 0; v < (1L << d); ++v) {
        Mat A = A0;
        for (int j = 0; j < d; ++j) A += ((v >> j) & 1 ? r[j] : -r[j]) * Aj[j];
        es.compute(A, Eigen::EigenvaluesOnly);
        lowest = std::min(lowest, es.eigenvalues().minCoeff());
        if (lowest - slack < target) break;
      }
      return lowest;
    };

    Vec contrib(d);
    if (box.has_h) {
      const double slack = w_centre + remainder(&contrib) + safety;
      const double lb = affine_min(slack) - slack;
      if (lb >= target) {
        Eigen::Index best = 0;
        contrib.maxCoeff(&best);
        *split_dim = dims[static_cast<std::size_t>(best)];
        return lb;
      }
    }

    // Compute fresh second derivatives over this box.
    std::array<Dual2, n> xb;
    std::array<Dual2, U> yb;
    for (int i = 0; i < n; ++i) xb[i] = Dual2(xc[i].v.lo);
    for (int j = 0; j < d; ++j) {
      const int i = dims[j];
      xb[i].v = Interval(box.lo[i], box.hi[i]);
      xb[i].g[j] = Interval(1.0);
    }
    CrbaTemplate<R>::template eval<Dual2>(xb.data(), yb.data());
    Mat Hmid(m, m), Hrad(m, m);
    for (int p = 0; p < pairs; ++p) {
      for (int a = 0; a < m; ++a) {
        for (int b = a; b < m; ++b) {
          const Interval& h = yb[detail::upper_index(n, idx[a], idx[b])].h[p];
          Hmid(a, b) = Hmid(b, a) = h.mid();
          Hrad(a, b) = Hrad(b, a) = up(std::max(h.hi - h.mid(), h.mid() - h.lo));
        }
      }
      box.hn[p] = sym_norm(Cb * Hmid * Cb.transpose()) +
                  sym_norm(Cb_abs * Hrad * Cb_abs.transpose());
    }
    box.has_h = true;
    const double slack = w_centre + remainder(&contrib) + safety;
    Eigen::Index best = 0;
    contrib.maxCoeff(&best);
    *split_dim = dims[static_cast<std::size_t>(best)];
    return affine_min(slack) - slack;
  };

  // Run a depth-first branch and bound of one block against a target.
  enum class Outcome { Proved, Counterexample, Budget };
  auto subtree = [&](const Box& root, std::size_t c, double target, std::atomic<long>* spent,
                     long budget, double* proved_min, long* boxes, double* counter) {
    std::vector<Box> stack{root};
    *proved_min = std::numeric_limits<double>::infinity();
    *boxes = 0;
    while (!stack.empty()) {
      Box box = stack.back();
      stack.pop_back();
      ++*boxes;
      if (spent->fetch_add(1, std::memory_order_relaxed) >= budget) return Outcome::Budget;
      int dim = 0;
      const double lb = bound(box, c, target, &dim);
      if (lb >= target) {
        *proved_min = std::min(*proved_min, lb);
        continue;
      }
      Eigen::VectorXd centre = 0.5 * (box.lo + box.hi);
      for (int i = 0; i < n; ++i) {
        if (held[i]) centre[i] = mid[i];
      }
      const double fc = lambda_at(centre, c);
      if (fc < target) {
        *counter = fc;
        return Outcome::Counterexample;
      }
      Box left = box, right = box;
      const double split_at = 0.5 * (box.lo[dim] + box.hi[dim]);
      left.hi[dim] = split_at;
      right.lo[dim] = split_at;
      stack.push_back(right);
      stack.push_back(left);
    }
    return Outcome::Proved;
  };

  // Cut the box into subtrees by bisecting each split coordinate a fixed number of times, and
  // run the subtrees on all hardware threads. Each subtree stops at its first counterexample.
  // The outcome does not depend on the thread schedule.
  auto branch = [&](std::size_t c, double target, long total_budget, double* proved_min,
                    long* boxes, double* counter) {
    const auto& dims = split[c];
    const int cuts = std::max(1, 12 / static_cast<int>(dims.size()));
    Box whole;
    whole.lo = lo;
    whole.hi = proof_hi;
    std::vector<Box> roots{whole};
    for (int d : dims) {
      for (int k = 0; k < cuts; ++k) {
        std::vector<Box> next;
        for (const Box& b : roots) {
          Box l = b, r = b;
          const double at = 0.5 * (b.lo[d] + b.hi[d]);
          l.hi[d] = at;
          r.lo[d] = at;
          next.push_back(l);
          next.push_back(r);
        }
        roots.swap(next);
      }
    }
    // All subtrees share one budget of boxes. An attempt that runs out of it fails.
    std::atomic<long> spent{0};
    std::vector<Outcome> outcome(roots.size(), Outcome::Proved);
    std::vector<double> mins(roots.size()), counters(roots.size());
    std::vector<long> counts(roots.size());
    std::atomic<std::size_t> next{0};
    auto work = [&] {
      for (std::size_t i = next++; i < roots.size(); i = next++) {
        outcome[i] =
            subtree(roots[i], c, target, &spent, total_budget, &mins[i], &counts[i], &counters[i]);
      }
    };
    const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t) pool.emplace_back(work);
    for (auto& t : pool) t.join();
    *proved_min = std::numeric_limits<double>::infinity();
    *boxes = 0;
    *counter = std::numeric_limits<double>::infinity();
    Outcome result = Outcome::Proved;
    for (std::size_t i = 0; i < roots.size(); ++i) {
      *boxes += counts[i];
      if (outcome[i] == Outcome::Counterexample) {
        *counter = std::min(*counter, counters[i]);
        result = Outcome::Counterexample;
      } else if (outcome[i] == Outcome::Proved) {
        *proved_min = std::min(*proved_min, mins[i]);
      }
    }
    // An exhausted budget always reports Budget. A counterexample found before the budget
    // ran out depends on the schedule.
    if (spent.load() > total_budget) return Outcome::Budget;
    return result;
  };

  for (const auto& [margin, total_budget] : settings.attempts) {
    std::vector<BlockResult> done;
    bool ok = true;
    for (std::size_t c = 0; c < blocks.size() && ok; ++c) {
      for (int attempt = 0; attempt < 32; ++attempt) {
        const double target = std::floor(sampled[c] * (1.0 - margin) * 1e6) / 1e6;
        double proved_min = 0.0, counter = 0.0;
        long boxes = 0;
        const Outcome o = branch(c, target, total_budget, &proved_min, &boxes, &counter);
        std::cerr << "[certify] block " << c << " margin " << margin << " target " << target
                  << " boxes " << boxes << " at "
                  << std::chrono::duration<double>(clock::now() - t0).count() << " s"
                  << (o == Outcome::Proved       ? " proved"
                      : o == Outcome::Budget     ? " over budget"
                                                 : " counterexample")
                  << "\n";
        if (o == Outcome::Counterexample) {
          sampled[c] = std::min(sampled[c], counter);
          continue;
        }
        if (o == Outcome::Budget) {
          ok = false;
          break;
        }
        BlockResult b;
        b.coordinates = blocks[c];
        b.split = split[c];
        b.sampled_min = sampled[c];
        b.scale = target;
        b.proved_min = proved_min;
        b.boxes = boxes;
        done.push_back(b);
        break;
      }
      if (ok && done.size() != c + 1) ok = false;
    }
    if (!ok) continue;
    result.proved = true;
    result.margin = margin;
    result.blocks = done;
    result.M_lower = L;
    for (const auto& b : done) {
      for (int i : b.coordinates) {
        for (int j = 0; j < n; ++j) {
          if (block_of[j] == block_of[i]) result.M_lower(i, j) = b.scale * L(i, j);
        }
      }
    }
    break;
  }
  if (!result.proved) {
    for (std::size_t c = 0; c < blocks.size(); ++c) {
      BlockResult b;
      b.coordinates = blocks[c];
      b.split = split[c];
      b.sampled_min = sampled[c];
      result.blocks.push_back(b);
    }
  }
  result.seconds = std::chrono::duration<double>(clock::now() - t0).count();
  return result;
}

}  // namespace geodex::robots::certify
