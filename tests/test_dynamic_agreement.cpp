// Tests that the type-erased manifolds of the Python bindings compute the same bits as the
// typed C++ manifolds they mirror, in the smoother and in plan().

#include <cmath>
#include <cstdlib>
#include <cstring>

#include <random>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "geodex/algorithm/path_smoothing.hpp"
#include "geodex/manifold/configuration_space.hpp"
#include "geodex/manifold/euclidean.hpp"
#include "geodex/metrics/jacobi.hpp"
#include "geodex/metrics/kinetic_energy.hpp"
#include "geodex/utils/ordered_sum.hpp"

#include "wrappers/py_config_space.hpp"
#include "wrappers/py_euclidean.hpp"
#include "wrappers/py_metrics.hpp"

#ifdef GEODEX_TEST_ROBOTS
#include "geodex/robots/joint_space.hpp"

#include "wrappers/py_robot_model.hpp"
#endif

#ifdef GEODEX_TEST_OMPL
#include "geodex/heuristics/matrix_lower_bound.hpp"
#include "geodex/planning/plan.hpp"
#endif

namespace ga = geodex::algorithm;
namespace gpy = geodex::python;

namespace {

// An SPD mass matrix whose entries depend on every coordinate. Its rows are diagonally
// dominant for N up to 7.
template <int N>
struct CoupledMass {
  Eigen::Matrix<double, N, N> operator()(const Eigen::Vector<double, N>& q) const {
    Eigen::Matrix<double, N, N> m;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        m(i, j) =
            i == j ? 2.0 + std::sin(q[i]) : 0.3 * std::cos(q[i] - q[j]) / (1 + std::abs(i - j));
      }
    }
    return m;
  }
};

// The same matrix for dynamic-size configurations, as a Python callable returns it.
template <int N>
Eigen::MatrixXd coupled_mass_dynamic(const Eigen::VectorXd& q) {
  return CoupledMass<N>{}(Eigen::Vector<double, N>(q));
}

// Gravitational potential of a two-link arm.
double potential(const Eigen::VectorXd& q) {
  return 9.81 * 0.5 * std::sin(q[0]) + 9.81 * (std::sin(q[0]) + 0.5 * std::sin(q[0] + q[1]));
}

// A ball obstacle of radius r at the origin of the first two coordinates.
struct Ball {
  double r;
  template <typename P>
  bool operator()(const P& q) const {
    return geodex::utils::ordered_norm(q.head(2)) > r;
  }
};

// A path from start to goal that bends around the ball through the second coordinate.
template <typename P>
std::vector<P> detour(const P& start, const P& goal, const int per_edge) {
  std::vector<P> corners;
  for (const double lift : {0.0, 0.8, 0.9, 0.8, 0.0}) {
    const double t = static_cast<double>(corners.size()) / 4.0;
    P c = (1.0 - t) * start + t * goal;
    c[1] += lift;
    corners.push_back(c);
  }
  std::vector<P> path{corners.front()};
  for (std::size_t k = 1; k < corners.size(); ++k) {
    for (int j = 1; j <= per_edge; ++j) {
      const double s = static_cast<double>(j) / per_edge;
      path.push_back((1.0 - s) * corners[k - 1] + s * corners[k]);
    }
  }
  return path;
}

// A configuration at x0 on the first axis, with small offsets in the other coordinates.
template <int N>
Eigen::Vector<double, N> spread(const double x0) {
  Eigen::Vector<double, N> p;
  for (int i = 0; i < N; ++i) p[i] = i == 0 ? x0 : (i == 1 ? 0.0 : 0.1 * (i % 3) - 0.1 * x0);
  return p;
}

// Expects every coordinate of every point to match bit for bit.
template <typename PA, typename PB>
void expect_same_bits(const std::vector<PA>& a, const std::vector<PB>& b) {
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t k = 0; k < a.size(); ++k) {
    ASSERT_EQ(a[k].size(), b[k].size());
    for (Eigen::Index i = 0; i < a[k].size(); ++i) {
      ASSERT_EQ(a[k][i], b[k][i]) << "point " << k << ", coordinate " << i;
    }
  }
}

// Smooths `path` on the typed manifold and on its wrapper and expects the same result.
template <typename MA, typename MB, typename Valid>
void expect_same_smoothing(const MA& typed, const MB& erased, const Valid& valid,
                           const std::vector<typename MA::Point>& path, const bool round) {
  ga::PathSmoothingSettings s;
  s.collision_check_resolution = 0.01;
  s.output_spacing = 0.05;
  s.round_corners = round;
  const auto a = ga::smooth_path(typed, valid, path, s);
  const std::vector<Eigen::VectorXd> dynamic_path(path.begin(), path.end());
  const auto b = ga::smooth_path(erased, valid, dynamic_path, s);
  ASSERT_TRUE(a.collision_free);
  ASSERT_TRUE(b.collision_free);
  expect_same_bits(a.path, b.path);
  EXPECT_EQ(a.length, b.length);
  EXPECT_EQ(a.profile.relax_moves, b.profile.relax_moves);
  EXPECT_EQ(a.profile.rounded_corners, b.profile.rounded_corners);
  EXPECT_EQ(a.profile.rounding_retries, b.profile.rounding_retries);
  EXPECT_EQ(a.profile.kept_corners, b.profile.kept_corners);
}

}  // namespace

TEST(OrderedSums, FixedAndDynamicSizesGiveTheSameBits) {
  std::mt19937_64 rng(3);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  for (int trial = 0; trial < 2000; ++trial) {
    Eigen::Vector<double, 7> u, v;
    Eigen::Matrix<double, 7, 7> a;
    for (int i = 0; i < 7; ++i) {
      u[i] = uniform(rng);
      v[i] = uniform(rng);
      for (int j = 0; j < 7; ++j) a(i, j) = uniform(rng);
    }
    const Eigen::VectorXd ud = u, vd = v;
    const Eigen::MatrixXd ad = a;
    ASSERT_EQ(geodex::utils::ordered_norm(u), geodex::utils::ordered_norm(ud));
    ASSERT_EQ(geodex::utils::ordered_dot(u, v), geodex::utils::ordered_dot(ud, vd));
    ASSERT_EQ(geodex::utils::ordered_quadratic_form(u, a, v),
              geodex::utils::ordered_quadratic_form(ud, ad, vd));
  }
}

TEST(DynamicAgreement, EuclideanSmoothingMatchesThePythonWrapper) {
  const geodex::Euclidean<7> typed;
  const gpy::PyEuclidean wrapper(7);
  const auto erased = wrapper.to_dynamic_manifold();
  const auto path = detour(spread<7>(-1.0), spread<7>(1.0), 12);
  for (const bool round : {false, true}) {
    expect_same_smoothing(typed, erased, Ball{0.5}, path, round);
  }
}

TEST(DynamicAgreement, KineticEnergySmoothingMatchesThePythonComposition) {
  using Space =
      geodex::ConfigurationSpace<geodex::Euclidean<7>, geodex::KineticEnergyMetric<CoupledMass<7>>>;
  const Space typed{geodex::Euclidean<7>{}, geodex::KineticEnergyMetric<CoupledMass<7>>{{}}};
  gpy::PyEuclidean base(7);
  auto dm = base.to_dynamic_manifold();
  dm.probe_sizes();
  const gpy::PyKineticEnergyMetric metric(&coupled_mass_dynamic<7>);
  const gpy::PyConfigurationSpace space(dm, metric.to_dynamic_metric(), "Euclidean", "KE");
  const auto erased = space.to_dynamic_manifold();
  const auto path = detour(spread<7>(-1.0), spread<7>(1.0), 12);
  for (const bool round : {false, true}) {
    expect_same_smoothing(typed, erased, Ball{0.5}, path, round);
  }
}

TEST(DynamicAgreement, JacobiSmoothingMatchesThePythonComposition) {
  // The typed space freezes its metric through a Gram matrix and the composition through
  // the metric's own inner product.
  using Mass = CoupledMass<2>;
  using Metric = geodex::JacobiMetric<Mass, double (*)(const Eigen::VectorXd&)>;
  using Space = geodex::ConfigurationSpace<geodex::Euclidean<2>, Metric>;
  const double h = 1.2 * 9.81 * 2.0;
  const Space typed{geodex::Euclidean<2>{}, Metric{Mass{}, &potential, h}};
  gpy::PyEuclidean base(2);
  auto dm = base.to_dynamic_manifold();
  dm.probe_sizes();
  const gpy::PyJacobiMetric metric(&coupled_mass_dynamic<2>, &potential, h);
  const gpy::PyConfigurationSpace space(dm, metric.to_dynamic_metric(), "Euclidean", "Jacobi");
  const auto erased = space.to_dynamic_manifold();
  const auto path = detour(spread<2>(-1.0), spread<2>(1.0), 12);
  for (const bool round : {false, true}) {
    expect_same_smoothing(typed, erased, Ball{0.5}, path, round);
  }
}

#ifdef GEODEX_TEST_ROBOTS
namespace {

namespace gr = geodex::robots;

constexpr auto kArm = gr::Robot::Fr3Gripper;

Eigen::Vector<double, 7> arm_start() {
  return (Eigen::Vector<double, 7>() << 0.1758, -0.1952, 0.4451, -2.1536, 1.9130, 2.0966, 1.0981)
      .finished();
}

Eigen::Vector<double, 7> arm_goal() {
  return (Eigen::Vector<double, 7>() << 0.1591, -0.0746, 0.4710, -1.3579, 1.4042, 2.1810, 1.9137)
      .finished();
}

// A joint-space ball between the two configurations that the straight edge crosses.
struct JointBall {
  Eigen::Vector<double, 7> c = 0.5 * (arm_start() + arm_goal());
  double r = 0.15;
  template <typename P>
  bool operator()(const P& q) const {
    double s = 0.0;
    for (int i = 0; i < 7; ++i) s += (q[i] - c[i]) * (q[i] - c[i]);
    return std::sqrt(s) > r;
  }
};

// A path from the start through a point beside the ball to the goal, in short edges.
std::vector<Eigen::Vector<double, 7>> arm_detour() {
  Eigen::Vector<double, 7> mid = 0.5 * (arm_start() + arm_goal());
  mid[0] += 0.3;
  mid[2] -= 0.3;
  std::vector<Eigen::Vector<double, 7>> path{arm_start()};
  for (const auto& [a, b] : {std::pair{arm_start(), mid}, std::pair{mid, arm_goal()}}) {
    for (int j = 1; j <= 20; ++j) path.push_back(a + (static_cast<double>(j) / 20) * (b - a));
  }
  return path;
}

}  // namespace

TEST(DynamicAgreement, RobotArmSmoothingMatchesThePythonRobotModel) {
  using gr::ArmMetric;
  for (const auto arm : {ArmMetric::KineticEnergy, ArmMetric::Euclidean}) {
    gpy::RobotModelOptions options;
    options.metric = arm;
    const auto erased = gpy::make_robot_model<kArm>(options).to_dynamic_manifold();
    for (const bool round : {false, true}) {
      if (arm == ArmMetric::KineticEnergy) {
        expect_same_smoothing(gr::joint_space<kArm, ArmMetric::KineticEnergy>(), erased,
                              JointBall{}, arm_detour(), round);
      } else {
        expect_same_smoothing(gr::joint_space<kArm, ArmMetric::Euclidean>(), erased, JointBall{},
                              arm_detour(), round);
      }
    }
  }
}

#ifdef GEODEX_TEST_OMPL
TEST(DynamicAgreement, RobotArmPlanMatchesThePythonRobotModel) {
  namespace gp = geodex::planning;
  gp::PlanSettings settings;
  settings.iterations = 400;
  settings.seed = 3;
  const auto [lo, hi] = gr::MassMatrix<kArm>::joint_limits();
  settings.limits.emplace(lo, hi);
  const geodex::heuristics::MatrixLowerBound<Eigen::Dynamic> heuristic(
      gr::joint_lower_bound<kArm>(gr::ArmMetric::KineticEnergy));
  const auto typed = gp::plan(gr::joint_space<kArm, gr::ArmMetric::KineticEnergy>(), arm_start(),
                              arm_goal(), JointBall{}, settings, heuristic);
  const auto erased =
      gp::plan(gpy::make_robot_model<kArm>().to_dynamic_manifold(), Eigen::VectorXd(arm_start()),
               Eigen::VectorXd(arm_goal()), JointBall{}, settings, heuristic);
  ASSERT_TRUE(typed.solved);
  ASSERT_TRUE(erased.solved);
  expect_same_bits(typed.raw_path, erased.raw_path);
  expect_same_bits(typed.path, erased.path);
  EXPECT_EQ(typed.cost, erased.cost);
}
#endif
#endif
