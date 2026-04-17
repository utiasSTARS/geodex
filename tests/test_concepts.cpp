#include <Eigen/Core>
#include <gtest/gtest.h>

#include "geodex/geodex.hpp"

// A mock Euclidean manifold that satisfies all concepts.
struct MockEuclidean {
  using Scalar = double;
  using Point = Eigen::Vector2d;
  using Tangent = Eigen::Vector2d;

  int dim() const { return 2; }

  Point random_point() const { return Point::Random(); }

  Scalar inner(const Point& /*p*/, const Tangent& u, const Tangent& v) const { return u.dot(v); }

  Scalar norm(const Point& p, const Tangent& v) const { return std::sqrt(inner(p, v, v)); }

  Scalar distance(const Point& p, const Point& q) const { return (q - p).norm(); }

  Point geodesic(const Point& p, const Point& q, Scalar t) const { return (1.0 - t) * p + t * q; }

  Point exp(const Point& p, const Tangent& v) const { return p + v; }

  Tangent log(const Point& p, const Point& q) const { return q - p; }
};

// A struct that should NOT satisfy RiemannianManifold (missing operations).
struct NotAManifold {
  using Scalar = double;
  int dim() const { return 1; }
};

// Compile-time concept checks
static_assert(geodex::Manifold<MockEuclidean>);
static_assert(geodex::RiemannianManifold<MockEuclidean>);
static_assert(geodex::HasMetric<MockEuclidean>);
static_assert(geodex::HasDistance<MockEuclidean>);
static_assert(geodex::HasGeodesic<MockEuclidean>);

static_assert(!geodex::Manifold<NotAManifold>);
static_assert(!geodex::RiemannianManifold<NotAManifold>);

TEST(Concepts, MockEuclideanSatisfiesAllConcepts) {
  // Runtime confirmation — if this compiles, concepts are satisfied.
  EXPECT_TRUE(geodex::Manifold<MockEuclidean>);
  EXPECT_TRUE(geodex::RiemannianManifold<MockEuclidean>);
  EXPECT_TRUE(geodex::HasMetric<MockEuclidean>);
  EXPECT_TRUE(geodex::HasDistance<MockEuclidean>);
  EXPECT_TRUE(geodex::HasGeodesic<MockEuclidean>);
}

TEST(Concepts, NotAManifoldFailsConcepts) {
  EXPECT_FALSE(geodex::Manifold<NotAManifold>);
  EXPECT_FALSE(geodex::RiemannianManifold<NotAManifold>);
}

TEST(MockEuclidean, BasicOperations) {
  MockEuclidean m;
  EXPECT_EQ(m.dim(), 2);

  Eigen::Vector2d p(0.0, 0.0);
  Eigen::Vector2d q(3.0, 4.0);

  EXPECT_DOUBLE_EQ(m.distance(p, q), 5.0);

  auto mid = m.geodesic(p, q, 0.5);
  EXPECT_DOUBLE_EQ(mid.x(), 1.5);
  EXPECT_DOUBLE_EQ(mid.y(), 2.0);

  auto v = m.log(p, q);
  EXPECT_EQ(v, q);

  auto r = m.exp(p, v);
  EXPECT_EQ(r, q);
}
