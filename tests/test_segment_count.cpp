/// @file test_segment_count.cpp
/// @brief Edge collision checks are spaced in the coordinates the edge moves through as
/// well as in the planning metric.

#include <cmath>
#include <memory>

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <ompl/base/ScopedState.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorBounds.h>

#include "geodex/integration/ompl/geodex_state_space.hpp"
#include "geodex/manifold/configuration_space.hpp"
#include "geodex/manifold/euclidean.hpp"
#include "geodex/metrics/constant_spd.hpp"

namespace {

using Manifold = geodex::ConfigurationSpace<geodex::Euclidean<2>, geodex::ConstantSPDMetric<2>>;
using Space = geodex::integration::ompl::GeodexStateSpace<Manifold>;

std::shared_ptr<Space> make_space() {
  // Joint 1 is nearly free in the metric, like a light wrist under kinetic energy.
  Eigen::Matrix2d A = Eigen::Matrix2d::Identity();
  A(1, 1) = 1e-6;
  const Manifold m(geodex::Euclidean<2>{}, geodex::ConstantSPDMetric<2>(A));
  ompl::base::RealVectorBounds bounds(2);
  bounds.setLow(-3.0);
  bounds.setHigh(3.0);
  auto space = std::make_shared<Space>(m, bounds);
  space->setInterpolationMode(geodex::integration::ompl::InterpolationMode::BaseGeodesic);
  space->setup();
  return space;
}

}  // namespace

TEST(SegmentCount, FollowsTheCoordinatesWithoutACollisionResolution) {
  const auto space = make_space();
  ompl::base::ScopedState<Space> a(space), b(space);
  a->values[0] = 0.0;
  a->values[1] = -2.0;
  b->values[0] = 0.0;
  b->values[1] = 2.0;
  // Joint 1 moves 4 rad. The spacing is 1% of the 8.49 rad box diagonal.
  const double spacing = space->getLongestValidSegmentFraction() * std::sqrt(72.0);
  EXPECT_GE(space->validSegmentCount(a.get(), b.get()),
            static_cast<unsigned int>(std::ceil(4.0 / spacing)));
}

TEST(SegmentCount, AnExplicitResolutionSetsTheCoordinateSpacing) {
  const auto space = make_space();
  space->setCollisionResolution(0.5);
  ompl::base::ScopedState<Space> a(space), b(space);
  a->values[0] = a->values[1] = 0.0;
  b->values[0] = 0.0;
  b->values[1] = 2.0;
  EXPECT_GE(space->validSegmentCount(a.get(), b.get()), 4u);
}
