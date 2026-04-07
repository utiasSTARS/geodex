#include <Eigen/Core>
#include <geodex/core/sampler.hpp>
#include <gtest/gtest.h>

using geodex::HaltonSampler;
using geodex::Sampler;
using geodex::SeedableSampler;
using geodex::StochasticSampler;

// ---------------------------------------------------------------------------
// Halton sampler
// ---------------------------------------------------------------------------

TEST(HaltonSampler, DeterministicAcrossInstances) {
  HaltonSampler a, b;
  Eigen::VectorXd out_a(3), out_b(3);
  for (int i = 0; i < 100; ++i) {
    a.sample_box(3, out_a);
    b.sample_box(3, out_b);
    EXPECT_EQ(out_a, out_b);
  }
}

TEST(HaltonSampler, ReseedResetsSequence) {
  HaltonSampler a;
  Eigen::VectorXd expected(2), out(2);
  a.sample_box(2, expected);
  a.seed(0);
  a.sample_box(2, out);
  EXPECT_EQ(expected, out);
}

TEST(HaltonSampler, FirstFewValuesMatchKnownSequence) {
  // Known van-der-Corput values:
  //   index=1: base2=0.5,  base3=1/3
  //   index=2: base2=0.25, base3=2/3
  //   index=3: base2=0.75, base3=1/9
  HaltonSampler s;
  Eigen::VectorXd out(2);

  s.sample_box(2, out);
  EXPECT_NEAR(out[0], 0.5, 1e-15);
  EXPECT_NEAR(out[1], 1.0 / 3.0, 1e-15);

  s.sample_box(2, out);
  EXPECT_NEAR(out[0], 0.25, 1e-15);
  EXPECT_NEAR(out[1], 2.0 / 3.0, 1e-15);

  s.sample_box(2, out);
  EXPECT_NEAR(out[0], 0.75, 1e-15);
  EXPECT_NEAR(out[1], 1.0 / 9.0, 1e-15);
}

TEST(HaltonSampler, OutputsInUnitBox) {
  HaltonSampler s;
  Eigen::VectorXd out(5);
  for (int i = 0; i < 1000; ++i) {
    s.sample_box(5, out);
    for (int j = 0; j < 5; ++j) {
      EXPECT_GE(out[j], 0.0);
      EXPECT_LT(out[j], 1.0);
    }
  }
}

// ---------------------------------------------------------------------------
// Stochastic sampler
// ---------------------------------------------------------------------------

TEST(StochasticSampler, SeedReproducibility) {
  StochasticSampler a{42};
  StochasticSampler b{42};
  Eigen::VectorXd out_a(3), out_b(3);
  for (int i = 0; i < 100; ++i) {
    a.sample_box(3, out_a);
    b.sample_box(3, out_b);
    EXPECT_EQ(out_a, out_b);
  }
}

TEST(StochasticSampler, DifferentSeedsDiverge) {
  StochasticSampler a{1};
  StochasticSampler b{2};
  Eigen::VectorXd out_a(3), out_b(3);
  a.sample_box(3, out_a);
  b.sample_box(3, out_b);
  EXPECT_NE(out_a, out_b);
}

TEST(StochasticSampler, OutputsInUnitBox) {
  StochasticSampler s{7};
  Eigen::VectorXd out(4);
  for (int i = 0; i < 1000; ++i) {
    s.sample_box(4, out);
    for (int j = 0; j < 4; ++j) {
      EXPECT_GE(out[j], 0.0);
      EXPECT_LT(out[j], 1.0);
    }
  }
}

TEST(StochasticSampler, DefaultShareThreadLocalState) {
  // Default-constructed samplers share the same thread_local generator, so
  // back-to-back draws from two instances differ with overwhelming probability.
  StochasticSampler a, b;
  Eigen::VectorXd out_a(3), out_b(3);
  a.sample_box(3, out_a);
  b.sample_box(3, out_b);
  EXPECT_NE(out_a, out_b);
}

// ---------------------------------------------------------------------------
// Concept satisfaction
// ---------------------------------------------------------------------------

static_assert(Sampler<StochasticSampler>);
static_assert(Sampler<HaltonSampler>);
static_assert(SeedableSampler<StochasticSampler>);
static_assert(SeedableSampler<HaltonSampler>);
