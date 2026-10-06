#include <stdexcept>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "geodex/core/sampler.hpp"
#include "geodex/manifold/euclidean.hpp"
#include "geodex/manifold/se2.hpp"
#include "geodex/manifold/se3.hpp"
#include "geodex/manifold/so2.hpp"
#include "geodex/manifold/so3.hpp"
#include "geodex/manifold/sphere.hpp"
#include "geodex/manifold/torus.hpp"

using geodex::DynamicSampler;
using geodex::HaltonSampler;
using geodex::PseudoRandomSampler;
using geodex::Sampler;
using geodex::ScrambledHaltonSampler;
using geodex::SeedableSampler;

// ---------------------------------------------------------------------------
// Halton sampler
// ---------------------------------------------------------------------------

TEST(HaltonSampler, DeterministicAcrossInstances) {
  HaltonSampler a, b;
  Eigen::VectorXd out_a(3), out_b(3);
  for (int i = 0; i < 100; ++i) {
    a.sample(3, out_a);
    b.sample(3, out_b);
    EXPECT_EQ(out_a, out_b);
  }
}

TEST(HaltonSampler, ReseedResetsSequence) {
  HaltonSampler a;
  Eigen::VectorXd expected(2), out(2);
  a.sample(2, expected);
  a.seed(0);
  a.sample(2, out);
  EXPECT_EQ(expected, out);
}

TEST(HaltonSampler, FirstFewValuesMatchKnownSequence) {
  // Known van-der-Corput values:
  //   index=1: base2=0.5,  base3=1/3
  //   index=2: base2=0.25, base3=2/3
  //   index=3: base2=0.75, base3=1/9
  HaltonSampler s;
  Eigen::VectorXd out(2);

  s.sample(2, out);
  EXPECT_NEAR(out[0], 0.5, 1e-15);
  EXPECT_NEAR(out[1], 1.0 / 3.0, 1e-15);

  s.sample(2, out);
  EXPECT_NEAR(out[0], 0.25, 1e-15);
  EXPECT_NEAR(out[1], 2.0 / 3.0, 1e-15);

  s.sample(2, out);
  EXPECT_NEAR(out[0], 0.75, 1e-15);
  EXPECT_NEAR(out[1], 1.0 / 9.0, 1e-15);
}

TEST(HaltonSampler, OutputsInUnitBox) {
  HaltonSampler s;
  Eigen::VectorXd out(5);
  for (int i = 0; i < 1000; ++i) {
    s.sample(5, out);
    for (int j = 0; j < 5; ++j) {
      EXPECT_GE(out[j], 0.0);
      EXPECT_LT(out[j], 1.0);
    }
  }
}

// ---------------------------------------------------------------------------
// Stochastic sampler
// ---------------------------------------------------------------------------

TEST(PseudoRandomSampler, SeedReproducibility) {
  PseudoRandomSampler a{42};
  PseudoRandomSampler b{42};
  Eigen::VectorXd out_a(3), out_b(3);
  for (int i = 0; i < 100; ++i) {
    a.sample(3, out_a);
    b.sample(3, out_b);
    EXPECT_EQ(out_a, out_b);
  }
}

TEST(PseudoRandomSampler, DifferentSeedsDiverge) {
  PseudoRandomSampler a{1};
  PseudoRandomSampler b{2};
  Eigen::VectorXd out_a(3), out_b(3);
  a.sample(3, out_a);
  b.sample(3, out_b);
  EXPECT_NE(out_a, out_b);
}

TEST(PseudoRandomSampler, OutputsInUnitBox) {
  PseudoRandomSampler s{7};
  Eigen::VectorXd out(4);
  for (int i = 0; i < 1000; ++i) {
    s.sample(4, out);
    for (int j = 0; j < 4; ++j) {
      EXPECT_GE(out[j], 0.0);
      EXPECT_LT(out[j], 1.0);
    }
  }
}

TEST(PseudoRandomSampler, DefaultShareThreadLocalState) {
  // Default-constructed samplers share one thread_local generator. Back-to-back samples from
  // two instances differ with overwhelming probability.
  PseudoRandomSampler a, b;
  Eigen::VectorXd out_a(3), out_b(3);
  a.sample(3, out_a);
  b.sample(3, out_b);
  EXPECT_NE(out_a, out_b);
}

// ---------------------------------------------------------------------------
// Scrambled Halton sampler
// ---------------------------------------------------------------------------

TEST(ScrambledHaltonSampler, SeedReproducibility) {
  ScrambledHaltonSampler a{42};
  ScrambledHaltonSampler b{42};
  Eigen::VectorXd out_a(4), out_b(4);
  for (int i = 0; i < 100; ++i) {
    a.sample(4, out_a);
    b.sample(4, out_b);
    EXPECT_EQ(out_a, out_b);
  }
}

TEST(ScrambledHaltonSampler, DifferentSeedsDiverge) {
  ScrambledHaltonSampler a{1};
  ScrambledHaltonSampler b{2};
  Eigen::VectorXd out_a(4), out_b(4);
  a.sample(4, out_a);
  b.sample(4, out_b);
  EXPECT_NE(out_a, out_b);
}

TEST(ScrambledHaltonSampler, ReseedResetsSequence) {
  ScrambledHaltonSampler a{7};
  Eigen::VectorXd expected(3), out(3);
  a.sample(3, expected);
  a.seed(7);
  a.sample(3, out);
  EXPECT_EQ(expected, out);
}

TEST(ScrambledHaltonSampler, OutputsInUnitBox) {
  ScrambledHaltonSampler s{13};
  Eigen::VectorXd out(6);
  for (int i = 0; i < 1000; ++i) {
    s.sample(6, out);
    for (int j = 0; j < 6; ++j) {
      EXPECT_GE(out[j], 0.0);
      EXPECT_LT(out[j], 1.0);
    }
  }
}

// ---------------------------------------------------------------------------
// Dynamic sampler
// ---------------------------------------------------------------------------

TEST(DynamicSampler, WrapsHaltonDeterministically) {
  DynamicSampler d{HaltonSampler{}};
  HaltonSampler ref;
  Eigen::VectorXd out_d(3), out_ref(3);
  for (int i = 0; i < 50; ++i) {
    d.sample(3, out_d);
    ref.sample(3, out_ref);
    EXPECT_EQ(out_d, out_ref);
  }
}

TEST(DynamicSampler, WrapsSeededPseudoRandom) {
  DynamicSampler d{PseudoRandomSampler{42}};
  PseudoRandomSampler ref{42};
  Eigen::VectorXd out_d(3), out_ref(3);
  for (int i = 0; i < 50; ++i) {
    d.sample(3, out_d);
    ref.sample(3, out_ref);
    EXPECT_EQ(out_d, out_ref);
  }
}

TEST(DynamicSampler, SeedForwardsToWrappedSampler) {
  DynamicSampler d{HaltonSampler{}};
  d.seed(5);
  HaltonSampler ref;
  ref.seed(5);
  Eigen::VectorXd out_d(2), out_ref(2);
  d.sample(2, out_d);
  ref.sample(2, out_ref);
  EXPECT_EQ(out_d, out_ref);
}

// ---------------------------------------------------------------------------
// Scrambled radical inverse (known values)
// ---------------------------------------------------------------------------

TEST(ScrambledVanDerCorput, KnownValuesWithSwapPermutation) {
  const std::vector<int> swap2{1, 0};
  EXPECT_NEAR(geodex::detail::scrambled_van_der_corput(1, 2, swap2), 0.0, 1e-15);
  EXPECT_NEAR(geodex::detail::scrambled_van_der_corput(2, 2, swap2), 0.5, 1e-15);
  EXPECT_NEAR(geodex::detail::scrambled_van_der_corput(3, 2, swap2), 0.0, 1e-15);
  // Identity permutation reproduces the plain van der Corput sequence.
  const std::vector<int> id2{0, 1};
  EXPECT_NEAR(geodex::detail::scrambled_van_der_corput(1, 2, id2),
              geodex::detail::van_der_corput(1, 2), 1e-15);
  EXPECT_NEAR(geodex::detail::scrambled_van_der_corput(5, 2, id2),
              geodex::detail::van_der_corput(5, 2), 1e-15);
}

// ---------------------------------------------------------------------------
// Scrambled Halton sampler (independence and boundary)
// ---------------------------------------------------------------------------

TEST(ScrambledHaltonSampler, DefaultInstancesDiverge) {
  ScrambledHaltonSampler a, b;
  Eigen::VectorXd out_a(4), out_b(4);
  a.sample(4, out_a);
  b.sample(4, out_b);
  EXPECT_NE(out_a, out_b);
}

TEST(ScrambledHaltonSampler, OutputsInUnitBoxAtMaxDim) {
  ScrambledHaltonSampler s{99};
  Eigen::VectorXd out(30);
  for (int i = 0; i < 200; ++i) {
    s.sample(30, out);
    for (int j = 0; j < 30; ++j) {
      EXPECT_GE(out[j], 0.0);
      EXPECT_LT(out[j], 1.0);
    }
  }
}

// ---------------------------------------------------------------------------
// Dynamic sampler (value semantics and default)
// ---------------------------------------------------------------------------

TEST(DynamicSampler, WrapsScrambledHaltonDeterministically) {
  DynamicSampler d{ScrambledHaltonSampler{42}};
  ScrambledHaltonSampler ref{42};
  Eigen::VectorXd out_d(4), out_ref(4);
  for (int i = 0; i < 50; ++i) {
    d.sample(4, out_d);
    ref.sample(4, out_ref);
    EXPECT_EQ(out_d, out_ref);
  }
}

TEST(DynamicSampler, CopyIsIndependentSequence) {
  DynamicSampler a{HaltonSampler{}};
  Eigen::VectorXd warm(2);
  a.sample(2, warm);
  DynamicSampler b = a;
  Eigen::VectorXd out_a(2), out_b(2);
  a.sample(2, out_a);
  b.sample(2, out_b);
  EXPECT_EQ(out_a, out_b);
}

TEST(DynamicSampler, DefaultOutputsInUnitBox) {
  DynamicSampler d;
  Eigen::VectorXd out(3);
  for (int i = 0; i < 100; ++i) {
    d.sample(3, out);
    for (int j = 0; j < 3; ++j) {
      EXPECT_GE(out[j], 0.0);
      EXPECT_LT(out[j], 1.0);
    }
  }
}

// ---------------------------------------------------------------------------
// High dimension (beyond a small prime table)
// ---------------------------------------------------------------------------

TEST(HaltonSampler, HighDimensionStaysInUnitBox) {
  HaltonSampler s;
  Eigen::VectorXd out(200);
  s.sample(200, out);
  for (int j = 0; j < 200; ++j) {
    EXPECT_GE(out[j], 0.0);
    EXPECT_LT(out[j], 1.0);
  }
}

TEST(ScrambledHaltonSampler, HighDimensionStaysInUnitBox) {
  ScrambledHaltonSampler s{5};
  Eigen::VectorXd out(200);
  for (int trial = 0; trial < 5; ++trial) {
    s.sample(200, out);
    for (int j = 0; j < 200; ++j) {
      EXPECT_GE(out[j], 0.0);
      EXPECT_LT(out[j], 1.0);
    }
  }
}

TEST(ScrambledHaltonSampler, HighDimensionSeedReproducible) {
  ScrambledHaltonSampler a{9}, b{9};
  Eigen::VectorXd out_a(100), out_b(100);
  for (int i = 0; i < 5; ++i) {
    a.sample(100, out_a);
    b.sample(100, out_b);
    EXPECT_EQ(out_a, out_b);
  }
}

TEST(HaltonSampler, ThrowsBeyondPrimeTable) {
  HaltonSampler s;
  Eigen::VectorXd out(2000);
  EXPECT_THROW(s.sample(2000, out), std::invalid_argument);
}

TEST(ScrambledHaltonSampler, ThrowsBeyondPrimeTable) {
  ScrambledHaltonSampler s{1};
  Eigen::VectorXd out(2000);
  EXPECT_THROW(s.sample(2000, out), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// from_unit_cube reads only the coordinates it is given
// ---------------------------------------------------------------------------

TEST(FromUnitCube, ShortVectorThrows) {
  const Eigen::VectorXd one = Eigen::VectorXd::Constant(1, 0.5);
  EXPECT_THROW(geodex::Euclidean<2>{}.from_unit_cube(one), std::invalid_argument);
  EXPECT_THROW(geodex::Torus<2>{}.from_unit_cube(one), std::invalid_argument);
  EXPECT_THROW(geodex::SE2<>{}.from_unit_cube(one), std::invalid_argument);
  EXPECT_THROW(geodex::SO3<>{}.from_unit_cube(one), std::invalid_argument);
  EXPECT_THROW(geodex::SE3<>{}.from_unit_cube(one), std::invalid_argument);
  EXPECT_THROW(geodex::Sphere<>{}.from_unit_cube(one), std::invalid_argument);
  EXPECT_NO_THROW(geodex::SO2<>{}.from_unit_cube(one));
  EXPECT_NO_THROW(geodex::Euclidean<2>{}.from_unit_cube(Eigen::VectorXd::Constant(3, 0.5)));
}

// ---------------------------------------------------------------------------
// Concept satisfaction
// ---------------------------------------------------------------------------

static_assert(Sampler<PseudoRandomSampler>);
static_assert(Sampler<HaltonSampler>);
static_assert(Sampler<ScrambledHaltonSampler>);
static_assert(Sampler<DynamicSampler>);
static_assert(SeedableSampler<PseudoRandomSampler>);
static_assert(SeedableSampler<HaltonSampler>);
static_assert(SeedableSampler<ScrambledHaltonSampler>);
static_assert(SeedableSampler<DynamicSampler>);
