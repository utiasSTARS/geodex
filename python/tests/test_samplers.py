"""Standalone sampler classes: shape, range, determinism, and seeding."""

import numpy as np
import pytest

import geodex


def test_halton_is_deterministic_across_instances():
    a = geodex.HaltonSampler()
    b = geodex.HaltonSampler()
    for _ in range(20):
        assert np.array_equal(a.sample(3), b.sample(3))


def test_halton_known_first_values():
    s = geodex.HaltonSampler()
    assert np.allclose(s.sample(2), [0.5, 1.0 / 3.0])
    assert np.allclose(s.sample(2), [0.25, 2.0 / 3.0])


def test_scrambled_halton_seed_reproducible():
    a = geodex.ScrambledHaltonSampler(42)
    b = geodex.ScrambledHaltonSampler(42)
    for _ in range(20):
        assert np.array_equal(a.sample(4), b.sample(4))


def test_scrambled_halton_different_seeds_diverge():
    a = geodex.ScrambledHaltonSampler(1)
    b = geodex.ScrambledHaltonSampler(2)
    assert not np.array_equal(a.sample(4), b.sample(4))


def test_reseed_resets_scrambled_sequence():
    s = geodex.ScrambledHaltonSampler(5)
    a = s.sample(3)
    s.seed(5)
    b = s.sample(3)
    assert np.array_equal(a, b)


def test_pseudorandom_seed_reproducible():
    a = geodex.PseudoRandomSampler(7)
    b = geodex.PseudoRandomSampler(7)
    for _ in range(20):
        assert np.array_equal(a.sample(3), b.sample(3))


@pytest.mark.parametrize(
    "make",
    [
        lambda: geodex.ScrambledHaltonSampler(0),
        geodex.HaltonSampler,
        lambda: geodex.PseudoRandomSampler(0),
    ],
)
def test_sample_shape_and_range(make):
    s = make()
    for n in (1, 3, 8):
        u = s.sample(n)
        assert u.shape == (n,)
        assert np.all(u >= 0.0) and np.all(u < 1.0)
