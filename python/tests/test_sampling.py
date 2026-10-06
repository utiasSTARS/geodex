"""Sampler seeding and reproducibility across the Python manifolds."""

import numpy as np
import pytest

import geodex


def _manifolds():
    return [
        geodex.Euclidean(3),
        geodex.Torus(2),
        geodex.SO2(),
        geodex.SE2(),
        geodex.SO3(),
        geodex.SE3(),
        geodex.Sphere(),
        geodex.SphereN(4),
    ]


@pytest.mark.parametrize("m", _manifolds())
def test_seed_is_reproducible(m):
    m.seed(42)
    a = np.array([m.random_point() for _ in range(6)])
    m.seed(42)
    b = np.array([m.random_point() for _ in range(6)])
    assert np.array_equal(a, b)


def test_different_seeds_diverge():
    e = geodex.Euclidean(3)
    e.seed(1)
    a = np.array([e.random_point() for _ in range(4)])
    e.seed(2)
    b = np.array([e.random_point() for _ in range(4)])
    assert not np.array_equal(a, b)


def test_module_seed_makes_default_construction_reproducible():
    geodex.seed(7)
    e1 = geodex.Euclidean(3)
    x = np.array([e1.random_point() for _ in range(5)])
    geodex.seed(7)
    e2 = geodex.Euclidean(3)
    y = np.array([e2.random_point() for _ in range(5)])
    assert np.array_equal(x, y)


def test_product_reproducible_after_module_seed():
    # A product built after geodex.seed(n) reproduces, since each block's default
    # sampler takes its seed from the reseeded source at construction.
    geodex.seed(11)
    p1 = geodex.Product([geodex.Euclidean(2), geodex.SO2()])
    a = np.array([p1.random_point() for _ in range(4)])
    geodex.seed(11)
    p2 = geodex.Product([geodex.Euclidean(2), geodex.SO2()])
    b = np.array([p2.random_point() for _ in range(4)])
    assert np.array_equal(a, b)


def test_high_dimension_euclidean_is_valid():
    p = geodex.Euclidean(100).random_point()
    assert p.shape == (100,)
    assert np.all(np.isfinite(p))
    assert np.all(p >= -1.0) and np.all(p <= 1.0)


def test_sphere_and_so3_stay_on_manifold_after_seed():
    s = geodex.Sphere()
    s.seed(3)
    assert abs(np.linalg.norm(s.random_point()) - 1.0) < 1e-9
    q = geodex.SO3()
    q.seed(3)
    assert abs(np.linalg.norm(q.random_point()) - 1.0) < 1e-9


def test_halton_kwarg_is_deterministic():
    # Plain Halton is deterministic, so two instances match without any seeding.
    a = geodex.Euclidean(3, sampler="halton")
    b = geodex.Euclidean(3, sampler="halton")
    xa = np.array([a.random_point() for _ in range(5)])
    xb = np.array([b.random_point() for _ in range(5)])
    assert np.array_equal(xa, xb)


def test_set_sampler_switches_to_halton():
    m = geodex.Euclidean(2)
    m.set_sampler("halton")
    ref = geodex.Euclidean(2, sampler="halton")
    xa = np.array([m.random_point() for _ in range(5)])
    xb = np.array([ref.random_point() for _ in range(5)])
    assert np.array_equal(xa, xb)


def test_random_sampler_kwarg_reproducible_with_seed():
    a = geodex.Euclidean(3, sampler="random")
    a.seed(123)
    xa = np.array([a.random_point() for _ in range(5)])
    b = geodex.Euclidean(3, sampler="random")
    b.seed(123)
    xb = np.array([b.random_point() for _ in range(5)])
    assert np.array_equal(xa, xb)


def test_sampler_kwarg_on_variant_manifold():
    # Variant wrappers (SE2) accept the sampler kwarg too.
    a = geodex.SE2(sampler="halton")
    b = geodex.SE2(sampler="halton")
    xa = np.array([a.random_point() for _ in range(5)])
    xb = np.array([b.random_point() for _ in range(5)])
    assert np.array_equal(xa, xb)


def test_unknown_sampler_raises():
    with pytest.raises(ValueError):
        geodex.Euclidean(3, sampler="nope")


def test_product_seed_is_reproducible():
    p1 = geodex.Product([geodex.SO3(), geodex.Euclidean(3)])
    p1.seed(5)
    a = np.array([p1.random_point() for _ in range(4)])
    p2 = geodex.Product([geodex.SO3(), geodex.Euclidean(3)])
    p2.seed(5)
    b = np.array([p2.random_point() for _ in range(4)])
    assert np.array_equal(a, b)


def test_product_set_sampler_halton_is_deterministic():
    p1 = geodex.Product([geodex.Euclidean(2), geodex.SO2()])
    p1.set_sampler("halton")
    a = np.array([p1.random_point() for _ in range(4)])
    p2 = geodex.Product([geodex.Euclidean(2), geodex.SO2()])
    p2.set_sampler("halton")
    b = np.array([p2.random_point() for _ in range(4)])
    assert np.array_equal(a, b)
