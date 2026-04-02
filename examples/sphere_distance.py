#!/usr/bin/env python3
"""Distance computation on the sphere with different metrics and retractions.

Compares geodesic distance (exact) with the midpoint distance approximation
across three setups:
  1. Round metric + exponential map
  2. Anisotropic metric (A=diag(4,1,1)) + exponential map
  3. Round metric + projection retraction
"""

import numpy as np

import geodex


def point_at_theta(theta: float) -> np.ndarray:
    """Point on the great circle in the xz-plane at angle theta from the north pole."""
    return np.array([np.sin(theta), 0.0, np.cos(theta)])


def print_distance_table(label: str, manifold, p: np.ndarray, thetas: list[float]):
    print(f"\n=== {label} ===")
    print(f"{'theta':>10s}{'exact':>15s}{'midpoint':>15s}{'error':>15s}")
    print("-" * 55)

    for theta in thetas:
        q = point_at_theta(theta)
        exact = manifold.distance(p, q)
        midpoint = geodex.distance_midpoint(manifold, p, q)
        error = abs(midpoint - exact)
        print(f"{theta:10.4f}{exact:15.4f}{midpoint:15.4f}{error:15.2e}")


north = np.array([0.0, 0.0, 1.0])
thetas = [0.1, 0.5, 1.0, np.pi / 2, 2.0, 3.0, np.pi]

# 1. Round metric + true exp/log
round_sphere = geodex.Sphere()
print_distance_table("Round metric + Exponential map", round_sphere, north, thetas)

# 2. Anisotropic metric + true exp/log
A = np.diag([4.0, 1.0, 1.0])
weighted = geodex.ConstantSPDMetric(A)
aniso_sphere = geodex.ConfigurationSpace(geodex.Sphere(), weighted)
print_distance_table("Anisotropic metric (A=diag(4,1,1)) + Exponential map",
                     aniso_sphere, north, thetas)

# 3. Round metric + projection retraction
proj_sphere = geodex.Sphere(retraction="projection")
print_distance_table("Round metric + Projection retraction", proj_sphere, north, thetas)
