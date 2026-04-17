#!/usr/bin/env python3
"""Discrete geodesic interpolation on the sphere across metric/retraction variants.

Python-API parallel of `examples/sphere_interpolation.cpp`. Runs
`discrete_geodesic` from the north pole to a shared target under five
configurations and prints per-case summary statistics:

  1. Round metric + exponential map (textbook great-circle).
  2. Round metric + projection retraction.
  3. Anisotropic metric A = diag(4, 1, 1).
  4. Anisotropic metric A = diag(25, 1, 1), smaller step size.
  5. Round metric near-antipodal target (graceful degradation near cut locus).

For each run we report status, iteration count, initial/final distance, path
length (waypoint count), arc length, and max angular deviation from the
great-circle plane through start and target. Anisotropic runs should show a
non-zero deviation — the path bends to trade expensive x-motion for cheaper
y-motion.
"""

import math

import numpy as np

import geodex


def spherical_point(theta: float, phi: float) -> np.ndarray:
    """Point on the unit sphere at polar angle theta (from north) and azimuth phi."""
    return np.array(
        [math.sin(theta) * math.cos(phi),
         math.sin(theta) * math.sin(phi),
         math.cos(theta)]
    )


def arc_length(points: list[np.ndarray]) -> float:
    """Sum of successive angular (great-circle) distances along the polyline."""
    total = 0.0
    for i in range(len(points) - 1):
        dot = np.clip(np.dot(points[i], points[i + 1]), -1.0, 1.0)
        total += math.acos(dot)
    return total


def max_plane_deviation(points: list[np.ndarray], start: np.ndarray,
                        target: np.ndarray) -> float:
    """Max angular deviation (radians) of any path point from the great circle
    plane spanned by `start` and `target`. Zero for a true great-circle arc."""
    normal = np.cross(start, target)
    n = np.linalg.norm(normal)
    if n < 1e-12:
        return 0.0  # degenerate (antipodal or identical endpoints)
    normal /= n
    return max(math.asin(min(1.0, abs(float(np.dot(p, normal))))) for p in points)


def run(label: str, manifold, start: np.ndarray, target: np.ndarray,
        settings: geodex.InterpolationSettings):
    result = geodex.discrete_geodesic(manifold, start, target, settings)
    pts = [np.asarray(p) for p in result.path]
    arc = arc_length(pts)
    dev = max_plane_deviation(pts, start, target)
    print(
        f"  [{label:48s}] status={str(result.status):24s} "
        f"iters={result.iterations:4d} "
        f"d_init={result.initial_distance:.4f} d_final={result.final_distance:.2e} "
        f"pts={len(pts):4d} arc={arc:.4f} rad  dev={math.degrees(dev):6.2f} deg"
    )


# Common start (north pole) and a shared target off the y=0 plane so anisotropy
# can bend the path along y.
north = np.array([0.0, 0.0, 1.0])
shared = spherical_point(theta=1.3, phi=0.9)  # ~52 deg azimuth

print("Running discrete_geodesic scenarios on the sphere:\n")

# 1. Round metric + exponential map (default Sphere).
s1 = geodex.InterpolationSettings(step_size=0.1)
run("1. Round metric, true exp/log", geodex.Sphere(), north, shared, s1)

# 2. Round metric + projection retraction.
s2 = geodex.InterpolationSettings(step_size=0.1)
run("2. Round metric, projection retraction",
    geodex.Sphere(retraction="projection"), north, shared, s2)

# 3. Moderate anisotropic metric via ConfigurationSpace wrapper.
A1 = np.diag([4.0, 1.0, 1.0])
aniso1 = geodex.ConfigurationSpace(geodex.Sphere(), geodex.ConstantSPDMetric(A1))
s3 = geodex.InterpolationSettings(step_size=0.1)
run("3. Anisotropic A=diag(4,1,1)", aniso1, north, shared, s3)

# 4. Heavy anisotropic metric — smaller step_size for stability.
A2 = np.diag([25.0, 1.0, 1.0])
aniso2 = geodex.ConfigurationSpace(geodex.Sphere(), geodex.ConstantSPDMetric(A2))
s4 = geodex.InterpolationSettings(step_size=0.05, max_steps=500)
run("4. Anisotropic A=diag(25,1,1)", aniso2, north, shared, s4)

# 5. Near-antipodal target (just inside the cut locus).
near_antipodal = spherical_point(theta=math.pi - 0.1, phi=0.9)
s5 = geodex.InterpolationSettings(step_size=0.2)
run("5. Round metric, near-antipodal", geodex.Sphere(), north, near_antipodal, s5)
