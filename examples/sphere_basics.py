#!/usr/bin/env python3
"""Basic operations on the 2-sphere using geodex Python bindings.

Demonstrates:
  - Creating manifolds with default and custom policies
  - Exponential and logarithmic maps
  - Geodesic distance and interpolation
  - Swapping metrics (ConstantSPDMetric) and retractions (projection retraction)
"""

import numpy as np

import geodex


def fmt(arr):
    """Format a numpy array like Eigen's IOFormat: [0.1234, 0.5678, 0.9012]."""
    return "[" + ", ".join(f"{x:.4f}" for x in arr) + "]"


# --- 1. Default sphere: round metric + exponential map ---
sphere = geodex.Sphere()
print("=== Sphere (round metric, exponential map) ===")
print(f"dim = {sphere.dim()}\n")

p = np.array([0.0, 0.0, 1.0])  # north pole
q = np.array([1.0, 0.0, 0.0])  # equator

# Logarithmic map: tangent vector at p pointing toward q
v = sphere.log(p, q)
print(f"p = {fmt(p)}")
print(f"q = {fmt(q)}")
print(f"log(p, q) = {fmt(v)}")

# Exponential map: recover q from p and the tangent vector
q_recovered = sphere.exp(p, v)
print(f"exp(p, v) = {fmt(q_recovered)}\n")

# Geodesic distance: should be pi/2
d = sphere.distance(p, q)
print(f"distance(p, q) = {d:.4f}  (expected: {np.pi / 2:.4f})\n")

# Geodesic interpolation: trace the great circle
print("Geodesic interpolation:")
for i in range(6):
    t = i / 5.0
    pt = sphere.geodesic(p, q, t)
    print(f"  t={t:.1f}: {fmt(pt)}")

# --- 2. Anisotropic metric: ConstantSPDMetric ---
print("\n=== Sphere (anisotropic metric A=diag(4,1,1)) ===")

A = np.diag([4.0, 1.0, 1.0])
weighted = geodex.ConstantSPDMetric(A)
aniso_sphere = geodex.ConfigurationSpace(geodex.Sphere(), weighted)

# The anisotropic metric changes norms and distances
u = np.array([1.0, 0.0, 0.0])
norm_round = sphere.norm(p, u)
norm_aniso = aniso_sphere.norm(p, u)
print(f"norm_round(p, [1,0,0]) = {norm_round:.4f}")
print(f"norm_aniso(p, [1,0,0]) = {norm_aniso:.4f}  (scaled by sqrt(4))")

d_aniso = aniso_sphere.distance(p, q)
print(f"distance_round(p, q) = {d:.4f}")
print(f"distance_aniso(p, q) = {d_aniso:.4f}")

# --- 3. Projection retraction ---
print("\n=== Sphere (round metric, projection retraction) ===")

proj_sphere = geodex.Sphere(retraction="projection")

# Projection retraction: cheaper but approximate
v_proj = proj_sphere.log(p, q)
q_proj = proj_sphere.exp(p, v_proj)
print(f"log(p, q) = {fmt(v_proj)}")
print(f"exp(p, log(p, q)) = {fmt(q_proj)}  (approximate round-trip)")

d_proj = proj_sphere.distance(p, q)
print(f"distance_proj(p, q) = {d_proj:.4f}")
print(f"distance_exact(p, q) = {d:.4f}")

# --- 4. Random sampling ---
print("\n=== Random sampling ===")
for i in range(3):
    rp = sphere.random_point()
    print(f"  random_point {i}: {fmt(rp)}  (norm={np.linalg.norm(rp):.4f})")
