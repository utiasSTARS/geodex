#!/usr/bin/env python3
"""Evaluate KE and Jacobi metrics on a grid over T^2 (2-link planar arm).

Outputs a JSON file with:
  - fine grid (50x50) of potential and det(M) values for background heatmaps
  - coarse grid (12x12) of inverse metric tensors for ellipse visualization

Usage:
  python minimum_energy_grid.py [output.json]
"""

import json
import sys

import numpy as np

import geodex


# --- 2-link planar arm parameters ---
L1, L2 = 1.0, 1.0       # link lengths (m)
M1, M2 = 1.0, 1.0       # link masses (kg)
LC1, LC2 = 0.5, 0.5     # CoM distances from joint (m)
I1, I2 = 1/12, 1/12     # moments of inertia (kg*m^2)
G = 9.81                 # gravitational acceleration (m/s^2)


def mass_matrix(q: np.ndarray) -> np.ndarray:
    """Compute the 2x2 mass matrix M(q) for the planar arm."""
    c2 = np.cos(q[1])
    h = L1 * LC2 * c2
    m00 = I1 + I2 + M1*LC1**2 + M2*(L1**2 + LC2**2 + 2*h)
    m01 = I2 + M2*(LC2**2 + h)
    m11 = I2 + M2*LC2**2
    return np.array([[m00, m01], [m01, m11]])


def potential(q: np.ndarray) -> float:
    """Gravitational potential energy P(q)."""
    return (M1 * G * LC1 * np.sin(q[0])
            + M2 * G * (L1 * np.sin(q[0]) + LC2 * np.sin(q[0] + q[1])))


# Analytical upper bound: arm fully extended upward
PMAX = G * (M1*LC1 + M2*(L1 + LC2))

# Build geodex configuration spaces using the mass matrix callable
ke_metric = geodex.KineticEnergyMetric(mass_matrix)
cspace_ke = geodex.ConfigurationSpace(geodex.Euclidean(2), ke_metric)

# Grid parameters
N_FINE = 50
N_ELLIPSE = 12
LO, HI = -np.pi, np.pi

q1_fine = np.linspace(LO, HI, N_FINE)
q2_fine = np.linspace(LO, HI, N_FINE)
q1_coarse = np.array([LO + (HI - LO) * (i + 0.5) / N_ELLIPSE for i in range(N_ELLIPSE)])
q2_coarse = np.array([LO + (HI - LO) * (i + 0.5) / N_ELLIPSE for i in range(N_ELLIPSE)])

# Evaluate on fine grid
pot_grid = np.zeros((N_FINE, N_FINE))
det_grid = np.zeros((N_FINE, N_FINE))
for i, q1 in enumerate(q1_fine):
    for j, q2 in enumerate(q2_fine):
        q = np.array([q1, q2])
        pot_grid[i, j] = potential(q)
        det_grid[i, j] = np.linalg.det(mass_matrix(q))

# KE metric inverse tensors on coarse grid
ke_ellipses = []
for q1 in q1_coarse:
    for q2 in q2_coarse:
        q = np.array([q1, q2])
        M = mass_matrix(q)
        M_inv = np.linalg.inv(M)
        ke_ellipses.append({
            "q": [float(q1), float(q2)],
            "M_inv": M_inv.tolist(),
        })

# Jacobi metric inverse tensors for multiple energy levels
H_FACTORS = [1.2, 2.0, 5.0]
jacobi_data = []
for alpha in H_FACTORS:
    H = alpha * PMAX
    ellipses = []
    for q1 in q1_coarse:
        for q2 in q2_coarse:
            q = np.array([q1, q2])
            M_inv = np.linalg.inv(mass_matrix(q))
            scale = 2.0 * (H - potential(q))
            J_inv = M_inv / scale
            ellipses.append({
                "q": [float(q1), float(q2)],
                "J_inv": J_inv.tolist(),
            })
    jacobi_data.append({
        "H_over_Pmax": alpha,
        "H": float(H),
        "ellipses": ellipses,
    })

# Write JSON output
output_file = sys.argv[1] if len(sys.argv) > 1 else "minimum_energy_grid.json"
result = {
    "arm": {"l1": L1, "l2": L2, "m1": M1, "m2": M2,
            "lc1": LC1, "lc2": LC2, "I1": I1, "I2": I2, "g": G},
    "pmax": float(PMAX),
    "grid_fine": {
        "n": N_FINE,
        "q1": q1_fine.tolist(),
        "q2": q2_fine.tolist(),
        "potential": pot_grid.tolist(),
        "det_M": det_grid.tolist(),
    },
    "grid_ellipse": {
        "n": N_ELLIPSE,
        "q1": q1_coarse.tolist(),
        "q2": q2_coarse.tolist(),
        "ke": ke_ellipses,
        "jacobi": jacobi_data,
    },
}

with open(output_file, "w") as f:
    json.dump(result, f, indent=2)

print(f"Wrote {output_file}")
print(f"  Pmax (analytical) = {PMAX:.2f} J")
print(f"  Fine grid:   {N_FINE}x{N_FINE}")
print(f"  Coarse grid: {N_ELLIPSE}x{N_ELLIPSE}")
print(f"\n  KE config space dim = {cspace_ke.dim()}")
