"""Draw the landing-page illustration, a geodesic of a Riemannian metric on the plane.

The metric is g(p) = c(p) R(phi(p)) diag(1, k) R(phi(p))^T, a costly bump c around one point
with a rotating anisotropy. The figure draws the unit ball of g at a grid of points (its
Tissot indicatrix), the straight chord between two points, and the geodesic between them,
found by minimizing the discrete energy of a polyline with fixed ends.

Usage: python docs/tools/landing_hero.py [output.svg]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

OUT = Path(__file__).resolve().parents[1] / "_static" / "landing" / "hero.svg"

WIDTH, HEIGHT = 480, 350
X0, Y0, SCALE = 20.0, 335.0, 110.0
K = 0.45
BUMP, BUMP_AT, BUMP_SIGMA = 4.0, np.array([2.05, 1.3]), 0.55
START, GOAL = np.array([0.4, 0.5]), np.array([3.6, 2.3])
POINTS = 64


def cost(p: np.ndarray) -> np.ndarray:
    """The conformal factor c at points of shape (..., 2)."""
    d2 = np.sum((p - BUMP_AT) ** 2, axis=-1)
    return 1.0 + BUMP * np.exp(-d2 / (2.0 * BUMP_SIGMA**2))


def angle(p: np.ndarray) -> np.ndarray:
    """The direction phi of the stiff axis at points of shape (..., 2)."""
    return 0.3 * np.sin(1.1 * p[..., 0]) - 0.3 * p[..., 1]


def metric(p: np.ndarray) -> np.ndarray:
    """The metric matrices of shape (..., 2, 2)."""
    a = angle(p)
    ca, sa = np.cos(a), np.sin(a)
    rot = np.stack([np.stack([ca, -sa], -1), np.stack([sa, ca], -1)], -2)
    lam = np.array([1.0, K])
    return cost(p)[..., None, None] * (rot * lam) @ np.swapaxes(rot, -1, -2)


def energy(flat: np.ndarray) -> float:
    """The discrete energy of the polyline with the interior points `flat`."""
    path = np.vstack([START, flat.reshape(-1, 2), GOAL])
    steps = np.diff(path, axis=0)
    mids = 0.5 * (path[1:] + path[:-1])
    return float(np.einsum("ni,nij,nj->", steps, metric(mids), steps))


def geodesic() -> np.ndarray:
    """The geodesic from START to GOAL, the lower-energy of the two polylines started from
    the chord bent to either side of the bump."""
    t = np.linspace(0.0, 1.0, POINTS)[1:-1, None]
    normal = np.array([GOAL[1] - START[1], START[0] - GOAL[0]])
    normal /= np.linalg.norm(normal)
    results = []
    for side in (1.0, -1.0):
        guess = START + t * (GOAL - START) + side * 0.6 * np.sin(np.pi * t) * normal
        results.append(minimize(energy, guess.ravel(), method="L-BFGS-B",
                                options={"maxiter": 20000, "maxfun": 10**7,
                                         "ftol": 1e-15, "gtol": 1e-10}))
    best = min(results, key=lambda r: r.fun)
    return np.vstack([START, best.x.reshape(-1, 2), GOAL])


def svg_point(p: np.ndarray) -> tuple[float, float]:
    """The SVG coordinates of the plane point `p`."""
    return X0 + SCALE * p[0], Y0 - SCALE * p[1]


def mix(a: str, b: str, s: float) -> str:
    """The hex color a fraction `s` of the way from `a` to `b`."""
    ca = np.array([int(a[i:i + 2], 16) for i in (1, 3, 5)])
    cb = np.array([int(b[i:i + 2], 16) for i in (1, 3, 5)])
    return "#" + "".join(f"{int(round(v)):02x}" for v in (1 - s) * ca + s * cb)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
             f'width="{WIDTH}" height="{HEIGHT}">']
    parts.append('<g stroke="#2980b9" stroke-opacity="0.55" stroke-width="1">')
    for x in np.linspace(0.2, 3.8, 9):
        for y in np.linspace(0.2, 2.8, 7):
            p = np.array([x, y])
            c, a = float(cost(p)), float(angle(p))
            rx, ry = 0.105 / np.sqrt(c), 0.105 / np.sqrt(c * K)
            cx, cy = svg_point(p)
            fill = mix("#f1f6fb", "#b3cfe8", (c - 1.0) / BUMP)
            parts.append(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{SCALE * rx:.1f}" '
                         f'ry="{SCALE * ry:.1f}" fill="{fill}" '
                         f'transform="rotate({-np.degrees(a):.1f} {cx:.1f} {cy:.1f})"/>')
    parts.append("</g>")
    (sx, sy), (gx, gy) = svg_point(START), svg_point(GOAL)
    parts.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{gx:.1f}" y2="{gy:.1f}" '
                 'stroke="#8c99a6" stroke-width="1.6" stroke-dasharray="5 5"/>')
    points = " ".join(f"{u:.1f},{v:.1f}" for u, v in map(svg_point, geodesic()))
    parts.append(f'<polyline points="{points}" fill="none" stroke="#1c5cab" '
                 'stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>')
    parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="7" fill="#1baf7a" '
                 'stroke="#ffffff" stroke-width="2"/>')
    parts.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="7" fill="#eb6834" '
                 'stroke="#ffffff" stroke-width="2"/>')
    parts.append("</svg>")
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
