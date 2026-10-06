"""Assets of docs/ros2/moveit.rst.

``ros2-moveit-attached-object``
    The demo's attached box in its enclosing sphere and in foam's cover of 16 spheres, from
    docs/ros2/captures/attached_box_foam16.yaml.
"""

from __future__ import annotations

import numpy as np
import yaml

from common import ROOT
from plots import write_plotly
from style import BLUE, INK_2, ORANGE, plotly_layout

CAPTURES = ROOT / "docs" / "ros2" / "captures"

# The attached box of the FR3 demo, in 2f85_tcp: its size and center.
BOX = np.array([0.04, 0.24, 0.16])
CENTER = np.array([0.0, 0.0, 0.055])


def _sphere_mesh(go, center, radius, color, opacity, rows=14, cols=28):
    """A sphere as a plotly mesh of `rows` latitude bands."""
    theta = np.linspace(0.0, np.pi, rows + 1)
    phi = np.linspace(0.0, 2.0 * np.pi, cols, endpoint=False)
    t, f = np.meshgrid(theta, phi, indexing="ij")
    xyz = np.stack([np.sin(t) * np.cos(f), np.sin(t) * np.sin(f), np.cos(t)], -1)
    xyz = xyz.reshape(-1, 3) * radius + center
    tri = []
    for a in range(rows):
        for b in range(cols):
            p, q = a * cols + b, a * cols + (b + 1) % cols
            tri += [(p, q, p + cols), (q, q + cols, p + cols)]
    i, j, k = np.array(tri).T
    return go.Mesh3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], i=i, j=j, k=k, color=color,
                     opacity=opacity, flatshading=False, hoverinfo="skip", showscale=False,
                     lighting=dict(ambient=0.7, diffuse=0.5, specular=0.1))


def _box_mesh(go):
    """The attached box as a plotly mesh."""
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    corners = corners * BOX / 2.0 + CENTER
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    i, j, k = np.array(faces).T
    return go.Mesh3d(x=corners[:, 0], y=corners[:, 1], z=corners[:, 2], i=i, j=j, k=k,
                     color=ORANGE, opacity=1.0, flatshading=True, hoverinfo="skip",
                     lighting=dict(ambient=0.75, diffuse=0.5))


def moveit_attached_object():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    doc = yaml.safe_load((CAPTURES / "attached_box_foam16.yaml").read_text())
    cover = np.array(doc["attached_object"]["spheres"], dtype=float).reshape(-1, 4)
    radius = 0.5 * float(np.linalg.norm(BOX))
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=(f"One enclosing sphere, r = {radius:.3f} m",
                                        f"foam, {len(cover)} spheres"),
                        horizontal_spacing=0.02)
    fig.add_trace(_box_mesh(go), row=1, col=1)
    fig.add_trace(_sphere_mesh(go, CENTER, radius, BLUE, 0.22), row=1, col=1)
    fig.add_trace(_box_mesh(go), row=1, col=2)
    for x, y, z, r in cover:
        fig.add_trace(_sphere_mesh(go, np.array([x, y, z]), r, BLUE, 0.22, rows=10, cols=20),
                      row=1, col=2)
    half = radius * 1.02
    axis = dict(range=[-half, half], showbackground=False, showgrid=True, gridcolor="#e6e5e1",
                zeroline=False, showticklabels=False, title="", color=INK_2)
    scene = dict(xaxis=axis, yaxis=axis,
                 zaxis=dict(axis, range=[CENTER[2] - half, CENTER[2] + half]),
                 aspectmode="cube", camera=dict(eye=dict(x=1.5, y=0.9, z=0.7)))
    fig.update_layout(**plotly_layout(margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
                                      scene=scene, scene2=scene))
    write_plotly(fig, "ros2-moveit-attached-object", height=460)


def generate(parts=("attached",)):
    if "attached" in parts:
        moveit_attached_object()
