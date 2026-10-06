"""Assets of docs/getting-started/installation.rst.

``installation-first-plan``
    The plan of examples/getting_started/first_plan.py, a differential-drive robot driving
    around a disc in a 4 m by 4 m room, drawn along SE(2) geodesics with its heading.
"""

from __future__ import annotations

import numpy as np

from common import densify, run_example
from plots import write_plotly
from style import AQUA, BLUE, INK_2, OBSTACLE, OBSTACLE_EDGE, ORANGE, plotly_layout

DISC = (2.0, 2.0, 0.8)


def generate(parts=("first-plan",)):
    import geodex
    import plotly.graph_objects as go

    data = run_example("getting_started/first_plan")
    space = geodex.SE2(wx=1.0, wy=100.0, wtheta=1.0, x_lo=0.0, x_hi=4.0, y_lo=0.0, y_hi=4.0)
    path = densify(space, data["path"], step=0.05)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines", name="path",
                             line=dict(color=BLUE, width=3),
                             customdata=np.degrees(path[:, 2]),
                             hovertemplate="x %{x:.2f} m<br>y %{y:.2f} m"
                                           "<br>heading %{customdata:.0f} deg<extra></extra>"))
    poses = path[:: max(1, len(path) // 12)][1:-1]
    fig.add_trace(go.Scatter(x=poses[:, 0], y=poses[:, 1], mode="markers", name="heading",
                             hoverinfo="skip",
                             marker=dict(symbol="arrow", size=15, color=INK_2,
                                         angle=90.0 - np.degrees(poses[:, 2]))))
    ends = path[[0, -1]]
    fig.add_trace(go.Scatter(x=ends[:, 0], y=ends[:, 1], mode="markers", showlegend=False,
                             marker=dict(size=13, color=[AQUA, ORANGE],
                                         line=dict(color="#ffffff", width=2)),
                             hovertext=["start", "goal"], hoverinfo="text"))

    cx, cy, r = DISC
    shapes = [
        dict(type="rect", x0=0.0, x1=4.0, y0=0.0, y1=4.0, line=dict(color=INK_2, width=2)),
        dict(type="circle", x0=cx - r, x1=cx + r, y0=cy - r, y1=cy + r, fillcolor=OBSTACLE,
             opacity=0.55, line=dict(color=OBSTACLE_EDGE)),
    ]
    fig.update_layout(**plotly_layout(
        shapes=shapes, legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08),
        margin=dict(l=40, r=20, t=50, b=50), hovermode="closest",
        xaxis=dict(title="x (m)", range=[-0.2, 4.2], gridcolor="#e6e5e1", linecolor=INK_2,
                   ticks="outside", constrain="domain"),
        yaxis=dict(title="y (m)", range=[-0.2, 4.2], gridcolor="#e6e5e1", linecolor=INK_2,
                   ticks="outside", scaleanchor="x", constrain="domain")))
    write_plotly(fig, "installation-first-plan", height=460)
