"""Assets of docs/getting-started/reproducibility.rst.

``reproducibility-budgets``
    The first plan of the landing page run ten times per seed for ten seeds, once with a
    budget of 2000 iterations and once with a budget of 0.02 s. Each distinct cost of a seed is
    one point, with its number of runs. The iteration budget gives one cost per seed, and under
    the time budget some seeds return more than one, depending on the machine.
"""

from __future__ import annotations

import sys

import numpy as np

from common import ROOT
from plots import write_plotly
from style import BLUE, INK_2, ORANGE, plotly_layout

SEEDS = range(1, 11)
REPEATS = 10
TIME_BUDGET = 0.02


def generate():
    import geodex
    import plotly.graph_objects as go

    sys.path.insert(0, str(ROOT / "examples" / "getting_started"))
    import reproducibility as example

    rows = []
    for seed in SEEDS:
        for _ in range(REPEATS):
            for label, settings in (
                    ("iterations", geodex.PlanSettings(iterations=2000, seed=seed)),
                    ("time", geodex.PlanSettings(time=TIME_BUDGET, seed=seed))):
                result = geodex.plan(example.space, example.start, example.goal,
                                     example.is_valid, settings=settings)
                if result.solved:
                    rows.append((label, seed, result.cost))

    fig = go.Figure()
    # One point per distinct cost of a seed, on the seed's line. A time-budget point is an open
    # circle around the iteration-budget dot it may coincide with.
    for label, name, marker in (
            ("iterations", "2000 iterations", dict(color=BLUE, size=10,
                                                   line=dict(color="#ffffff", width=1))),
            ("time", f"{TIME_BUDGET:g} s", dict(color="rgba(0,0,0,0)", size=15,
                                                 line=dict(color=ORANGE, width=2.5)))):
        counts = {}
        for lab, seed, cost in rows:
            if lab == label:
                key = (seed, round(cost, 6))
                counts[key] = counts.get(key, 0) + 1
        seeds = np.array([s for s, _ in counts], dtype=float)
        costs = np.array([c for _, c in counts])
        runs = np.array(list(counts.values()), dtype=float)
        fig.add_trace(go.Scatter(
            x=seeds, y=costs, mode="markers", name=f"budget {name}", marker=marker,
            customdata=np.stack([seeds, costs, runs], axis=1),
            hovertemplate=(f"budget {name}<br>seed %{{customdata[0]:.0f}}"
                           "<br>cost %{customdata[1]:.4f}<br>%{customdata[2]:.0f} of "
                           f"{REPEATS} runs<extra></extra>")))
    fig.update_layout(**plotly_layout(
        xaxis=dict(title="seed (ten runs each)", tickvals=list(SEEDS), gridcolor="#e6e5e1",
                   linecolor=INK_2, ticks="outside"),
        yaxis=dict(title="metric length of the path", gridcolor="#e6e5e1",
                   linecolor=INK_2, ticks="outside"),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08),
        hovermode="closest"))
    write_plotly(fig, "reproducibility-budgets", height=440)
