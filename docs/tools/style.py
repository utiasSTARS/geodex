"""One visual style for every generated docs figure.

Figures use the body font of the docs site, Lato from docs/_static/fonts, and sans-serif
math in STIX Sans. Series take the validated categorical colors below in a fixed order and
never cycle. The first three stay distinct for color-blind readers. A figure shows at most
three series and labels them directly.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FONTS = ROOT / "docs" / "_static" / "fonts"

# Categorical slots, adjacent pairs contrast-checked on the light surface.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SERIES = (BLUE, ORANGE, AQUA)

INK = "#0b0b0b"          # titles and values
INK_2 = "#52514e"        # axis labels and annotations
GRID = "#e6e5e1"         # recessive grid lines
OBSTACLE = "#9aa0a6"     # obstacles and walls, in a neutral gray
OBSTACLE_EDGE = "#6b7178"
SURFACE_OBSTACLE = "#3d5a80"  # obstacles on a manifold's surface, a light shade of the theme ink
SURFACE = "#ffffff"
FONT = "Lato"


def matplotlib_style():
    """Apply the docs style to matplotlib and return the pyplot module."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    for path in sorted(FONTS.glob("*.ttf")):
        font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 14,
        "mathtext.fontset": "stixsans",
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.grid": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "legend.frameon": False,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "svg.fonttype": "path",
        "svg.hashsalt": "geodex-docs",
        "path.simplify": True,
    })
    return plt


def plotly_layout(**overrides) -> dict:
    """Layout defaults for a plotly figure in the docs."""
    axis = dict(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=INK_2,
                ticks="outside", tickcolor=INK_2, title_font=dict(color=INK_2))
    layout = dict(
        font=dict(family=f"{FONT}, Helvetica, Arial, sans-serif", size=15, color=INK),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=dict(l=70, r=30, t=40, b=60),
        xaxis=axis,
        yaxis=dict(axis),
        hoverlabel=dict(font=dict(family=f"{FONT}, sans-serif", size=14)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    layout.update(overrides)
    return layout
