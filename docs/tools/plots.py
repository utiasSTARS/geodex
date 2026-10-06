"""Write interactive plotly figures for the docs.

``write_plotly(fig, name)`` writes ``docs/_static/plots/<name>.html``, a fragment without
plotly.js that the ``plotly-figure`` directive inlines, and ``<name>.png``, a still of the
same figure for readers without JavaScript. The fragment has a fixed element id, and an
unchanged figure regenerates to identical bytes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from browser import Browser
from still import serve

ROOT = Path(__file__).resolve().parents[2]
PLOTS = ROOT / "docs" / "_static" / "plots"
CONFIG = {"displaylogo": False, "responsive": True,
          "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]}


def write_plotly(fig, name: str, height: int = 520, still_width: int = 960) -> Path:
    """Write the fragment and still of `fig` under `name` and return the fragment path."""
    import plotly
    import plotly.io as pio

    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.update_layout(height=height, autosize=True)
    fragment = pio.to_html(fig, full_html=False, include_plotlyjs=False, include_mathjax=False,
                           div_id=f"plot-{name}", config=CONFIG, default_width="100%",
                           default_height=f"{height}px", auto_play=False)
    target = PLOTS / f"{name}.html"
    target.write_text(fragment + "\n", encoding="utf-8")

    library = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    with tempfile.TemporaryDirectory() as tmp:
        site = Path(tmp)
        (site / "plotly.min.js").write_bytes(library.read_bytes())
        (site / "index.html").write_text(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<script src='plotly.min.js'></script>"
            f"<style>@font-face{{font-family:Lato;src:url('Lato-Regular.ttf')}}"
            "body{margin:0;background:#fff}.modebar{display:none!important}</style></head>"
            f"<body>{fragment}</body></html>", encoding="utf-8")
        fonts = ROOT / "docs" / "_static" / "fonts"
        (site / "Lato-Regular.ttf").write_bytes((fonts / "Lato-Regular.ttf").read_bytes())
        with serve(site) as url, Browser(still_width, height, scale=2.0) as browser:
            browser.open(f"{url}/index.html")
            browser.evaluate("document.fonts.ready.then(() => true)")
            browser.screenshot(PLOTS / f"{name}.png")
    return target
