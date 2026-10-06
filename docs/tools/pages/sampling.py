"""Assets of docs/concepts/sampling.rst.

``docs/concepts/figs/cube-to-manifold.svg``
    Scrambled Halton points in the unit square and their image on the sphere under
    ``from_unit_cube``.
``docs/concepts/figs/low-discrepancy-comparison.svg``
    100 points of the pseudo-random, Halton and scrambled Halton samplers in the unit square.

figures/visualize_sampling.py draws both with seed 0.
"""

from __future__ import annotations

import runpy
import sys

from common import DOCS, ROOT
from style import matplotlib_style

SCRIPT = ROOT / "docs" / "tools" / "figures" / "visualize_sampling.py"


def generate(parts=("figures",)):
    matplotlib_style()
    argv = sys.argv
    sys.argv = [SCRIPT.name, "--out-dir", str(DOCS / "concepts" / "figs")]
    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    finally:
        sys.argv = argv
