"""The examples behind the getting-started, concepts and tutorial pages.

Run with ``pixi run test-docs``. See ``pairs.py`` for what a pair test checks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pairs import Example, check_example

EXAMPLES = [
    Example("getting_started/first_plan"),
    Example("getting_started/quickstart"),
    Example("getting_started/reproducibility"),
    Example("tutorials/geodex_basics"),
    Example("concepts/sampling"),
    Example("concepts/metrics"),
    Example("concepts/discrete_geodesic"),
    Example("concepts/planning"),
    Example("concepts/smoothing"),
    Example("tutorials/minimum_energy_planning"),
    Example("tutorials/se2_planning"),
]


@pytest.mark.parametrize("example", EXAMPLES, ids=str)
def test_python_matches_cpp(example: Example, tmp_path: Path):
    check_example(example, tmp_path)


def test_se2_figures_show_the_snippet_plans(tmp_path: Path):
    """The SE(2) tutorial draws its holonomic and differential-drive figures with
    docs/tools/figures/se2_tutorial.py. Every waypoint of the page's snippet program lies
    on the figure's densified path."""
    import json
    import subprocess
    import sys

    import numpy as np

    from pairs import ROOT

    tutorials = ROOT / "examples" / "tutorials"
    tool = ROOT / "docs" / "tools" / "figures" / "se2_tutorial.py"
    subprocess.run([sys.executable, "se2_planning.py", "--json", str(tmp_path / "s.json")],
                   cwd=tutorials, check=True, capture_output=True)
    snippets = json.loads((tmp_path / "s.json").read_text())
    for scenario, key in (("holonomic", "holonomic"), ("diff_drive", "directional")):
        out = tmp_path / f"{scenario}.json"
        subprocess.run([sys.executable, str(tool), f"--scenario={scenario}", "-o", str(out),
                        "--seed=1"], check=True, capture_output=True)
        figure = np.array(json.loads(out.read_text())["runs"][0]["smoothed_path"])
        for waypoint in np.array(snippets[key]["path"]):
            d = figure - waypoint
            turn = np.abs(np.arctan2(np.sin(d[:, 2]), np.cos(d[:, 2])))
            gap = np.min(np.hypot(d[:, 0], d[:, 1]) + turn)
            assert gap < 1e-9, f"{scenario}: a snippet waypoint is {gap} from the figure path"
