"""Assets of docs/tutorials/se2-planning.rst.

Runs the five scenarios of figures/se2_tutorial.py with seed 1 and their fixed iteration
budgets, then draws every figure and animation of the page with the plotting scripts in
figures/, in the docs style. The animations are VP9 WebM videos with an H.264 MP4 fallback
and a poster, encoded by ffmpeg (``GEODEX_FFMPEG``, or ``ffmpeg`` on the PATH).
"""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

from common import ROOT, figure_path, video_paths
from style import matplotlib_style

SCENARIOS = ("holonomic", "holo_clearance", "diff_drive", "diff_clearance", "parking")
FIGURES = ROOT / "docs" / "tools" / "figures"
MAP = str(FIGURES / "willow_corridor.png")
VIDEO_DPI = 150


def _script(name: str, *args: str):
    """Run a plotting script in this process, after the docs style registered Lato."""
    plt = matplotlib_style()
    plt.rcParams["animation.ffmpeg_path"] = os.environ.get("GEODEX_FFMPEG", "ffmpeg")
    argv = sys.argv
    sys.argv = [name, *args]
    try:
        runpy.run_path(str(FIGURES / name), run_name="__main__")
    except SystemExit as exit_:
        if exit_.code not in (0, None):
            raise
    finally:
        sys.argv = argv


def _animate(run: str, name: str, *args: str):
    """Write the animation of a run as a video pair and poster under _static/videos."""
    webm, mp4, poster = video_paths(name)
    _script("animate_se2_tutorial.py", run, "-o", str(webm), str(mp4), "--poster", str(poster),
            "--fps", "15", "--dpi", str(VIDEO_DPI), *args)


def generate(parts=("figures",)):
    with tempfile.TemporaryDirectory() as tmp:
        runs = {}
        for scenario in SCENARIOS:
            out = Path(tmp) / f"{scenario}.json"
            subprocess.run([sys.executable, str(FIGURES / "se2_tutorial.py"),
                            f"--scenario={scenario}", "-o", str(out), "--seed=1"], check=True,
                           stdout=subprocess.DEVNULL)
            runs[scenario] = str(out)

        def fig(name):
            return str(figure_path("tutorials", "se2-planning", name))

        for scenario, name in (("holonomic", "holonomic"), ("diff_drive", "diff_drive"),
                               ("diff_clearance", "diff_clearance")):
            _script("visualize_se2_tutorial.py", runs[scenario], "-o", fig(f"{name}_result.svg"),
                    "--map", MAP)
            _animate(runs[scenario], f"se2-{name.replace('_', '-')}-sweep", "--map", MAP)
        _script("visualize_se2_tutorial.py", runs["holo_clearance"], "-o",
                fig("conformal_factor.svg"), "--map", MAP, "--mode", "conformal")
        _script("visualize_se2_tutorial.py", runs["holonomic"], runs["holo_clearance"], "-o",
                fig("clearance_comparison.svg"), "--map", MAP, "--mode", "comparison")
        _script("visualize_se2_tutorial.py", runs["parking"], "-o", fig("parking_result.svg"))
        _script("visualize_se2_tutorial.py", runs["parking"], "-o", fig("parking_lot.svg"),
                "--mode", "env")
        _animate(runs["parking"], "se2-parking-footprints")
        _script("visualize_se2_tutorial.py", runs["holonomic"], "-o", fig("willow_corridor.svg"),
                "--map", MAP, "--mode", "map")
        figs = Path(fig("poses_and_footprints.svg")).parent
        _script("draw_se2_diagrams.py", "--output-dir", str(figs))
        for name in ("poses_and_footprints", "inflation", "footprint_checking"):
            (figs / f"se2_{name}.svg").replace(figs / f"{name}.svg")
