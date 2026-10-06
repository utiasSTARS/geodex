"""Helpers shared by the page generators."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def run_example(stem: str, *args: str) -> dict:
    """Run ``examples/<stem>.py`` with ``--json`` and return its results.

    Figures use the output of the documented example itself. A figure and the snippet next
    to it show the same computation.
    """
    script = ROOT / "examples" / f"{stem}.py"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.json"
        subprocess.run([sys.executable, str(script), *args, "--json", str(out)],
                       cwd=script.parent, check=True, stdout=subprocess.DEVNULL)
        return json.loads(out.read_text())


def densify(manifold, path, per_edge: int | None = None, step: float | None = None):
    """Points along ``manifold.geodesic`` between consecutive waypoints.

    A planner joins its waypoints by geodesics. Straight segments between them show a path
    the robot does not take.
    """
    path = [np.asarray(p, dtype=float) for p in path]
    points = []
    for a, b in zip(path[:-1], path[1:]):
        n = per_edge or max(1, int(np.ceil(manifold.distance(a, b) / step)))
        points += [manifold.geodesic(a, b, k / n) for k in range(n)]
    points.append(path[-1])
    return np.array(points)


def video_paths(name: str) -> tuple[Path, Path, Path]:
    """Paths of an animation under ``docs/_static/videos/<name>``, a VP9 ``.webm``, an H.264
    ``.mp4`` fallback and a ``.png`` poster. ``video-figure`` loads these three files."""
    base = DOCS / "_static" / "videos" / name
    base.parent.mkdir(parents=True, exist_ok=True)
    return tuple(base.with_suffix(suffix) for suffix in (".webm", ".mp4", ".png"))


def figure_path(section: str, page: str, name: str) -> Path:
    """Path of a static figure of a page, ``docs/<section>/figs/<page>/<name>``."""
    target = DOCS / section / "figs" / page / name
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
