"""Check the math of the JavaScript widgets against geodex.

The SE(2) metric explorer on the Metrics page computes the logarithm, the exponential and
the metric norm in JavaScript. This test runs those functions with Node.js on a grid of
poses and compares them with geodex's own SE(2), so the widget cannot drift from the
library. It is skipped where Node.js is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

WIDGET = Path(__file__).resolve().parents[1] / "_static" / "se2-metric-explorer.html"


def _widget_math() -> str:
    text = WIDGET.read_text(encoding="utf-8")
    start = text.index("// [se2-math-start]")
    end = text.index("// [se2-math-end]")
    return text[start:end]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_se2_explorer_matches_geodex():
    import geodex

    rng = np.random.default_rng(0)
    poses = np.column_stack([rng.uniform(-3, 3, 200), rng.uniform(-3, 3, 200),
                             rng.uniform(-3.1, 3.1, 200)])
    weights = [1.0, 10.0, 0.7]
    script = _widget_math() + (
        "const poses = " + json.dumps(poses.tolist()) + ";\n"
        "const w = " + json.dumps(weights) + ";\n"
        "const out = poses.map(p => { const v = se2Log(p[0], p[1], p[2]);"
        " return {log: v, back: se2Exp(v, 1), norm: metricNorm(v, w)}; });\n"
        "console.log(JSON.stringify(out));\n")
    result = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True,
                                       check=True).stdout)

    se2 = geodex.SE2(wx=weights[0], wy=weights[1], wtheta=weights[2])
    origin = np.zeros(3)
    for pose, js in zip(poses, result):
        expected = se2.log(origin, pose)
        np.testing.assert_allclose(js["log"], expected, atol=1e-9)
        np.testing.assert_allclose(js["back"], pose, atol=1e-9)
        assert js["norm"] == pytest.approx(se2.norm(origin, expected), abs=1e-9)
