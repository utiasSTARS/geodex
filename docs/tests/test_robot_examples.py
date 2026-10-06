"""The examples behind the robot guides (docs/robots).

Run with ``pixi run test-docs``. See ``pairs.py`` for what a pair test checks. Every
example plans under a fixed seed and iteration budget, so Python and C++ return the same
path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pairs import Example, check_example

EXAMPLES = [
    Example("robots/catalog"),
    Example("robots/manipulation/arm_ke"),
    Example("robots/navigation/bases"),
    Example("robots/mobile_manipulation/stretch"),
    Example("robots/mobile_manipulation/clearpath"),
]


@pytest.mark.parametrize("example", EXAMPLES, ids=str)
def test_python_matches_cpp(example: Example, tmp_path: Path):
    check_example(example, tmp_path)
