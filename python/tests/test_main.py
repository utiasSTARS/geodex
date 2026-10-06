"""Tests for ``python -m geodex``."""

import subprocess
import sys

import geodex
from geodex import _geodex_core


def test_prints_the_version_and_the_components():
    out = subprocess.run([sys.executable, "-m", "geodex"], check=True, capture_output=True,
                         text=True).stdout.splitlines()
    assert out[0] == f"geodex {geodex.__version__}"
    expected = {
        "planning": hasattr(_geodex_core, "plan"),
        "built-in robots": hasattr(_geodex_core, "robots"),
        "collision checking": hasattr(_geodex_core, "Scene"),
    }
    assert out[1:] == [f"{label}: {'yes' if has else 'no'}" for label, has in expected.items()]
