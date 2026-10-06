"""Run a documentation example in Python and in C++ and compare the two.

Every snippet on a docs page is a marked region of a file under ``examples/``. A test module
lists its examples as :class:`Example` values and calls :func:`check_example` for each. The
Python file runs the way a reader would run it, the C++ executable built from its twin
(under ``$GEODEX_BUILD_DIR``, default ``build``) runs next, and both write their results as
JSON with ``--json``. The two must agree value by value, with the same booleans and strings
and numbers within ``RTOL`` and ``ATOL``. Under one seed and one iteration budget both
languages run the same planner, so the paths agree to rounding. Keys ending in ``_ms`` hold
wall-clock times and are not compared.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(os.environ.get("GEODEX_BUILD_DIR", ROOT / "build"))
RTOL = 1e-9
ATOL = 1e-8


@dataclass(frozen=True)
class Example:
    """A docs example under examples/, by path without suffix."""

    stem: str
    args: tuple[str, ...] = ()
    cpp: bool = True
    timeout: int = 600
    ignore: tuple[str, ...] = field(default=())
    # A known disagreement and its reason. With `xfail_keys`, only those top-level keys may
    # differ and every other key is still compared; without, the whole result may differ.
    # The mark is strict: the test fails once the two languages agree, as a reminder to
    # remove it.
    xfail: str = ""
    xfail_keys: tuple[str, ...] = ()

    @property
    def python(self) -> Path:
        return ROOT / "examples" / f"{self.stem}.py"

    @property
    def binary(self) -> Path:
        return BUILD / "examples" / self.stem

    def __str__(self) -> str:
        return self.stem + ("" if not self.args else ":" + ",".join(self.args))


def _run(command: list[str], out: Path, timeout: int, cwd: Path) -> dict:
    proc = subprocess.run(command + ["--json", str(out)], cwd=cwd, capture_output=True,
                          text=True, timeout=timeout)
    assert proc.returncode == 0, (
        f"{' '.join(command)} exited {proc.returncode}\n{proc.stdout[-4000:]}\n"
        f"{proc.stderr[-4000:]}")
    return json.loads(out.read_text())


def _compare(a, b, where: str, ignore: tuple[str, ...]) -> list[str]:
    """Differences between two JSON values, as readable lines."""
    if isinstance(a, dict) and isinstance(b, dict):
        diffs = []
        if set(a) != set(b):
            diffs.append(f"{where}: keys {sorted(set(a) ^ set(b))} differ")
        for key in sorted(set(a) & set(b)):
            if key.endswith("_ms") or key in ignore:
                continue
            diffs += _compare(a[key], b[key], f"{where}.{key}", ignore)
        return diffs
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{where}: lengths {len(a)} and {len(b)}"]
        diffs = []
        for i, (x, y) in enumerate(zip(a, b)):
            diffs += _compare(x, y, f"{where}[{i}]", ignore)
            if len(diffs) > 5:
                break
        return diffs
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or a is None:
        return [] if a == b else [f"{where}: {a!r} != {b!r}"]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL):
            return []
        return [f"{where}: {a!r} != {b!r} (difference {abs(a - b):.3g})"]
    return [f"{where}: {a!r} != {b!r}"]


def check_example(example: Example, tmp_path: Path):
    """Run `example` in Python and, when it has a C++ twin, in C++, and compare."""
    py = _run([sys.executable, str(example.python), *example.args], tmp_path / "py.json",
              example.timeout, example.python.parent)
    if not example.cpp:
        return
    if not example.binary.exists():
        pytest.fail(f"{example.binary} is missing; build the C++ examples first "
                    "(pixi run build-cpp)")
    cpp = _run([str(example.binary), *example.args], tmp_path / "cpp.json", example.timeout,
               example.python.parent)
    if example.xfail and example.xfail_keys:
        known = {k: (py.pop(k, None), cpp.pop(k, None)) for k in example.xfail_keys}
        diffs = _compare(py, cpp, "result", example.ignore)
        assert not diffs, "Python and C++ disagree:\n" + "\n".join(diffs)
        if any(_compare(a, b, k, example.ignore) for k, (a, b) in known.items()):
            pytest.xfail(example.xfail)
        pytest.fail(f"{example} now agrees in Python and C++; remove its xfail mark")
    diffs = _compare(py, cpp, "result", example.ignore)
    if example.xfail and diffs:
        pytest.xfail(example.xfail)
    if example.xfail:
        pytest.fail(f"{example} now agrees in Python and C++; remove its xfail mark")
    assert not diffs, "Python and C++ disagree:\n" + "\n".join(diffs)
