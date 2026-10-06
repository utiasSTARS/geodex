"""Tests of the docs asset tools that run without the planning stack."""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from fetch import fetch  # noqa: E402


def test_fetch_checks_the_hash_and_unpacks(tmp_path: Path):
    archive = tmp_path / "robot.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("robot/model.urdf", "<robot name='r'/>")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    table = {"robot": {"url": archive.as_uri(), "sha256": digest}}

    out = fetch("robot", table, cache=tmp_path / "cache")
    assert (out / "robot" / "model.urdf").read_text() == "<robot name='r'/>"
    assert fetch("robot", table, cache=tmp_path / "cache") == out  # cached

    table["robot"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="does not match the pin"):
        fetch("robot", table, cache=tmp_path / "other")
