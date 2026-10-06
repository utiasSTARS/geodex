"""The API pages cover every public name of the Python module and the C++ headers.

The Python check compares the names the installed module exports with the autodoc entries of
docs/api/python.rst, both ways. The C++ checks read the Doxygen XML the pixi ``docs`` task
writes: every header that declares a public entity must be included by docs/api/cpp.rst,
and, once the site is built, every public entity must have an anchor on the rendered page.
The C++ checks skip until ``pixi run docs`` has run, and fail instead when
``GEODEX_DOCS_REQUIRE_SITE=1`` is set, as the docs workflow does after building the site.
"""

from __future__ import annotations

import importlib.util
import os
import re
import types
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1]
ROOT = DOCS.parent
BUILD = Path(os.environ.get("GEODEX_BUILD_DIR", ROOT / "build"))
XML = BUILD / "docs" / "doxygen" / "xml"
HTML = BUILD / "docs" / "sphinx" / "api" / "cpp.html"
REQUIRE_SITE = os.environ.get("GEODEX_DOCS_REQUIRE_SITE") == "1"


def _not_built(reason: str):
    """Skip a check whose input the docs build writes, or fail when the site is required."""
    if REQUIRE_SITE:
        pytest.fail(reason)
    pytest.skip(reason)


def _api_cpp():
    spec = importlib.util.spec_from_file_location("api_cpp", DOCS / "tools" / "api_cpp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def python_surface() -> set[str]:
    """Public names of the extension module and of its submodules, as ``geodex.<name>``."""
    geodex = pytest.importorskip("geodex")
    core = geodex._geodex_core
    names = set()
    for name in dir(core):
        if name.startswith("_"):
            continue
        value = getattr(core, name)
        if isinstance(value, types.ModuleType):
            if value.__name__.startswith(core.__name__ + "."):
                names |= {f"geodex.{name}.{n}" for n in dir(value) if not n.startswith("_")}
        else:
            names.add(f"geodex.{name}")
    return names


def test_python_page_covers_the_module():
    page = (DOCS / "api" / "python.rst").read_text(encoding="utf-8")
    documented = set(re.findall(r"^\.\. auto(?:class|function|data|exception):: (\S+)", page,
                                re.MULTILINE))
    exported = python_surface()
    missing = sorted(exported - documented)
    assert not missing, f"missing from api/python.rst: {missing}"
    assert not documented - exported, f"not in the module: {sorted(documented - exported)}"


@pytest.fixture(scope="module")
def entities():
    if not (XML / "index.xml").exists():
        _not_built(f"no Doxygen XML in {XML}, run pixi run docs first")
    return _api_cpp().collect(XML)


def test_cpp_page_includes_every_public_header(entities):
    page = (DOCS / "api" / "cpp.rst").read_text(encoding="utf-8")
    included = set(re.findall(r"^\.\. include:: \S*/api/cpp/(\S+)\.inc\s*$", page, re.MULTILINE))
    declared = set(entities)
    assert not declared - included, f"missing from api/cpp.rst: {sorted(declared - included)}"
    assert not included - declared, f"no public entities: {sorted(included - declared)}"


def _encoded(name: str) -> str:
    """The nested-name part of a Sphinx C++ anchor, N6geodex10heuristics11FactorBoundE."""
    return "N" + "".join(f"{len(part)}{part}" for part in name.split("::")) + "E"


def test_cpp_page_renders_every_public_entity(entities):
    if not HTML.exists():
        _not_built(f"{HTML} is missing, run pixi run docs first")
    html = HTML.read_text(encoding="utf-8")
    ids = " ".join(re.findall(r'id="([^"]+)"', html))
    missing = []
    for items in entities.values():
        for _, directive in items:
            kind, target = re.match(r"\.\. (\w+):: ([^\s(]+)", directive).groups()
            if kind == "doxygendefine":
                found = f"c.{target}" in ids
            elif "<" in target:  # a specialization, whose anchor continues with its arguments
                found = _encoded(target.split("<")[0])[:-1] + "I" in ids
            else:
                found = _encoded(target) in ids
            if not found:
                missing.append(target)
    assert not missing, f"public entities without an anchor on api/cpp.html: {missing}"
