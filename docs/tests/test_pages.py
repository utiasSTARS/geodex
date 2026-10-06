"""Checks on the page sources that the Sphinx build does not make.

Every Python or C++ snippet on a page must come from a tested example through
``code-pair``, and a page holds no inline Python or C++ block. Pages hold no em-dash.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1]
PAGES = sorted(DOCS.rglob("*.rst"))
INLINE = re.compile(r"^\s*\.\. (code-block|code|sourcecode)::\s*(python|py|cpp|c\+\+)\s*$",
                    re.MULTILINE)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(DOCS)))
def test_no_untested_code(page: Path):
    found = [m.group(0).strip() for m in INLINE.finditer(page.read_text(encoding="utf-8"))]
    assert not found, f"inline code on {page.name}; use code-pair with a tested example"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(DOCS)))
def test_no_em_dash(page: Path):
    lines = [i + 1 for i, line in enumerate(page.read_text(encoding="utf-8").splitlines())
             if "—" in line]
    assert not lines, f"em-dash on lines {lines} of {page.name}"
