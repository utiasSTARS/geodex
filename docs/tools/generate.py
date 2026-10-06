#!/usr/bin/env python3
"""Regenerate the figures, interactive plots and 3D recordings of the docs.

The assets come from seeded runs with fixed iteration budgets, most of them runs of the
documented examples. Each page is a module under ``pages/`` with a ``generate(parts=...)``
function, which by default regenerates everything the page shows. Without an argument, the
script regenerates every page. The robot models (``robot_meshes/build.py``) and the landing
hero (``landing_hero.py``) have tools of their own.

Usage:
  pixi run docs-assets                   # every page
  pixi run docs-assets planning          # one page
  pixi run docs-assets planning:sphere   # one part of a page
  python docs/tools/generate.py --list
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))



def pages() -> list[str]:
    """Every page module under pages/, in name order."""
    return sorted(p.stem for p in (HERE / "pages").glob("*.py") if p.stem != "__init__")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pages", nargs="*", help="pages to regenerate (default: all)")
    parser.add_argument("--list", action="store_true", help="list the pages and exit")
    args = parser.parse_args()
    available = pages()
    if args.list:
        print("\n".join(available))
        return
    requests = [r.split(":", 1) for r in args.pages] or [[p] for p in available]
    unknown = sorted({r[0] for r in requests} - set(available))
    if unknown:
        parser.error(f"unknown pages {unknown}; choose from {available}")
    for request in requests:
        start = time.perf_counter()
        module = importlib.import_module(f"pages.{request[0]}")
        if len(request) == 2:
            module.generate(parts=tuple(request[1].split(",")))
        else:
            module.generate()
        print(f"[docs-assets] {':'.join(request)}: {time.perf_counter() - start:.1f} s",
              flush=True)


if __name__ == "__main__":
    main()
