#!/usr/bin/env python3
"""Vendor MathJax core and a named font package into ``docs/_static/mathjax/``.

The script downloads ``tex-mml-chtml.js`` of MathJax ``MATHJAX_VERSION`` and the CHTML
files of the font package ``@mathjax/mathjax-<name>-font`` at the same version from
jsdelivr.

Usage::

    python3 docs/tools/vendor_mathjax.py             # refresh current (termes)
    python3 docs/tools/vendor_mathjax.py newcm       # swap to a different font
    python3 docs/tools/vendor_mathjax.py --clean ... # remove other fonts first

After swapping, also update ``docs/conf.py`` so the three occurrences of the
font name match the one you just vendored.

Supported font names: termes, newcm, modern, stix2, tex, pagella, fira,
schola, bonum, asana (anything published as ``@mathjax/mathjax-<name>-font``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MATHJAX_DIR = REPO_ROOT / "docs" / "_static" / "mathjax"
MATHJAX_VERSION = "4.1.1"


def fetch(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r:
        out_path.write_bytes(r.read())


def vendor_core() -> None:
    version = MATHJAX_VERSION
    url = f"https://cdn.jsdelivr.net/npm/mathjax@{version}/tex-mml-chtml.js"
    out = MATHJAX_DIR / "tex-mml-chtml.js"
    print(f"core: mathjax@{version}")
    fetch(url, out)
    print(f"  -> {out.relative_to(REPO_ROOT)} ({out.stat().st_size:,} bytes)")


def vendor_font(font_name: str) -> None:
    package = f"@mathjax/mathjax-{font_name}-font"
    resolved_version = MATHJAX_VERSION
    print(f"font: {package}@{resolved_version}")
    listing_url = f"https://data.jsdelivr.com/v1/packages/npm/{package}@{resolved_version}"
    with urllib.request.urlopen(listing_url) as r:
        data = json.load(r)

    base = f"https://cdn.jsdelivr.net/npm/{package}@{resolved_version}"
    dest_root = MATHJAX_DIR / "fonts" / font_name

    total = 0
    count = 0

    def walk(files, rel_prefix: str, dest_prefix: Path) -> None:
        nonlocal total, count
        for f in files:
            p = rel_prefix + f["name"]
            dest = dest_prefix / f["name"]
            if f["type"] == "file":
                # Keep the CHTML runtime only, the top-level chtml.js entry point and
                # the on-demand chtml/ chunks and woff2 files. The build does not load
                # svg.js, cjs/, mjs/, tex-mml-*-font.js or the SVG tree.
                keep = p == "chtml.js" or p.startswith("chtml/")
                if keep:
                    fetch(f"{base}/{p}", dest)
                    total += dest.stat().st_size
                    count += 1
            else:
                walk(f.get("files", []), p + "/", dest)

    walk(data["files"], "", dest_root)
    print(f"  -> {count} files, {total:,} bytes under "
          f"{dest_root.relative_to(REPO_ROOT)}/")


def clean_other_fonts(keep: str) -> None:
    fonts_dir = MATHJAX_DIR / "fonts"
    if not fonts_dir.exists():
        return
    for child in fonts_dir.iterdir():
        if child.is_dir() and child.name != keep:
            print(f"clean: removing {child.relative_to(REPO_ROOT)}/")
            shutil.rmtree(child)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("font", nargs="?", default="termes",
                    help="font package name (default: termes)")
    ap.add_argument("--clean", action="store_true",
                    help="remove other vendored fonts to keep the repo lean")
    args = ap.parse_args()

    print(f"Vendoring MathJax core + {args.font} font")
    print(f"Target:    {MATHJAX_DIR.relative_to(REPO_ROOT)}/")
    vendor_core()
    vendor_font(args.font)
    if args.clean:
        clean_other_fonts(keep=args.font)
    print("Done. Update docs/conf.py so the font name matches if you swapped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
