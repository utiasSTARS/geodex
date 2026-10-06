#!/usr/bin/env python3
"""Vendor the Lato body font of the docs into ``docs/_static/fonts``.

The site sets its body text in Lato, and the figure scripts under docs/tools set
their labels in the same files, so figures and prose match. The script downloads the four
Lato 2.015 faces from the google/fonts repository at a pinned commit, checks each against
its SHA-256, and keeps the Latin, Latin-1, Greek and common symbol ranges, which cuts each
face from about 650 KB to about 115 KB. The OFL licence text is copied next to the fonts.

Run it again only to change the pin. It needs fontTools, which comes with matplotlib.

Usage:
  python docs/tools/vendor_fonts.py
"""

import hashlib
import urllib.request
from pathlib import Path

from fontTools import subset

COMMIT = "5d3b76120a319730fda218cc7410174a462b32cb"
BASE = f"https://raw.githubusercontent.com/google/fonts/{COMMIT}/ofl/lato/"
FILES = {
    "Lato-Regular.ttf": "d636e4683231f931eda222d588e944d082bfd3bdba02f928bee461c0f185b251",
    "Lato-Italic.ttf": "e399c44efe1387100531d26c7e4800c5d12251b890d6654a3098c7c679cb1786",
    "Lato-Bold.ttf": "8a0aace75d33794eece4b28187bfc1df0bbd2888b5d8a56e01788c8d65d16be1",
    "Lato-BoldItalic.ttf": "62c1b7f0d2e74b45960154c3520efc337b553db0961bfdc950d5618334596cc8",
    "OFL.txt": "74ba064d03f1f1c4a952da936c3eb71866c34404916734de3cae73b34357e59e",
}
UNICODES = ("U+0000-00FF,U+0131,U+0152-0153,U+02C6,U+02DA,U+02DC,U+0370-03FF,U+2000-206F,"
            "U+2070-209F,U+20AC,U+2122,U+2190-2199,U+2202,U+2206,U+220F,U+2211-2212,U+221A,"
            "U+221E,U+2248,U+2260,U+2264-2265,U+22C5")
OUT = Path(__file__).resolve().parents[1] / "_static" / "fonts"


def fetch(name: str, sha256: str) -> bytes:
    data = urllib.request.urlopen(BASE + name, timeout=60).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise SystemExit(f"{name}: sha256 {digest} does not match the pin {sha256}")
    return data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, sha256 in FILES.items():
        data = fetch(name, sha256)
        target = OUT / name
        if not name.endswith(".ttf"):
            target.write_bytes(data)
            continue
        source = OUT / f".{name}.full"
        source.write_bytes(data)
        options = subset.Options()
        options.layout_features = ["kern", "liga", "lnum", "tnum", "pnum"]
        options.name_IDs = ["*"]
        options.notdef_outline = True
        font = subset.load_font(str(source), options)
        subsetter = subset.Subsetter(options)
        subsetter.populate(unicodes=subset.parse_unicodes(UNICODES))
        subsetter.subset(font)
        subset.save_font(font, str(target), options)
        source.unlink()
        print(f"{name}: {len(data) // 1024} KB -> {target.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
