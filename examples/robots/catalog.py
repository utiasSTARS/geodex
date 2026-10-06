#!/usr/bin/env python3
"""Example of the Robot Guides page, the Python version of catalog.cpp.

Lists the built-in robots and the number of configuration coordinates of each. The
count of a mobile robot includes its base pose.

Usage:
  python examples/robots/catalog.py [--json out.json]
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    # [docs-start:catalog]
    import geodex

    for name in geodex.robots.available():
        print(f"{name:>16}: {geodex.vamp.robot_dimension(name)} coordinates")
    # [docs-end:catalog]

    if args.json:
        args.json.write_text(json.dumps({"robots": [
            {"name": n, "dim": geodex.vamp.robot_dimension(n)}
            for n in geodex.robots.available()]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
