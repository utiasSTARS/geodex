#!/usr/bin/env python3
"""Fail if a wheel, sdist or install tree contains content that must not ship.

The rules live in third_party/excluded-content.toml.

    check_distribution.py dist/*.whl dist/*.tar.gz     # archives
    check_distribution.py --tree <prefix>             # an install tree
    check_distribution.py --installed                 # the geodex importable here
"""

import argparse
import glob
import re
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

RULES = Path(__file__).resolve().parents[2] / "third_party" / "excluded-content.toml"
BINARY = re.compile(r"\.(so(\.\d+)*|dylib|pyd|a|lib)$")


def load_rules(path):
    with open(path, "rb") as f:
        return tomllib.load(f).get("exclude", [])


def members(target):
    """Yield (path, read) for every file in a wheel, sdist or directory."""
    target = Path(target)
    if target.is_dir():
        for p in sorted(target.rglob("*")):
            if p.is_file():
                yield str(p.relative_to(target)), p.read_bytes
    elif target.suffix == ".whl" or target.suffix == ".zip":
        with zipfile.ZipFile(target) as z:
            for info in z.infolist():
                if not info.is_dir():
                    yield info.filename, (lambda n=info.filename: z.read(n))
    else:
        with tarfile.open(target) as t:
            for info in t.getmembers():
                if info.isfile():
                    yield info.name, (lambda i=info: t.extractfile(i).read())


def check_artifact(target, rules):
    problems = []
    for path, read in members(target):
        for rule in rules:
            for pattern in rule.get("paths", []):
                if re.search(pattern, path):
                    problems.append(f"{target}: {path} ({rule['what']})")
            if BINARY.search(path) and rule.get("binaries"):
                data = read()
                for needle in rule["binaries"]:
                    if needle.encode() in data:
                        problems.append(f"{target}: {path} contains '{needle}' ({rule['what']})")
    return problems


def check_installed(rules):
    import geodex

    names = set()
    for namespace, function in (("robots", "available"), ("vamp", "registered_robots")):
        try:
            names |= {n.lower() for n in getattr(getattr(geodex, namespace), function)()}
        except ImportError:
            pass
    return [
        f"installed geodex offers robot '{robot}' ({rule['what']})"
        for rule in rules
        for robot in rule.get("robots", [])
        if robot.lower() in names
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("artifacts", nargs="*", help="wheels and sdists")
    parser.add_argument("--tree", action="append", default=[], help="an install tree")
    parser.add_argument("--installed", action="store_true", help="check the importable geodex")
    parser.add_argument("--rules", default=str(RULES))
    args = parser.parse_args()

    rules = load_rules(args.rules)
    # Expand glob patterns here. PowerShell passes them unexpanded.
    artifacts = []
    for pattern in args.artifacts:
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        if not matches:
            parser.error(f"no file matches {pattern}")
        artifacts += matches
    problems = []
    for target in artifacts + args.tree:
        problems += check_artifact(target, rules)
    if args.installed:
        problems += check_installed(rules)
    for problem in problems:
        print(problem, file=sys.stderr)
    checked = len(artifacts) + len(args.tree) + int(args.installed)
    print(f"{checked} checked, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
