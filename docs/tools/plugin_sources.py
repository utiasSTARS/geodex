#!/usr/bin/env python3
"""Keep the ROS 2 pages in step with the plugin repositories.

The ROS 2 pages show configuration files of geodex_nav2_planner and geodex_moveit and tables
of their parameters. Both come from verbatim copies of the plugins' own files under
docs/ros2/sources/, listed in docs/ros2/sources.toml, and the tables are generated from the
copies into docs/ros2/generated/. Nothing on the pages is written by hand.

  python docs/tools/plugin_sources.py sync    # copy from the checkouts, record hashes,
                                                # regenerate the tables
  python docs/tools/plugin_sources.py check   # fail on any drift

check fails when a copy no longer matches its recorded SHA-256, when a generated table
differs from what the copies produce, or, for every plugin checkout it finds, when a copy
differs from the plugin's file. A checkout that is absent is reported and skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import sys
import tomllib
from pathlib import Path

import yaml

DOCS = Path(__file__).resolve().parents[1]
ROOT = DOCS.parent
ROS2 = DOCS / "ros2"
MANIFEST = ROS2 / "sources.toml"
SOURCES = ROS2 / "sources"
GENERATED = ROS2 / "generated"


def manifest() -> dict:
    with open(MANIFEST, "rb") as f:
        return tomllib.load(f)


def checkout(name: str, repo: dict) -> Path:
    """The plugin checkout, from its environment variable or the sibling directory."""
    return Path(os.environ.get(repo["env"], ROOT.parent / name)).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_hashes(hashes: dict[str, str]) -> None:
    text = MANIFEST.read_text(encoding="utf-8")
    head = text[:text.index("[sha256]")]
    body = "".join(f'"{key}" = "{value}"\n' for key, value in sorted(hashes.items()))
    MANIFEST.write_text(head + "[sha256]\n" + body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _groups(text: str) -> dict[str, str]:
    """Top-level parameter name to the heading of the comment block above it, the comment's
    first sentence, for parameter files that group their entries with comments."""
    groups, heading = {}, None
    for line in text.splitlines():
        comment = re.match(r"^  # (.+)$", line)
        key = re.match(r"^  ([a-z_][a-z0-9_]*):\s*$", line)
        if comment:
            heading = comment.group(1).split(". ")[0].rstrip(".")
        elif key and heading:
            groups[key.group(1)] = heading
    return groups


def _leaves(node: dict, prefix: str = ""):
    """Parameters of a generate_parameter_library table in file order, with dotted names."""
    for key, value in node.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and "type" in value:
            yield name, value
        elif isinstance(value, dict):
            yield from _leaves(value, name + ".")


def _number(value) -> str:
    """A number as the tables show it, with the multiples of pi by name."""
    if isinstance(value, float):
        for multiple, name in ((2.0, "2π"), (1.0, "π"), (0.5, "π/2"), (0.25, "π/4")):
            if abs(abs(value) - multiple * math.pi) < 1e-9:
                return ("-" if value < 0 else "") + name
        text = repr(value)
        return text if len(text) <= 10 else f"{value:.6g}"
    return str(value)


def _value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value == "" or value == []:
        return "empty"
    if isinstance(value, list):
        return ", ".join(_value(v) for v in value)
    return _number(value)


def _validation(spec: dict) -> str:
    parts = []
    for rule, args in (spec or {}).items():
        if rule == "bounds<>":
            parts.append(f"{_value(args[0])} to {_value(args[1])}")
        elif rule == "gt_eq<>":
            parts.append(f"≥ {_value(args[0])}")
        elif rule == "gt<>":
            parts.append(f"> {_value(args[0])}")
        elif rule == "lt_eq<>":
            parts.append(f"≤ {_value(args[0])}")
        elif rule == "one_of<>":
            parts.append(", ".join(f"``{v}``" for v in args[0]))
        elif rule == "size_lt<>":
            parts.append(f"fewer than {args[0]} entries")
        else:
            parts.append(f"``{rule}`` {args}")
    return "; ".join(parts)


def _cell(text: str) -> str:
    """Text for one list-table cell, on one line."""
    return " ".join(str(text).split()) or " "


def parameter_table(path: Path) -> str:
    """A list-table per group of a generate_parameter_library file."""
    text = path.read_text(encoding="utf-8")
    (params,) = yaml.safe_load(text).values()
    groups = _groups(text)
    order: list[str] = []
    rows: dict[str, list[str]] = {}
    for name, spec in _leaves(params):
        top = name.split(".")[0]
        group = groups.get(top) or (top.replace("_", " ").capitalize() if "." in name
                                    else "General")
        if group not in rows:
            order.append(group)
            rows[group] = []
        rows[group] += [
            f"   * - ``{name}``",
            f"     - {spec['type'].replace('_', ' ')}",
            f"     - {_cell(_value(spec.get('default_value', '')))}",
            f"     - {_cell(_validation(spec.get('validation')))}",
            f"     - {_cell(spec.get('description', ''))}",
        ]
    out = [f".. Generated by docs/tools/plugin_sources.py from {path.relative_to(SOURCES)}.",
           ".. Do not edit; run the script's sync mode instead.", ""]
    for group in order:
        out += [f".. list-table:: {group}", "   :header-rows: 1", "   :widths: 31 9 11 16 33",
                "   :class: geodex-param-table docutils", "",
                "   * - Parameter", "     - Type", "     - Default", "     - Range",
                "     - Description", *rows[group], ""]
    return "\n".join(out)


def robot_table(paths: list[Path]) -> str:
    """The metric weights, direction stage and footprint of each Clearpath base file."""
    out = [".. Generated by docs/tools/plugin_sources.py from config/robots/*.yaml.",
           ".. Do not edit; run the script's sync mode instead.", "",
           ".. list-table::", "   :header-rows: 1", "   :widths: 16 30 8 8 10 12 16", "",
           "   * - File", "     - Base", "     - ``wx``", "     - ``wy``", "     - ``wtheta``",
           "     - Direction stage", "     - Footprint (m)"]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        first = text.splitlines()[0].lstrip("# ").rstrip(".")
        base = re.sub(r"^an? ", "", first.split(", ", 1)[1] if ", " in first else first)
        base = base[:1].upper() + base[1:]
        config = yaml.safe_load(text)
        planner = config["planner_server"]["ros__parameters"]["GridBased"]
        footprint = yaml.safe_load(
            config["global_costmap"]["global_costmap"]["ros__parameters"]["footprint"])
        xs, ys = [p[0] for p in footprint], [p[1] for p in footprint]
        size = f"{max(xs) - min(xs):.3f} x {max(ys) - min(ys):.3f}"
        stage = "on" if planner.get("max_reverse_run", -1.0) >= 0.0 else "off"
        out += [f"   * - ``{path.name}``", f"     - {_cell(base)}",
                f"     - {planner['wx']}", f"     - {planner['wy']}",
                f"     - {planner['wtheta']}", f"     - {stage}", f"     - {size}"]
    return "\n".join(out) + "\n"


def tables() -> dict[Path, str]:
    """Every generated file and its content, from the copies."""
    nav2 = SOURCES / "geodex_nav2_planner"
    moveit = SOURCES / "geodex_moveit"
    robots = [nav2 / "config" / "robots" / f"{n}.yaml"
              for n in ("jackal", "husky", "ridgeback", "dingo_o")]
    return {
        GENERATED / "nav2_parameters.inc": parameter_table(
            nav2 / "params" / "geodex_nav2_planner_parameters.yaml"),
        GENERATED / "nav2_robots.inc": robot_table(robots),
        GENERATED / "moveit_parameters.inc": parameter_table(
            moveit / "geodex_moveit" / "params" / "geodex_moveit_parameters.yaml"),
    }


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def sync() -> int:
    data = manifest()
    hashes = {}
    for name, repo in data["repositories"].items():
        source = checkout(name, repo)
        if not source.is_dir():
            print(f"error: no checkout of {name} at {source}; set {repo['env']}",
                  file=sys.stderr)
            return 1
        for rel in repo["files"]:
            target = SOURCES / name / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source / rel).read_bytes())
            hashes[f"{name}/{rel}"] = sha256(target)
    write_hashes(hashes)
    GENERATED.mkdir(parents=True, exist_ok=True)
    for path, content in tables().items():
        path.write_text(content, encoding="utf-8")
    print(f"copied {len(hashes)} files and wrote {len(tables())} tables")
    return 0


def check() -> list[str]:
    """Every drift found, as readable lines; empty when the pages are in step."""
    data = manifest()
    problems = []
    listed = set()
    for name, repo in data["repositories"].items():
        source = checkout(name, repo)
        present = source.is_dir()
        if not present:
            print(f"note: no checkout of {name} at {source}, comparing copies with their "
                  "recorded hashes only")
        for rel in repo["files"]:
            key = f"{name}/{rel}"
            listed.add(key)
            copy = SOURCES / name / rel
            if not copy.exists():
                problems.append(f"{key}: copy missing, run sync")
                continue
            if data["sha256"].get(key) != sha256(copy):
                problems.append(f"{key}: copy differs from its recorded SHA-256")
            if present and (source / rel).read_bytes() != copy.read_bytes():
                problems.append(f"{key}: the plugin's file changed, run sync")
    stale = set(data["sha256"]) - listed
    problems += [f"{key}: recorded but no longer listed" for key in sorted(stale)]
    for path, content in tables().items():
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            problems.append(f"{path.relative_to(ROOT)}: out of date, run sync")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("sync", "check"))
    args = parser.parse_args()
    if args.mode == "sync":
        return sync()
    problems = check()
    for line in problems:
        print(line, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
