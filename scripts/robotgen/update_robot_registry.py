#!/usr/bin/env python3
"""Add a robot to the built-in robot manifest and the C++ registries.

It rewrites the robot lists in cmake/robots_manifest.cmake and the marked blocks of
mass_matrix.hpp, mass_matrix.cpp, mass_lower_bound.hpp and precompute_robot_bound.cpp.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


NAMES_RE = re.compile(r"set\(GEODEX_ROBOT_NAMES\s*(.*?)\)", re.DOTALL)
URDFS_RE = re.compile(r"set\(GEODEX_ROBOT_URDFS\s*(.*?)\)", re.DOTALL)
PUBLIC_RE = re.compile(r"set\(GEODEX_ROBOT_PUBLIC_NAMES\s*(.*?)\)", re.DOTALL)
BASES_RE = re.compile(r"set\(GEODEX_ROBOT_BASES\s*(.*?)\)", re.DOTALL)
DRIVES = {"holonomic": "Holonomic", "differential": "Differential"}


def parse_names(body: str) -> list[str]:
    return [
        token.strip()
        for token in body.splitlines()
        if token.strip() and not token.strip().startswith("#")
    ]


def parse_urdfs(body: str) -> list[str]:
    return re.findall(r'"([^"]+)"', body)


def enum_name(robot: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in robot.split("_"))


def cmake_urdf_path(repo_root: Path, urdf_path: Path) -> str:
    try:
        rel = urdf_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return urdf_path.resolve().as_posix()
    return "${CMAKE_CURRENT_LIST_DIR}/../" + rel.as_posix()


def replace_between(text: str, begin: str, end: str, replacement: str) -> str:
    pattern = re.compile(
        rf"({re.escape(begin)}\n)(.*?)(\n{re.escape(end)})",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise SystemExit(f"could not find marker block {begin} ... {end}")
    return pattern.sub(rf"\1{replacement}\3", text, count=1)


def parse_pairs(text: str, regex: re.Pattern) -> dict[str, str]:
    match = regex.search(text)
    if not match:
        return {}
    pairs = {}
    for token in parse_names(match.group(1)):
        key, _, value = token.partition("=")
        pairs[key] = value
    return pairs


def load_manifest(path: Path) -> tuple[list[str], list[str], dict[str, str]]:
    text = path.read_text()
    names_match = NAMES_RE.search(text)
    urdfs_match = URDFS_RE.search(text)
    if not names_match or not urdfs_match:
        raise SystemExit(f"could not parse {path}")
    names = parse_names(names_match.group(1))
    urdfs = parse_urdfs(urdfs_match.group(1))
    if len(names) != len(urdfs):
        raise SystemExit(
            f"{path} has {len(names)} robot names but {len(urdfs)} URDF paths"
        )
    return names, urdfs, parse_pairs(text, PUBLIC_RE)


def write_manifest(path: Path, names: list[str], urdfs: list[str]) -> None:
    """Rewrite the two robot lists in place and keep every other line."""
    names_block = "\n".join(f"  {name}" for name in names)
    urdfs_block = "\n".join(f'  "{urdf}"' for urdf in urdfs)
    text = path.read_text()
    text = NAMES_RE.sub(lambda _: f"set(GEODEX_ROBOT_NAMES\n{names_block}\n)", text, count=1)
    text = URDFS_RE.sub(lambda _: f"set(GEODEX_ROBOT_URDFS\n{urdfs_block}\n)", text, count=1)
    path.write_text(text)


def write_header(path: Path, names: list[str], public: dict[str, str],
                 bases: dict[str, str]) -> None:
    text = path.read_text()
    includes = "\n".join(f'#include "generated/{name}_crba.hpp"' for name in names)
    enum_values = "\n".join(f"  {enum_name(name)}," for name in names)
    visit_cases = "\n".join(
        f"    case Robot::{enum_name(name)}: "
        f"return f.template operator()<Robot::{enum_name(name)}>();"
        for name in names
    )
    traits = "\n\n".join(
        f"""template <>
struct RobotTraits<Robot::{enum_name(name)}> {{
  static constexpr std::string_view name = "{public.get(name, name)}";
  static constexpr BaseDrive drive = BaseDrive::{DRIVES.get(bases.get(name, ""), "None")};
  static constexpr int Nq = generated::{name}_nq;
  static constexpr int Nv = generated::{name}_nv;
  static constexpr int UpperCount = generated::{name}_upper_count;
  static constexpr const double* lower_limit = generated::{name}_lower_limit;
  static constexpr const double* upper_limit = generated::{name}_upper_limit;
  static void crba(const double* q, double* M_upper) {{ ::{name}_crba(q, M_upper); }}
}};"""
        for name in names
    )

    text = replace_between(
        text,
        "// GEODEX_ROBOT_INCLUDES_BEGIN",
        "// GEODEX_ROBOT_INCLUDES_END",
        includes,
    )
    text = replace_between(
        text,
        "// GEODEX_ROBOT_ENUM_BEGIN",
        "// GEODEX_ROBOT_ENUM_END",
        enum_values,
    )
    text = replace_between(
        text,
        "// GEODEX_ROBOT_TRAITS_BEGIN",
        "// GEODEX_ROBOT_TRAITS_END",
        traits,
    )
    text = replace_between(
        text,
        "// GEODEX_ROBOT_VISIT_BEGIN",
        "// GEODEX_ROBOT_VISIT_END",
        visit_cases,
    )
    path.write_text(text)


def write_cpp(path: Path, names: list[str]) -> None:
    text = path.read_text()
    registered = "\n".join(f"    Robot::{enum_name(name)}," for name in names)
    text = replace_between(
        text,
        "    // GEODEX_ROBOT_REGISTERED_BEGIN",
        "    // GEODEX_ROBOT_REGISTERED_END",
        registered,
    )
    path.write_text(text)


def write_lower_bound_header(path: Path, names: list[str]) -> None:
    text = path.read_text()
    includes = "\n".join(f'#include "generated/{name}_bound.hpp"' for name in names)
    traits = "\n\n".join(
        f"""template <>
struct RobotBoundTraits<Robot::{enum_name(name)}> {{
  static constexpr const double* data = generated::{name}_mass_lower_bound;
  static constexpr int count = generated::{name}_lower_bound_count;
  static constexpr double certificate = generated::{name}_mass_lower_bound_certificate;
  static constexpr bool converged = generated::{name}_mass_lower_bound_converged;
  static constexpr bool proved = generated::{name}_mass_lower_bound_proved;
}};"""
        for name in names
    )
    text = replace_between(
        text,
        "// GEODEX_ROBOT_BOUND_INCLUDES_BEGIN",
        "// GEODEX_ROBOT_BOUND_INCLUDES_END",
        includes,
    )
    text = replace_between(
        text,
        "// GEODEX_ROBOT_BOUND_TRAITS_BEGIN",
        "// GEODEX_ROBOT_BOUND_TRAITS_END",
        traits,
    )
    path.write_text(text)


def write_bound_dispatch(path: Path, names: list[str]) -> None:
    text = path.read_text()
    dispatch = "\n".join(
        f'  if (name == "{name}") return bake<Robot::{enum_name(name)}>(name, out_dir);'
        for name in names
    )
    text = replace_between(
        text,
        "  // GEODEX_ROBOT_BOUND_DISPATCH_BEGIN",
        "  // GEODEX_ROBOT_BOUND_DISPATCH_END",
        dispatch,
    )
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--name", required=True, help="lowercase robot id, e.g. ur5")
    parser.add_argument("--urdf", required=True, type=Path, help="canonical URDF path")
    args = parser.parse_args()

    name = args.name.strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise SystemExit(
            f"invalid robot name {name!r}; use lowercase [a-z][a-z0-9_]*"
        )

    repo_root = args.repo_root.resolve()
    manifest = repo_root / "cmake" / "robots_manifest.cmake"
    header = repo_root / "include" / "geodex" / "robots" / "mass_matrix.hpp"
    cpp = repo_root / "src" / "robots" / "mass_matrix.cpp"
    lower_bound_header = repo_root / "include" / "geodex" / "robots" / "mass_lower_bound.hpp"
    bound_tool = repo_root / "scripts" / "robotgen" / "precompute_robot_bound.cpp"

    names, urdfs, public = load_manifest(manifest)
    bases = parse_pairs(manifest.read_text(), BASES_RE)
    urdf_entry = cmake_urdf_path(repo_root, args.urdf)
    if name in names:
        urdfs[names.index(name)] = urdf_entry
    else:
        names.append(name)
        urdfs.append(urdf_entry)
    # The registry lists the robots in alphabetical order.
    names, urdfs = map(list, zip(*sorted(zip(names, urdfs))))

    write_manifest(manifest, names, urdfs)
    write_header(header, names, public, bases)
    write_cpp(cpp, names)
    # geodex_robots and the bound tool do not include generated/<robot>_bound.hpp. A robot can
    # appear in these two files before its bound header exists.
    write_lower_bound_header(lower_bound_header, names)
    write_bound_dispatch(bound_tool, names)


if __name__ == "__main__":
    main()
