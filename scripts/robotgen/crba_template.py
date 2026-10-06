#!/usr/bin/env python3
"""Turn a shipped CRBA source into a C++ template over the scalar type.

    python scripts/robotgen/crba_template.py src/robots/generated/<robot>_crba.cpp OUT.hpp <robot>

It copies the body of `<robot>_crba_forward_zero`, keeps the non-Apple trigonometry branch
and replaces `double` with a template parameter. The result specializes
`geodex::robots::certify::CrbaTemplate` for the robot, and the bound tool evaluates it in
interval arithmetic.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def enum_name(robot: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in robot.split("_"))


def extract_body(source: str, robot: str) -> str:
    """Return the statements of `<robot>_crba_forward_zero` between its braces."""
    head = re.search(rf"void {re.escape(robot)}_crba_forward_zero\([^)]*\)\s*\{{\n", source)
    if head is None:
        raise SystemExit(f"{robot}: no {robot}_crba_forward_zero in the source")
    end = source.find("\n}\n", head.end())
    if end < 0:
        raise SystemExit(f"{robot}: unterminated {robot}_crba_forward_zero")
    return source[head.end():end + 1]


def periodic_coordinates(body: str, nq: int) -> list[bool]:
    """Return the coordinates that the expression reads only through sine and cosine.

    The post-processed source reads such a coordinate only in the trigonometry prelude
    (`_trig_in[k] = x[j];`).
    """
    uses = {j: [] for j in range(nq)}
    for line in body.splitlines():
        code = line.split("//", 1)[0]
        for j in re.findall(r"\bx\[(\d+)\]", code):
            uses[int(j)].append(code.strip())
    return [bool(uses[j]) and all(re.fullmatch(r"_trig_in\[\d+\] = x\[%d\];" % j, u)
                                  for u in uses[j]) for j in range(nq)]


def templatize(body: str) -> str:
    """Replace the scalar type and keep the non-Apple trigonometry branch."""
    lines = []
    skipping = False
    for line in body.splitlines():
        stripped = line.strip()
        if stripped == "#if defined(__APPLE__)":
            skipping = True
            continue
        if stripped == "#else" and skipping:
            skipping = False
            continue
        if stripped == "#endif":
            continue
        if skipping:
            continue
        if stripped in ("const double* x = in[0];", "double* y = out[0];"):
            continue
        line = line.replace("alignas(32) double", "T").replace("double v[", "T v[")
        if re.search(r"\bdouble\b", line):
            raise SystemExit(f"unexpected double in: {line}")
        lines.append(line)
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("robot")
    args = parser.parse_args()

    source = args.source.read_text()
    raw = extract_body(source, args.robot)
    nq = re.search(rf"void {re.escape(args.robot)}_crba\(const double q\[(\d+)\]", source)
    if nq is None:
        raise SystemExit(f"{args.robot}: no {args.robot}_crba wrapper in the source")
    periodic = periodic_coordinates(raw, int(nq.group(1)))
    body = templatize(raw)
    text = (
        f"// Generated at build time by scripts/robotgen/crba_template.py from\n"
        f"// {args.source.name}. DO NOT EDIT.\n\n"
        "#pragma once\n\n"
        "#include <cmath>\n\n"
        '#include "crba_certify.hpp"\n\n'
        "namespace geodex::robots::certify {\n\n"
        "template <>\n"
        f"struct CrbaTemplate<Robot::{enum_name(args.robot)}> {{\n"
        "  /// Coordinates the expression reads only through sine and cosine.\n"
        f"  static constexpr bool periodic[{len(periodic)}] = {{"
        + ", ".join("true" if v else "false" for v in periodic) + "};\n"
        "  template <class T>\n"
        "  static void eval(const T* x, T* y) {\n"
        "    using std::cos;\n"
        "    using std::sin;\n"
        f"{body}"
        "  }\n"
        "};\n\n"
        "}  // namespace geodex::robots::certify\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_text() != text:
        args.output.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
