#!/usr/bin/env python3
"""Replace mesh geometry in a URDF with the meshes' bounding boxes.

For every <visual> and <collision> whose mesh file name appears in --meshes, the mesh
becomes the axis-aligned bounding box of its vertices in the geometry frame, rounded
outward to 10 micrometers, and the element's origin moves to the box center. With
--header, one comment with that text replaces the comments that follow the XML
declaration. Everything else in the file stays byte for byte, including kinematics and
inertials.

    python scripts/robotgen/mesh_boxes.py in.urdf out.urdf --meshes DIR [DIR ...] [--header TEXT]

The script finds each mesh file by its base name in the given directories. An element with
a rotated origin is an error.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import trimesh

LEADING_COMMENTS = re.compile(r"\A(<\?xml[^>]*\?>\n)((?:[ \t]*<!--.*?-->[ \t]*\n)+)",
                              re.DOTALL)
BLOCK = re.compile(r"(?P<indent>[ \t]*)<(?P<tag>visual|collision)>(?P<body>.*?)</(?P=tag)>",
                   re.DOTALL)
MESH = re.compile(r'<mesh\s+filename="(?P<file>[^"]+)"\s*(?:/>|>\s*</mesh>)')
ATTR = re.compile(r'(?P<name>xyz|rpy)="(?P<value>[^"]*)"')


def box_of(path: Path) -> tuple[list[float], list[float]]:
    """Return the center and size of the mesh's bounding box, rounded outward to 1e-5 m."""
    lo, hi = trimesh.load(path, force="mesh").bounds
    lo = [math.floor(v * 1e5) / 1e5 for v in lo]
    hi = [math.ceil(v * 1e5) / 1e5 for v in hi]
    center = [round((a + b) / 2, 6) for a, b in zip(lo, hi)]
    size = [round(b - a, 6) for a, b in zip(lo, hi)]
    return center, size


def fmt(values: list[float]) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--meshes", type=Path, nargs="+", required=True)
    parser.add_argument("--header", help="text of the comment after the XML declaration")
    args = parser.parse_args()

    files = {p.name: p for d in args.meshes for p in sorted(d.rglob("*")) if p.is_file()}
    text = args.input.read_text()
    replaced = 0

    def replace(block: re.Match) -> str:
        nonlocal replaced
        body = block["body"]
        mesh = MESH.search(body)
        if not mesh or Path(mesh["file"]).name not in files:
            return block[0]
        origin = re.search(r"<origin\b[^>]*/>", body)
        attrs = dict(ATTR.findall(origin[0])) if origin else {}
        if any(float(v) != 0.0 for v in attrs.get("rpy", "0 0 0").split()):
            sys.exit(f"{mesh['file']}: rotated origin is not supported")
        offset = [float(v) for v in attrs.get("xyz", "0 0 0").split()]
        center, size = box_of(files[Path(mesh["file"]).name])
        xyz = [round(o + c, 6) for o, c in zip(offset, center)]
        new_origin = f'<origin xyz="{fmt(xyz)}" rpy="0 0 0"/>'
        if origin:
            body = body.replace(origin[0], new_origin, 1)
        else:
            lead = re.match(r"\n([ \t]*)", body)
            body = f"\n{lead[1]}{new_origin}{body}" if lead else new_origin + body
        body = MESH.sub(f'<box size="{fmt(size)}"/>', body)
        replaced += 1
        return f"{block['indent']}<{block['tag']}>{body}</{block['tag']}>"

    out = BLOCK.sub(replace, text)
    if args.header is not None:
        lead = LEADING_COMMENTS.match(out)
        if not lead:
            sys.exit(f"{args.input}: no comment follows the XML declaration")
        out = f"{lead[1]}<!-- {args.header} -->\n" + out[lead.end():]
    args.output.write_text(out)
    print(f"replaced {replaced} mesh elements", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
