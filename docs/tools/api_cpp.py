#!/usr/bin/env python3
"""Write the Breathe directives of the C++ API page from the Doxygen XML.

For every public header under include/geodex, this writes one include file,
build/docs/api/cpp/<header path>.inc, holding one Breathe directive per entity the header
declares (class, struct, concept, function, enum, type alias, variable, macro), in the
order the header declares them. docs/api/cpp.rst includes one file per header, so the page
follows the headers and renders each entity once. Entities in a ``detail`` namespace and
classes nested in another class (which their parent renders) are left out.

  python docs/tools/api_cpp.py [--xml build/docs/doxygen/xml] [--out build/docs/api/cpp]

The pixi ``docs`` task runs it between Doxygen and Sphinx. It also writes ``headers.txt``,
the headers that declare a public entity, which docs/tests/test_api_coverage.py compares with
the headers the page includes.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "include/geodex/"
DIRECTIVE = {"class": "doxygenclass", "struct": "doxygenstruct", "union": "doxygenunion",
             "concept": "doxygenconcept", "function": "doxygenfunction", "enum": "doxygenenum",
             "typedef": "doxygentypedef", "variable": "doxygenvariable",
             "define": "doxygendefine"}


def public(name: str) -> bool:
    parts = name.split("::")
    return parts[0] == "geodex" and "detail" not in parts


def header_of(location: ET.Element | None) -> str | None:
    if location is None:
        return None
    path = location.get("file", "").replace("\\", "/")
    index = path.find(PREFIX)
    return path[index + len(PREFIX):] if index >= 0 else None


def text(node: ET.Element | None) -> str:
    return "".join(node.itertext()).strip() if node is not None else ""


def signature(member: ET.Element) -> str:
    """Parameter types of a function, the form Breathe resolves an overload by."""
    types = []
    for param in member.findall("param"):
        kind = re.sub(r"\s+", " ", text(param.find("type")))
        array = text(param.find("array"))
        types.append(kind + array)
    args = "(" + ", ".join(types) + ")"
    argsstring = text(member.find("argsstring"))
    tail = argsstring[argsstring.rfind(")") + 1:] if ")" in argsstring else ""
    for qualifier in ("const", "noexcept"):
        if re.search(rf"\b{qualifier}\b", tail):
            args += f" {qualifier}"
    return args


def collect(xml: Path) -> dict[str, list[tuple[int, str]]]:
    """Header to its (line, directive) entries."""
    index = ET.parse(xml / "index.xml").getroot()
    classes = {c.findtext("name") for c in index.findall("compound")
               if c.get("kind") in ("class", "struct", "union", "concept")}
    entries: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for compound in index.findall("compound"):
        kind, name = compound.get("kind"), compound.findtext("name")
        if kind in ("class", "struct", "union", "concept"):
            if not public(name) or "::".join(name.split("::")[:-1]) in classes:
                continue
            root = ET.parse(xml / f"{compound.get('refid')}.xml").getroot()
            location = root.find("compounddef/location")
            header = header_of(location)
            if header is None:
                continue
            options = "" if kind == "concept" else "   :members:\n   :undoc-members:\n"
            entries[header].append((int(location.get("line", 0)),
                                    f".. {DIRECTIVE[kind]}:: {name}\n{options}"))
        elif kind in ("namespace", "file"):
            if kind == "namespace" and not public(name):
                continue
            root = ET.parse(xml / f"{compound.get('refid')}.xml").getroot()
            members = root.findall("compounddef/sectiondef/memberdef")
            if kind == "file":
                members = [m for m in members if m.get("kind") == "define"]
            names = defaultdict(int)
            for member in members:
                if member.get("kind") == "function":
                    names[member.findtext("name")] += 1
            for member in members:
                mkind = member.get("kind")
                if mkind not in DIRECTIVE:
                    continue
                header = header_of(member.find("location"))
                if header is None:
                    continue
                short = member.findtext("name")
                qualified = short if mkind == "define" else f"{name}::{short}"
                if mkind == "function" and qualified in classes:
                    continue  # a deduction guide, which its class documents
                target = qualified
                if mkind == "function" and names[short] > 1:
                    target += signature(member)
                line = int(member.find("location").get("line", 0))
                entries[header].append((line, f".. {DIRECTIVE[mkind]}:: {target}\n"))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--xml", type=Path, default=ROOT / "build" / "docs" / "doxygen" / "xml")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "docs" / "api" / "cpp")
    args = parser.parse_args()
    if not (args.xml / "index.xml").exists():
        print(f"error: no Doxygen XML in {args.xml}; build the docs target first",
              file=sys.stderr)
        return 1
    entries = collect(args.xml)
    for header, items in entries.items():
        target = args.out / f"{header}.inc"
        target.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(directive for _, directive in sorted(items, key=lambda e: e[0]))
        target.write_text(f".. Generated by docs/tools/api_cpp.py from {header}.\n\n{body}",
                          encoding="utf-8")
    (args.out / "headers.txt").write_text("\n".join(sorted(entries)) + "\n", encoding="utf-8")
    count = sum(len(items) for items in entries.values())
    print(f"wrote {count} entities of {len(entries)} headers to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
