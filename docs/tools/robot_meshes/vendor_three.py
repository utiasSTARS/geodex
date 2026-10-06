#!/usr/bin/env python3
"""Copy the three.js files of the docs robot viewer into ``docs/_static/three``.

The files come from the pinned three.js package in ``sources.toml``. Each copied addon
imports the module build by a relative path in place of the bare name ``three``. The license
texts of three.js and of the meshoptimizer decoder go along as ``LICENSE``.

Usage::

    python docs/tools/robot_meshes/vendor_three.py
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

from fetch import fetch, sources  # noqa: E402

OUT = ROOT / "docs" / "_static" / "three"
ADDONS = ("loaders/GLTFLoader.js", "utils/BufferGeometryUtils.js", "controls/OrbitControls.js",
          "libs/meshopt_decoder.module.js", "environments/RoomEnvironment.js", "lines/Line2.js",
          "lines/LineSegments2.js", "lines/LineGeometry.js", "lines/LineSegmentsGeometry.js",
          "lines/LineMaterial.js")
BARE = re.compile(r"""from\s+['"]three['"]""")


def main():
    package = fetch("three") / "package"
    version = sources()["three"]["url"].rsplit("-", 1)[1].removesuffix(".tgz")
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "addons").mkdir(parents=True)
    shutil.copy2(package / "build" / "three.module.min.js", OUT / "three.module.min.js")
    for rel in ADDONS:
        text = (package / "examples" / "jsm" / rel).read_text(encoding="utf-8")
        depth = "../" * (rel.count("/") + 1)
        text = BARE.sub(f"from '{depth}three.module.min.js'", text)
        target = OUT / "addons" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    decoder = (package / "examples" / "jsm" / "libs" / "meshopt_decoder.module.js")
    notice = "".join(line[3:] + "\n" for line in decoder.read_text().splitlines()[:2])
    (OUT / "LICENSE").write_text(
        f"three.js {version}, https://github.com/mrdoob/three.js (npm package three).\n\n"
        + (package / "LICENSE").read_text(encoding="utf-8")
        + "\n\naddons/libs/meshopt_decoder.module.js is part of meshoptimizer,\n"
        "https://github.com/zeux/meshoptimizer, under the MIT License.\n" + notice,
        encoding="utf-8")
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"three.js {version} into {OUT.relative_to(ROOT)}, {total / 1024:.0f} KB")


if __name__ == "__main__":
    main()
