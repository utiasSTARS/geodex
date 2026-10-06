"""The ROS 2 pages show the plugins' own files and tables generated from them.

docs/tools/plugin_sources.py keeps verbatim copies of the plugin files under
docs/ros2/sources/ and the tables generated from them under docs/ros2/generated/. This test
fails when a copy was edited, when a table is stale, or, where a plugin checkout is present
(a sibling directory, or GEODEX_NAV2_PLANNER_DIR and GEODEX_MOVEIT_DIR), when the plugin's file
changed since the copy was taken. The pages must also read only files the script copies.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location(
        "plugin_sources", DOCS / "tools" / "plugin_sources.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_copies_and_tables_match_their_sources():
    problems = _script().check()
    assert not problems, "ROS 2 pages drifted, run docs/tools/plugin_sources.py sync:\n" + \
        "\n".join(problems)


def test_pages_read_only_copied_files():
    script = _script()
    listed = {f"{name}/{rel}" for name, repo in script.manifest()["repositories"].items()
              for rel in repo["files"]}
    used = set()
    for page in (DOCS / "ros2").glob("*.rst"):
        for target in re.findall(r"^\s*\.\. literalinclude:: sources/(\S+)", page.read_text(),
                                 re.MULTILINE):
            used.add(target)
    assert used, "no page includes a plugin file"
    assert not used - listed, f"pages include files the script does not copy: {used - listed}"
