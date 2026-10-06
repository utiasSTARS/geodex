#!/usr/bin/env python3
"""Generate, install and check the geodex type stubs.

The stubs come from nanobind's stubgen run on a built extension module, one file for the
module (``_geodex_core.pyi``) and one ``<submodule>/__init__.pyi`` per submodule.

    stubs.py generate --stubgen <nanobind>/src/stubgen.py --path <dir> --output <dir>
    stubs.py sync <generated dir> python/geodex     # replace the checked-in stubs
    stubs.py check <generated dir> python/geodex    # fail if they differ
"""

import argparse
import difflib
import importlib.machinery
import importlib.util
import shutil
import sys
import types
from pathlib import Path

MODULE = "geodex._geodex_core"


def stub_files(root):
    """Return the relative paths of the stub files under root."""
    return sorted(p.relative_to(root) for p in Path(root).rglob("*.pyi"))


def generate(args):
    spec = importlib.util.spec_from_file_location("nanobind_stubgen", args.stubgen)
    stubgen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stubgen)

    # Load the built module, not an installed geodex.
    package_dir = Path(args.path) / "geodex"
    package = types.ModuleType("geodex")
    package.__path__ = [str(package_dir)]
    sys.modules["geodex"] = package
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        path = package_dir / ("_geodex_core" + suffix)
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"no built _geodex_core module in {package_dir}")
    loader = importlib.machinery.ExtensionFileLoader(MODULE, str(path))
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location(MODULE, path, loader=loader)
    )
    sys.modules[MODULE] = module
    loader.exec_module(module)

    output = Path(args.output)
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    top = output / "_geodex_core.pyi"
    generator = stubgen.StubGen(
        module=module, recursive=True, include_docstrings=True, output_file=top, quiet=True
    )
    generator.put(module)
    top.write_text(generator.get(), encoding="utf-8")
    return 0


def sync(args):
    generated, target = Path(args.generated), Path(args.target)
    for rel in stub_files(target):
        if not (generated / rel).exists():
            (target / rel).unlink()
    for rel in stub_files(generated):
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(generated / rel, dest)
    return 0


def check(args):
    generated, target = Path(args.generated), Path(args.target)
    wanted, present = stub_files(generated), stub_files(target)
    status = 0
    for rel in sorted(set(wanted) | set(present)):
        new = (generated / rel).read_text().splitlines(True) if rel in wanted else []
        old = (target / rel).read_text().splitlines(True) if rel in present else []
        if new != old:
            status = 1
            sys.stdout.writelines(
                difflib.unified_diff(old, new, f"{target / rel}", f"generated/{rel}")
            )
    if status:
        print("\nThe checked-in stubs differ from the built module. Run `pixi run stubs`.")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="run stubgen on a built module")
    gen.add_argument("--stubgen", required=True, help="path to nanobind's stubgen.py")
    gen.add_argument("--path", required=True, help="directory that contains geodex/")
    gen.add_argument("--output", required=True, help="directory for the stub files")
    for name in ("sync", "check"):
        cmd = sub.add_parser(name)
        cmd.add_argument("generated")
        cmd.add_argument("target")
    args = parser.parse_args()
    return {"generate": generate, "sync": sync, "check": check}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
