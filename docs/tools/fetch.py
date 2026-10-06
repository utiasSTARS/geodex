"""Fetch the pinned external inputs of the docs assets.

``fetch(name)`` downloads the source named ``name`` in ``sources.toml`` once, checks its
SHA-256, unpacks it when it is an archive, and returns the local directory or file. The
cache lives in ``build/docs-assets-cache`` and is safe to delete.

.. code-block:: python

   from fetch import fetch
   stretch = fetch("stretch4_urdf")   # directory of the unpacked, hash-checked wheel
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / "build" / "docs-assets-cache"


def sources(path: Path = HERE / "sources.toml") -> dict:
    """The pinned sources, by name."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name: str, table: dict | None = None, cache: Path = CACHE) -> Path:
    """Local copy of the source `name`, downloaded and checked on first use."""
    entry = (table or sources())[name]
    target = cache / name
    stamp = target / ".sha256"
    if stamp.exists() and stamp.read_text().strip() == entry["sha256"]:
        return _result(target, entry)
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    filename = entry["url"].rsplit("/", 1)[-1] or name
    download = target / filename
    with urllib.request.urlopen(entry["url"], timeout=120) as response, open(download, "wb") as f:
        shutil.copyfileobj(response, f)
    digest = _sha256(download)
    if digest != entry["sha256"]:
        shutil.rmtree(target, ignore_errors=True)
        raise RuntimeError(f"{name}: sha256 {digest} does not match the pin {entry['sha256']}")
    if entry.get("unpack", True) and _is_archive(filename):
        _unpack(download, target / "src")
    stamp.write_text(entry["sha256"] + "\n")
    return _result(target, entry)


def _is_archive(filename: str) -> bool:
    return filename.endswith((".tar.gz", ".tgz", ".tar.xz", ".zip", ".whl"))


def _unpack(archive: Path, into: Path):
    into.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith((".zip", ".whl")):
        with zipfile.ZipFile(archive) as z:
            z.extractall(into)
    else:
        with tarfile.open(archive) as t:
            t.extractall(into, filter="data")


def _result(target: Path, entry: dict) -> Path:
    unpacked = target / "src"
    if unpacked.exists():
        return unpacked
    return next(p for p in target.iterdir() if p.name != ".sha256")
