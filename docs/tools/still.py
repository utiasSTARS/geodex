"""Still images of interactive docs assets, taken with headless Chrome.

A recorded viser scene and a plotly figure both need a browser to draw. The docs show a
still of each until the reader opens it, and the still is the page's content when
JavaScript is off. This module serves a directory over HTTP on localhost and screenshots
one page of it with Chrome or Chromium in headless mode. It uses the software WebGL
renderer and works on machines without a GPU.

Set ``GEODEX_CHROME`` to the browser executable when it is not on the PATH.
"""

from __future__ import annotations

import contextlib
import functools
import http.server
import os
import shutil
import socketserver
import subprocess
import threading
from pathlib import Path

_CANDIDATES = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)


def find_chrome() -> str:
    """Path of a Chrome or Chromium executable. Raises a RuntimeError that says how to set one."""
    explicit = os.environ.get("GEODEX_CHROME")
    if explicit:
        return explicit
    for candidate in _CANDIDATES:
        path = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if path:
            return path
    raise RuntimeError(
        "docs stills need Chrome or Chromium; install one or set GEODEX_CHROME to its path")


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@contextlib.contextmanager
def serve(directory: Path):
    """Serve `directory` on a free localhost port for the duration of the block."""
    handler = functools.partial(_QuietHandler, directory=str(directory))
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{httpd.server_address[1]}"
        finally:
            httpd.shutdown()


def screenshot(url: str, out: Path, width: int, height: int, wait_ms: int = 4000,
               scale: float = 2.0) -> Path:
    """Render `url` at `width` x `height` CSS pixels and write a PNG to `out`."""
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        find_chrome(),
        "--headless=new",
        "--hide-scrollbars",
        "--no-first-run",
        "--no-default-browser-check",
        "--use-angle=swiftshader",
        "--enable-unsafe-swiftshader",
        "--ignore-gpu-blocklist",
        f"--force-device-scale-factor={scale}",
        f"--window-size={width},{height}",
        f"--virtual-time-budget={wait_ms}",
        f"--screenshot={out}",
        url,
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=180)
    if not out.exists():
        raise RuntimeError(f"Chrome wrote no screenshot for {url}")
    return out
