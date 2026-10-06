"""A headless Chrome session driven over the DevTools protocol.

``Browser`` starts Chrome or Chromium in headless mode with the DevTools protocol on a pipe,
opens one page at a fixed size, evaluates JavaScript in it and captures screenshots. The
robot viewer's stills, the landing animation and the in-browser kinematics check use it. It
needs only the standard library and the browser that ``still.find_chrome`` finds.

.. code-block:: python

   with serve(site) as url, Browser(1280, 720) as browser:
       browser.open(f"{url}/scene.html")
       browser.evaluate("window.geodexRobotViewers[0].ready.then(() => true)")
       browser.screenshot(out)
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
from pathlib import Path

from still import find_chrome


class Browser:
    """One headless page of `width` by `height` CSS pixels at `scale` device pixels each."""

    def __init__(self, width: int, height: int, scale: float = 1.0, timeout: float = 120.0):
        self.size = (width, height, scale)
        self.timeout = timeout
        self._id = 0
        self._buffer = b""
        self._events = []

    def __enter__(self):
        self._profile = tempfile.TemporaryDirectory()
        to_chrome_r, self._to_chrome = os.pipe()
        self._from_chrome, from_chrome_w = os.pipe()

        def fds():
            # Chrome reads commands from fd 3 and writes replies to fd 4.
            read, write = os.dup(to_chrome_r), os.dup(from_chrome_w)
            os.dup2(read, 3)
            os.dup2(write, 4)
            # os.dup returns close-on-exec descriptors, and dup2 onto the same number keeps that flag.
            os.set_inheritable(3, True)
            os.set_inheritable(4, True)

        command = [find_chrome(), "--headless=new", "--remote-debugging-pipe", "--no-first-run",
                   "--no-default-browser-check", "--hide-scrollbars", "--mute-audio",
                   "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                   "--ignore-gpu-blocklist", f"--user-data-dir={self._profile.name}",
                   "about:blank"]
        self._process = subprocess.Popen(command, close_fds=False, preexec_fn=fds,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.close(to_chrome_r)
        os.close(from_chrome_w)
        targets = self._call("Target.getTargets")["targetInfos"]
        page = next(t["targetId"] for t in targets if t["type"] == "page")
        self._session = self._call("Target.attachToTarget", targetId=page,
                                   flatten=True)["sessionId"]
        width, height, scale = self.size
        self.call("Page.enable")
        self.call("Runtime.enable")
        self.call("Emulation.setDeviceMetricsOverride", width=width, height=height,
                  deviceScaleFactor=scale, mobile=False)
        return self

    def __exit__(self, *exc):
        try:
            self._call("Browser.close")
        except Exception:
            pass
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
        os.close(self._to_chrome)
        os.close(self._from_chrome)
        self._profile.cleanup()

    def _send(self, message: dict) -> int:
        self._id += 1
        message["id"] = self._id
        os.write(self._to_chrome, json.dumps(message).encode() + b"\0")
        return self._id

    def _receive(self) -> dict:
        while b"\0" not in self._buffer:
            chunk = os.read(self._from_chrome, 1 << 20)
            if not chunk:
                raise RuntimeError("Chrome closed the DevTools pipe")
            self._buffer += chunk
        raw, self._buffer = self._buffer.split(b"\0", 1)
        return json.loads(raw)

    def _call(self, method: str, session: str | None = None, **params) -> dict:
        message = {"method": method, "params": params}
        if session:
            message["sessionId"] = session
        wanted = self._send(message)
        while True:
            reply = self._receive()
            if reply.get("id") == wanted:
                if "error" in reply:
                    raise RuntimeError(f"{method}: {reply['error']}")
                return reply.get("result", {})
            self._events.append(reply)

    def call(self, method: str, **params) -> dict:
        """Send a DevTools command to the page and return its result."""
        return self._call(method, self._session, **params)

    def open(self, url: str):
        """Load `url` and wait for its load event."""
        self._events.clear()
        self.call("Page.navigate", url=url)
        while not any(e.get("method") == "Page.loadEventFired" for e in self._events):
            self._events.append(self._receive())

    def evaluate(self, expression: str):
        """Value of a JavaScript expression in the page, awaited when it is a promise."""
        result = self.call("Runtime.evaluate", expression=expression, awaitPromise=True,
                           returnByValue=True, timeout=int(self.timeout * 1000))
        if "exceptionDetails" in result:
            raise RuntimeError(f"{expression}: {result['exceptionDetails']}")
        return result["result"].get("value")

    def screenshot(self, out: Path, fmt: str = "png", quality: int = 90) -> Path:
        """Write a screenshot of the page to `out`."""
        params = {"format": fmt, "captureBeyondViewport": False}
        if fmt == "jpeg":
            params["quality"] = quality
        data = self.call("Page.captureScreenshot", **params)["data"]
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(data))
        return out
