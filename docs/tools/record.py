"""Record viser scenes for static playback in the docs.

A recording is a ``.viser`` file that the viser client (copied into the built site by the
docs extension) plays without a Python server. Everything added to the scene before
``start()`` is the opening frame, and every change made after it, separated by
``sleep()``, becomes the animation. ``save()`` writes ``<name>.viser`` and a still
``<name>.png`` into ``docs/_static/scenes``, where the ``viser-scene`` directive finds them.

.. code-block:: python

   with SceneRecorder("sphere-path", camera_position=(2.2, -2.2, 1.6)) as rec:
       rec.scene.add_icosphere("/sphere", radius=1.0, color=(220, 228, 236))
       dot = rec.scene.add_icosphere("/dot", radius=0.03, color=(42, 120, 214))
       rec.start()
       for p in path:
           dot.position = p
           rec.sleep(1 / 30)
   # writes docs/_static/scenes/sphere-path.viser and .png

Record and replay with the same viser, the one pinned in pixi.toml. A client that sees a
recording from another version shows a version warning over the scene.
"""

from __future__ import annotations

import shutil
import socket
import tempfile
from pathlib import Path

from still import screenshot, serve

ROOT = Path(__file__).resolve().parents[2]
SCENES = ROOT / "docs" / "_static" / "scenes"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SceneRecorder:
    """A headless viser server whose scene is recorded to a file."""

    def __init__(self, name: str, *, camera_position=(2.0, -2.0, 1.5), camera_look_at=(0, 0, 0),
                 up: str = "+z", still_size: tuple[int, int] = (960, 600)):
        import viser

        self.name = name
        self.still_size = still_size
        self.server = viser.ViserServer(host="127.0.0.1", port=_free_port(), verbose=False)
        self.server.scene.set_up_direction(up)
        self.server.initial_camera.position = tuple(float(v) for v in camera_position)
        self.server.initial_camera.look_at = tuple(float(v) for v in camera_look_at)
        self.server.gui.configure_theme(control_layout="collapsed", show_logo=False,
                                        dark_mode=False)
        self._serializer = None

    @property
    def scene(self):
        """The viser scene API (``add_mesh_simple``, ``add_line_segments``, ...)."""
        return self.server.scene

    def start(self):
        """Freeze the opening frame. Later changes form the animation."""
        self._serializer = self.server.get_scene_serializer()

    def sleep(self, seconds: float):
        """Advance the recording clock between two changes."""
        if self._serializer is None:
            raise RuntimeError("call start() before sleep()")
        self._serializer.insert_sleep(seconds)

    def save(self, directory: Path = SCENES) -> Path:
        """Write ``<name>.viser`` and ``<name>.png``, a still of the final frame, and stop
        the server."""
        if self._serializer is None:
            self.start()
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{self.name}.viser"
        target.write_bytes(self._serializer.serialize())
        final_frame = self.server.get_scene_serializer().serialize()
        self.server.stop()
        render_still(final_frame, directory / f"{self.name}.png", self.still_size)
        return target

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is None:
            self.save()
        else:
            self.server.stop()


# The still shows the scene alone, without the playback bar or the notice the client
# raises under the software renderer the screenshot uses.
_STILL_CSS = ("<style>.mantine-Notifications-root,.mantine-Paper-root"
              "{display:none!important}</style>")


def _blank(png: Path) -> bool:
    import matplotlib.image

    pixels = matplotlib.image.imread(png)
    return float(pixels[..., :3].std()) < 1e-3


def render_still(recording: bytes, out: Path, size: tuple[int, int], wait_ms: int = 8000,
                 attempts: int = 3):
    """Screenshot the viser client playing the serialized `recording`.

    After a blank screenshot, taken before the scene loaded, it tries again with twice the
    time budget, up to `attempts` times.
    """
    import viser

    client = Path(viser.__file__).parent / "client" / "build"
    with tempfile.TemporaryDirectory() as tmp:
        site = Path(tmp)
        shutil.copytree(client, site / "viser")
        index = site / "viser" / "index.html"
        page = index.read_text(encoding="utf-8")
        index.write_text(page.replace("</head>", _STILL_CSS + "</head>", 1), encoding="utf-8")
        (site / "still.viser").write_bytes(recording)
        with serve(site) as url:
            for attempt in range(attempts):
                screenshot(f"{url}/viser/index.html?playbackPath=../still.viser", out,
                           *size, wait_ms=wait_ms * 2**attempt)
                if not _blank(out):
                    return
        raise RuntimeError(f"the still {out.name} stayed blank after {attempts} attempts")
