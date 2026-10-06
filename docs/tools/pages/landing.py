"""The robot guides card of the landing page, docs/index.rst.

``docs/_static/landing/robots.webm``, ``robots.mp4``, ``robots.jpg``
    A short loop of the Stretch 3 with its meshes driving the documented differential-drive
    plan of examples/robots/mobile_manipulation/stretch.py around the kitchen island, drawn by the
    robot viewer in headless Chrome with a camera that follows the base. The WebM is VP9, the MP4
    H.264, and the JPEG poster shows the robot at the end of its path.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from browser import Browser
from common import DOCS, run_example
from robot_scene import RobotScene, harness, sample, scene_objects, site, wait_ready
from robots_draw import densify, whole_body_geodesic, whole_body_pieces
from style import BLUE, INK_2

KITCHEN = DOCS.parent / "examples" / "robots" / "mobile_manipulation" / "scenes" / "kitchen.yaml"
SIZE = (800, 400)
FPS = 30
# Camera offset from the smoothed base position, and the height it looks at.
OFFSET = np.array([1.7, -1.9, 3.1])
LOOK_HEIGHT = 0.5
FOLLOW = 0.12
# Seconds from the start of the loop to the poster, less the length of the path. The viewer
# pauses 0.6 s before the path and 1.4 s after it.
POSTER_TIME = 0.6 + 1.2


def scene() -> RobotScene:
    documented = run_example("robots/mobile_manipulation/stretch")
    path = np.array(documented["stretch3"]["path"])
    dense = densify(path, whole_body_geodesic(5), whole_body_pieces)
    s = RobotScene("landing-robots", "stretch3", sample(dense, 150), fps=FPS,
                   camera_position=OFFSET, camera_target=(0.0, 0.0, LOOK_HEIGHT), fov=40.0,
                   floor={"center": [0.6, 0.6], "half": 4.0, "cell": 0.25, "major": 4})
    # The kitchen without its see-through walls.
    s.add_objects([o for o in scene_objects(KITCHEN) if "opacity" not in o])
    s.ghosts(3)
    s.trace_base(BLUE, width=3.0)
    s.trace_end_effector(INK_2, width=2.0)
    return s


SETUP = """window.__frame = (t, draw) => {{
  const v = window.geodexRobotViewers[0];
  v.seek(t);
  const c = window.__follow || (window.__follow = {{x: v.q[0], y: v.q[1]}});
  c.x += {k} * (v.q[0] - c.x);
  c.y += {k} * (v.q[1] - c.y);
  v.stage.look([c.x + {ox}, c.y + {oy}, {oz}], [c.x, c.y, {h}]);
  if (draw) v.stage.render();
  return true;
}}; true"""


def frames(doc: dict, out: Path) -> tuple[int, int]:
    """Write the frames of one loop to `out` and return their count and the poster's index."""
    with site(harness(["landing-robots"], "capture"), {"landing-robots": doc}) as url, \
            Browser(*SIZE) as browser:
        browser.open(url)
        wait_ready(browser)
        browser.evaluate(SETUP.format(k=FOLLOW, ox=OFFSET[0], oy=OFFSET[1], oz=OFFSET[2],
                                      h=LOOK_HEIGHT))
        period, move = browser.evaluate(
            "[window.geodexRobotViewers[0].period, window.geodexRobotViewers[0].move]")
        count = int(round(period * FPS))
        # A first pass without drawing settles the following camera. The loop then starts
        # with the camera where it ends.
        for i in range(count):
            browser.evaluate(f"window.__frame({i / FPS}, false)")
        for i in range(count):
            browser.evaluate(f"window.__frame({i / FPS}, true)")
            browser.screenshot(out / f"{i:04d}.png")
        # The poster is the frame near the end of the pause after the path.
        return count, int(round((POSTER_TIME + move) * FPS))


def encode(frames_dir: Path, poster: int):
    ffmpeg = os.environ.get("GEODEX_FFMPEG", "ffmpeg")
    target = DOCS / "_static" / "landing" / "robots"
    source = ["-framerate", str(FPS), "-i", str(frames_dir / "%04d.png")]
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", *source, "-c:v", "libvpx-vp9",
                    "-b:v", "0", "-crf", "40", "-row-mt", "1", "-pix_fmt", "yuv420p", "-an",
                    str(target.with_suffix(".webm"))], check=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", *source, "-c:v", "libx264",
                    "-preset", "slow", "-crf", "28", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-an", str(target.with_suffix(".mp4"))],
                   check=True)
    from PIL import Image

    Image.open(frames_dir / f"{poster:04d}.png").convert("RGB").save(
        target.with_suffix(".jpg"), quality=86, optimize=True, progressive=True)


def generate(parts=("robots",)):
    if "robots" not in parts:
        return
    doc = scene().document()
    with tempfile.TemporaryDirectory() as tmp:
        _, poster = frames(doc, Path(tmp))
        encode(Path(tmp), poster)
    for suffix in (".webm", ".mp4", ".jpg"):
        path = (DOCS / "_static" / "landing" / "robots").with_suffix(suffix)
        print(f"[landing] {path.name} {path.stat().st_size / 1024:.0f} KB")
