"""Assets of docs/concepts/metrics.rst.

``metrics-arm-shoulder``
    The two-link arm of examples/concepts/metrics.py, stretched and folded, turning its
    shoulder back and forth at 1 rad/s. The arrows are the velocities of the two link centers,
    and each panel shows the cost of the motion under the kinetic-energy metric, from the
    example's output. A VP9 WebM video with an H.264 MP4 fallback and a poster, encoded by
    ffmpeg (``GEODEX_FFMPEG``, or ``ffmpeg`` on the PATH).
"""

from __future__ import annotations

import os

import numpy as np

from common import run_example, video_paths
from style import AQUA, BLUE, GRID, INK, INK_2, ORANGE, matplotlib_style

FPS = 30
DPI = 120  # 1260 by 600 pixels, even for H.264
SWING = 0.5  # rad to each side of the middle
ELBOWS = (("stretched", 0.0, BLUE), ("folded", 2.8, ORANGE))  # the example's two poses


def _joints(q1: float, q2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Elbow, tip and the two link centers of the arm with unit links."""
    elbow = np.array([np.cos(q1), np.sin(q1)])
    tip = elbow + np.array([np.cos(q1 + q2), np.sin(q1 + q2)])
    return elbow, tip, 0.5 * elbow, elbow + 0.5 * (tip - elbow)


def generate(parts=("arm-shoulder",)):
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    plt = matplotlib_style()
    plt.rcParams["animation.ffmpeg_path"] = os.environ.get("GEODEX_FFMPEG", "ffmpeg")
    costs = run_example("concepts/metrics")["kinetic_energy"]

    # The shoulder turns at 1 rad/s from the middle to one side, back to the other and back.
    period = 4.0 * SWING
    frames = int(round(period * FPS))
    t = np.arange(frames) / FPS
    shoulder = SWING * (1.0 - 2.0 * np.abs(((t / period + 0.25) % 1.0) * 2.0 - 1.0))
    direction = np.where(((t / period + 0.25) % 1.0) < 0.5, 1.0, -1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04, wspace=0.08)
    artists = []
    for ax, (label, q2, color), cost in zip(axes, ELBOWS, costs):
        ax.set_xlim(-0.6, 2.3)
        ax.set_ylim(-1.45, 1.45)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{label}, elbow at {q2:g} rad\n1 rad/s at the shoulder costs {cost:.3f}",
                     fontsize=15, color=INK)
        # The arcs the link centers and the tip sweep.
        sweep = np.linspace(-SWING, SWING, 100)
        for index, style in ((2, ":"), (3, ":"), (1, "-")):  # link centers dotted, tip solid
            arc = np.array([_joints(a, q2)[index] for a in sweep])
            ax.plot(arc[:, 0], arc[:, 1], style, color=color, lw=1.2, alpha=0.45)
        ax.plot([0.0], [0.0], "o", ms=13, color=INK_2, zorder=4)
        ax.plot([-0.25, 0.25], [-0.08, -0.08], color=GRID, lw=6, solid_capstyle="butt", zorder=1)
        (links,) = ax.plot([], [], "-", color=color, lw=9, solid_capstyle="round", zorder=3)
        (centers,) = ax.plot([], [], "o", ms=9, color="#ffffff", mec=INK, mew=1.5, zorder=5)
        arrows = [ax.annotate("", xy=(0, 0), xytext=(0, 0), zorder=6,
                              arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=2.4,
                                              mutation_scale=18)) for _ in range(2)]
        artists.append((q2, links, centers, arrows))

    def draw(i):
        changed = []
        for q2, links, centers, arrows in artists:
            elbow, tip, c1, c2 = _joints(shoulder[i], q2)
            links.set_data([0.0, elbow[0], tip[0]], [0.0, elbow[1], tip[1]])
            centers.set_data([c1[0], c2[0]], [c1[1], c2[1]])
            # A point at r from the shoulder moves at |r| m/s, normal to r.
            for arrow, c in zip(arrows, (c1, c2)):
                v = direction[i] * np.array([-c[1], c[0]])
                arrow.xy = tuple(c + 0.45 * v)
                arrow.set_position(tuple(c))
            changed += [links, centers, *arrows]
        return changed

    webm, mp4, poster = video_paths("metrics-arm-shoulder")
    draw(frames // 8)
    fig.savefig(poster, dpi=DPI)
    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / FPS, blit=False)
    anim.save(str(webm), writer=FFMpegWriter(
        fps=FPS, codec="libvpx-vp9",
        extra_args=["-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0", "-pix_fmt", "yuv420p",
                    "-row-mt", "1"]), dpi=DPI)
    anim.save(str(mp4), writer=FFMpegWriter(
        fps=FPS, codec="libx264",
        extra_args=["-crf", "20", "-preset", "slow", "-tune", "animation", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart"]), dpi=DPI)
    plt.close(fig)
