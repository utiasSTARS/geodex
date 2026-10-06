"""Sphinx directives for the geodex docs.

``code-pair``
    A Python and C++ tab pair filled from a tested example. The Python tab comes first and
    both tabs join the ``lang`` sync group, so the reader's language stays selected across
    pages. Each tab shows the lines between ``[docs-start:<name>]`` and
    ``[docs-end:<name>]`` comment markers. A name may mark several regions of one file,
    which are shown in order, each dedented on its own, so a tab can show the includes of a
    file with the body of a function, or a whole program without the lines that only serve
    the tests. Below the code, each tab links the whole example file at the release tag
    (``geodex_source_url`` in conf.py).

``viser-scene``
    A recorded viser scene from ``_static/scenes/<name>.viser``. The page shows the still
    ``_static/scenes/<name>.png`` until the reader opens the interactive view, which plays
    the recording in the viser client copied from the pinned viser package.

``plotly-figure``
    An interactive plotly figure from ``_static/plots/<name>.html`` (a fragment written by
    ``pixi run docs-assets``) with ``_static/plots/<name>.png`` as its still for readers
    without JavaScript. plotly.js loads only on pages that hold a figure.

``video-figure``
    A short looping animation from ``_static/videos/<name>.webm`` (VP9), with
    ``<name>.mp4`` (H.264) for browsers without VP9 and ``<name>.png`` as its poster. It
    plays muted and inline like an animated image, inside a numbered-caption figure.

``robot-scene``
    A robot following a planned path in the three.js robot viewer, from
    ``_static/robot-scenes/<name>.json`` with ``<name>.jpg`` as its poster, and the robot
    model under ``_static/robots/<robot>/``. It loads and plays when it scrolls into view,
    and a ``:hero:`` scene at the top of a page loads and plays with the page. ``:spheres:``
    overlays the robot's VAMP collision spheres and uses the poster ``<name>-spheres.jpg``.
    ``:spheres: hidden`` loads the spheres hidden behind the Spheres button and uses the
    poster ``<name>.jpg``. A scene without a robot, such as a path on a sphere, moves small
    balls along its frames. ``:width:`` narrows the scene.
    The viewer and three.js load only on pages that hold a scene.

``landing-video``
    A looping muted video for a card of the landing page, from
    ``_static/landing/<name>.webm`` and ``<name>.mp4`` with ``<name>.jpg`` as its poster.
    Until all three files exist the card shows the ``:fallback:`` image.

The viser client and plotly.js are copied from the installed, pinned packages at build
time, so the recordings, the fragments and the players always come from one version.
"""

from __future__ import annotations

import html
import shutil
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from sphinx.util import logging

logger = logging.getLogger(__name__)

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
LANGS = (("Python", "python", "py", "python"), ("C++", "cpp", "cpp", "cpp"))


def _static_prefix(docname: str) -> str:
    """Relative path from the page of `docname` to the output root."""
    return "../" * docname.count("/")


def extract_regions(path: Path, name: str) -> str:
    """The lines of every ``[docs-start:name]`` to ``[docs-end:name]`` region of `path`,
    each dedented, joined by blank lines. Raises ValueError when there is no region or a
    region is left open."""
    start, end = f"[docs-start:{name}]", f"[docs-end:{name}]"
    regions, current = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if start in line:
            if current is not None:
                raise ValueError(f"{path}: region {name!r} opened twice")
            current = []
        elif end in line:
            if current is None:
                raise ValueError(f"{path}: region {name!r} closed before it opened")
            regions.append(current)
            current = None
        elif current is not None:
            current.append(line)
    if current is not None:
        raise ValueError(f"{path}: region {name!r} is never closed")
    if not regions:
        raise ValueError(f"{path}: no region {name!r}")
    lines = []
    for region in regions:
        while region and not region[-1].strip():
            region.pop()
        indents = [len(line) - len(line.lstrip()) for line in region if line.strip()]
        cut = min(indents, default=0)
        region = [line[cut:] for line in region]
        # A region that only closes a block continues the one before it.
        closes = region and region[0].strip()[:1] in ("}", ")", "]")
        if lines and not closes:
            lines.append("")
        lines += region
    return "\n".join(lines)


class CodePairDirective(Directive):
    """``.. code-pair:: <example> <marker>``, where <example> is a path under examples/
    without its suffix (for example ``concepts/planning``). ``:python:`` or ``:cpp:``
    name a different file for one language, ``:python-marker:`` or ``:cpp-marker:`` a
    different marker, and ``:only:`` restricts the pair to one language."""

    required_arguments = 2
    option_spec = {
        "python": directives.unchanged,
        "cpp": directives.unchanged,
        "python-marker": directives.unchanged,
        "cpp-marker": directives.unchanged,
        "only": lambda arg: directives.choice(arg, ("python", "cpp")),
    }

    def run(self):
        stem, marker = self.arguments
        env = self.state.document.settings.env
        lines = [".. tab-set::", "   :sync-group: lang", ""]
        for label, key, suffix, language in LANGS:
            if self.options.get("only") not in (None, key):
                continue
            rel = self.options.get(key, f"{stem}.{suffix}")
            path = EXAMPLES / rel
            if not path.exists():
                raise self.error(f"code-pair: {path} does not exist")
            try:
                code = extract_regions(path, self.options.get(f"{key}-marker", marker))
            except ValueError as err:
                raise self.error(f"code-pair: {err}") from None
            env.note_dependency(str(path))
            lines += [f"   .. tab-item:: {label}", f"      :sync: {key}", "",
                      f"      .. code-block:: {language}", ""]
            lines += [f"         {line}" if line else "" for line in code.splitlines()]
            lines.append("")
            source = env.config.geodex_source_url
            if source:
                rel_path = f"examples/{rel}"
                lines += ["      .. rst-class:: geodex-example-link", "",
                          f"      Full example: `{rel_path} <{source}{rel_path}>`__", ""]
        container = nodes.container(classes=["geodex-code-pair"])
        self.state.nested_parse(StringList(lines, source=f"code-pair {stem}"),
                                self.content_offset, container)
        return [container]


def _caption(directive: Directive) -> list[nodes.Node]:
    if not directive.content:
        return []
    caption = nodes.container(classes=["geodex-caption"])
    directive.state.nested_parse(directive.content, directive.content_offset, caption)
    return [caption]


class ViserSceneDirective(Directive):
    """``.. viser-scene:: <name>`` with ``:alt:`` (required) and an optional caption as
    content. ``:aspect:`` sets the frame's width to height ratio (default 16/10)."""

    required_arguments = 1
    has_content = True
    option_spec = {"alt": directives.unchanged_required, "aspect": directives.unchanged}

    def run(self):
        env = self.state.document.settings.env
        name = self.arguments[0]
        if "alt" not in self.options:
            raise self.error("viser-scene needs :alt:")
        static = Path(env.srcdir) / "_static" / "scenes"
        for suffix in (".viser", ".png"):
            path = static / f"{name}{suffix}"
            if not path.exists():
                raise self.error(f"viser-scene: {path} is missing, run pixi run docs-assets")
            env.note_dependency(str(path))
        root = _static_prefix(env.docname)
        client = f"{root}_static/viser/index.html?playbackPath=../scenes/{name}.viser"
        alt = html.escape(self.options["alt"], quote=True)
        aspect = html.escape(self.options.get("aspect", "16/10"), quote=True)
        markup = (
            f'<div class="geodex-scene" data-src="{html.escape(client, quote=True)}" '
            f'data-title="{alt}" style="aspect-ratio:{aspect}">'
            f'<img src="{root}_static/scenes/{name}.png" alt="{alt}" loading="lazy">'
            '<button type="button" class="geodex-scene-open">'
            '<span class="geodex-scene-icon" aria-hidden="true">&#9654;</span>'
            "Open the interactive 3D view</button></div>"
        )
        figure = nodes.container(classes=["geodex-figure"])
        figure += nodes.raw("", markup, format="html")
        figure += _caption(self)
        return [figure]


class PlotlyFigureDirective(Directive):
    """``.. plotly-figure:: <name>`` with ``:alt:`` (required) and an optional caption."""

    required_arguments = 1
    has_content = True
    option_spec = {"alt": directives.unchanged_required}

    def run(self):
        env = self.state.document.settings.env
        name = self.arguments[0]
        if "alt" not in self.options:
            raise self.error("plotly-figure needs :alt:")
        static = Path(env.srcdir) / "_static" / "plots"
        fragment = static / f"{name}.html"
        still = static / f"{name}.png"
        for path in (fragment, still):
            if not path.exists():
                raise self.error(f"plotly-figure: {path} is missing, run pixi run docs-assets")
            env.note_dependency(str(path))
        if not hasattr(env, "geodex_plotly_docs"):
            env.geodex_plotly_docs = set()
        env.geodex_plotly_docs.add(env.docname)
        root = _static_prefix(env.docname)
        alt = html.escape(self.options["alt"], quote=True)
        markup = (
            '<div class="geodex-plotly">'
            f'<noscript><img src="{root}_static/plots/{name}.png" alt="{alt}"></noscript>'
            f'{fragment.read_text(encoding="utf-8")}'
            "</div>"
        )
        figure = nodes.container(classes=["geodex-figure"])
        figure += nodes.raw("", markup, format="html")
        figure += _caption(self)
        return [figure]


class VideoFigureDirective(Directive):
    """``.. video-figure:: <name>`` with ``:alt:`` (required), ``:width:`` (default 100%) and
    an optional caption as content, like ``figure``."""

    required_arguments = 1
    has_content = True
    option_spec = {"alt": directives.unchanged_required,
                   "width": directives.length_or_percentage_or_unitless}

    def run(self):
        env = self.state.document.settings.env
        name = self.arguments[0]
        if "alt" not in self.options:
            raise self.error("video-figure needs :alt:")
        static = Path(env.srcdir) / "_static" / "videos"
        for suffix in (".webm", ".mp4", ".png"):
            path = static / f"{name}{suffix}"
            if not path.exists():
                raise self.error(f"video-figure: {path} is missing, run pixi run docs-assets")
            env.note_dependency(str(path))
        base = f"{_static_prefix(env.docname)}_static/videos/{name}"
        alt = html.escape(self.options["alt"], quote=True)
        width = html.escape(self.options.get("width", "100%"), quote=True)
        markup = (
            f'<video class="geodex-video" style="width:{width}" poster="{base}.png" '
            f'aria-label="{alt}" autoplay loop muted playsinline disablepictureinpicture>'
            f'<source src="{base}.webm" type="video/webm">'
            f'<source src="{base}.mp4" type="video/mp4">'
            f'<img src="{base}.png" alt="{alt}"></video>'
        )
        figure = nodes.figure(align="center")
        figure += nodes.raw("", markup, format="html")
        if self.content:
            body = nodes.Element()
            self.state.nested_parse(self.content, self.content_offset, body)
            first = body[0]
            if not isinstance(first, nodes.paragraph):
                raise self.error("video-figure: the caption must start with a paragraph")
            figure += nodes.caption(first.rawsource, "", *first.children)
            if len(body) > 1:
                figure += nodes.legend("", *body[1:])
        return [figure]


VIEWER_FILES = ("robot-viewer/robot-viewer.js", "robot-viewer/robot-viewer.css",
                "three/three.module.min.js", "three/LICENSE")


def robot_scene_files(static: Path, name: str, spheres: bool | str = False) -> list[Path]:
    """Every file a ``robot-scene`` of `name` loads, under the ``_static`` directory
    `static`. `spheres` is False, True or "shown" for spheres shown at the start, or "hidden"
    for spheres behind the Spheres button. Raises ValueError when the scene file is missing,
    or when it asks for spheres and does not name a robot."""
    import json

    scene = static / "robot-scenes" / f"{name}.json"
    if not scene.exists():
        raise ValueError(f"{scene} is missing, run pixi run docs-assets")
    robot = json.loads(scene.read_text(encoding="utf-8")).get("robot")
    if spheres and not robot:
        raise ValueError(f"{scene} does not name a robot for its spheres")
    shown = spheres is True or spheres == "shown"
    poster = f"{name}-spheres.jpg" if shown else f"{name}.jpg"
    files = [scene, static / "robot-scenes" / poster]
    if robot:
        model = static / "robots" / robot
        files += [model / "chain.json", model / f"{robot}.glb", model / f"{robot}_ghost.glb"]
    if spheres:
        files.append(model / "spheres.json")
    return files + [static / rel for rel in VIEWER_FILES]


_PLAY = ('<svg class="geodex-robot-icon-play" viewBox="0 0 16 16" aria-hidden="true">'
         '<path d="M4 2.5v11l9-5.5z"/></svg>'
         '<svg class="geodex-robot-icon-pause" viewBox="0 0 16 16" aria-hidden="true">'
         '<path d="M3.5 2.5h3v11h-3zM9.5 2.5h3v11h-3z"/></svg>')


def _spheres_option(argument: str | None) -> str:
    """The ``:spheres:`` option. Empty or ``shown`` shows the spheres at the start, and
    ``hidden`` starts with them hidden."""
    return directives.choice((argument or "").strip() or "shown", ("shown", "hidden"))


class RobotSceneDirective(Directive):
    """``.. robot-scene:: <name>`` with ``:alt:`` (required), the flag ``:hero:``,
    ``:spheres:`` (empty, ``shown`` or ``hidden``), ``:aspect:`` (the width to height ratio,
    default 16/9), ``:width:`` (default 100%) and an optional caption as content."""

    required_arguments = 1
    has_content = True
    option_spec = {"alt": directives.unchanged_required, "hero": directives.flag,
                   "spheres": _spheres_option, "aspect": directives.unchanged,
                   "width": directives.length_or_percentage_or_unitless}

    def run(self):
        env = self.state.document.settings.env
        name = self.arguments[0]
        if "alt" not in self.options:
            raise self.error("robot-scene needs :alt:")
        hero, spheres = "hero" in self.options, self.options.get("spheres", False)
        static = Path(env.srcdir) / "_static"
        try:
            files = robot_scene_files(static, name, spheres)
        except ValueError as err:
            raise self.error(f"robot-scene: {err}") from None
        for path in files:
            if not path.exists():
                raise self.error(f"robot-scene: {path} is missing, run pixi run docs-assets")
            env.note_dependency(str(path))
        if not hasattr(env, "geodex_robot_docs"):
            env.geodex_robot_docs = set()
        env.geodex_robot_docs.add(env.docname)
        root = _static_prefix(env.docname)
        alt = html.escape(self.options["alt"], quote=True)
        aspect = html.escape(self.options.get("aspect", "16/9"), quote=True)
        width = html.escape(self.options.get("width", "100%"), quote=True)
        poster = files[1].name
        shown = "true" if spheres == "shown" else "false"
        toggle = (f'<button type="button" class="geodex-robot-spheres" aria-pressed="{shown}">'
                  "Spheres</button>" if spheres else "")
        classes = "geodex-robot-scene geodex-robot-hero" if hero else "geodex-robot-scene"
        mode = {"shown": "1", "hidden": "hidden"}.get(spheres, "")
        flags = (' data-hero="1"' if hero else "") + (f' data-spheres="{mode}"' if mode else "")
        loading = "eager" if hero else "lazy"
        markup = (
            f'<div class="{classes}" data-scene="{root}_static/robot-scenes/{name}.json" '
            f'data-robots="{root}_static/robots/" data-alt="{alt}"{flags} '
            f'style="aspect-ratio:{aspect};width:{width};margin:0 auto">'
            f'<img class="geodex-robot-poster" src="{root}_static/robot-scenes/{poster}" '
            f'alt="{alt}" loading="{loading}">'
            '<div class="geodex-robot-stage"></div>'
            '<div class="geodex-robot-hint" aria-hidden="true">Drag to orbit</div>'
            '<div class="geodex-robot-controls" hidden>'
            f'<button type="button" class="geodex-robot-play" aria-label="Play">{_PLAY}</button>'
            '<input type="range" class="geodex-robot-scrub" min="0" max="1000" value="1000" '
            'aria-label="Position along the path">'
            f"{toggle}</div></div>"
        )
        classes = ["geodex-figure", "geodex-robot-figure"]
        if hero:
            classes.append("geodex-robot-figure-hero")
        figure = nodes.container(classes=classes)
        figure += nodes.raw("", markup, format="html")
        figure += _caption(self)
        return [figure]


class LandingVideoDirective(Directive):
    """``.. landing-video:: <name>`` with ``:alt:`` (required), and ``:fallback:`` and
    ``:fallback-alt:`` for the image shown until the video files exist."""

    required_arguments = 1
    option_spec = {"alt": directives.unchanged_required,
                   "fallback": directives.uri,
                   "fallback-alt": directives.unchanged}

    def run(self):
        env = self.state.document.settings.env
        name = self.arguments[0]
        if "alt" not in self.options:
            raise self.error("landing-video needs :alt:")
        static = Path(env.srcdir) / "_static" / "landing"
        files = [static / f"{name}{suffix}" for suffix in (".webm", ".mp4", ".jpg")]
        for path in files:
            env.note_dependency(str(path))
        if not all(path.exists() for path in files):
            if "fallback" not in self.options:
                raise self.error(f"landing-video: {files[0].with_suffix('')}.* are missing "
                                 "and there is no :fallback:")
            alt = self.options.get("fallback-alt", self.options["alt"])
            return [nodes.image(uri=self.options["fallback"], alt=alt,
                                classes=["landing-media"])]
        base = f"{_static_prefix(env.docname)}_static/landing/{name}"
        alt = html.escape(self.options["alt"], quote=True)
        markup = (
            f'<video class="landing-media" poster="{base}.jpg" aria-label="{alt}" '
            'autoplay loop muted playsinline disablepictureinpicture>'
            f'<source src="{base}.webm" type="video/webm">'
            f'<source src="{base}.mp4" type="video/mp4">'
            f'<img src="{base}.jpg" alt="{alt}"></video>'
        )
        return [nodes.raw("", markup, format="html")]


def _purge(app, env, docname):
    for key in ("geodex_plotly_docs", "geodex_robot_docs"):
        if hasattr(env, key):
            getattr(env, key).discard(docname)


def _merge(app, env, docnames, other):
    for key in ("geodex_plotly_docs", "geodex_robot_docs"):
        if not hasattr(env, key):
            setattr(env, key, set())
        getattr(env, key).update(getattr(other, key, set()) & set(docnames))


def _page_context(app, pagename, templatename, context, doctree):
    if pagename in getattr(app.env, "geodex_plotly_docs", set()):
        app.add_js_file("plotly/plotly.min.js", priority=400)
    if pagename in getattr(app.env, "geodex_robot_docs", set()):
        app.add_css_file("robot-viewer/robot-viewer.css")
        app.add_js_file("robot-viewer/robot-viewer.js", type="module")


def _copy_players(app, exception):
    """Copy the viser client and plotly.js of the installed, pinned packages."""
    if exception is not None or app.builder.format != "html":
        return
    static = Path(app.outdir) / "_static"
    try:
        import viser
    except ImportError:
        logger.warning("viser is not installed, so recorded scenes cannot play")
    else:
        client = Path(viser.__file__).parent / "client" / "build"
        shutil.copytree(client, static / "viser", dirs_exist_ok=True)
    try:
        import plotly
    except ImportError:
        logger.warning("plotly is not installed, so interactive figures cannot render")
    else:
        source = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
        (static / "plotly").mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, static / "plotly" / "plotly.min.js")


def setup(app):
    app.add_config_value("geodex_source_url", "", "env")
    app.add_directive("code-pair", CodePairDirective)
    app.add_directive("viser-scene", ViserSceneDirective)
    app.add_directive("plotly-figure", PlotlyFigureDirective)
    app.add_directive("video-figure", VideoFigureDirective)
    app.add_directive("robot-scene", RobotSceneDirective)
    app.add_directive("landing-video", LandingVideoDirective)
    app.connect("env-purge-doc", _purge)
    app.connect("env-merge-info", _merge)
    app.connect("html-page-context", _page_context)
    app.connect("build-finished", _copy_players)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
