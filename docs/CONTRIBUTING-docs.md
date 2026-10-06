# Writing the geodex docs

The site is Sphinx with pydata-sphinx-theme (light mode only), sphinx-design and a local
extension, `docs/_ext/geodex_docs.py`. Every page builds with `-W`, every snippet is code
the tests run, and every figure comes from one command. This note gives a writer the
conventions behind those three rules.

## Commands

| command | does |
|---|---|
| `pixi run docs` | builds the site into `build/docs/sphinx` with warnings as errors |
| `pixi run test-docs` | runs every docs example in Python and C++ and compares the two (`docs/tests/`) |
| `pixi run docs-assets [page[:part]]` | regenerates figures, plots and recordings (`docs/tools/`) |
| `pixi run stubs` | regenerates the Python type stubs from the built module after a binding changes |
| `pixi run check-stubs` | fails when the checked-in type stubs differ from the built module |

Build and run on a Linux or macOS machine with planning, collision checking and the built-in
robots. Stills need Chrome or Chromium, and `GEODEX_CHROME` names it when it is not on the
`PATH`. Animations need ffmpeg with libvpx-vp9 and libx264, and `GEODEX_FFMPEG` names it when it
is not on the `PATH`.

## Snippets

A snippet is a region of an example file between comment markers.

```python
# [docs-start:plan]
result = geodex.plan(space, start, goal, is_valid, settings=settings)
# [docs-end:plan]
```

```cpp
// [docs-start:plan]
auto result = geodex::planning::plan(space, start, goal, is_valid, settings);
// [docs-end:plan]
```

A page shows the pair as synchronized Python and C++ tabs, Python first.

```rst
.. code-pair:: concepts/planning plan
```

- The first argument is the file under `examples/` without its suffix, the second the marker.
  `:python:` or `:cpp:` name another file for one language, `:python-marker:` or
  `:cpp-marker:` another marker, and `:only: cpp` or `:only: python` shows one tab (for code
  that exists in one language only, such as a custom C++ sampler).
- A marker name may open several regions in one file. They are shown in order and dedented
  together. A C++ tab can show a whole program (includes, `main`, `return 0; }`) while the
  lines that only write the test JSON stay outside the markers.
- Code outside the markers stays out of sight. Comments inside the markers follow the prose rules
  too. Every name a snippet uses is defined in a snippet shown earlier on the same page (imports,
  helpers, scene paths, seeds). Under each tab, the page links the whole example file at the
  release tag.
- Snippet lines fit 84 columns, the width of the content column. The docs example
  directories have a `.clang-format` with that limit. Wrap Python by hand.
- A missing marker is a build error. Plain `code-block` is for shell, CMake, YAML and TOML.
  A Python or C++ block that no test runs does not belong on a page.

## Examples and their tests

- Put examples in a directory per docs section, for example `examples/robots/navigation/`,
  with a `.py` and a `.cpp` of the same stem. Both accept `--json PATH` and write the numbers
  they print as one JSON object. `examples/common/json_output.hpp` has a small writer for C++.
- Add the directory to the `CMakeLists.txt` of its parent (`add_subdirectory`), list it in
  `examples/README.md`, and register each C++ file with
  `geodex_docs_example(<target> <file>.cpp [OMPL] [ROBOTS])` from
  `examples/common/DocsExamples.cmake`. That builds it under its stem and adds a ctest.
- List the examples in a test module of your own, `docs/tests/test_<section>.py`, with
  `Example("robots/navigation/bases")` values and `pairs.check_example(example, tmp_path)`. The
  pair must agree to a relative 1e-9. The check skips keys ending in `_ms`, and `ignore=(...)`
  skips named keys, with a comment that names the reason.
- Fix the seed and the iteration budget of every plan. A time budget gives a different path
  on every run and cannot be compared.

## Figures, plots and 3D recordings

Generated assets come from `docs/tools/pages/<page>.py`, one module per page, each
with `generate(parts=(...))`. `generate.py` finds the modules by itself. Draw every figure from the
output of the documented example (`common.run_example("concepts/planning")`).

- Static figures go to `docs/<section>/figs/<page>/` through `common.figure_path`, drawn with
  `style.matplotlib_style()` (Lato from `docs/_static/fonts`, STIX Sans math, 14 pt, 9 to
  16 in wide) and embedded with `:width:` of 70 to 100 percent.
- Interactive plots use `plots.write_plotly(fig, name)`, which writes
  `docs/_static/plots/<name>.html` and a still `<name>.png`, and `style.plotly_layout()`.
  Embed with `.. plotly-figure:: <name>` and an `:alt:`. Only that page loads plotly.js.
- Short 2D animations are videos, not GIFs. `common.video_paths(name)` gives the three files
  under `docs/_static/videos/`, a VP9 `.webm`, an H.264 `.mp4` fallback and a `.png` poster. Embed with `.. video-figure:: <name>`, an `:alt:`, an optional `:width:`
  and a caption. The video plays muted and loops like an animated image.
- Robots move in the robot viewer, a three.js player under `docs/_static/robot-viewer/`.
  `robot_scene.RobotScene(name, robot, frames, camera_position=..., camera_target=...)` takes
  the configurations of a plan in the robot's joint order, one per frame at 30 frames per
  second. Add the obstacles (`scene_objects`, `box`), the traces
  (`trace_end_effector`, `trace_base`), the markers and the faint copies (`ghosts`), then
  `save()` writes `docs/_static/robot-scenes/<name>.json` and a poster `<name>.jpg` that the
  viewer draws in headless Chrome. Embed with `.. robot-scene:: <name>` and an `:alt:`. The
  scene plays when it scrolls into view. `:hero:` puts a scene that plays with the page at
  its top. `:spheres:` overlays the robot's VAMP collision spheres and needs the poster of
  `save(spheres_poster=True)`. Show the spheres at the start only where the page explains
  collision checking.
  `:spheres: hidden` loads the spheres hidden behind the Spheres button and keeps the
  mesh-only poster, as in the gallery of `robots/index`.
- The viewer's robot models live in `docs/_static/robots/<robot>/`. Build them with
  `pixi run --manifest-path docs/tools/robot_meshes/pixi.toml build [robot...]`,
  which reads the pinned descriptions and writes the kinematic tree, the meshes and the
  `NOTICE` with their sources and licenses. The `FINISHES` and `MATERIALS` tables of
  `build.py` give the parts realistic colors per robot and source link, and a robot without
  rules keeps its upstream colors. Then `pixi run docs-assets robot_models` writes
  the sphere overlays and checks every model against geodex (`robot_models:browser` repeats
  the check in the viewer itself). `vendor_three.py` in the same directory copies the pinned
  three.js files into `docs/_static/three/`.
- 3D scenes without a robot, such as a path on the sphere, play in the same viewer.
  `robot_scene.PathScene(name, frames, movers, camera_position=..., camera_target=...)` moves
  one ball per entry of `movers`, three coordinates per ball and frame. Add the objects
  (`sphere`, `cap`), the traces and the markers, then `save()`. Embed with
  `.. robot-scene:: <name>`, where `:width:` and `:aspect:` size the scene.
- Colors come from `style.py`, three categorical slots at most per figure, with direct
  labels or a legend. Obstacles are neutral grey, and obstacles on the surface of a manifold,
  such as caps on the sphere, take `SURFACE_OBSTACLE`.
- Paths are drawn along the manifold's geodesics (`common.densify`), never as straight lines
  between waypoints.

## What is committed

The docs build (locally, in CI and for Read the Docs) embeds assets but never regenerates them.
Commit every file a page loads. That covers the hand-written widgets in
`docs/_static/*.html`, the plot fragments and stills in `docs/_static/plots/`, the recordings
and stills in `docs/_static/scenes/`, the robot viewer with its scenes, posters, models and
three.js (`docs/_static/{robot-viewer,robot-scenes,robots,three}/`), the videos and posters in
`docs/_static/videos/`, the figures in `docs/<section>/figs/`, and the fonts.
`.gitignore` ignores `*.html` and `*.json` everywhere else, with narrow exceptions for the
`docs/_static` paths above. A page that loads a `.json` or another ignored type needs its own
exception, added as narrowly as the ones there. Check a new asset with
`git check-ignore <file>`, which prints the path only when git would ignore it. The inputs that
`docs-assets` downloads (under `build/docs-assets-cache`) are not committed, and `sources.toml`
pins them.

## Pinned sources

External inputs of the assets (robot descriptions, meshes, maps) are pinned.
`docs/tools/sources.toml` lists each with a fixed URL, its SHA-256 and its license, and
`fetch.fetch("<name>")` downloads it once into `build/docs-assets-cache`, checks the hash and
unpacks the archive. Do not add an input without a fixed version or one whose license forbids
showing it.

## Generated pages

- The C++ API page is generated. The `docs` task runs Doxygen and then
  `docs/tools/api_cpp.py`, which writes one Breathe directive per public entity into
  `build/docs/api/cpp/`, in the order each header declares them. A new public header needs
  its include line in `docs/api/cpp.rst`, which `docs/tests/test_api_coverage.py` checks, and
  every public entity needs a short doc comment.
- The ROS 2 pages show verbatim copies of the plugins' own files under `docs/ros2/sources/`.
  After a plugin changes, run `python docs/tools/plugin_sources.py sync` with the plugin
  checkouts present. It copies the files, records their SHA-256 and regenerates the parameter
  tables in `docs/ros2/generated/`. Then reread the paragraphs that describe the copied files.
  `plugin_sources.py check` fails on any drift.

## ROS 2 captures

The stills and videos of the ROS 2 pages are screen captures of the demos in the two plugin
repositories with RViz on mock hardware. `docs/ros2/captures/` holds the sphere cover of the
attached box, and `pixi run docs-assets ros2` draws the attached-object plot from it.

## Prose

Pages, captions and code comments use complete sentences in the active voice and the field's
own terms. They do not use em-dashes, cleft constructions, rhetorical questions or colons inside
prose sentences (end a lead-in sentence with a period, or with a comma before display math).
Numbers on a page say where they come from (the command, the seeds, the machine when it
matters).

## Landing page

`docs/index.rst` opens with the title, a row of tags and the lead paragraphs inside a
`landing-hero` container. `theme.css` sets them on a tinted panel with
`docs/_static/landing/hero.svg` on its right, which `python docs/tools/landing_hero.py`
draws. Below them, the page holds one sphinx-design card per section, with `:link:` and
`:link-type: doc`, and the section's index sits in the hidden toctree.

A card shows a looping video with `.. landing-video:: <name>` as its first content and
`:class-card: landing-card-media` on the card. The video is `docs/_static/landing/<name>.webm`
with an `.mp4` fallback and a `.jpg` poster, and until all three files exist the card shows
the `:fallback:` image.
