"""Sphinx configuration for geodex documentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "geodex"

# The release the pages describe, from pyproject.toml. Every code-pair links its example file
# at this release's tag.
import tomllib as _tomllib

with open(Path(__file__).parents[1] / "pyproject.toml", "rb") as _f:
    release = _tomllib.load(_f)["project"]["version"]
version = release
geodex_source_url = f"https://github.com/utiasSTARS/geodex/blob/v{release}/"
copyright = (
    "2026, Space and Terrestrial Autonomous Robotic Systems (STARS) Lab"
)
author = "geodex contributors"

extensions = [
    "breathe",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.graphviz",
    "sphinxcontrib.mermaid",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinxcontrib.bibtex",
    "geodex_docs",
    "geodex_autodoc",
]

# Python API autodoc. The Python bindings are a compiled nanobind module, so
# autodoc imports geodex at build time and the full planning stack must be
# importable; the docs build task installs it and puts the OMPL/VAMP libraries on
# the loader path. napoleon renders the Google-style docstrings the bindings
# carry. autoclass_content "both" merges each class docstring with its __init__
# so the constructor signature and parameters appear.
autoclass_content = "both"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# nanobind writes each function's signature into the first docstring line, so
# autodoc surfaces the bound self parameter and nanobind's verbose ndarray
# annotations. Trim both for a clean rendered signature.
import re as _re

_NDARRAY_RE = _re.compile(r"numpy\.ndarray\[[^\]]*\]")
_CORE_PREFIX_RE = _re.compile(r"geodex\._geodex_core\.")
_OBJECT_REPR_RE = _re.compile(r"<[\w.]+ object at 0x[0-9a-fA-F]+>")


def _clean_signature(text):
    if not text:
        return text
    text = _NDARRAY_RE.sub("ndarray", text)
    text = _CORE_PREFIX_RE.sub("geodex.", text)
    text = _OBJECT_REPR_RE.sub("...", text)
    return text


def _strip_self(signature):
    if not signature:
        return signature
    if signature.startswith("(self, "):
        return "(" + signature[len("(self, ") :]
    if signature == "(self)":
        return "()"
    return signature


def _autodoc_process_signature(app, what, name, obj, options, signature, return_annotation):
    return _strip_self(_clean_signature(signature)), _clean_signature(return_annotation)


def setup(app):
    app.connect("autodoc-process-signature", _autodoc_process_signature)

# MathJax renders math in TeX Gyre Termes (Times family).
#
# The core bundle and the font package of MathJax 4 are vendored under
# ``docs/_static/mathjax/`` and the build does not load them from a CDN. To swap fonts,
# run ``python3 docs/tools/vendor_mathjax.py <name>`` and change the three ``termes``
# occurrences below to the new name. The ``[mathjax]`` marker in ``loader.paths`` is a
# MathJax built-in that resolves to the directory of the main JS at runtime, and it
# works from every page.
mathjax_path = "mathjax/tex-mml-chtml.js"
# The vendored bundle carries no speech rule engine (``sre/``), so speech, Braille and the
# semantic enrichment they need stay off. Screen readers still get MathML.
mathjax3_config = {
    "loader": {
        "paths": {
            "mathjax-termes": "[mathjax]/fonts/termes",
        },
    },
    "chtml": {"font": "mathjax-termes"},
    "options": {
        "enableSpeech": False,
        "enableBraille": False,
        "enableEnrichment": False,
        "enableExplorer": False,
        "menuOptions": {"settings": {"speech": False, "braille": False, "enrich": False}},
    },
}

# Bibliography
bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "alpha"
bibtex_reference_style = "author_year"

# Breathe configuration
breathe_projects = {"geodex": "../build/docs/doxygen/xml"}
breathe_default_project = "geodex"

# Graphviz: render diagrams as SVG so they stay crisp at any zoom level,
# and pass -Gdpi to tools that still emit raster.
graphviz_output_format = "svg"
graphviz_dot_args = [
    "-Gfontname=Helvetica",
    "-Nfontname=Helvetica",
    "-Efontname=Helvetica",
]

# Mermaid: force every diagram to render at its intrinsic viewBox width
# (1 viewBox unit = 1 CSS pixel) instead of stretching to width="100%".
# Without this, mermaid emits width="100%" on the SVG element, which makes
# diagrams with smaller viewBoxes scale up more than diagrams with bigger
# viewBoxes, so two class diagrams on the same page render with visibly
# different text sizes. Setting useMaxWidth=false on every diagram type
# yields a consistent per-character pixel size across the page.
#
# This dict is consumed by sphinxcontrib-mermaid and serialized into
# mermaid.initialize({...}) at page-load time. The pages load this Mermaid version.
mermaid_version = "11.12.1"
mermaid_init_config = {
    "startOnLoad": False,
    "theme": "base",
    "themeVariables": {
        "primaryColor": "#e7f0fa",
        "primaryTextColor": "#1a1a1a",
        "primaryBorderColor": "#2980b9",
        "lineColor": "#2980b9",
        "secondaryColor": "#e7f0fa",
        "tertiaryColor": "#f7fbfe",
        "background": "transparent",
        "fontFamily": "Helvetica,Arial,sans-serif",
    },
    "class": {"useMaxWidth": False},
    "classDiagram": {"useMaxWidth": False},
    "flowchart": {"useMaxWidth": False},
    "sequence": {"useMaxWidth": False},
}

# HTML theme. pydata-sphinx-theme, pinned to light mode: every figure, diagram and widget
# is drawn for a light page, so the theme switcher is left out of the navbar.
html_theme = "pydata_sphinx_theme"
html_title = "geodex"
html_show_sphinx = True
html_context = {"default_mode": "light"}
html_theme_options = {
    "navbar_end": ["navbar-icon-links"],
    "navigation_depth": 3,
    "collapse_navigation": True,
    "show_toc_level": 2,
    "show_prev_next": True,
    "secondary_sidebar_items": {"**": ["page-toc"], "index": []},
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "logo": {"text": "geodex", "image_light": "_static/logo-light.svg",
             "image_dark": "_static/logo-dark.svg"},
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/utiasSTARS/geodex",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pygeodex/",
            "icon": "fa-brands fa-python",
        },
    ],
}
html_sidebars = {"index": []}

html_static_path = ["_static"]
html_favicon = "_static/favicon.svg"
# custom.css sets the fonts and the widgets, theme.css the colors and the page chrome.
html_css_files = ["custom.css", "theme.css"]
html_js_files = ["mermaid-intrinsic-size.js", "geodex.js"]

# The snippet tests and the docs tools live under docs/ but are not pages.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "tests", "tools", "rtd",
                    "CONTRIBUTING-docs.md"]
