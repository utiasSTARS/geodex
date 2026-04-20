"""Sphinx configuration for geodex documentation."""

project = "geodex"
copyright = (
    "2026, Space and Terrestrial Autonomous Robotic Systems (STARS) Lab"
)
author = "geodex contributors"

extensions = [
    "breathe",
    "sphinx.ext.mathjax",
    "sphinx.ext.graphviz",
    "sphinxcontrib.mermaid",
    "sphinx_tabs.tabs",
    "sphinx_togglebutton",
    "sphinxcontrib.bibtex",
]

# MathJax: render math with TeX Gyre Termes (Times family). MathJax's default
# TeX font diverges from what the project's LaTeX manuscripts render, most
# visibly on the calligraphic alphabets. Termes is the closest match we found
# across body italics, digits, and the script letters.
#
# Alternate fonts ship only in MathJax v4 (v3 carries the TeX font alone), and
# we vendor both the core bundle and the active font package under
# ``docs/_static/mathjax/`` so offline builds work without hitting jsdelivr.
# To swap fonts: run ``python3 docs/scripts/vendor_mathjax.py <name>`` and
# change the three ``termes`` occurrences below to the new name. The
# ``[mathjax]`` marker in ``loader.paths`` is a MathJax built-in that resolves
# to the directory of the main JS at runtime, so it works from every page.
mathjax_path = "mathjax/tex-mml-chtml.js"
mathjax3_config = {
    "loader": {
        "paths": {
            "mathjax-termes": "[mathjax]/fonts/termes",
        },
    },
    "chtml": {"font": "mathjax-termes"},
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
# This dict is consumed by sphinxcontrib-mermaid and serialised into
# mermaid.initialize({...}) at page-load time.
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

# HTML theme
html_theme = "sphinx_rtd_theme"
html_show_sphinx = True
html_theme_options = {
    'navigation_depth': 3,
    'collapse_navigation': True
}

html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
html_js_files = [
    'mermaid-intrinsic-size.js',
]

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
