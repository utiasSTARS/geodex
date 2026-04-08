"""Sphinx configuration for geodex documentation."""

project = "geodex"
copyright = "2026, geodex contributors"
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
