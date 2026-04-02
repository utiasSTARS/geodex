"""Sphinx configuration for geodex documentation."""

project = "geodex"
copyright = "2026, geodex contributors"
author = "geodex contributors"

extensions = [
    "breathe",
    "sphinx.ext.mathjax",
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

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
