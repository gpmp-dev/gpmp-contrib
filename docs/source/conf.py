# Configuration file for the Sphinx documentation builder.

import datetime
import os
import sys

_DOCS_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_DOCS_SOURCE_DIR, "..", ".."))
sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault("GPMP_LOG_LEVEL", "WARNING")

project = "GPmp-contrib"
current_year = datetime.date.today().year
copyright = (
    f"2022-{current_year}, CentraleSupelec"
    if current_year > 2022
    else "2022, CentraleSupelec"
)
author = "Emmanuel Vazquez"
with open(os.path.join(_REPO_ROOT, "VERSION"), encoding="utf-8") as f:
    release = f.read().strip()
version = release
language = "en"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "numpydoc",
]

templates_path = ["_templates"]
source_suffix = [".rst"]
exclude_patterns = []

pygments_style = "witchhazel.WitchHazelStyle"
pygments_dark_style = "witchhazel.WitchHazelStyle"

autosummary_generate = True
autodoc_typehints = "description"
numpydoc_class_members_toctree = False
numpydoc_show_class_members = False

html_theme = "furo"
html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_css_files = ["jupyter-sphinx.css"]

html_theme_options = {
    "source_repository": "https://github.com/gpmp-dev/gpmp-contrib/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "light_css_variables": {
        "color-brand-primary": "#003d45",
        "color-brand-content": "#027bab",
        "color-api-name": "#003d45",
        "color-api-pre-name": "#027bab",
    },
    "dark_css_variables": {
        "color-brand-primary": "#80d8e6",
        "color-brand-content": "#80d8e6",
        "color-api-name": "#80d8e6",
        "color-api-pre-name": "#9ca0a5",
    },
}

bibtex_bibfiles = ["references.bib"]
