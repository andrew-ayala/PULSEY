"""Sphinx configuration for the PULSEY documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "PULSEY"
copyright = "2025, Andrew Ayala"
author = "Andrew Ayala"
release = "0.0.8"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Mock heavier optional/scientific imports so Read the Docs can build the API
# pages without compiling or importing the full runtime stack.
autodoc_mock_imports = [
    "jax",
    "jaxoplanet",
    "matplotlib",
    "IPython",
    "tqdm",
    "PIL",
    "s2fft",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- Options for HTML output -------------------------------------------------

html_theme = "alabaster"
html_static_path = []
