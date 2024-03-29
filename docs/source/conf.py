# Configuration file for the Sphinx documentation builder.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../src/"))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PySLAM"
copyright = "2023, Adithya Yerramsetty"
author = "Adithya Yerramsetty"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "myst_parser",
]

# Set paths for auto-api
autoapi_dirs = ["../../src/pyslam"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
