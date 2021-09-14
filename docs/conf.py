# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
_HERE = os.path.dirname(__file__)
print(_HERE)
print([f.path for f in os.scandir(_HERE) if f.is_dir()])
_ROOT_DIR = os.path.abspath(os.path.join(_HERE, '..'))
print(_ROOT_DIR)
print([f.path for f in os.scandir(_ROOT_DIR) if f.is_dir()])
_PACKAGE_DIR = os.path.abspath(os.path.join(_HERE, '../evrptwv2g'))
print(_PACKAGE_DIR)
print([f.path for f in os.scandir(_PACKAGE_DIR) if f.is_dir()])
_GITHUB_DIR = os.path.abspath('/home/runner/work/E-VRP-TW-V2G/E-VRP-TW-V2G')
_GITHUB_PACKAGE_DIR = os.path.abspath('/home/runner/work/E-VRP-TW-V2G/E-VRP-TW-V2G/evrptwv2g')

sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _PACKAGE_DIR)
sys.path.insert(0, _GITHUB_DIR)
sys.path.insert(0, _GITHUB_PACKAGE_DIR)

# import evrptwv2g


# -- Project information -----------------------------------------------------

project = 'E-VRP-TW-V2G'
copyright = '2021, Rami Ariss'
author = 'Rami Ariss'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary'
]

autosummary_generate = True

autosummary_mock_imports = []

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []  # '_static'