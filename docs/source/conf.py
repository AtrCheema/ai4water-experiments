# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os

project = 'ai4water-experiments'
copyright = '2022, Ather Abbas'
author = 'Ather Abbas'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
'sphinx.ext.todo',
'sphinx.ext.viewcode',
'sphinx.ext.autodoc',
'sphinx.ext.autosummary',
'sphinx.ext.doctest',
'sphinx.ext.intersphinx',
'sphinx.ext.imgconverter',
'sphinx_issues',
'sphinx.ext.mathjax',
'sphinx.ext.napoleon',
'sphinx.ext.githubpages',
"sphinx-prompt",
"sphinx_gallery.gen_gallery",
'sphinx.ext.ifconfig',
]

toggleprompt_offset_right  = 30

# specify the master doc, otherwise the build at read the docs fails
master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

sphinx_gallery_conf = {
    'backreferences_dir': 'gen_modules/backreferences',
    #'doc_module': ('sphinx_gallery', 'numpy'),
    'reference_url': {
        'sphinx_gallery': None,
    },
    'examples_dirs': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'examples'),
    'gallery_dirs': 'auto_examples',
    'compress_images': ('images', 'thumbnails'),
    'filename_pattern': 'example',

    # 'binder': {'org': 'sphinx-gallery',
    #            'repo': 'sphinx-gallery.github.io',
    #            'branch': 'master',
    #            'binderhub_url': 'https://mybinder.org',
    #            'dependencies': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.binder', 'requirements.txt'),
    #            'notebooks_dir': 'notebooks',
    #            'use_jupyter_lab': True,
    #            },
    #'show_memory': True,
    #'junit': os.path.join('sphinx-gallery', 'junit-results.xml'),
    # capture raw HTML or, if not present, __repr__ of last expression in
    # each code block
    'capture_repr': ('_repr_html_', '__repr__'),
    'matplotlib_animations': True,
    'image_srcset': ["2x"]
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
