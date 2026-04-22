# docs/conf.py

project = "hextraj"
author = "Willi Rath"
copyright = "2026, Willi Rath"

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

# Notebook execution - use pre-executed notebooks
nb_execution_mode = "off"

# AutoAPI
autoapi_dirs = ["../src/hextraj"]
autoapi_type = "python"
autoapi_root = "api"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
suppress_warnings = ["autoapi.python_import_resolution"]

# Napoleon (Google docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "geopandas": ("https://geopandas.org/en/stable", None),
}

# Theme
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/willirath/hextraj",
    "use_repository_button": True,
    "use_issues_button": True,
    "repository_branch": "main",
    "path_to_docs": "docs",
}
html_title = "hextraj"
