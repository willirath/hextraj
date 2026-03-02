# Packaging design

## Goals

- Modern, standards-compliant packaging (PEP 517/518/621)
- Single source of truth: `pyproject.toml`
- Reproducible environments via pixi lockfile
- `src/` layout to prevent accidental imports of the uninstalled package

feedback:

---

## Layout

```
hextraj/
├── src/
│   └── hextraj/
│       ├── __init__.py
│       ├── _version.py        # written by setuptools_scm
│       ├── hexproj.py
│       ├── _proj.py           # centralised pyproj handling (see below)
│       ├── redblobhex.py
│       ├── redblobhex_array.py
│       ├── aux.py
│       └── data/
│           └── trajs/
│               └── nwshelf.nc  # 1.3 MB example data
├── tests/
├── notebooks/
├── dev/
├── pyproject.toml
├── pixi.toml
├── pixi.lock
└── AGENTS.md
```

Files to delete: `setup.py`, `setup.cfg`, root-level `data/` (moved into package)

The example data (`nwshelf.nc`, 1.3 MB) is well under the 10 MB threshold and is
bundled inside the package for use in notebooks and tests via `importlib.resources`:

```python
from importlib.resources import files
nwshelf = files("hextraj.data.trajs") / "nwshelf.nc"
```

feedback:

---

## pyproject.toml

Python floor: **3.10**. Version management: **setuptools_scm** (git-tag-driven,
written to `src/hextraj/_version.py`). Release workflow via pixi task (see pixi section).

```toml
[build-system]
requires = ["setuptools>=64", "setuptools_scm[toml]>=8"]
build-backend = "setuptools.build_meta"

[tool.setuptools_scm]
version_scheme = "no-guess-dev"
write_to = "src/hextraj/_version.py"
write_to_template = '__version__ = "{version}"'
tag_regex = '^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$'

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
hextraj = ["data/**/*"]

[project]
name = "hextraj"
dynamic = ["version"]
description = "Hex labelling of trajectory data"
readme = "README.md"
license = { file = "LICENSE.txt" }
authors = [{ name = "Willi Rath", email = "wrath@geomar.de" }]
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "pyproj",
    "xarray",
    "geopandas",
    "shapely",
]

[project.optional-dependencies]
full = ["hextraj", "dask", "scipy", "cartopy"]
dev  = ["hextraj[full]", "pytest", "pytest-cov"]

[project.urls]
Homepage = "https://github.com/willirath/hextraj"
```

feedback:

### Dependency notes

`geopandas` and `shapely` are promoted to **core** dependencies. The rationale:
- After M2/M3, geopandas is the primary output format; it's not optional
- Users of this package will be in environments with scipy/dask/cartopy anyway
- Keeping the install simple beats a nuanced optional-dep matrix
- `dask`, `scipy`, `cartopy` remain in `[full]` — heavier and not always needed for basic labelling

feedback:

---

## Environment management: pixi

pixi manages conda-forge environments with a lockfile. Chosen over uv because
`pyproj`, `geopandas`, `cartopy` have C/Fortran deps that are smoother via
conda-forge than PyPI wheels.

feedback:

### pixi.toml sketch

```toml
[project]
name = "hextraj"
channels = ["conda-forge"]
platforms = ["linux-64", "osx-arm64", "osx-64"]

[dependencies]
python = ">=3.10,<3.13"
numpy = "*"
pyproj = "*"
xarray = "*"
geopandas = "*"
shapely = "*"
pip = "*"

[feature.full.dependencies]
dask = "*"
scipy = "*"
cartopy = "*"

[feature.dev.dependencies]
pytest = "*"
pytest-cov = "*"

[feature.dev.tasks]
test = "pytest"
install = "pip install -e . --no-build-isolation"

[environments]
default = { features = ["full", "dev"], solve-group = "default" }
minimal = { features = [], solve-group = "minimal" }
```

feedback:

---

## Centralised projection handling (`_proj.py`)

All pyproj interaction moves into `src/hextraj/_proj.py`. `hexproj.py` imports from
there. This makes future pyproj API changes a one-file fix and keeps `hexproj.py`
focused on hex logic.

The immediate fix for the deprecated `init=` API:

```python
# _proj.py
import pyproj

def make_transformer(projection_name, lat_origin, lon_origin):
    proj = pyproj.Proj(
        f"+proj={projection_name} +lat_0={lat_origin} +lon_0={lon_origin} "
        "+datum=WGS84 +units=m"
    )
    transformer = pyproj.Transformer.from_crs(
        "epsg:4326", proj.crs, always_xy=True
    )
    return proj, transformer
```

`always_xy=True` preserves the existing (lon, lat) argument order.

feedback:

---

## Open questions

_Resolved: Python 3.10, setuptools_scm for versioning, pixi for env and release tasks._
