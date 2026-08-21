# Fix the PyPI install and the dependency declarations

Issue: https://github.com/willirath/hextraj/issues/40
Branch: `fix-pypi-dependencies`

A wheel installed from PyPI cannot be imported. `hextraj/__init__.py` imports
`hex_analysis`, which imports dask at module level, and dask sits in the
`[full]` extra. `pip install "hextraj[full]"` then fails on `pyarrow`, because
the PyPI `dask` distribution ships the core scheduler only. Three further
declaration errors surfaced during the audit. All of them land in one pull
request.

Reproduce the base-install failure locally:

```shell
pixi run -e minimal python -c "import hextraj"
```

The `minimal` pixi environment carries the core dependencies and no dask, so it
stands in for a bare PyPI install. It currently raises
`ModuleNotFoundError: No module named 'dask'`.

## Work items

### 1. Make dask optional at run time

`src/hextraj/hex_analysis.py` imports `dask`, `dask.array as da`, and
`dask.dataframe as dd` at module level (lines 9 to 11). Remove those three
imports. Four call sites use them:

| Site | Use |
|---|---|
| `hex_counts_lazy` | `isinstance(hex_ids, (pd.Series, dd.Series))`, and `dask.is_dask_collection` |
| `_attach_geometry` | `dask.is_dask_collection(counts)` |
| `hex_connectivity` | `dask.is_dask_collection`, `da.ones_like`, `dd.concat`, `dd.from_dask_array` |
| `hex_connectivity_dask` | none; it reaches dask through `xr.Dataset.to_dask_dataframe` |

Add two module-level helpers that answer their question without requiring dask:

```python
def _is_dask_collection(obj):
    """Return whether obj is a dask collection, False when dask is absent."""
    try:
        import dask
    except ImportError:
        return False
    return dask.is_dask_collection(obj)


def _dask_series_types():
    """Return (dd.Series,) when dask.dataframe imports, () otherwise."""
    try:
        import dask.dataframe as dd
    except ImportError:
        return ()
    return (dd.Series,)
```

`import dask.dataframe` raises `ImportError` when pyarrow is missing, so the
same guard covers the partial install as well.

Rewrite the call sites:

- `hex_counts_lazy`: `isinstance(hex_ids, (pd.Series, *_dask_series_types()))`.
- `_attach_geometry` and `hex_connectivity`: call `_is_dask_collection`.
- `hex_connectivity`: import `dask.array as da` and `dask.dataframe as dd`
  inside the function, once, immediately after `is_dask` is computed and only
  when it is true.

The annotations on `hex_counts_lazy` and `hex_counts` name `dd.Series` and
`dd.DataFrame`. The module already carries `from __future__ import annotations`,
so those names are never evaluated at run time. Keep them, and keep mypy happy
with a guarded import:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dask.dataframe as dd
```

Behaviour must not change when dask is installed.

### 2. Declare the dask extra as `dask[dataframe]`

`dask.dataframe` needs pyarrow, and the PyPI `dask` distribution does not pull
it in. In `pyproject.toml`, write the extra as `full = ["dask[dataframe]"]`.

### 3. Declare pandas

`hex_analysis.py:6` imports pandas at module level and nothing declares it. It
resolves today only because geopandas depends on it. Add `pandas` to
`dependencies` in `pyproject.toml`.

### 4. Drop scipy, move cartopy

`scipy` is imported nowhere in `src/`, `tests/`, or `notebooks/`. Remove it from
the `full` extra in `pyproject.toml` and from `feature.full.dependencies` in
`pixi.toml`.

`cartopy` is imported by `notebooks/hex_conn_dask.ipynb` and
`notebooks/hex_analysis.ipynb` only, as `cartopy.io.shapereader`. It builds
notebooks, so it belongs with the development tooling. Remove it from the `full`
extra in `pyproject.toml`, and move it from `feature.full.dependencies` to
`feature.dev.dependencies` in `pixi.toml`. Leave the `dev` extra in
`pyproject.toml` as it stands; that extra covers the test run, not the notebook
build.

Resulting declarations:

| | core | extra |
|---|---|---|
| before | numpy, pyproj, xarray, geopandas, shapely | dask, scipy, cartopy |
| after | numpy, pyproj, pandas, xarray, geopandas, shapely | `dask[dataframe]` |

`geopandas`, `pandas`, `xarray`, and `shapely` stay core. They reflect how the
package is normally used.

### 5. Remove the dead xarray import

`src/hextraj/hexproj.py:10` imports xarray and never uses it. Remove the import.
Confirm first that no annotation or docstring example in that file needs the
name.

### 6. Documentation

- `README.md:41` and `docs/index.md:19` both say "With dask, scipy, and
  cartopy". Both now describe dask alone.
- `dev/docs/packaging.md` records the old split under "Dependency notes". Append
  a short subsection that states the corrected policy and cites issue 40. Do not
  rewrite the historical text.

## Tests

Add `tests/test_optional_dask.py`. Plain pytest functions, no classes, per
`AGENTS.md`.

Simulate the absent dependency with a fixture that sets `sys.modules["dask"]`,
`sys.modules["dask.array"]`, and `sys.modules["dask.dataframe"]` to `None`
through `monkeypatch.setitem`. A `None` entry makes `import dask` raise
`ImportError`, which is what a bare install produces.

Cover:

- `_is_dask_collection` returns False for a numpy array and for a pandas Series.
- `hex_counts_lazy` on a `pd.Series` returns the same counts it returns with
  dask installed.
- `hex_counts` on a numpy-backed `xr.DataArray` returns a GeoDataFrame.
- `hex_connectivity` on a numpy-backed `xr.DataArray` returns a GeoDataFrame.

The existing dask tests must keep passing unchanged. They are the evidence that
the dask paths still work.

## Continuous integration

`tests.yml` runs through pixi, which installs from conda-forge. There, `dask` is
a metapackage that pulls `dask-dataframe` and pyarrow. That is why this shipped:
the pixi run cannot detect a PyPI-only gap.

Add a second job to `.github/workflows/tests.yml`, named `wheel-import`, that
runs independently of the `test` job:

1. `actions/checkout@v4` with `fetch-depth: 0`, because setuptools_scm reads the
   git history.
2. `actions/setup-python@v5` with Python 3.12.
3. `pip install build`, then `python -m build --wheel`.
4. Install the wheel into a fresh virtual environment, with no extras. Import
   `hextraj` and run a smoke call that exercises the dask-free path, for example
   `hex_counts` over a `pd.Series` of labels.
5. Install the wheel with `[full]` into a second fresh virtual environment.
   Import `hextraj`, then `import dask.dataframe` to prove pyarrow arrived.

Both installs must resolve from PyPI wheels alone.

## Verification

```shell
pixi run -e minimal python -c "import hextraj; print(hextraj.HexProj())"
pixi run test
pixi run mypy
```

The first command is the regression check. It must succeed where it now fails.
