"""Tests for optional dask dependency.

These tests verify that hex_analysis functions work correctly when dask
is not installed, simulating a bare PyPI install.
"""

import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hextraj import HexProj
from hextraj.hex_analysis import (
    _is_dask_collection,
    _dask_series_types,
    hex_counts,
    hex_counts_lazy,
    hex_connectivity,
)


@pytest.fixture
def dask_absent(monkeypatch):
    """Simulate absent dask by setting its modules to None.

    This makes `import dask` raise ImportError, which is what a bare
    PyPI install produces.
    """
    monkeypatch.setitem(sys.modules, "dask", None)
    monkeypatch.setitem(sys.modules, "dask.array", None)
    monkeypatch.setitem(sys.modules, "dask.dataframe", None)


def test_is_dask_collection_returns_false_numpy(dask_absent):
    """_is_dask_collection returns False for a numpy array when dask absent."""
    assert _is_dask_collection(np.array([1, 2, 3])) is False


def test_is_dask_collection_returns_false_pandas(dask_absent):
    """_is_dask_collection returns False for a pandas Series when dask absent."""
    ser = pd.Series([1, 2, 3])
    assert _is_dask_collection(ser) is False


def test_dask_series_types_empty(dask_absent):
    """_dask_series_types returns empty tuple when dask absent."""
    assert _dask_series_types() == ()


def test_hex_counts_lazy_pandas_series(dask_absent):
    """hex_counts_lazy on pd.Series returns correct counts with dask absent."""
    hex_ids = pd.Series([1, 1, 2, 2, 2, 3])
    result = hex_counts_lazy(hex_ids)
    expected = pd.Series(
        [2, 3, 1],
        index=pd.Index([1, 2, 3], name="hex_id"),
        name="count",
    )
    pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())


def test_hex_counts_pandas_series(dask_absent):
    """hex_counts on pd.Series returns GeoDataFrame with dask absent."""
    hp = HexProj()
    hex_ids = pd.Series([1, 1, 2, 2, 2, 3])
    result = hex_counts(hex_ids, hp=hp)
    assert isinstance(result, pd.DataFrame)
    assert hasattr(result, "geometry")
    assert len(result) == 3
    assert list(result.index) == [1, 2, 3]
    assert list(result.columns) == ["count", "geometry"]


def test_hex_counts_xarray_dataarray(dask_absent):
    """hex_counts on numpy-backed xr.DataArray returns GeoDataFrame with dask absent."""
    hp = HexProj()
    hex_ids = xr.DataArray([1, 1, 2, 2, 3, 4], dims=["a"])
    result = hex_counts(hex_ids, hp=hp)
    assert isinstance(result, pd.DataFrame)
    assert hasattr(result, "geometry")
    assert len(result) == 4


def test_hex_connectivity_xarray_dataarray(dask_absent):
    """hex_connectivity on numpy-backed xr.DataArray returns GeoDataFrame with dask absent."""
    hp = HexProj()
    hex_ids = xr.DataArray([[1, 2], [3, 4]], dims=["a", "b"])
    result = hex_connectivity(hex_ids, from_dim="a", from_idx=0, to_dim="b", to_idx=1, hp=hp)
    assert isinstance(result, pd.DataFrame)
    assert hasattr(result, "geometry")
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["from_id", "to_id"]


def test_is_dask_collection_dask_array():
    """_is_dask_collection returns True for dask array."""
    import dask.array as da

    arr = da.from_array([1, 2, 3], chunks=3)
    assert _is_dask_collection(arr) is True


def test_dask_series_types_returns_tuple():
    """_dask_series_types returns (dd.Series,) when dask is installed."""
    import dask.dataframe as dd

    result = _dask_series_types()
    assert result == (dd.Series,)
