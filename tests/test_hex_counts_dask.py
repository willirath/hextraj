"""Tests for hex_counts and hex_counts_lazy with dask-backed inputs.

Patterned on tests/test_hex_connectivity_dask.py.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import dask.array as da
import dask.dataframe as dd
import geopandas as gpd

from hextraj.hexproj import HexProj
from hextraj.hex_id import INVALID_HEX_ID
from hextraj.hex_analysis import hex_counts, hex_counts_lazy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hp():
    return HexProj(hex_size_meters=2_000_000)


@pytest.fixture(params=[(2, 3), (3, 3), (6, 1)])
def chunks(request):
    """(chunk_traj, chunk_obs) pairs to parametrize chunk-independence tests."""
    return request.param


@pytest.fixture
def hex_ids_dask(hp, chunks):
    """6 traj × 3 obs dask-backed hex-ID DataArray.

    NaN-injected at (traj=0, obs=1) and (traj=3, obs=2) so INVALID_HEX_ID
    appears.  Labels via xr.apply_ufunc — the canonical lazy path.
    """
    rng = np.random.default_rng(42)
    n_traj, n_obs = 6, 3
    chunk_traj, chunk_obs = chunks

    lon_np = rng.uniform(-10, 10, size=(n_traj, n_obs)).astype(np.float64)
    lat_np = rng.uniform(-5, 5, size=(n_traj, n_obs)).astype(np.float64)
    lon_np[0, 1] = np.nan
    lat_np[0, 1] = np.nan
    lon_np[3, 2] = np.nan
    lat_np[3, 2] = np.nan

    lon = xr.DataArray(
        da.from_array(lon_np, chunks=(chunk_traj, chunk_obs)),
        dims=("traj", "obs"),
    )
    lat = xr.DataArray(
        da.from_array(lat_np, chunks=(chunk_traj, chunk_obs)),
        dims=("traj", "obs"),
    )
    return xr.apply_ufunc(
        hp.label, lon, lat, dask="parallelized", output_dtypes=[np.int64]
    )


@pytest.fixture
def hex_ids_numpy(hex_ids_dask):
    """Eager (numpy-backed) equivalent — oracle for correctness tests."""
    return hex_ids_dask.compute()


# ---------------------------------------------------------------------------
# Correctness / type
# ---------------------------------------------------------------------------


def test_returns_geodataframe_for_dask_input(hex_ids_dask, hp):
    """hex_counts on a dask-backed DataArray must return a GeoDataFrame."""
    result = hex_counts(hex_ids_dask, hp=hp)
    assert isinstance(result, gpd.GeoDataFrame), (
        f"Expected gpd.GeoDataFrame, got {type(result)}"
    )


def test_matches_numpy_full_reduction(hex_ids_dask, hex_ids_numpy, hp):
    """Full-reduction dask result must match the numpy oracle (reduce_dims=None)."""
    result_dask = hex_counts(hex_ids_dask, hp=hp).sort_index()
    result_numpy = hex_counts(hex_ids_numpy, hp=hp).sort_index()
    pd.testing.assert_series_equal(result_dask["count"], result_numpy["count"])


def test_matches_numpy_partial_reduction_obs_kept(hex_ids_dask, hex_ids_numpy, hp):
    """Partial reduction over 'traj' (keep obs) must match numpy oracle."""
    result_dask = hex_counts(hex_ids_dask, reduce_dims=["traj"], hp=hp).sort_index()
    result_numpy = hex_counts(hex_ids_numpy, reduce_dims=["traj"], hp=hp).sort_index()
    pd.testing.assert_series_equal(result_dask["count"], result_numpy["count"])


def test_matches_numpy_partial_reduction_traj_kept(hex_ids_dask, hex_ids_numpy, hp):
    """Partial reduction over 'obs' (keep traj) must match numpy oracle."""
    result_dask = hex_counts(hex_ids_dask, reduce_dims=["obs"], hp=hp).sort_index()
    result_numpy = hex_counts(hex_ids_numpy, reduce_dims=["obs"], hp=hp).sort_index()
    pd.testing.assert_series_equal(result_dask["count"], result_numpy["count"])


# ---------------------------------------------------------------------------
# INVALID_HEX_ID preservation
# ---------------------------------------------------------------------------


def test_invalid_hex_id_preserved(hex_ids_dask, hp):
    """INVALID_HEX_ID (-1) must appear in the index with geometry=None."""
    result = hex_counts(hex_ids_dask, hp=hp)
    # Handle both single-level index and MultiIndex
    if isinstance(result.index, pd.MultiIndex):
        all_hex_ids = result.index.get_level_values("hex_id")
        assert INVALID_HEX_ID in all_hex_ids, (
            f"Expected INVALID_HEX_ID in MultiIndex 'hex_id' level, "
            f"got {all_hex_ids.unique().tolist()}"
        )
        invalid_rows = result[all_hex_ids == INVALID_HEX_ID]
    else:
        assert INVALID_HEX_ID in result.index, (
            f"Expected INVALID_HEX_ID in index, got {result.index.tolist()}"
        )
        invalid_rows = result.loc[[INVALID_HEX_ID]]
    assert all(g is None for g in invalid_rows["geometry"]), (
        "Expected geometry=None for all INVALID_HEX_ID rows"
    )


# ---------------------------------------------------------------------------
# Series inputs
# ---------------------------------------------------------------------------


def test_dd_series_input(hp):
    """dd.Series input must produce correct counts, index name 'hex_id', INVALID preserved."""
    h1 = hp.label(np.array([0.0]), np.array([0.0]))[0]
    h2 = hp.label(np.array([30.0]), np.array([0.0]))[0]
    s = dd.from_pandas(
        pd.Series([h1, h2, np.int64(INVALID_HEX_ID), h1], dtype=np.int64),
        npartitions=2,
    )
    result = hex_counts(s, hp=hp)
    assert isinstance(result, gpd.GeoDataFrame), (
        f"Expected GeoDataFrame, got {type(result)}"
    )
    assert result.index.name == "hex_id", (
        f"Expected index name 'hex_id', got {result.index.name!r}"
    )
    assert INVALID_HEX_ID in result.index, "INVALID_HEX_ID must be preserved"
    assert result.loc[INVALID_HEX_ID, "geometry"] is None, (
        "INVALID_HEX_ID row must have geometry=None"
    )
    assert result.loc[h1, "count"] == 2, f"h1 should have count=2, got {result.loc[h1, 'count']}"
    assert result.loc[h2, "count"] == 1, f"h2 should have count=1, got {result.loc[h2, 'count']}"


# ---------------------------------------------------------------------------
# Chunk independence
# ---------------------------------------------------------------------------


def test_chunk_independence(hex_ids_dask, hex_ids_numpy, hp):
    """Result must be identical regardless of dask chunk layout (parametrized via chunks fixture)."""
    result_dask = hex_counts(hex_ids_dask, hp=hp).sort_index()
    result_numpy = hex_counts(hex_ids_numpy, hp=hp).sort_index()
    pd.testing.assert_series_equal(result_dask["count"], result_numpy["count"])


# ---------------------------------------------------------------------------
# Custom dim names
# ---------------------------------------------------------------------------


def test_custom_dim_names(hp):
    """hex_counts must work with arbitrary dim names; must not hardcode 'traj'/'obs'."""
    rng = np.random.default_rng(7)
    n_p, n_t = 6, 3
    lon_np = rng.uniform(-10, 10, size=(n_p, n_t)).astype(np.float64)
    lat_np = rng.uniform(-5, 5, size=(n_p, n_t)).astype(np.float64)

    lon = xr.DataArray(
        da.from_array(lon_np, chunks=(2, n_t)), dims=("particle", "time")
    )
    lat = xr.DataArray(
        da.from_array(lat_np, chunks=(2, n_t)), dims=("particle", "time")
    )
    hex_ids = xr.apply_ufunc(
        hp.label, lon, lat, dask="parallelized", output_dtypes=[np.int64]
    )
    hex_ids_np = hex_ids.compute()

    # Full reduction — must succeed and match numpy oracle
    r_full_dask = hex_counts(hex_ids, hp=hp).sort_index()
    r_full_np = hex_counts(hex_ids_np, hp=hp).sort_index()
    pd.testing.assert_series_equal(r_full_dask["count"], r_full_np["count"])

    # Partial — reduce "time", keep "particle"
    r_part_dask = hex_counts(hex_ids, reduce_dims=["time"], hp=hp).sort_index()
    r_part_np = hex_counts(hex_ids_np, reduce_dims=["time"], hp=hp).sort_index()
    pd.testing.assert_series_equal(r_part_dask["count"], r_part_np["count"])

    # Partial — reduce "particle", keep "time"
    r_part2_dask = hex_counts(hex_ids, reduce_dims=["particle"], hp=hp).sort_index()
    r_part2_np = hex_counts(hex_ids_np, reduce_dims=["particle"], hp=hp).sort_index()
    pd.testing.assert_series_equal(r_part2_dask["count"], r_part2_np["count"])


# ---------------------------------------------------------------------------
# New defaults
# ---------------------------------------------------------------------------


def test_reduce_dims_none_reduces_all(hex_ids_dask, hp):
    """reduce_dims=None (default) must reduce all dims → single-level 'hex_id' index."""
    result = hex_counts(hex_ids_dask, hp=hp)
    assert not isinstance(result.index, pd.MultiIndex), (
        "Full reduction must produce a flat 'hex_id' index, not a MultiIndex"
    )
    assert result.index.name == "hex_id", (
        f"Index name must be 'hex_id', got {result.index.name!r}"
    )


def test_reduce_dims_empty_list_reduces_all(hex_ids_dask, hp):
    """reduce_dims=[] must be identical to reduce_dims=None (both mean 'reduce all')."""
    result_none = hex_counts(hex_ids_dask, hp=hp).sort_index()
    result_empty = hex_counts(hex_ids_dask, reduce_dims=[], hp=hp).sort_index()
    pd.testing.assert_frame_equal(result_none[["count"]], result_empty[["count"]])


# ---------------------------------------------------------------------------
# hex_counts_lazy
# ---------------------------------------------------------------------------


def test_hex_counts_lazy_returns_dd_series_full_reduction(hex_ids_dask):
    """hex_counts_lazy with no reduce_dims must return a dd.Series (not computed)."""
    result = hex_counts_lazy(hex_ids_dask)
    assert isinstance(result, dd.Series), (
        f"Expected dd.Series for full reduction, got {type(result)}"
    )


def test_hex_counts_lazy_returns_dd_dataframe_partial_reduction(hex_ids_dask):
    """hex_counts_lazy with reduce_dims=['obs'] must return a dd.DataFrame with correct columns."""
    result = hex_counts_lazy(hex_ids_dask, reduce_dims=["obs"])
    assert isinstance(result, dd.DataFrame), (
        f"Expected dd.DataFrame for partial reduction, got {type(result)}"
    )
    assert set(result.columns) == {"traj", "hex_id", "count"}, (
        f"Expected columns {{'traj', 'hex_id', 'count'}}, got {set(result.columns)}"
    )


def test_hex_counts_lazy_graph_nonempty(hex_ids_dask):
    """hex_counts_lazy must return a dask collection with a non-empty task graph."""
    result = hex_counts_lazy(hex_ids_dask)
    assert len(result.__dask_graph__()) > 0, (
        "Expected a non-empty dask graph (result should be lazy)"
    )


def test_preserves_coord_values_in_multiindex(hp):
    """Non-trivial coordinate values on kept dims must appear in the output MultiIndex."""
    rng = np.random.default_rng(42)
    n_traj, n_obs = 6, 3
    traj_coords = np.array([100, 200, 300, 400, 500, 600])
    obs_coords = np.array([10, 20, 30])

    lon_np = rng.uniform(-10, 10, size=(n_traj, n_obs)).astype(np.float64)
    lat_np = rng.uniform(-5, 5, size=(n_traj, n_obs)).astype(np.float64)

    lon = xr.DataArray(
        da.from_array(lon_np, chunks=(2, n_obs)),
        dims=("traj", "obs"),
        coords={"traj": traj_coords, "obs": obs_coords},
    )
    lat = xr.DataArray(
        da.from_array(lat_np, chunks=(2, n_obs)),
        dims=("traj", "obs"),
        coords={"traj": traj_coords, "obs": obs_coords},
    )
    hex_ids = xr.apply_ufunc(
        hp.label, lon, lat, dask="parallelized", output_dtypes=[np.int64]
    )

    result = hex_counts(hex_ids, reduce_dims=["obs"], hp=hp)
    assert isinstance(result.index, pd.MultiIndex), (
        "Expected MultiIndex for partial reduction"
    )
    traj_level = set(result.index.get_level_values("traj").tolist())
    expected = set(traj_coords.tolist())
    assert traj_level == expected, (
        f"Expected traj coord values {expected}, got {traj_level}. "
        "Coord values must be preserved, not replaced with 0-based integers."
    )


# ---------------------------------------------------------------------------
# Regression guards
# ---------------------------------------------------------------------------


def test_does_not_call_values_on_input(hex_ids_dask, hp, monkeypatch):
    """hex_counts must not call .values on a dask-backed DataArray input.

    Patches xr.DataArray.values as a property that raises on the class level —
    the only reliable way to intercept property access without subclassing.
    hex_counts must complete without triggering it.
    """
    def _raise(self):
        raise AssertionError("hex_counts called .values on input")

    # Patch the property on the class; restore after test via monkeypatch.
    monkeypatch.setattr(type(hex_ids_dask), "values", property(_raise), raising=True)
    # If hex_counts accesses .values it will raise AssertionError.
    result = hex_counts(hex_ids_dask, hp=hp)
    assert isinstance(result, gpd.GeoDataFrame)


def test_geometry_batched_not_per_hex(hex_ids_dask, hp, monkeypatch):
    """hex_counts must use the batched HexProj.to_geodataframe path, not hex_corners_lon_lat.

    The batched path in HexProj.to_geodataframe never calls hex_corners_lon_lat.
    Patching it to raise verifies that hex_counts does not fall through to the
    old per-hex loop in _build_counts_geodataframe.
    """
    def _raise(self, hex_tuple=None):
        raise AssertionError("hex_counts called hex_corners_lon_lat (per-hex path)")

    monkeypatch.setattr(HexProj, "hex_corners_lon_lat", _raise)
    result = hex_counts(hex_ids_dask, hp=hp)
    assert isinstance(result, gpd.GeoDataFrame)
