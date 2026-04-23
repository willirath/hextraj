import numpy as np
import pandas as pd
import pytest
import xarray as xr
import geopandas as gpd
from shapely.geometry import LineString, Polygon

from hextraj.hexproj import HexProj
from hextraj.hex_id import INVALID_HEX_ID
from hextraj.hex_analysis import hex_counts, hex_connectivity, hex_connectivity_power


@pytest.fixture
def hp():
    return HexProj(hex_size_meters=2_000_000)


@pytest.fixture
def hex_ids(hp):
    """5 trajs × 2 obs; no invalid positions."""
    lon = np.array([[0.0, 10.0], [0.0, 10.0], [20.0, 30.0], [20.0, 30.0], [5.0, 5.0]])
    lat = np.zeros((5, 2))
    return xr.DataArray(
        hp.label(lon, lat),
        dims=("traj", "obs"),
        coords={"traj": np.arange(5), "obs": np.arange(2)},
    )


@pytest.fixture
def hex_ids_invalid(hp):
    """3 trajs × 2 obs; obs=1 of trajs 0 and 2 are invalid."""
    lon = np.array([[0.0, np.nan], [10.0, 20.0], [0.0, np.nan]])
    lat = np.array([[0.0, np.nan], [0.0, 0.0], [0.0, np.nan]])
    return xr.DataArray(
        hp.label(lon, lat),
        dims=("traj", "obs"),
        coords={"traj": np.arange(3), "obs": np.arange(2)},
    )


# ---------------------------------------------------------------------------
# hex_counts — return type and columns
# ---------------------------------------------------------------------------


def test_hex_counts_returns_geodataframe(hex_ids):
    result = hex_counts(hex_ids, reduce_dims=["traj", "obs"])
    assert isinstance(result, gpd.GeoDataFrame)


def test_hex_counts_has_count_column(hex_ids):
    result = hex_counts(hex_ids, reduce_dims=["traj", "obs"])
    assert "count" in result.columns


def test_hex_counts_has_geometry_column(hex_ids):
    result = hex_counts(hex_ids, reduce_dims=["traj", "obs"])
    assert "geometry" in result.columns


# ---------------------------------------------------------------------------
# hex_counts — correct counts
# ---------------------------------------------------------------------------


def test_hex_counts_correct_counts(hex_ids):
    result = hex_counts(hex_ids, reduce_dims=["traj", "obs"])
    # Compare against numpy value_counts — don't hardcode hex collisions
    vals, cnts = np.unique(hex_ids.values, return_counts=True)
    for hid, cnt in zip(vals, cnts):
        assert result.loc[hid, "count"] == cnt


# ---------------------------------------------------------------------------
# hex_counts — index structure
# ---------------------------------------------------------------------------


def test_hex_counts_all_dims_single_level_index(hex_ids):
    result = hex_counts(hex_ids, reduce_dims=["traj", "obs"])
    assert result.index.name == "hex_id"
    assert not isinstance(result.index, pd.MultiIndex)


def test_hex_counts_partial_dims_multiindex_traj_kept(hex_ids):
    result = hex_counts(hex_ids, reduce_dims=["obs"])
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["traj", "hex_id"]


def test_hex_counts_partial_dims_multiindex_obs_kept(hex_ids):
    result = hex_counts(hex_ids, reduce_dims=["traj"])
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["obs", "hex_id"]


# ---------------------------------------------------------------------------
# hex_counts — INVALID_HEX_ID handling
# ---------------------------------------------------------------------------


def test_hex_counts_invalid_present_in_index(hex_ids_invalid):
    result = hex_counts(hex_ids_invalid, reduce_dims=["traj", "obs"])
    assert INVALID_HEX_ID in result.index


def test_hex_counts_invalid_has_none_geometry(hex_ids_invalid):
    result = hex_counts(hex_ids_invalid, reduce_dims=["traj", "obs"])
    assert result.loc[INVALID_HEX_ID, "geometry"] is None


def test_hex_counts_valid_hex_has_polygon_geometry(hex_ids, hp):
    result = hex_counts(hex_ids, reduce_dims=["traj", "obs"])
    h0 = hp.label(np.array([0.0]), np.array([0.0]))[0]
    assert isinstance(result.loc[h0, "geometry"], Polygon)


@pytest.mark.parametrize("include_invalid", [True, False])
def test_hex_counts_with_and_without_invalid(hp, include_invalid):
    h0 = hp.label(np.array([0.0]), np.array([0.0]))[0]
    h1 = hp.label(np.array([10.0]), np.array([0.0]))[0]
    if include_invalid:
        data = np.array([[h0, INVALID_HEX_ID], [h1, h1]])
    else:
        data = np.array([[h0, h0], [h1, h1]])
    ids = xr.DataArray(data, dims=("traj", "obs"))
    result = hex_counts(ids, reduce_dims=["traj", "obs"])
    if include_invalid:
        assert INVALID_HEX_ID in result.index
    else:
        assert INVALID_HEX_ID not in result.index


# ---------------------------------------------------------------------------
# hex_counts — Series input
# ---------------------------------------------------------------------------


def test_hex_counts_pd_series_returns_geodataframe(hex_ids):
    s = pd.Series(hex_ids.values.flatten().astype(np.int64))
    result = hex_counts(s)
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.index.name == "hex_id"


def test_hex_counts_pd_series_correct_counts(hex_ids):
    flat = hex_ids.values.flatten().astype(np.int64)
    s = pd.Series(flat)
    result = hex_counts(s)
    vals, cnts = np.unique(flat, return_counts=True)
    for hid, cnt in zip(vals, cnts):
        assert result.loc[hid, "count"] == cnt


# ---------------------------------------------------------------------------
# hex_connectivity — return type and columns
# ---------------------------------------------------------------------------


def test_hex_connectivity_returns_geodataframe(hex_ids):
    result = hex_connectivity(hex_ids, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1)
    assert isinstance(result, gpd.GeoDataFrame)


def test_hex_connectivity_has_count_column(hex_ids):
    result = hex_connectivity(hex_ids, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1)
    assert "count" in result.columns


def test_hex_connectivity_has_geometry_column(hex_ids):
    result = hex_connectivity(hex_ids, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1)
    assert "geometry" in result.columns


# ---------------------------------------------------------------------------
# hex_connectivity — index structure
# ---------------------------------------------------------------------------


def test_hex_connectivity_multiindex_names(hex_ids):
    result = hex_connectivity(hex_ids, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1)
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["from_id", "to_id"]


# ---------------------------------------------------------------------------
# hex_connectivity — correct counts
# ---------------------------------------------------------------------------


def test_hex_connectivity_correct_pair_counts(hex_ids):
    result = hex_connectivity(hex_ids, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1)
    from_ids = hex_ids.isel(obs=0).values.ravel()
    to_ids   = hex_ids.isel(obs=1).values.ravel()
    from collections import Counter
    expected = Counter(zip(from_ids.tolist(), to_ids.tolist()))
    for (f, t), cnt in expected.items():
        assert result.loc[(f, t), "count"] == cnt


def test_hex_connectivity_loc_forward_lookup(hex_ids, hp):
    result = hex_connectivity(hex_ids, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1)
    h0 = hp.label(np.array([0.0]), np.array([0.0]))[0]
    sub = result.loc[(h0, slice(None))]
    assert len(sub) > 0


# ---------------------------------------------------------------------------
# hex_connectivity — INVALID_HEX_ID handling
# ---------------------------------------------------------------------------


def test_hex_connectivity_invalid_as_destination(hex_ids_invalid):
    result = hex_connectivity(hex_ids_invalid, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1)
    to_ids = result.index.get_level_values("to_id")
    assert INVALID_HEX_ID in to_ids


def test_hex_connectivity_invalid_destination_none_geometry(hex_ids_invalid):
    result = hex_connectivity(hex_ids_invalid, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1)
    invalid_rows = result[result.index.get_level_values("to_id") == INVALID_HEX_ID]
    assert len(invalid_rows) > 0
    assert all(g is None for g in invalid_rows["geometry"])


def test_hex_connectivity_valid_pair_has_linestring_geometry(hex_ids, hp):
    result = hex_connectivity(hex_ids, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1)
    h0 = hp.label(np.array([0.0]), np.array([0.0]))[0]
    h1 = hp.label(np.array([10.0]), np.array([0.0]))[0]
    assert isinstance(result.loc[(h0, h1), "geometry"], LineString)


# ---------------------------------------------------------------------------
# hex_connectivity — weight kwarg
# ---------------------------------------------------------------------------


def test_hex_connectivity_weight_sums_correctly(hex_ids):
    weight = xr.DataArray(
        np.full_like(hex_ids.values, 2.0, dtype=float),
        dims=hex_ids.dims,
        coords=hex_ids.coords,
    )
    unweighted = hex_connectivity(hex_ids, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1)
    weighted   = hex_connectivity(hex_ids, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1, weight=weight)
    # uniform weight=2 → weighted count == 2× unweighted count for every pair
    for idx in unweighted.index:
        assert weighted.loc[idx, "count"] == pytest.approx(2.0 * unweighted.loc[idx, "count"])


# ---------------------------------------------------------------------------
# hex_connectivity — self-loop case
# ---------------------------------------------------------------------------


def test_hex_connectivity_self_loop_all_diagonal(hex_ids):
    result = hex_connectivity(hex_ids, from_dim="obs", from_idx=0, to_dim="obs", to_idx=0)
    for from_id, to_id in result.index:
        assert from_id == to_id


# ---------------------------------------------------------------------------
# hex_connectivity_power — return type and columns
# ---------------------------------------------------------------------------


def test_hex_connectivity_power_returns_geodataframe(hex_ids, hp):
    conn = hex_connectivity(hex_ids, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1, hp=hp)
    result = hex_connectivity_power(conn, n=1, hp=hp)
    assert isinstance(result, gpd.GeoDataFrame)


def test_hex_connectivity_power_has_probability_column(hex_ids, hp):
    conn = hex_connectivity(hex_ids, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1, hp=hp)
    result = hex_connectivity_power(conn, n=1, hp=hp)
    assert "probability" in result.columns


def test_hex_connectivity_power_has_geometry_column(hex_ids, hp):
    conn = hex_connectivity(hex_ids, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1, hp=hp)
    result = hex_connectivity_power(conn, n=1, hp=hp)
    assert "geometry" in result.columns


# ---------------------------------------------------------------------------
# hex_connectivity_power — index structure
# ---------------------------------------------------------------------------


def test_hex_connectivity_power_multiindex_structure(hex_ids, hp):
    conn = hex_connectivity(hex_ids, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1, hp=hp)
    result = hex_connectivity_power(conn, n=1, hp=hp)
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["from_id", "to_id"]


# ---------------------------------------------------------------------------
# hex_connectivity_power — n=1 should match normalized conn
# ---------------------------------------------------------------------------


def test_hex_connectivity_power_n1_same_pairs_as_conn(hex_ids, hp):
    conn = hex_connectivity(hex_ids, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1, hp=hp)
    result = hex_connectivity_power(conn, n=1, hp=hp)
    assert set(result.index) == set(conn.index)


def test_hex_connectivity_power_n1_probabilities_sum_to_one(hex_ids, hp):
    conn = hex_connectivity(hex_ids, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1, hp=hp)
    result = hex_connectivity_power(conn, n=1, hp=hp)
    from_ids = result.index.get_level_values("from_id").unique()
    for f_id in from_ids:
        total = result.loc[(f_id, slice(None)), "probability"].sum()
        assert total == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# hex_connectivity_power — condition_on_valid behavior
# ---------------------------------------------------------------------------


def test_hex_connectivity_power_condition_on_valid_removes_invalid(hex_ids_invalid, hp):
    conn = hex_connectivity(hex_ids_invalid, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1, hp=hp)
    result = hex_connectivity_power(conn, n=1, hp=hp, condition_on_valid=True)
    to_ids = result.index.get_level_values("to_id")
    assert INVALID_HEX_ID not in to_ids


def test_hex_connectivity_power_condition_on_valid_sums_to_one(hex_ids_invalid, hp):
    conn = hex_connectivity(hex_ids_invalid, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1, hp=hp)
    result = hex_connectivity_power(conn, n=1, hp=hp, condition_on_valid=True)
    from_ids = result.index.get_level_values("from_id").unique()
    for f_id in from_ids:
        total = result.loc[(f_id, slice(None)), "probability"].sum()
        assert total == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# hex_connectivity_power — n=2 and higher powers
# ---------------------------------------------------------------------------


def test_hex_connectivity_power_n2_returns_valid_geodataframe(hex_ids, hp):
    conn = hex_connectivity(hex_ids, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1, hp=hp)
    result = hex_connectivity_power(conn, n=2, hp=hp)
    assert isinstance(result, gpd.GeoDataFrame)
    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["from_id", "to_id"]
    assert "probability" in result.columns
    assert "geometry" in result.columns


# ---------------------------------------------------------------------------
# hex_connectivity_power — geometry
# ---------------------------------------------------------------------------


def test_hex_connectivity_power_valid_pair_has_linestring(hex_ids, hp):
    conn = hex_connectivity(hex_ids, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1, hp=hp)
    result = hex_connectivity_power(conn, n=1, hp=hp)
    # Find a valid (non-INVALID) pair
    valid_pairs = [idx for idx in result.index if idx[0] != INVALID_HEX_ID and idx[1] != INVALID_HEX_ID]
    if valid_pairs:
        idx = valid_pairs[0]
        geom = result.loc[idx, "geometry"]
        assert isinstance(geom, LineString)


def test_hex_connectivity_power_invalid_endpoint_has_none_geometry(hex_ids_invalid, hp):
    conn = hex_connectivity(hex_ids_invalid, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1, hp=hp)
    result = hex_connectivity_power(conn, n=1, hp=hp)
    # Find INVALID endpoints
    invalid_rows = result[(result.index.get_level_values("from_id") == INVALID_HEX_ID) |
                          (result.index.get_level_values("to_id") == INVALID_HEX_ID)]
    if len(invalid_rows) > 0:
        assert all(g is None for g in invalid_rows["geometry"])


# ---------------------------------------------------------------------------
# hex_connectivity_power — sparsity (zero-probability entries dropped)
# ---------------------------------------------------------------------------


def test_hex_connectivity_power_zero_probability_pairs_absent(hex_ids, hp):
    conn = hex_connectivity(hex_ids, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1, hp=hp)
    result = hex_connectivity_power(conn, n=1, hp=hp)
    assert all(result["probability"] > 0)


# ---------------------------------------------------------------------------
# hex_connectivity_power — n=0 edge case
# ---------------------------------------------------------------------------


def test_hex_connectivity_power_n0(hex_ids, hp):
    conn = hex_connectivity(hex_ids, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1, hp=hp)
    result = hex_connectivity_power(conn, n=0, hp=hp)
    # n=0 should return identity matrix: from_hex == to_hex has probability 1.0, all others 0.0
    from_ids = result.index.get_level_values("from_id").unique()
    for f_id in from_ids:
        # For this from_id, only (f_id, f_id) should be present with probability 1.0
        pairs = result.loc[f_id].index if isinstance(result.loc[f_id].index, pd.Index) else pd.Index([result.loc[f_id].name])
        # Handle case where result.loc[f_id] is a Series (single pair) vs DataFrame (multiple pairs)
        if isinstance(result.loc[f_id], pd.Series):
            to_id = result.loc[f_id].name
            prob = result.loc[(f_id, to_id), "probability"]
            assert to_id == f_id
            assert prob == pytest.approx(1.0)
        else:
            # Multiple pairs for this from_id (shouldn't happen for identity matrix)
            to_ids = result.loc[f_id].index.get_level_values("to_id")
            for to_id in to_ids:
                prob = result.loc[(f_id, to_id), "probability"]
                if to_id == f_id:
                    assert prob == pytest.approx(1.0)
                else:
                    # Zero probability pairs should be dropped, so this shouldn't occur
                    assert False, f"Unexpected non-diagonal pair ({f_id}, {to_id}) in n=0 result"


# ---------------------------------------------------------------------------
# hex_connectivity_power — condition_on_valid drops invalid column
# ---------------------------------------------------------------------------


def test_hex_connectivity_power_condition_drops_invalid_column(hex_ids_invalid, hp):
    conn = hex_connectivity(hex_ids_invalid, from_dim="obs", from_idx=0, to_dim="obs", to_idx=1, hp=hp)
    result = hex_connectivity_power(conn, n=1, hp=hp, condition_on_valid=True)
    # After condition_on_valid=True, no INVALID_HEX_ID should appear in either index level
    from_ids = result.index.get_level_values("from_id")
    to_ids = result.index.get_level_values("to_id")
    assert INVALID_HEX_ID not in from_ids
    assert INVALID_HEX_ID not in to_ids
