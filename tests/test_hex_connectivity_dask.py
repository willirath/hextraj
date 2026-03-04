import numpy as np
import pytest
import xarray as xr
import dask.array as da
import dask.dataframe as dd

from hextraj.hexproj import HexProj
from hextraj.hex_id import INVALID_HEX_ID
from hextraj.hex_analysis import hex_connectivity_dask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hp():
    return HexProj(hex_size_meters=2_000_000)


@pytest.fixture
def ds_small():
    """Small synthetic dask-backed xr.Dataset, chunked along traj.

    Shape: 6 traj × 3 obs.  Two NaN positions to trigger INVALID_HEX_ID.
    Includes a weight variable 'w' and a traj-dimension 'release_region'.
    Chunked (2, 3) along (traj, obs).
    """
    rng = np.random.default_rng(42)

    n_traj, n_obs = 6, 3
    chunk_traj = 2

    lon_np = rng.uniform(-10, 10, size=(n_traj, n_obs)).astype(np.float64)
    lat_np = rng.uniform(-5, 5, size=(n_traj, n_obs)).astype(np.float64)

    # Inject NaNs so that at least one INVALID_HEX_ID will appear.
    lon_np[0, 1] = np.nan
    lat_np[0, 1] = np.nan
    lon_np[3, 2] = np.nan
    lat_np[3, 2] = np.nan

    lon = da.from_array(lon_np, chunks=(chunk_traj, n_obs))
    lat = da.from_array(lat_np, chunks=(chunk_traj, n_obs))

    w_np = rng.uniform(0.5, 2.0, size=(n_traj, n_obs)).astype(np.float64)
    w = da.from_array(w_np, chunks=(chunk_traj, n_obs))

    region_np = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    release_region = da.from_array(region_np, chunks=chunk_traj)

    ds = xr.Dataset(
        {
            "lon": (["traj", "obs"], lon),
            "lat": (["traj", "obs"], lat),
            "w": (["traj", "obs"], w),
            "release_region": (["traj"], release_region),
        },
        coords={"obs": np.array([10, 20, 30], dtype=np.int64)},
    )
    return ds


# ---------------------------------------------------------------------------
# test_returns_dask_dataframe
# ---------------------------------------------------------------------------


def test_returns_dask_dataframe(ds_small, hp):
    """hex_connectivity_dask must return a dask DataFrame, not a computed result."""
    result = hex_connectivity_dask(ds_small, hp)
    assert isinstance(result, dd.DataFrame), (
        f"Expected dask.dataframe.DataFrame, got {type(result)}"
    )


# ---------------------------------------------------------------------------
# test_result_is_lazy
# ---------------------------------------------------------------------------


def test_result_is_lazy(ds_small, hp):
    """The dask task graph must be non-empty before .compute() is called."""
    result = hex_connectivity_dask(ds_small, hp)
    graph_len = len(result.__dask_graph__())
    assert graph_len > 0, "Expected a non-empty dask graph (result should be lazy)"


# ---------------------------------------------------------------------------
# test_has_from_id_and_to_id_columns
# ---------------------------------------------------------------------------


def test_has_from_id_and_to_id_columns(ds_small, hp):
    """Both 'from_id' and 'to_id' must be columns (not index) in the result."""
    result = hex_connectivity_dask(ds_small, hp)
    assert "from_id" in result.columns, (
        f"'from_id' missing from columns: {list(result.columns)}"
    )
    assert "to_id" in result.columns, (
        f"'to_id' missing from columns: {list(result.columns)}"
    )


# ---------------------------------------------------------------------------
# test_has_obs_column
# ---------------------------------------------------------------------------


def test_has_obs_column(ds_small, hp):
    """The obs coordinate must be retained as a column in the result.

    After compute(), the set of obs values in the result must exactly match
    the obs coordinate values from the input dataset.  ds_small uses
    obs=[10, 20, 30] (non-trivial, non-0-based) so this test is not
    vacuously true with default integer ranges.
    """
    result = hex_connectivity_dask(ds_small, hp)
    assert "obs" in result.columns, (
        f"'obs' missing from columns: {list(result.columns)}"
    )

    result_df = result.compute()
    expected_obs_vals = set(ds_small["obs"].values.tolist())
    result_obs_vals = set(result_df["obs"].unique().tolist())
    assert result_obs_vals == expected_obs_vals, (
        f"obs column values {result_obs_vals!r} do not match the original "
        f"obs coordinate values {expected_obs_vals!r}"
    )


# ---------------------------------------------------------------------------
# test_row_count
# ---------------------------------------------------------------------------


def test_row_count(ds_small, hp):
    """Result must have n_traj × n_obs rows total (one per traj/obs combination)."""
    result_df = hex_connectivity_dask(ds_small, hp).compute()
    n_traj = ds_small.sizes["traj"]
    n_obs = ds_small.sizes["obs"]
    expected_rows = n_traj * n_obs
    assert len(result_df) == expected_rows, (
        f"Expected {expected_rows} rows (n_traj={n_traj} × n_obs={n_obs}), "
        f"got {len(result_df)}"
    )


# ---------------------------------------------------------------------------
# test_from_id_is_constant_per_traj
# ---------------------------------------------------------------------------


def test_from_id_is_constant_per_traj(ds_small, hp):
    """All rows for the same trajectory must share the same from_id (obs=0 hex)."""
    result_df = hex_connectivity_dask(ds_small, hp).compute()

    # We need a traj identifier — the result should have a traj column or index.
    # Group by traj and check that from_id is constant within each group.
    assert "traj" in result_df.columns or result_df.index.name == "traj", (
        "Expected a 'traj' column or index to identify trajectories"
    )

    traj_col = result_df["traj"] if "traj" in result_df.columns else result_df.index

    for traj_id, group in result_df.groupby(traj_col):
        unique_from_ids = group["from_id"].unique()
        assert len(unique_from_ids) == 1, (
            f"traj={traj_id} has multiple from_id values: {unique_from_ids}; "
            "from_id should be constant per trajectory (always obs=0 hex)"
        )


# ---------------------------------------------------------------------------
# test_from_id_at_obs0_matches_to_id
# ---------------------------------------------------------------------------


def test_from_id_at_obs0_matches_to_id(ds_small, hp):
    """For rows at the first obs value, from_id must equal to_id."""
    result_df = hex_connectivity_dask(ds_small, hp).compute()

    first_obs = ds_small["obs"].values[0]
    obs0_rows = result_df[result_df["obs"] == first_obs]

    assert len(obs0_rows) > 0, (
        f"No rows found with obs == {first_obs}"
    )

    mismatches = obs0_rows[obs0_rows["from_id"] != obs0_rows["to_id"]]
    assert len(mismatches) == 0, (
        f"At obs={first_obs}, from_id != to_id for {len(mismatches)} rows; "
        "from_id should equal to_id at obs=0 since both refer to the same hex"
    )


# ---------------------------------------------------------------------------
# test_with_weight_column
# ---------------------------------------------------------------------------


def test_with_weight_column(ds_small, hp):
    """Passing weight='w' must produce a 'w' column in the result."""
    result = hex_connectivity_dask(ds_small, hp, weight="w")
    assert isinstance(result, dd.DataFrame)

    assert "w" in result.columns, (
        f"Expected 'w' column in result, got columns: {list(result.columns)}"
    )

    result_df = result.compute()
    assert "w" in result_df.columns, (
        f"After compute(), expected 'w' column, got: {list(result_df.columns)}"
    )


# ---------------------------------------------------------------------------
# test_with_groupby_cols
# ---------------------------------------------------------------------------


def test_with_groupby_cols(ds_small, hp):
    """Passing groupby_cols=['release_region'] must produce that column in the result."""
    result = hex_connectivity_dask(ds_small, hp, groupby_cols=["release_region"])
    assert isinstance(result, dd.DataFrame)

    assert "release_region" in result.columns, (
        f"Expected 'release_region' column, got columns: {list(result.columns)}"
    )

    result_df = result.compute()
    assert "release_region" in result_df.columns, (
        f"After compute(), expected 'release_region' column, got: {list(result_df.columns)}"
    )


# ---------------------------------------------------------------------------
# test_invalid_hex_ids_present
# ---------------------------------------------------------------------------


def test_invalid_hex_ids_present(ds_small, hp):
    """INVALID_HEX_ID (-1) must appear in to_id when inputs contain NaNs.

    The fixture injects NaN positions at (traj=0, obs=1) and (traj=3, obs=2),
    so INVALID_HEX_ID should appear in to_id for those rows.
    """
    result_df = hex_connectivity_dask(ds_small, hp).compute()

    to_ids = result_df["to_id"]
    has_invalid = INVALID_HEX_ID in to_ids.values
    assert has_invalid, (
        "Expected INVALID_HEX_ID to appear in to_id when NaN positions "
        "are present in the input dataset"
    )


# ---------------------------------------------------------------------------
# test_groupby_cols_values_align
# ---------------------------------------------------------------------------


def test_groupby_cols_values_align(hp):
    """release_region values in the result must correspond to the correct trajectories.

    Uses a fresh fixture with a deliberately non-symmetric region assignment:
    each trajectory has a unique region value (0..5).  This makes any axis
    transposition or partition misalignment immediately visible as a wrong
    region value on a row.

    For each traj, every row in the result must carry the same release_region
    as that trajectory has in the input dataset.  Specifically:

      traj 0 → release_region 0
      traj 1 → release_region 1
      traj 2 → release_region 2
      traj 3 → release_region 3
      traj 4 → release_region 4
      traj 5 → release_region 5

    This test exposes the expand_dims alignment bug: when expand_dims is used
    to broadcast a traj-only variable to (obs, traj), the region values for a
    given partition are drawn from the wrong axis and produce wrong region
    labels for the trajectories in that partition.
    """
    rng = np.random.default_rng(99)
    n_traj, n_obs = 6, 4
    chunk_traj = 2

    lon_np = rng.uniform(-10, 10, size=(n_traj, n_obs)).astype(np.float64)
    lat_np = rng.uniform(-5, 5, size=(n_traj, n_obs)).astype(np.float64)
    lon = da.from_array(lon_np, chunks=(chunk_traj, n_obs))
    lat = da.from_array(lat_np, chunks=(chunk_traj, n_obs))

    # Each trajectory has a UNIQUE region value — no two are alike.
    # Any transposition of (traj) and (obs) axes will produce wrong values.
    region_np = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    release_region = da.from_array(region_np, chunks=chunk_traj)

    ds = xr.Dataset(
        {
            "lon": (["traj", "obs"], lon),
            "lat": (["traj", "obs"], lat),
            "release_region": (["traj"], release_region),
        },
        coords={"obs": np.array([10, 20, 30, 40], dtype=np.int64)},
    )

    result = hex_connectivity_dask(ds, hp, groupby_cols=["release_region"])
    result_df = result.compute()

    assert "traj" in result_df.columns or result_df.index.name == "traj", (
        "Expected a 'traj' column or index to identify trajectories"
    )
    traj_col = result_df["traj"] if "traj" in result_df.columns else result_df.index

    # For each traj, all rows must carry the correct region value.
    for traj_id, group in result_df.groupby(traj_col):
        expected_region = int(region_np[traj_id])
        actual_regions = group["release_region"].unique().tolist()
        assert actual_regions == [expected_region], (
            f"traj={traj_id} should have release_region={expected_region}, "
            f"but found {actual_regions}. "
            "This indicates the expand_dims broadcast misaligned region values "
            "across trajectories (obs and traj axes were transposed)."
        )


# ---------------------------------------------------------------------------
# test_custom_lon_lat_var_names
# ---------------------------------------------------------------------------


def test_custom_lon_lat_var_names(hp):
    """hex_connectivity_dask must accept custom lon/lat variable names.

    Build a small dataset with variables named 'longitude' and 'latitude'
    (not 'lon'/'lat'), call hex_connectivity_dask with lon_var and lat_var
    parameters, and assert the result is a dask DataFrame.
    """
    rng = np.random.default_rng(42)
    n_traj, n_obs = 4, 3
    chunk_traj = 2

    lon_np = rng.uniform(-10, 10, size=(n_traj, n_obs)).astype(np.float64)
    lat_np = rng.uniform(-5, 5, size=(n_traj, n_obs)).astype(np.float64)

    lon = da.from_array(lon_np, chunks=(chunk_traj, n_obs))
    lat = da.from_array(lat_np, chunks=(chunk_traj, n_obs))

    ds = xr.Dataset(
        {
            "longitude": (["traj", "obs"], lon),
            "latitude": (["traj", "obs"], lat),
        },
        coords={"obs": np.array([10, 20, 30], dtype=np.int64)},
    )

    result = hex_connectivity_dask(
        ds, hp, lon_var="longitude", lat_var="latitude"
    )
    assert isinstance(result, dd.DataFrame), (
        f"Expected dask.dataframe.DataFrame, got {type(result)}"
    )


# ---------------------------------------------------------------------------
# test_custom_traj_dim
# ---------------------------------------------------------------------------


def test_custom_traj_dim(hp):
    """hex_connectivity_dask must respect custom traj dimension names.

    Build a dataset with dimension 'particle' instead of 'traj', call with
    traj_dim='particle', and assert the result has correct row count
    (n_particle × n_obs).
    """
    rng = np.random.default_rng(42)
    n_particle, n_obs = 4, 3
    chunk_particle = 2

    lon_np = rng.uniform(-10, 10, size=(n_particle, n_obs)).astype(np.float64)
    lat_np = rng.uniform(-5, 5, size=(n_particle, n_obs)).astype(np.float64)

    lon = da.from_array(lon_np, chunks=(chunk_particle, n_obs))
    lat = da.from_array(lat_np, chunks=(chunk_particle, n_obs))

    ds = xr.Dataset(
        {
            "lon": (["particle", "obs"], lon),
            "lat": (["particle", "obs"], lat),
        },
        coords={"obs": np.array([10, 20, 30], dtype=np.int64)},
    )

    result = hex_connectivity_dask(ds, hp, traj_dim="particle")
    result_df = result.compute()

    expected_rows = n_particle * n_obs
    assert len(result_df) == expected_rows, (
        f"Expected {expected_rows} rows (n_particle={n_particle} × n_obs={n_obs}), "
        f"got {len(result_df)}"
    )


# ---------------------------------------------------------------------------
# test_custom_obs_dim
# ---------------------------------------------------------------------------


def test_custom_obs_dim(hp):
    """hex_connectivity_dask must respect custom obs dimension names.

    Build a dataset with dimension 'time' instead of 'obs', call with
    obs_dim='time', and assert the result has a 'time' column (the obs
    coordinate under its original name).
    """
    rng = np.random.default_rng(42)
    n_traj, n_time = 4, 3
    chunk_traj = 2

    lon_np = rng.uniform(-10, 10, size=(n_traj, n_time)).astype(np.float64)
    lat_np = rng.uniform(-5, 5, size=(n_traj, n_time)).astype(np.float64)

    lon = da.from_array(lon_np, chunks=(chunk_traj, n_time))
    lat = da.from_array(lat_np, chunks=(chunk_traj, n_time))

    ds = xr.Dataset(
        {
            "lon": (["traj", "time"], lon),
            "lat": (["traj", "time"], lat),
        },
        coords={"time": np.array([100, 200, 300], dtype=np.int64)},
    )

    result = hex_connectivity_dask(ds, hp, obs_dim="time")
    assert "time" in result.columns, (
        f"'time' missing from columns: {list(result.columns)}"
    )
