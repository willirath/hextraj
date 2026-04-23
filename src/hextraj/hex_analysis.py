"""Aggregation and connectivity analysis functions for hex-labelled trajectory data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import dask
import dask.array as da
import dask.dataframe as dd
from shapely.geometry import LineString

from . import redblobhex_array as redblobhex
from .hex_id import INVALID_HEX_ID, decode_hex_id
from .hexproj import HexProj


def hex_connectivity_dask(
    ds,
    hp,
    weight=None,
    groupby_cols=None,
    obs_dim="obs",
    traj_dim="traj",
    lon_var="lon",
    lat_var="lat",
):
    """Build a lazy obs=0 to obs=all connectivity table: one row per (traj, obs).

    The function internally rechunks the obs dimension to -1 (full) so that
    each dask partition contains complete trajectories.  For inputs already
    chunked that way this is essentially a no-op; for obs-chunked inputs the
    rechunk happens once, upfront.

    Args:
        ds: xr.Dataset with at least two 2-D variables (traj x obs) for
            longitude and latitude, backed by dask arrays.
        hp: HexProj instance used to label lon/lat positions as hex IDs.
        weight: Optional name of a variable in ds to include as a weight
            column in the result.  The variable must have the same dimensions
            as the lon/lat arrays.
        groupby_cols: Optional list of variable names in ds to carry through
            as extra columns in the result.  Variables with fewer dims than
            the lon/lat arrays are broadcast automatically by
            ``to_dask_dataframe``.
        obs_dim: Name of the observation dimension in ds.  Defaults to "obs".
        traj_dim: Name of the trajectory dimension in ds.  Defaults to "traj".
        lon_var: Name of the longitude variable in ds.  Defaults to "lon".
        lat_var: Name of the latitude variable in ds.  Defaults to "lat".

    Returns:
        Lazy dask DataFrame with one row per (traj, obs) combination.
        Columns always include:
          - ``from_id``: hex ID at obs index 0 for each trajectory (int64)
          - ``to_id``: hex ID at the current obs position (int64)
          - obs coordinate column (named after obs_dim)
          - traj dimension column (named after traj_dim)
        Optional columns (present when the corresponding argument is given):
          - weight column (named after the weight argument)
          - one column per name in groupby_cols
        INVALID_HEX_ID (-1) appears in from_id or to_id wherever the input
        lon/lat values are NaN.
    """
    # Rechunk obs to full so each partition has all obs steps per trajectory.
    ds = ds.chunk({obs_dim: -1})

    var_dict = {"to_lon": ds[lon_var], "to_lat": ds[lat_var]}
    if weight is not None:
        var_dict[weight] = ds[weight]
    if groupby_cols:
        for col in groupby_cols:
            var_dict[col] = ds[col]

    coords = {obs_dim: ds.coords[obs_dim]} if obs_dim in ds.coords else {}
    mini_ds = xr.Dataset(var_dict, coords=coords)
    ddf = mini_ds.to_dask_dataframe(dim_order=[traj_dim, obs_dim])

    meta = ddf._meta.drop(columns=["to_lon", "to_lat"]).assign(
        to_id=pd.Series(dtype=np.int64),
        from_id=pd.Series(dtype=np.int64),
    )

    def _label(df):
        to_id = hp.label(df["to_lon"].values, df["to_lat"].values).astype(np.int64)
        obs0 = df.groupby(traj_dim, sort=False).head(1)
        from_id_map = dict(zip(
            obs0[traj_dim],
            hp.label(obs0["to_lon"].values, obs0["to_lat"].values).astype(np.int64),
        ))
        from_id = df[traj_dim].map(from_id_map).astype(np.int64)
        return df.assign(to_id=to_id, from_id=from_id).drop(columns=["to_lon", "to_lat"])

    return ddf.map_partitions(_label, meta=meta)


def hex_counts_lazy(
    hex_ids: xr.DataArray | pd.Series | dd.Series,
    reduce_dims: str | list[str] | None = None,
) -> dd.Series | dd.DataFrame | pd.Series | pd.DataFrame:
    """Count hex visits lazily, without attaching geometry.

    Companion to ``hex_counts`` for streaming to parquet or zarr:
    ``hex_counts_lazy(hex_ids).to_parquet("counts.parquet")``. The
    aggregation stays in dask.dataframe; the caller materialises when
    ready. For dask-backed inputs peak memory scales with unique hex IDs
    per partition, not total rows.

    Args:
        hex_ids: Hex IDs to count. ``xr.DataArray``, ``pd.Series``, or
            ``dd.Series`` of int64 values (as produced by
            ``HexProj.label``). INVALID_HEX_ID (-1) is preserved.
        reduce_dims: Dimensions to aggregate over. ``None`` (default) and
            ``[]`` both mean *reduce all dims*. A non-empty list collapses
            the named dims; remaining dims become columns in the returned
            DataFrame. Ignored for ``pd.Series`` / ``dd.Series`` inputs.

    Returns:
        Full reduction: a Series indexed by ``hex_id`` with count values
        (``dd.Series`` for dask-backed input, ``pd.Series`` otherwise).

        Partial reduction: a DataFrame with columns
        ``(*keep_dims, "hex_id", "count")``, parquet/zarr-writable
        without reshaping (``dd.DataFrame`` or ``pd.DataFrame``).

        Order is not guaranteed; call ``.sort_index()`` or use
        ``hex_counts`` for a sorted, geometry-attached GeoDataFrame.

    Raises:
        ValueError: When ``reduce_dims`` names a dim not on ``hex_ids``.
        TypeError: When ``hex_ids`` is not one of the accepted types.

    Notes:
        Performance is best when ``hex_ids`` is chunked along
        ``keep_dims``. Misalignment triggers a dask shuffle during
        aggregation; no silent rechunking is performed.
    """
    # Series inputs: short-circuit directly to value_counts.
    if isinstance(hex_ids, (pd.Series, dd.Series)):
        counts = hex_ids.value_counts(sort=False)
        counts.index.name = "hex_id"
        return counts

    if not isinstance(hex_ids, xr.DataArray):
        raise TypeError(
            f"hex_ids must be xr.DataArray, pd.Series, or dd.Series; got {type(hex_ids)}"
        )

    # Normalise reduce_dims.
    if isinstance(reduce_dims, str):
        reduce_dims = [reduce_dims]
    elif reduce_dims is None or len(reduce_dims) == 0:
        reduce_dims = list(hex_ids.dims)

    unknown = set(reduce_dims) - set(hex_ids.dims)
    if unknown:
        raise ValueError(
            f"reduce_dims contains dims not on hex_ids: {sorted(unknown)}. "
            f"Available dims: {list(hex_ids.dims)}"
        )

    all_dims = list(hex_ids.dims)
    keep_dims = [d for d in all_dims if d not in reduce_dims]
    is_dask_backed = dask.is_dask_collection(hex_ids)

    # Convert to dataframe (dask or pandas).
    ds = hex_ids.to_dataset(name="hex_id")
    if is_dask_backed:
        frame = ds.to_dask_dataframe(dim_order=list(hex_ids.dims))
    else:
        frame = ds.to_dataframe().reset_index()

    # Full reduction: collapse to a single Series.
    if not keep_dims:
        counts = frame["hex_id"].value_counts(sort=False)
        counts.index.name = "hex_id"
        return counts

    # Partial reduction: groupby on keep_dims and hex_id.
    counts = frame.groupby(keep_dims + ["hex_id"]).size().rename("count")
    counts = counts.reset_index()
    return counts


def _attach_geometry(counts, hp: HexProj) -> gpd.GeoDataFrame:
    """Attach hex polygon geometry to a counts Series or DataFrame.

    Computes the geometry once per unique hex via
    ``HexProj.to_geodataframe`` and broadcasts via ``reindex``.
    ``INVALID_HEX_ID`` rows carry ``geometry=None``. Result is sorted
    by index.
    """
    if dask.is_dask_collection(counts):
        counts = counts.compute()

    if isinstance(counts, pd.Series):
        hex_id_values = counts.index.to_numpy()
        unique_ids = np.unique(hex_id_values)
        geo = hp.to_geodataframe(unique_ids)
        geometries = geo.geometry.reindex(hex_id_values).values
        result = gpd.GeoDataFrame(
            {"count": counts.to_numpy(), "geometry": geometries},
            index=pd.Index(hex_id_values, name="hex_id"),
            crs="EPSG:4326",
        ).sort_index()
        return result

    # DataFrame path: columns (*keep_dims, "hex_id", "count")
    hex_id_values = counts["hex_id"].to_numpy()
    unique_ids = np.unique(hex_id_values)
    geo = hp.to_geodataframe(unique_ids)
    geometries = geo.geometry.reindex(hex_id_values).values
    keep_dims = [c for c in counts.columns if c not in ("hex_id", "count")]
    index = pd.MultiIndex.from_frame(counts[keep_dims + ["hex_id"]])
    result = gpd.GeoDataFrame(
        {"count": counts["count"].to_numpy(), "geometry": geometries},
        index=index,
        crs="EPSG:4326",
    ).sort_index()
    return result


def hex_counts(
    hex_ids: xr.DataArray | pd.Series | dd.Series,
    reduce_dims: str | list[str] | None = None,
    hp: HexProj | None = None,
) -> gpd.GeoDataFrame:
    """Count hex visits and attach polygon geometry to the result.

    Aggregation is lazy for dask-backed inputs — the small count table
    is materialised and decorated with geometry on return. For a fully
    lazy form (no geometry, streaming to parquet/zarr), use
    ``hex_counts_lazy``.

    Args:
        hex_ids: Hex IDs to count. ``xr.DataArray``, ``pd.Series``, or
            ``dd.Series`` of int64 values. INVALID_HEX_ID (-1) is
            preserved as a regular row with ``geometry=None``.
        reduce_dims: Dimensions to aggregate over. ``None`` (default) and
            ``[]`` both mean *reduce all dims*. A non-empty list
            collapses the named dims; remaining dims become leading
            levels of a MultiIndex. Ignored for ``pd.Series`` /
            ``dd.Series`` inputs.
        hp: Projection used to build polygon geometry. A default
            ``HexProj()`` is created when ``None``.

    Returns:
        GeoDataFrame with:
          - Index: ``"hex_id"`` (full reduction) or MultiIndex
            ``(*keep_dims, "hex_id")`` (partial reduction), sorted.
          - Column ``count``: int64 visit count.
          - Column ``geometry``: Polygon for valid hex IDs, ``None`` for
            INVALID_HEX_ID.

    Raises:
        ValueError: When ``reduce_dims`` names a dim not on ``hex_ids``.
        TypeError: When ``hex_ids`` is not one of the accepted types.
    """
    if hp is None:
        hp = HexProj()

    counts = hex_counts_lazy(hex_ids, reduce_dims=reduce_dims)
    return _attach_geometry(counts, hp)


def _build_edge_geometries(
    from_ids: np.ndarray,
    to_ids: np.ndarray,
    hp: HexProj,
) -> list:
    """Build LineString geometries for hex connectivity pairs.

    Args:
        from_ids: 1D array of int64 hex IDs (origins).
        to_ids: 1D array of int64 hex IDs (destinations).
        hp: HexProj instance for computing geometries.

    Returns:
        List of geometries: LineString between hex centers for valid pairs,
        None where either ID is INVALID_HEX_ID.
    """
    geometries = []
    for f_id, t_id in zip(from_ids, to_ids):
        if f_id == INVALID_HEX_ID or t_id == INVALID_HEX_ID:
            geometries.append(None)
        else:
            q_f, r_f = decode_hex_id(f_id)
            q_t, r_t = decode_hex_id(t_id)
            hex_f = redblobhex.Hex(q_f, r_f, -q_f - r_f)
            hex_t = redblobhex.Hex(q_t, r_t, -q_t - r_t)
            lon_f, lat_f = hp.hex_to_lon_lat_SoA(hex_f)
            lon_t, lat_t = hp.hex_to_lon_lat_SoA(hex_t)
            geometries.append(LineString([(lon_f, lat_f), (lon_t, lat_t)]))
    return geometries


def hex_connectivity(
    hex_ids: xr.DataArray,
    from_dim: str,
    from_idx: int,
    to_dim: str,
    to_idx: int,
    weight: xr.DataArray | None = None,
    hp: HexProj | None = None,
) -> gpd.GeoDataFrame:
    """Build connectivity matrix from hex IDs along specified dimensions.

    Args:
        hex_ids: xr.DataArray of int64 hex IDs.
        from_dim: Dimension name for origin position.
        from_idx: Index along from_dim for origin.
        to_dim: Dimension name for destination position.
        to_idx: Index along to_dim for destination.
        weight: Optional xr.DataArray of same shape as hex_ids for weighting pairs.
        hp: Optional HexProj instance to compute LineString geometries.
            If None, creates a default HexProj for geometry computation.

    Returns:
        GeoDataFrame with:
        - Index: MultiIndex of ("from_id", "to_id")
        - Column "count": pair count (or summed weights)
        - Column "geometry": LineString between centroids, or None if either ID is INVALID
    """
    # Use default HexProj if none provided
    if hp is None:
        hp = HexProj()

    # Select origin and destination slices
    from_slice = hex_ids.isel({from_dim: from_idx})
    to_slice = hex_ids.isel({to_dim: to_idx})

    is_dask = dask.is_dask_collection(hex_ids.data) or (
        weight is not None and dask.is_dask_collection(weight.data)
    )

    from_flat = from_slice.data.ravel()
    to_flat = to_slice.data.ravel()

    if weight is not None:
        w_flat = weight.isel({to_dim: to_idx}).data.ravel().astype(float)
    else:
        if is_dask:
            w_flat = da.ones_like(from_flat, dtype=float)
        else:
            w_flat = np.ones_like(from_flat, dtype=float)

    if is_dask:
        from_flat = from_flat.astype(np.int64)
        to_flat = to_flat.astype(np.int64)
        df = dd.concat(
            [
                dd.from_dask_array(from_flat, columns="from_id"),
                dd.from_dask_array(to_flat, columns="to_id"),
                dd.from_dask_array(w_flat, columns="w"),
            ],
            axis=1,
        )
        agg = df.groupby(["from_id", "to_id"])["w"].sum().compute()
    else:
        df = pd.DataFrame({
            "from_id": np.asarray(from_flat).astype(np.int64),
            "to_id": np.asarray(to_flat).astype(np.int64),
            "w": np.asarray(w_flat, dtype=float),
        })
        agg = df.groupby(["from_id", "to_id"])["w"].sum()

    from_ids_array = agg.index.get_level_values("from_id").to_numpy().astype(np.int64)
    to_ids_array = agg.index.get_level_values("to_id").to_numpy().astype(np.int64)
    counts_array = agg.to_numpy()

    geometries = _build_edge_geometries(from_ids_array, to_ids_array, hp)

    multi_index = pd.MultiIndex.from_arrays(
        [from_ids_array, to_ids_array],
        names=["from_id", "to_id"]
    )

    result_gdf = gpd.GeoDataFrame(
        {"count": counts_array, "geometry": geometries},
        index=multi_index,
        crs="EPSG:4326"
    )

    return result_gdf


def hex_connectivity_power(
    conn: gpd.GeoDataFrame,
    n: int,
    hp: HexProj,
    condition_on_valid: bool = False,
) -> gpd.GeoDataFrame:
    """Compute n-generation connectivity from a connectivity GeoDataFrame.

    Takes the connectivity matrix returned by hex_connectivity, row-normalises
    to get a transition matrix, raises to the n-th power, and returns the result
    in the same GeoDataFrame format.

    Args:
        conn: GeoDataFrame from hex_connectivity with MultiIndex ("from_id", "to_id"),
              "count" column, and "geometry" column.
        n: Power to raise the transition matrix to.
        hp: HexProj instance for reconstructing LineString geometries.
        condition_on_valid: If True, condition on staying in-domain by zeroing
                            the INVALID column and renormalising. INVALID is
                            removed from the output. Default False.

    Returns:
        GeoDataFrame with:
        - Index: MultiIndex of ("from_id", "to_id")
        - Column "probability": float in [0, 1], the (i,j) entry of T^n
        - Column "geometry": LineString between centroids, or None if either ID is INVALID
        - Zero-probability pairs dropped (sparse representation)
    """
    # Collect all unique IDs appearing in either index level
    all_ids = sorted(
        set(conn.index.get_level_values("from_id"))
        | set(conn.index.get_level_values("to_id"))
    )
    id_to_idx = {hid: i for i, hid in enumerate(all_ids)}
    N = len(all_ids)

    # Pivot conn["count"] to dense NxN array
    T = np.zeros((N, N), dtype=float)
    for (f_id, t_id), row in conn.iterrows():
        T[id_to_idx[f_id], id_to_idx[t_id]] = row["count"]

    # Ensure INVALID is present and set as self-loop
    if INVALID_HEX_ID in id_to_idx:
        inv_i = id_to_idx[INVALID_HEX_ID]
        T[inv_i, :] = 0.0
        T[inv_i, inv_i] = 1.0

    # Row-normalise to get transition matrix
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid divide-by-zero; zero rows stay zero
    T /= row_sums

    # Optionally condition on valid
    if condition_on_valid and INVALID_HEX_ID in id_to_idx:
        inv_i = id_to_idx[INVALID_HEX_ID]
        T[:, inv_i] = 0.0
        row_sums = T.sum(axis=1, keepdims=True)
        nonzero = row_sums[:, 0] > 0
        T[nonzero] /= row_sums[nonzero]
        # Remove INVALID from the ID list for output
        all_ids = [hid for hid in all_ids if hid != INVALID_HEX_ID]
        keep = [id_to_idx[hid] for hid in all_ids]
        T = T[np.ix_(keep, keep)]
        id_to_idx = {hid: i for i, hid in enumerate(all_ids)}

    # Raise to n-th power
    Tn = np.linalg.matrix_power(T, n)

    # Serialise back to sparse GeoDataFrame
    from_ids_out, to_ids_out, probs_out = [], [], []
    for i, f_id in enumerate(all_ids):
        for j, t_id in enumerate(all_ids):
            p = Tn[i, j]
            if p > 0:
                from_ids_out.append(f_id)
                to_ids_out.append(t_id)
                probs_out.append(p)

    from_ids_array = np.array(from_ids_out, dtype=np.int64)
    to_ids_array = np.array(to_ids_out, dtype=np.int64)
    geometries = _build_edge_geometries(from_ids_array, to_ids_array, hp)

    multi_index = pd.MultiIndex.from_arrays(
        [from_ids_array, to_ids_array],
        names=["from_id", "to_id"]
    )

    return gpd.GeoDataFrame(
        {"probability": probs_out, "geometry": geometries},
        index=multi_index,
        crs="EPSG:4326",
    )
