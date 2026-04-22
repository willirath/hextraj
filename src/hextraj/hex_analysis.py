"""Aggregation and connectivity analysis functions for hex-labelled trajectory data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import dask
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
            as extra columns in the result.  Variables with only the traj
            dimension are broadcast to (traj, obs) using broadcast_like.
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

    obs_dim_new = "__obs__"

    obs_vals = ds.coords.get(obs_dim, xr.DataArray(np.arange(ds.sizes[obs_dim]), dims=[obs_dim]))

    var_dict = {
        "to_lon": ds[lon_var],
        "to_lat": ds[lat_var],
    }

    if weight is not None:
        var_dict[weight] = ds[weight]

    if groupby_cols:
        for col in groupby_cols:
            col_arr = ds[col]
            if obs_dim not in col_arr.dims:
                col_arr = col_arr.broadcast_like(ds[lon_var])
            var_dict[col] = col_arr

    mini_ds = xr.Dataset(var_dict).assign_coords({obs_dim: obs_vals})
    mini_ds = mini_ds.rename_dims({obs_dim: obs_dim_new})
    ddf = mini_ds.to_dask_dataframe(dim_order=[traj_dim, obs_dim_new])

    # Build meta for map_partitions output: input meta + to_id + from_id columns.
    meta_in = ddf._meta
    meta = meta_in.assign(to_id=pd.Series(dtype=np.int64), from_id=pd.Series(dtype=np.int64))

    def _label(df):
        to_id = pd.Series(
            hp.label(df["to_lon"].values, df["to_lat"].values),
            index=df.index,
            name="to_id",
        )
        # obs_dim_new contains 0-based position indices; index 0 is obs step 0.
        obs0_rows = df[df[obs_dim_new] == 0]
        from_id_map = dict(zip(
            obs0_rows[traj_dim],
            hp.label(obs0_rows["to_lon"].values, obs0_rows["to_lat"].values),
        ))
        from_id = df[traj_dim].map(from_id_map).rename("from_id").astype(np.int64)
        return pd.concat([df, to_id, from_id], axis=1)

    ddf = ddf.map_partitions(_label, meta=meta)
    ddf = ddf.drop(columns=["to_lon", "to_lat", obs_dim_new])
    return ddf


def hex_counts_lazy(
    hex_ids,
    reduce_dims=None,
):
    """Lazy count-only form of hex_counts.  No geometry.

    Returns count tables without materialising geometry.  Suitable for
    streaming to parquet or zarr:
    ``hex_counts_lazy(hex_ids).to_parquet("counts.parquet")``.

    Parameters
    ----------
    hex_ids : xr.DataArray, pd.Series, or dd.Series
        Hex IDs to count.  For xr.DataArray inputs the array must contain
        int64 values produced by HexProj.label.  INVALID_HEX_ID (-1) is
        preserved as an ordinary value.
    reduce_dims : str, list[str], or None
        Dimensions to aggregate over.  ``None`` (default) and ``[]`` both
        mean *reduce all dims* — the result is a Series indexed by hex_id.
        A non-empty list specifies which dims to collapse; remaining dims
        become columns in the returned DataFrame.

        For pd.Series / dd.Series inputs this parameter is ignored.

    Returns
    -------
    dd.Series or pd.Series
        Full reduction (reduce_dims is None, [], or covers all dims):
        Series indexed by ``hex_id``, values are counts.  Returns a dask
        Series for dask-backed inputs, an eager pandas Series otherwise.
    dd.DataFrame or pd.DataFrame
        Partial reduction: DataFrame with columns ``(*keep_dims, "hex_id",
        "count")``.  Parquet/zarr-writable without further reshaping.
        Returns a dask DataFrame for dask-backed inputs, pandas otherwise.

    Notes
    -----
    INVALID_HEX_ID (-1) is preserved as an ordinary row.

    Performance is best when ``hex_ids`` is chunked along ``keep_dims``.
    Misalignment triggers a dask shuffle during aggregation.

    The lazy result is unsorted; call ``.compute().sort_index()`` or
    ``hex_counts(...)`` for a sorted, geometry-attached GeoDataFrame.
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

    all_dims = list(hex_ids.dims)
    keep_dims = [d for d in all_dims if d not in reduce_dims]

    # Full reduction: collapse to a single Series.
    if not keep_dims:
        # Wrap in dataset and convert to dask dataframe — never touch .values.
        if not dask.is_dask_collection(hex_ids):
            hex_ids = hex_ids.chunk(hex_ids.sizes)
        ds = hex_ids.to_dataset(name="__hex_id__")
        ddf = ds.to_dask_dataframe(dim_order=list(hex_ids.dims))
        counts = ddf["__hex_id__"].value_counts(sort=False)
        counts.index.name = "hex_id"
        # For numpy-backed DataArrays, compute eagerly.
        if not dask.is_dask_collection(hex_ids.data):
            counts = counts.compute()
        return counts

    # Partial reduction.
    if not dask.is_dask_collection(hex_ids):
        hex_ids = hex_ids.chunk(hex_ids.sizes)

    dim_order = keep_dims + [d for d in all_dims if d in reduce_dims]

    var_dict = {"__hex_id__": hex_ids}
    for d in keep_dims:
        coord = hex_ids.coords.get(d, xr.DataArray(np.arange(hex_ids.sizes[d]), dims=[d]))
        var_dict[d] = coord.broadcast_like(hex_ids)

    mini_ds = xr.Dataset(var_dict)
    ddf = mini_ds.to_dask_dataframe(dim_order=dim_order)
    counts = ddf.groupby(keep_dims + ["__hex_id__"]).size().rename("count")
    counts = counts.reset_index()
    counts = counts.rename(columns={"__hex_id__": "hex_id"})

    # For numpy-backed DataArrays, compute eagerly.
    if not dask.is_dask_collection(hex_ids.data):
        counts = counts.compute()

    return counts


def _attach_geometry(counts, hp):
    """Compute polygon geometry for each hex and attach to the counts table.

    Parameters
    ----------
    counts : pd.Series or pd.DataFrame (or dask equivalents)
        Output of hex_counts_lazy: Series indexed by hex_id (full reduction)
        or DataFrame with columns (*keep_dims, "hex_id", "count") (partial).
    hp : HexProj
        Used to build polygon geometries via the batched to_geodataframe path.

    Returns
    -------
    gpd.GeoDataFrame
        Eager GeoDataFrame with count and geometry columns.
        Full reduction: single-level index named "hex_id", sorted.
        Partial reduction: MultiIndex (*keep_dims, "hex_id"), sorted.
        INVALID_HEX_ID rows have geometry=None.
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
    hex_ids,
    reduce_dims=None,
    hp=None,
):
    """Count hex visits, optionally keeping a subset of dims as index levels.

    Parameters
    ----------
    hex_ids : xr.DataArray, pd.Series, or dd.Series
        Hex IDs to count.  Must contain int64 values (from HexProj.label).
        INVALID_HEX_ID (-1) is preserved as an ordinary row with
        geometry=None.
    reduce_dims : str, list[str], or None
        Dimensions to aggregate over.  ``None`` (default) and ``[]`` both
        mean *reduce all dims* — the result has a flat "hex_id" index.
        A non-empty list specifies which dims to collapse; remaining dims
        become leading levels of the MultiIndex.

        For pd.Series / dd.Series inputs this parameter is ignored.
    hp : HexProj or None
        Instance used for polygon geometry.  If None a default HexProj is
        created.

    Returns
    -------
    gpd.GeoDataFrame
        Full reduction: single-level index named "hex_id", sorted by hex_id.
        Columns: ``count`` (int64), ``geometry`` (Polygon or None).

        Partial reduction: MultiIndex (*keep_dims, "hex_id"), sorted.
        Same columns.

    Notes
    -----
    Aggregation is lazy for dask-backed inputs; the small count table is
    materialised and decorated with hex polygon geometry on return.

    INVALID_HEX_ID (-1) is preserved as an ordinary row with geometry=None.
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

    # Flatten to 1D
    from_ids = from_slice.values.ravel()
    to_ids = to_slice.values.ravel()

    # Get weights if provided
    if weight is not None:
        weight_slice = weight.isel({to_dim: to_idx})
        weights = weight_slice.values.ravel()
    else:
        weights = np.ones_like(from_ids, dtype=float)

    # Create (from_id, to_id) pairs and sum weights
    pairs = list(zip(from_ids, to_ids))
    pair_to_weight = {}
    for (f_id, t_id), w in zip(pairs, weights):
        key = (int(f_id), int(t_id))
        pair_to_weight[key] = pair_to_weight.get(key, 0.0) + float(w)

    from_ids_array = np.array([k[0] for k in pair_to_weight.keys()], dtype=np.int64)
    to_ids_array = np.array([k[1] for k in pair_to_weight.keys()], dtype=np.int64)
    counts_array = np.array(list(pair_to_weight.values()))

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
