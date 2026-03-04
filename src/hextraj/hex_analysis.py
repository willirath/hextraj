from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import LineString

from . import redblobhex_array as redblobhex
from .hex_id import INVALID_HEX_ID, decode_hex_id
from .hexproj import HexProj


def hex_counts(
    hex_ids: xr.DataArray | pd.Series,
    reduce_dims: str | list[str] | None = None,
    hp: HexProj | None = None,
) -> gpd.GeoDataFrame:
    """Count hex visits, optionally reducing over specified dimensions.

    Args:
        hex_ids: xr.DataArray of int64 hex IDs, or pd.Series of hex IDs.
        reduce_dims: Dimension name(s) to sum over. Ignored for Series input.
                     If None and hex_ids is a DataArray, no reduction occurs.
        hp: Optional HexProj instance to compute polygon geometries.
            If None, creates a default HexProj for geometry computation.

    Returns:
        GeoDataFrame with:
        - Index: "hex_id" (if all dims reduced) or MultiIndex with unreduced dims + "hex_id"
        - Column "count": visit count
        - Column "geometry": Polygon for valid hex, None for INVALID_HEX_ID
    """
    # Use default HexProj if none provided
    if hp is None:
        hp = HexProj()

    # Handle Series input: just do value_counts directly
    if isinstance(hex_ids, pd.Series):
        counts = hex_ids.value_counts(sort=False)
        result_gdf = _build_counts_geodataframe(counts.index.values, counts.values, hp)
        result_gdf.index.name = "hex_id"
        return result_gdf

    # Handle DataArray input
    if reduce_dims is None:
        reduce_dims = []
    elif isinstance(reduce_dims, str):
        reduce_dims = [reduce_dims]

    all_dims = list(hex_ids.dims)
    keep_dims = [d for d in all_dims if d not in reduce_dims]

    if not keep_dims:
        # Reduce over all dimensions
        hex_array = hex_ids.values.ravel()
        counts = pd.Series(hex_array).value_counts(sort=False)
        result_gdf = _build_counts_geodataframe(counts.index.values, counts.values, hp)
        result_gdf.index.name = "hex_id"
        return result_gdf

    # Partial reduction: group by keep_dims
    results = []
    index_tuples = []

    for coords, group_data in hex_ids.groupby(keep_dims):
        if isinstance(coords, (int, np.integer)):
            coords = (coords,)
        hex_array = group_data.values.ravel()
        counts = pd.Series(hex_array).value_counts(sort=False)
        for hex_id, count in counts.items():
            index_tuples.append(coords + (hex_id,))
            results.append(count)

    # Build result DataFrame
    if not results:
        # Empty case
        return gpd.GeoDataFrame(
            {"count": [], "geometry": []},
            index=pd.MultiIndex.from_tuples([], names=keep_dims + ["hex_id"]),
            crs="EPSG:4326"
        )

    # Convert to proper format for MultiIndex.from_tuples (list of tuples)
    hex_ids_flat = np.array([t[-1] for t in index_tuples], dtype=np.int64)
    counts_array = np.array(results)

    result_gdf = _build_counts_geodataframe(hex_ids_flat, counts_array, hp)

    # Construct MultiIndex from the index tuples
    multi_index = pd.MultiIndex.from_tuples(
        index_tuples,
        names=keep_dims + ["hex_id"]
    )
    result_gdf.index = multi_index
    return result_gdf


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


def _build_counts_geodataframe(
    hex_ids_array: np.ndarray,
    counts_array: np.ndarray,
    hp: HexProj,
) -> gpd.GeoDataFrame:
    """Build GeoDataFrame from hex IDs and their counts.

    Args:
        hex_ids_array: 1D array of int64 hex IDs.
        counts_array: 1D array of counts.
        hp: HexProj instance for computing geometries.

    Returns:
        GeoDataFrame with index=hex_ids_array, count and geometry columns.
    """
    from shapely.geometry import Polygon

    geometries = []
    for hex_id in hex_ids_array:
        if hex_id == INVALID_HEX_ID:
            geometries.append(None)
        else:
            q, r = decode_hex_id(hex_id)
            hex_obj = redblobhex.Hex(q, r, -q - r)
            corners_lon_lat = hp.hex_corners_lon_lat(hex_obj)
            # Convert to polygon (first and last corners are identical)
            coords = [(lon, lat) for lon, lat in corners_lon_lat[:-1]]
            geometries.append(Polygon(coords))

    result_gdf = gpd.GeoDataFrame(
        {"count": counts_array, "geometry": geometries},
        index=hex_ids_array,
        crs="EPSG:4326"
    )
    return result_gdf
