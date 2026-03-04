# M5: Connectivity and Dask recipes — implementation plan

## Status

M5 partial: `label()` and basic aggregation notebook done. Three items remain.

## Background: what is the O(N_traj × N_hex²) problem?

The old tutorial computed a hex-to-hex transition matrix as:

    (hex_labels.isel(obs=0) == unique_hex_labels_source)   # (N_traj, N_hex)
    & (hex_labels.isel(obs=-1) == unique_hex_labels_dest)  # (N_traj, N_hex)
    .sum("traj")                                            # (N_hex, N_hex)

The intermediate array has shape (N_traj, N_hex, N_hex). This is the scaling
problem. At 10 km resolution over the North Sea (~800 hexes), a 5k-trajectory
dataset produces a 3.2B-element boolean array.

## The efficient recipe (no new library code needed)

The fix is trivial once you have int64 hex IDs from `hp.label()`:

    from_ids = hp.label(lon[:, 0],  lat[:, 0])   # (N_traj,) int64
    to_ids   = hp.label(lon[:, -1], lat[:, -1])  # (N_traj,) int64

    import pandas as pd
    idx = pd.MultiIndex.from_arrays([from_ids, to_ids], names=["from_id", "to_id"])
    od_counts = pd.Series(1, index=idx).groupby(level=[0, 1]).sum()  # OD (Origin-Destination) counts

This is O(N_traj log N_traj). Memory is O(N_traj) not O(N_traj × N_hex²).

Convert to GeoDataFrame for visualisation:

    od_gdf = hp.edges_geodataframe(
        od_counts.index.get_level_values("from_id"),
        od_counts.index.get_level_values("to_id"),
        weight=od_counts.values,
    )

No new API method is warranted. This is user-side code.

## Dask recipes

`hp.label()` materialises at the pyproj boundary (`np.asarray()` at entry).
Dask input arrays are always computed before labelling.

The correct Dask recipe is `map_blocks`:

    import dask.array as da
    hex_ids = da.map_blocks(hp.label, lon_dask, lat_dask, dtype=np.int64)

This labels chunk-by-chunk, keeping intermediate data out of memory.
Aggregation (groupby) runs on materialised numpy arrays.

## Tasks

### Task 1: Modernise `hextraj_tutorial.ipynb`

- Replace `lon_lat_to_hex_AoS` / `hex_AoS_to_SoA` / `apply_ufunc` with `hp.label()`
- Replace boolean-broadcast connectivity with pandas groupby recipe (above)
- Show `map_blocks` recipe for dask labelling
- Load real trajectory data from the package: `importlib.resources.files('hextraj').joinpath('data/trajs/nwshelf.nc')` — 5000 trajectories × 20 obs, lon/lat/time/temperature/salinity over the NW Shelf
- Show `od_counts` → `hp.edges_geodataframe()` → plot
- The tutorial should no longer require a dask distributed Client setup

Deliverable: revised notebook, no library changes.

### Task 2: Add OD recipe to `hex_aggregation.ipynb`

The notebook already shows visit counts (choropleth) and per-route OD edges.
Add one section showing the general trajectory OD pattern:
label start/end positions of synthetic tracks, count pairs with pandas groupby,
plot weighted edges with `hp.edges_geodataframe()` over the choropleth.

Deliverable: one new notebook section (4–6 cells).

### Task 3: `scaling_benchmarks.ipynb` (new notebook)

Structure:
1. Setup: HexProj at varying resolutions, synthetic trajectory arrays.
2. Benchmark A — O(N_hex²) vs O(N_traj): time the two approaches as a function
   of N_hex (by varying `hex_size_meters`). N_hex values: 50, 100, 200, 400, 800.
   N_traj fixed at 10,000.
3. Benchmark B — labelling throughput: time `hp.label()` as a function of N_traj
   (1k to 1M). Show dask `map_blocks` wall time vs n_workers (2, 4, 8).
4. Plots: log-log axes. Each benchmark as a self-contained section.

Deliverable: new notebook at `notebooks/scaling_benchmarks.ipynb`.

## What is NOT in scope

- No new methods on `HexProj`.
- No changes to `hexproj.py`, `hex_id.py`, or tests.
- No scipy.sparse or networkx integration (user can do this from `od_counts`).
- No CI changes.

## Assessment of original roadmap items

**"Revisit connectivity computation scaling (O(N_traj × N_hex²))"**:
→ Solved by recipe in Tasks 1 and 2. No library code change needed.

**"Update tutorial with improved Dask-native recipes"**:
→ Task 1. Scope is modernising to `hp.label()` + `map_blocks` pattern.
  The tutorial currently refers to a missing data file — that must be fixed too.

**"Scaling benchmarks notebook"**:
→ Task 3. Straightforward; confirms the recipe scales as claimed.
