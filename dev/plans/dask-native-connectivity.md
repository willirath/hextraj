# Plan: Dask-native trajectory-to-connectivity pipeline

## Executive summary

The current `hex_connectivity` function materialises dask arrays via
`.values.ravel()`, breaking laziness. This plan designs a fully lazy pipeline
from dask-backed xarray Datasets (e.g. from Zarr) through hex labelling,
xarray-to-dask-DataFrame conversion, and lazy groupby aggregation, producing
a dask DataFrame of OD (Origin-Destination) counts that is only materialised at the user's
`.compute()` call. The goal is to encapsulate this pattern in a clean library
function that handles weights, metadata groupby keys, and normalisation.

---

## Recommended pipeline (pseudocode)

The function computes connectivity for all obs pairs so that the caller can
select or average over target observations afterwards. There is no `from_obs`
or `to_obs` parameter in the primary interface — those choices are left to the
caller via external slicing before passing the dataset in, or by requesting
obs-resolved output (see section 4).

Note that xarray dimensions and dask DataFrame columns are different concepts.
When a Dataset is converted via `.to_dask_dataframe()`, dims are flattened into
columns. The plan uses "columns" consistently when referring to the dask
DataFrame stage.

```python
def hex_connectivity(
    ds,                          # xr.Dataset with lon, lat (dask-backed)
    hp,                          # HexProj instance
    weight=None,                 # str (variable name in ds) or xr.DataArray
    groupby_cols=None,           # extra columns to keep in groupby (e.g. ["release_region"])
    obs_dim="obs",               # name of the observation dimension
    traj_dim="traj",             # name of the trajectory dimension
):
    # 1. Convert to dask DataFrame directly
    cols = ["lon", "lat"]
    if weight is not None:
        cols.append(weight)
    if groupby_cols:
        cols.extend(groupby_cols)
    ddf = ds[cols].to_dask_dataframe(dim_order=["traj", "obs"])  # dim_order controls axis ordering; when chunks are along traj, each traj-chunk maps cleanly to one DataFrame partition

    # 2. Label with map_partitions (stays lazy, operates on DataFrame columns)
    ddf["from_id"] = ddf.map_partitions(
        lambda df: hp.label(df["lon_from"], df["lat_from"]), meta=("from_id", np.int64)
    )
    ddf["to_id"] = ddf.map_partitions(
        lambda df: hp.label(df["lon_to"], df["lat_to"]), meta=("to_id", np.int64)
    )

    # 3. Lazy groupby aggregation
    group_keys = (groupby_cols or []) + ["from_id", "to_id"]
    if weight is not None:
        agg = ddf.groupby(group_keys).agg({weight: "sum"})
    else:
        agg = ddf.groupby(group_keys).size()

    # 4. Return lazy dask DataFrame (or Series)
    return agg
```

The user calls `.compute()` when ready, then optionally attaches geometry
via one of two post-compute paths described in section 6.

---

## 1. Synthetic dask-backed trajectory dataset

### Design

A test helper function builds a synthetic `xr.Dataset` with:

- `lon(traj, obs)` and `lat(traj, obs)` as dask arrays, chunked along `traj`.
- Optional metadata variables: `particle_id(traj)`, `release_region(traj)`,
  `ensemble(traj)`.
- Known hex assignments for deterministic assertions.

### Implementation

`da.random` can generate random arrays natively without going through numpy
first. The synthetic helper uses `da.random.uniform` and related functions
directly. Smoother trajectories built from a random walk would be more
realistic — this is noted as an option for tests that care about hex-cell
continuity rather than just correctness of aggregation.

```python
def make_synthetic_trajectories(
    n_traj=1000, n_obs=10, chunk_traj=100,
    lon_range=(-10, 10), lat_range=(-5, 5),
    n_regions=3, seed=42,
):
    rng = da.random.default_rng(seed)
    lon = rng.uniform(*lon_range, size=(n_traj, n_obs)).astype(np.float32)
    lat = rng.uniform(*lat_range, size=(n_traj, n_obs)).astype(np.float32)
    # Inject some NaN (beached particles)
    mask = rng.random((n_traj, n_obs)) < 0.05
    lon = da.where(mask, np.nan, lon)
    lat = da.where(mask, np.nan, lat)

    lon = lon.rechunk((chunk_traj, n_obs))
    lat = lat.rechunk((chunk_traj, n_obs))

    ds = xr.Dataset({
        "lon": (["traj", "obs"], lon),
        "lat": (["traj", "obs"], lat),
        "release_region": ("traj", rng.integers(0, n_regions, n_traj).rechunk(chunk_traj)),
    })
    return ds
```

An alternative builds smoother trajectories via a cumulative random walk
(`da.cumsum` of small increments) — useful when tests need realistic
hex-cell transitions rather than purely independent positions.

### Chunking rationale

The right chunking strategy depends on the analysis:

- **Chunk along `traj`** for trajectory-centric analyses (e.g. start-to-end
  connectivity). Each chunk contains complete trajectories, and `isel(obs=...)`
  is a cheap within-chunk slice.
- **Chunk along `obs`** for time-centric analyses (e.g. density snapshots at
  a given timestep). Each partition spans all trajectories at a subset of obs.
- **Chunk along both dimensions** for very large datasets where neither
  dimension fits in memory as a single chunk.

The function imposes no chunking requirement — it works with whatever chunking
the input Dataset carries.

---

## 2. Hex labelling timing

### The question

`hp.label(lon, lat)` calls pyproj internally, which materialises to numpy.
It cannot accept a raw dask array. Where should labelling happen?

### Options considered

| Approach | Lazy? | Chunk-aligned? | Complexity |
|----------|-------|----------------|------------|
| `xr.apply_ufunc(hp.label, ..., dask="parallelized")` | Yes | Yes | Medium — needs explicit output dtypes and shapes |
| `da.map_blocks(hp.label, lon.data, lat.data, ...)` | Yes | Yes | Low — but accesses `.data`, which is not a stable public API |
| Label inside `ddf.map_partitions(...)` | Yes | Yes | Low — operates on public DataFrame columns |
| Label before chunking (eager) | No | N/A | Defeats purpose |

### Recommendation: `ddf.map_partitions`

`ddf.map_partitions(hp.label, ...)` on the lon/lat columns of the dask
DataFrame is the recommended approach. It operates on public DataFrame columns,
stays lazy and chunk-aligned, and avoids accessing `.data` on xr.DataArray
(which is internal dask array access and not stable public API). This approach
is also consistent with what the reference heatmap notebook (`004_calculate_heatmaps.ipynb`)
does.

`xr.apply_ufunc` with `dask="parallelized"` is an alternative but adds
boilerplate for output dtype and core-dimension specifications. `da.map_blocks`
with `.data` is functional but discouraged as the primary pattern.

### When to label

Labelling happens **after** converting to dask DataFrame, inside
`map_partitions`. This is the most direct path:

1. `isel` on the Dataset to select the relevant obs positions (or pass the
   full dataset for obs-resolved output).
2. `.to_dask_dataframe()` — a metadata-only graph-rewiring step.
3. `ddf.map_partitions(hp.label, ...)` on the lon/lat columns to produce
   `from_id` and `to_id` columns.
4. `ddf.groupby(...)` for aggregation.

This avoids assembling intermediate `xr.Dataset` objects and stays in the
dask DataFrame world, where groupby is known to scale well (single sweep),
unlike xarray-based aggregation which can exhibit O(N²) or worse behaviour
for large unique-pair counts.

---

## 3. xarray to dask DataFrame conversion

### How `to_dask_dataframe()` handles chunking

`xr.Dataset.to_dask_dataframe()` maps each chunk of the underlying dask
arrays to one partition of the dask DataFrame. For a Dataset with variables
chunked as `(chunk_traj,)` along the `traj` dimension, each partition
contains `chunk_traj` rows. Chunk boundaries become partition boundaries
with no data movement.

Multiple variables in the Dataset become columns in each partition. The
conversion is a metadata operation — it rewires the dask graph without
computing anything.

**`dim_order` and partition alignment:** `to_dask_dataframe()` accepts a
`dim_order` keyword that controls which axis is the outermost (row-major)
dimension of the flattened DataFrame. When `dim_order=["traj", "obs"]`,
rows are ordered traj-major: all obs for trajectory 0 come first, then all
obs for trajectory 1, and so on. Because dask chunks are also traj-aligned
(each chunk holds a contiguous block of trajectories), each chunk maps to
exactly one partition with no cross-chunk row interleaving. This guarantee
breaks down if `dim_order` is omitted or set to `["obs", "traj"]` —
xarray may choose a different axis ordering, mixing trajectories across
partitions and forcing a shuffle. **Always pass `dim_order=["traj", "obs"]`
explicitly** (or `["traj"]` for purely traj-dimensioned variables) to make
the chunk-to-partition mapping deterministic.

### What columns to include

The xr.Dataset assembled before conversion should contain:

| Column | Source | Purpose |
|--------|--------|---------|
| `from_id` | `ddf.map_partitions(hp.label, lon_from, lat_from)` | Origin hex ID |
| `to_id` | `ddf.map_partitions(hp.label, lon_to, lat_to)` | Destination hex ID |
| `weight` | Variable name carried through from Dataset | Optional weight per pair |
| metadata columns | e.g. `release_region`, `ensemble` | Extra groupby keys |

Coordinate variables from the original Dataset (e.g. `traj`) become the
DataFrame index automatically. They can be reset if not needed for groupby.

**Forward-looking note (out of scope for this PR):** `obs` can be retained as
a regular DataFrame column — via `reset_index` or by including it explicitly
in `to_dask_dataframe` — to enable later mapping to particle age and
time-weighted connectivity aggregation. Worth keeping in mind during the
column assembly step so the door is not accidentally closed.

### Multi-dimensional case (obs-resolved)

For obs-resolved connectivity (triple index `(obs_step, from_id, to_id)`),
the Dataset is stacked `(traj, obs_step) -> event` before
`.to_dask_dataframe()`, and labelling is applied via `map_partitions` on the
resulting lon/lat columns. The stack is a graph-rewiring operation, not a
compute.

---

## 4. From/to hex ID construction

### Design: compute for all obs, let the caller slice

Rather than accepting `from_obs` / `to_obs` parameters that fix a single pair
of obs indices, the function computes connectivity across all obs (or the full
dataset as supplied). The caller slices the Dataset to the obs range of
interest before calling `hex_connectivity`, or requests obs-resolved output
and slices/averages the result afterwards. This keeps the function's scope
narrow and avoids encoding a particular analysis choice in the API.

### Start-to-end connectivity (caller pre-slices)

The caller selects obs indices externally:

```python
ds_slice = ds.isel(obs=[0, -1])   # first and last obs
result = hex_connectivity(ds_slice, hp, ...)
```

Inside the function, after `.to_dask_dataframe()`, labelling is applied via
`map_partitions` on the lon/lat columns for each obs step.

### Obs-resolved connectivity

For the `(obs_step, from_id, to_id)` pattern, the Dataset is stacked
`(traj, obs_step) -> event` before conversion to dask DataFrame. The
`from_id` at each step is the hex ID at the first obs, broadcast across all
steps:

```python
ds_stacked = ds.stack(event=("traj", "obs"))
ddf = ds_stacked.to_dask_dataframe()
ddf["hex_id"] = ddf.map_partitions(lambda df: hp.label(df["lon"], df["lat"]),
                                    meta=("hex_id", np.int64))
# from_id = hex_id at obs_step=0, repeated per traj; to_id = hex_id at current step
```

`broadcast_to` of the origin column is a zero-copy view — no data
duplication. The result is a `(from_id, to_id)` pair at each `(traj,
obs_step)` cell, ready for groupby.

### Recommendation

The primary function handles obs-resolved output (all obs pairs). Start-to-end
is the common special case achieved by pre-slicing the input Dataset to
`obs=[0, -1]`. Obs-resolved connectivity may become a separate function or a
mode flag if the output structure differs enough (triple-indexed vs
pair-indexed) to warrant it.

---

## 5. Lazy groupby aggregation

### Simple count

```python
ddf.groupby(["from_id", "to_id"]).size()
```

Dask aggregates per-partition first, then merges across partitions. Peak
memory is proportional to (unique pairs per chunk), not total trajectories.
This is the core operation.

### Weighted sum

```python
ddf.groupby(["from_id", "to_id"]).agg({"weight": "sum"})
```

Same lazy semantics. The `weight` column is carried through the dask graph
and summed per-partition before the final merge.

### Normalisation with a different denominator

Example: normalise by total trajectories per `from_hex`, not by sum of
weights. This requires a two-step aggregation:

```python
# Step 1: OD counts (lazy)
od_counts = ddf.groupby(["from_id", "to_id"]).size()

# Step 2: Marginal counts per from_id (lazy)
from_totals = ddf.groupby("from_id").size()

# Step 3: Normalise (triggers compute or stays lazy via merge)
od_probs = od_counts / from_totals
```

Steps 1 and 2 are independently lazy. Step 3 can be done lazily by joining
the two dask Series on the `from_id` index level, but in practice the
aggregated result is small enough that computing both and dividing in pandas
is simpler and equally fast. The function should return the raw counts and
let the user normalise, or accept a `normalize` kwarg that triggers
compute-and-divide.

### Multiple non-reduced dimensions (extra groupby keys)

```python
group_keys = ["release_region", "from_id", "to_id"]
ddf.groupby(group_keys).size()
```

Adding metadata columns to the groupby key is straightforward. The metadata
must be present in the DataFrame — either as a coordinate variable carried
through from xarray, or explicitly added before conversion. The function
should accept `groupby_dims` as a list of dimension/variable names to
include as additional groupby keys.

### Multiple reduced dimensions

When the input has dimensions beyond `(traj, obs)` — e.g.
`(ensemble, realization, traj, obs)` — the non-obs, non-groupby dimensions
are all reduced (summed over) in the groupby. This happens naturally: after
`isel(obs=...)` and `stack(event=(...))`, all remaining dimensions become
rows in the DataFrame, and the groupby aggregates over all of them.

The key insight: the user does not need to specify which dimensions to
reduce. Everything that is not in `groupby_dims` and not the obs dimension
gets flattened into the event axis by the stack+to_dask_dataframe step.

---

## 6. Return type

### Recommendation: return a dask DataFrame (lazy) by default

The function should return the lazy aggregated dask DataFrame (or Series).
Reasons:

1. **Laziness preserved**: the user controls when `.compute()` happens.
   For large datasets, this allows inspecting the task graph, profiling,
   or feeding the result into further dask operations.

2. **Geometry attachment and metadata enrichment**: After `.compute()`,
   the result can be enriched with geometry via two post-compute paths:

   **Option (a) — LineString geometry per OD pair (`edges_geodataframe`):**
   Connect origin and destination hex centres with a LineString. Useful for
   flow maps and directional overlays.

   ```python
   result = od_counts.compute()
   gdf = hp.edges_geodataframe(
       result.index.get_level_values("from_id"),
       result.index.get_level_values("to_id"),
       count=result.values,
   )
   ```

   **Option (b) — Destination hex polygon as active geometry:**
   Join the computed OD counts to a hex polygon GeoDataFrame so each row
   carries the destination hex polygon as its active geometry. This is more
   useful for choropleth maps (e.g., "how much flux arrives at each hex?").
   The `from_id` column is carried alongside as a plain attribute.

   ```python
   result = od_counts.compute().reset_index()
   hex_polygons = hp.hex_geodataframe(...)   # GeoDataFrame indexed by hex_id
   gdf = result.merge(hex_polygons, left_on="to_id", right_index=True)
   # gdf.geometry is the destination hex polygon (active, used for plotting)
   # gdf["from_id"] carries the origin hex ID as a regular column
   ```

   geopandas does not support two active geometry columns. If origin polygon
   geometry is also needed (e.g., for spatial indexing), it can be carried as
   an inactive `from_geometry` column:

   ```python
   gdf["from_geometry"] = hex_polygons.geometry.loc[gdf["from_id"]].values
   # from_geometry is a plain object column — not the active geometry,
   # but available for later spatial operations.
   ```

   The lazy pipeline stays pure; all geometry work is post-compute.

3. **Consistent with dask idioms**: dask users expect lazy return values
   from pipeline functions.

### Convenience wrapper

A separate convenience function (or method) can wrap the pipeline end-to-end
for users who want a GeoDataFrame immediately:

```python
def hex_connectivity_geodataframe(...) -> gpd.GeoDataFrame:
    """Computes and returns GeoDataFrame with destination hex polygon geometry."""
    result = hex_connectivity(...).compute().reset_index()
    hex_polygons = hp.hex_geodataframe(...)
    gdf = result.merge(hex_polygons, left_on="to_id", right_index=True)
    # Active geometry: destination hex polygon (for plotting / choropleth).
    # Optionally carry from_geometry as an inactive column for spatial indexing.
    gdf["from_geometry"] = hex_polygons.geometry.loc[gdf["from_id"]].values
    return gdf
```

The returned GeoDataFrame has the destination hex polygon as its active
geometry (used for plotting). The `from_geometry` column is inactive — not
used for plotting, but available for spatial indexing or filtering after the
fact. This keeps the lazy core composable while providing a one-call path for
users who do not need to control the compute step.

---

## 7. Integration with existing API

### Goal: a single `hex_connectivity`

The project is in early development and makes no backwards-compatibility
guarantees. The goal is a single `hex_connectivity` function that works for
both numpy/eager and dask/lazy inputs. There is no need for a parallel
`hex_connectivity_dask` — the two use cases should be unified under one name.

The function detects the input type and dispatches accordingly:

- If the input arrays are dask-backed (or the Dataset has dask-backed
  variables), the function returns a lazy dask DataFrame.
- If the inputs are numpy/eager, the function materialises the result
  immediately and returns a pandas DataFrame (or GeoDataFrame, depending
  on the convenience layer).

Detection can use `dask.is_dask_collection` or check
`isinstance(ds["lon"].data, da.Array)`. Dask is not required to be
installed for the numpy path — the import can be guarded.

### Removing the current `hex_connectivity`

The current `hex_connectivity` in `hex_analysis.py` operates on
pre-labelled `xr.DataArray` hex IDs and calls `.values.ravel()`. The new
unified function accepts a raw `xr.Dataset` with `lon`/`lat` variables and
handles labelling internally. The old function should be replaced without a
deprecation wrapper — if something is gone, it is gone.

### Module location

The function lives in `src/hextraj/hex_analysis.py`. No new module needed.

---

## Notebook sketch

A demonstration notebook should show:

1. **Open a chunked dataset** — `xr.open_dataset(path, chunks={"traj": N})`
   or `xr.open_zarr(store)`.

2. **Call `hex_connectivity`** — show the lazy return value, inspect
   `.visualize()` or `len(ddf.__dask_graph__())`.

3. **Compute and visualise** — `.compute()`, attach geometry with
   `hp.edges_geodataframe()`, plot with matplotlib.

4. **Weighted connectivity** — pass a weight column, show weighted sums.

5. **Grouped connectivity** — add `release_region` as a groupby key, show
   per-region OD matrices.

6. **Normalisation** — divide OD counts by marginal from-hex counts to get
   conditional probabilities.

7. **Timing comparison** — time the lazy pipeline vs running the same
   computation eagerly on a small materialised dataset.

### Performance optimization: `persist()` for intermediate results

When the lazy pipeline produces an intermediate dask DataFrame (e.g.,
aggregated OD counts before joining with metadata), calling `.persist()` can
avoid repeated graph evaluations if the result is used multiple times:

```python
od_counts = ddf.groupby(["from_id", "to_id"]).size()
od_counts_cached = od_counts.persist()  # Materialise into distributed memory

# Subsequent operations reuse the cached result
od_normalized = od_counts_cached / od_counts_cached.sum()
gdf = hp.edges_geodataframe(*od_counts_cached.compute().index.levels, ...)
```

**Caution**: only use `persist()` if the intermediate result fits comfortably
in memory. For large datasets, it defeats the purpose of lazy evaluation and
can cause OOM errors. If in doubt, compute and work with the pandas result
directly.

---

## TDD notes

### Test categories

**Unit tests for the lazy pipeline function:**

1. **Return type (dask input)**: when the Dataset is dask-backed, the result
   is a lazy dask DataFrame or Series (not yet computed).
2. **Return type (numpy input)**: when the Dataset is numpy-backed, the result
   is a pandas DataFrame or Series (already computed).
3. **Correct counts**: `.compute()` (or direct result for numpy path) matches
   a manually constructed count table for a small known dataset.
4. **INVALID_HEX_ID propagation**: NaN positions produce INVALID_HEX_ID
   in the result.
5. **Weight column**: weighted sum matches manual computation.
6. **Groupby columns**: extra groupby keys appear in the result index.
7. **Chunk independence**: result is identical regardless of chunk size
   (parametrize over `chunk_traj=10, 50, 100`).
8. **Single-chunk case**: works when the entire dataset fits in one chunk.

**Integration tests:**

9. **Round-trip with `edges_geodataframe`**: the computed result can be
   passed to `hp.edges_geodataframe()` without error.

**Edge cases:**

10. **All-NaN positions**: all trajectories are invalid at from or to obs.
11. **Empty dataset**: zero trajectories.
12. **Single trajectory**: one-row result.

### Test style

Per project conventions: plain `pytest` functions with `@pytest.mark.parametrize`,
no test classes. Tests use the synthetic trajectory builder from section 1.

### Fixture sketch

```python
@pytest.fixture
def hp():
    return HexProj(hex_size_meters=500_000)

@pytest.fixture(params=[10, 50, 100])
def chunk_traj(request):
    return request.param

@pytest.fixture
def ds(chunk_traj):
    return make_synthetic_trajectories(
        n_traj=100, n_obs=5, chunk_traj=chunk_traj, seed=42,
    )
```

---

## Open issues (post-implementation review)

From code review of `hex_connectivity_dask` (2026-03-04):

1. **`expand_dims` wrong for traj-only groupby cols** — `col_arr.expand_dims({obs_dim: n_obs})` produces shape `(obs, traj)` instead of `(traj, obs)`, misaligning values. Fix: `col_arr.broadcast_like(ds["lon"])`.

2. **Missing docstring** — `hex_connectivity_dask` has a one-liner; every other exported function has a full Args/Returns block. Add one.

3. **obs column fragility** — `ds.coords.get(obs_dim, ...)` relies on xarray internals for whether an implicit dimension coordinate survives `to_dask_dataframe` as a named column. Add an explicit test that obs values are correct (not just that the column exists), and verify the fallback path.

4. **`test_npartitions_matches_traj_chunks` fragility** — tests xarray/dask internal behaviour rather than user-visible semantics. May break across versions. Consider relaxing or removing.

5. **`groupby_cols` truthiness** — `if groupby_cols:` treats `[]` and `None` identically; document in docstring.

6. **Chained method call on one line** — `assign_coords(...).rename_dims(...)` should be split for readability.

---

## Open questions

1. **Obs-resolved connectivity**: should this be a separate function
   (`hex_connectivity_obs_resolved`) or a mode flag on `hex_connectivity`?
   The output structure is different (triple-indexed vs pair-indexed),
   which suggests a separate function to keep the primary API simple.

2. **`hp.label` dask-awareness**: should `HexProj.label` grow a dask-aware
   code path (detecting dask arrays and dispatching via `map_blocks`
   internally)? This would allow `label` to be called directly on dask
   arrays, but it couples `HexProj` to dask. Current recommendation: keep
   `hp.label` numpy-only; call it via `ddf.map_partitions` for the dask path.

3. **Filtering INVALID before groupby**: should the library function filter
   out INVALID_HEX_ID rows before groupby by default, with an opt-in
   `include_invalid=True`? Current recommendation: filter by default
   (most users want valid-only connectivity), with `include_invalid=True`
   to keep the INVALID bucket.
