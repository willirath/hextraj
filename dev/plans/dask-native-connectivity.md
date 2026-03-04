# Plan: Dask-native trajectory-to-connectivity pipeline

## Executive summary

The current `hex_connectivity` function materialises dask arrays via
`.values.ravel()`, breaking laziness. This plan designs a fully lazy pipeline
from dask-backed xarray Datasets (e.g. from Zarr) through hex labelling,
xarray-to-dask-DataFrame conversion, and lazy groupby aggregation, producing
a dask DataFrame of OD counts that is only materialised at the user's
`.compute()` call. The `hex_conn_dask.ipynb` notebook already demonstrates
this pattern manually; the goal is to encapsulate it in a clean library
function that handles weights, metadata groupby keys, and normalisation.

---

## Recommended pipeline (pseudocode)

```python
def hex_connectivity_lazy(
    ds,                          # xr.Dataset with lon, lat (dask-backed)
    hp,                          # HexProj instance
    from_obs=0,                  # obs index for origin
    to_obs=-1,                   # obs index for destination
    weight=None,                 # str (column in ds) or xr.DataArray
    groupby_dims=None,           # extra dims to keep in groupby (e.g. ["release_region"])
    obs_dim="obs",               # name of the observation dimension
    traj_dim="traj",             # name of the trajectory dimension
):
    # 1. Slice origin and destination positions
    lon_from = ds.lon.isel({obs_dim: from_obs})
    lat_from = ds.lat.isel({obs_dim: from_obs})
    lon_to   = ds.lon.isel({obs_dim: to_obs})
    lat_to   = ds.lat.isel({obs_dim: to_obs})

    # 2. Label with da.map_blocks (stays lazy)
    from_ids = da.map_blocks(hp.label, lon_from.data, lat_from.data, dtype=np.int64)
    to_ids   = da.map_blocks(hp.label, lon_to.data,   lat_to.data,   dtype=np.int64)

    # 3. Assemble into xr.Dataset, include weight and groupby columns
    result_ds = xr.Dataset({
        "from_id": xr.DataArray(from_ids, dims=non_obs_dims),
        "to_id":   xr.DataArray(to_ids,   dims=non_obs_dims),
    })
    if weight is not None:
        result_ds["weight"] = weight_da
    for dim in groupby_dims:
        result_ds[dim] = ds[dim]  # or ds.coords[dim]

    # 4. Convert to dask DataFrame (chunk boundaries → partitions)
    ddf = result_ds.to_dask_dataframe()

    # 5. Lazy groupby aggregation
    group_keys = groupby_dims + ["from_id", "to_id"]
    if weight is not None:
        agg = ddf.groupby(group_keys).agg({"weight": "sum"})
    else:
        agg = ddf.groupby(group_keys).size()

    # 6. Return lazy dask DataFrame (or Series)
    return agg
```

The user calls `.compute()` when ready, then optionally passes the result
to `hp.edges_geodataframe()` for geometry.

---

## 1. Synthetic dask-backed trajectory dataset

### Design

A test helper function builds a synthetic `xr.Dataset` with:

- `lon(traj, obs)` and `lat(traj, obs)` as dask arrays, chunked along `traj`.
- Optional metadata variables: `particle_id(traj)`, `release_region(traj)`,
  `ensemble(traj)`.
- Known hex assignments for deterministic assertions.

### Implementation

```python
def make_synthetic_trajectories(
    n_traj=1000, n_obs=10, chunk_traj=100,
    lon_range=(-10, 10), lat_range=(-5, 5),
    n_regions=3, seed=42,
):
    rng = np.random.default_rng(seed)
    lon = rng.uniform(*lon_range, size=(n_traj, n_obs)).astype(np.float32)
    lat = rng.uniform(*lat_range, size=(n_traj, n_obs)).astype(np.float32)
    # Inject some NaN (beached particles)
    mask = rng.random((n_traj, n_obs)) < 0.05
    lon[mask] = np.nan
    lat[mask] = np.nan

    ds = xr.Dataset({
        "lon": (["traj", "obs"], da.from_array(lon, chunks=(chunk_traj, n_obs))),
        "lat": (["traj", "obs"], da.from_array(lat, chunks=(chunk_traj, n_obs))),
        "release_region": ("traj", da.from_array(
            rng.integers(0, n_regions, n_traj), chunks=chunk_traj
        )),
    })
    return ds
```

### Chunking rationale

Chunk along `traj` (not `obs`) so that each chunk contains full trajectories.
The `isel(obs=...)` slice is then a cheap within-chunk operation. This mirrors
real-world Zarr stores where trajectories are the large dimension.

---

## 2. Hex labelling timing

### The question

`hp.label(lon, lat)` calls pyproj internally, which materialises to numpy.
It cannot accept a raw dask array. Where should labelling happen?

### Options considered

| Approach | Lazy? | Chunk-aligned? | Complexity |
|----------|-------|----------------|------------|
| `xr.apply_ufunc(hp.label, ..., dask="parallelized")` | Yes | Yes | Medium — needs explicit output dtypes and shapes |
| `da.map_blocks(hp.label, lon.data, lat.data, ...)` | Yes | Yes | Low — proven in notebook |
| Label inside `ddf.map_partitions(...)` | Yes | Yes | Medium — requires converting back from DataFrame columns |
| Label before chunking (eager) | No | N/A | Defeats purpose |

### Recommendation: `da.map_blocks` (already proven)

The `hex_conn_dask.ipynb` notebook already demonstrates this pattern
successfully. `da.map_blocks` calls `hp.label` on each chunk's numpy arrays,
keeping the result lazy and chunk-aligned. No data shuffles occur.

`xr.apply_ufunc` with `dask="parallelized"` would also work but adds
boilerplate for specifying output dtypes and core dimensions. Since `hp.label`
already handles arbitrary array shapes and returns a same-shape int64 array,
`da.map_blocks` is the simpler path.

### When to label

Two equivalent approaches:

**Option A: Label before DataFrame conversion (recommended)**

Label **before** converting to dask DataFrame. The labelling operates on
position arrays (lon, lat) which are naturally chunk-aligned in xarray. After
conversion to DataFrame, positions are in columns and labelling would require
`map_partitions` — functionally equivalent but less readable.

The pipeline:
1. `isel` to select the from/to obs positions (cheap slice per chunk).
2. `da.map_blocks(hp.label, ...)` on each slice.
3. Assemble labelled arrays into an `xr.Dataset`.
4. Convert to dask DataFrame.

**Option B: Label after DataFrame conversion**

Alternatively, `map_partitions(hp.label)` on lon/lat columns after
`.to_dask_dataframe()` conversion is equally lazy and arguably simpler — it
avoids needing to reassemble an intermediate xr.Dataset:

```python
ddf = result_ds.to_dask_dataframe()
ddf["from_id"] = ddf.map_partitions(
    lambda df: hp.label(df["lon"], df["lat"]), dtype=np.int64
)
```

Both approaches preserve laziness and produce identical results. Option B may
be preferred for its directness and fewer intermediate objects. Option A is
closer to the proven `hex_conn_dask.ipynb` notebook pattern.

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

### What columns to include

The xr.Dataset assembled before conversion should contain:

| Column | Source | Purpose |
|--------|--------|---------|
| `from_id` | `da.map_blocks(hp.label, lon_from, lat_from)` | Origin hex ID |
| `to_id` | `da.map_blocks(hp.label, lon_to, lat_to)` | Destination hex ID |
| `weight` | User-supplied DataArray, sliced to match | Optional weight per pair |
| metadata dims | e.g. `release_region`, `ensemble` | Extra groupby keys |

Coordinate variables from the original Dataset (e.g. `traj`) become the
DataFrame index automatically. They can be reset if not needed for groupby.

### Multi-dimensional case (obs-resolved)

For obs-resolved connectivity (triple index `(obs_step, from_id, to_id)`),
the full 2D `(traj, obs)` labelling is done first, then from_ids are
broadcast from obs=0 across all obs steps. The xr.Dataset is stacked
`(traj, obs_step) -> event` before `.to_dask_dataframe()`. The stack
is a graph rewiring operation, not a compute.

---

## 4. From/to hex ID construction

### Start-to-end connectivity

The simplest case: `isel(obs=from_idx)` and `isel(obs=to_idx)` on the
position arrays, then label each. Both slices share the same chunk structure
along `traj`, so the resulting dask arrays are aligned without shuffling.

```python
lon_from = ds.lon.isel(obs=0)    # shape (traj,), chunked
lon_to   = ds.lon.isel(obs=-1)   # shape (traj,), chunked — same chunks
from_ids = da.map_blocks(hp.label, lon_from.data, lat_from.data, dtype=np.int64)
to_ids   = da.map_blocks(hp.label, lon_to.data,   lat_to.data,   dtype=np.int64)
```

### Arbitrary from_idx / to_idx

The function accepts integer indices along the obs dimension. Negative
indexing works naturally via `isel`. The only constraint is that both slices
reduce the obs dimension to a scalar, producing 1D arrays along `traj`.

### Obs-resolved connectivity

For the `(obs_step, from_id, to_id)` pattern, label the full 2D array:

```python
all_ids = da.map_blocks(hp.label, ds.lon.data, ds.lat.data, dtype=np.int64)
from_da = da.broadcast_to(all_ids[:, 0:1], all_ids[:, 1:].shape)
to_da   = all_ids[:, 1:]
```

`broadcast_to` is a zero-copy view — no data duplication. The result is a
2D pair `(from_id, to_id)` at each `(traj, obs_step)` cell, ready for
stacking and DataFrame conversion.

### Recommendation

The primary function handles start-to-end (scalar from/to). Obs-resolved
connectivity is a separate function or a mode flag, since it produces a
fundamentally different output structure (triple-indexed vs pair-indexed).

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
   the result can be enriched with hex metadata (e.g., hex center coordinates,
   region labels) by joining back to a hex metadata DataFrame or xr.Dataset:

   ```python
   result = od_counts.compute()

   # Attach geometry
   gdf = hp.edges_geodataframe(
       result.index.get_level_values("from_id"),
       result.index.get_level_values("to_id"),
       count=result.values,
   )

   # Enrich with hex metadata (e.g., centres, region labels)
   from_centres = hex_metadata.set_index("hex_id").loc[gdf.index.get_level_values("from_id")]
   gdf["from_center_lon"] = from_centres["lon"].values
   gdf["from_center_lat"] = from_centres["lat"].values
   ```

   This is analogous to how the heatmap notebook reconstructs bin centers and
   labels from `pd.cut` categories after computing. The lazy pipeline stays
   pure; enrichment is a post-compute operation on pandas/geopandas objects.

3. **Consistent with dask idioms**: dask users expect lazy return values
   from pipeline functions.

### Convenience wrapper

A separate convenience function (or method) can wrap the pipeline end-to-end:

```python
def hex_connectivity_lazy(...) -> dask.dataframe.Series:
    """Returns lazy dask Series of OD counts."""
    ...

def hex_connectivity_geodataframe(...) -> gpd.GeoDataFrame:
    """Computes and returns GeoDataFrame with geometry."""
    result = hex_connectivity_lazy(...).compute()
    return hp.edges_geodataframe(...)
```

This keeps the lazy core composable while providing a one-call path for
users who want a GeoDataFrame immediately.

---

## 7. Integration with existing API

### Option A: Replace `hex_connectivity` (recommended)

The current `hex_connectivity` accepts `xr.DataArray` of pre-labelled hex
IDs and calls `.values.ravel()`. The new function should accept the raw
`xr.Dataset` with `lon`/`lat` and handle labelling internally. This is a
different interface.

Since the project does not maintain backwards compatibility, the
recommended approach is:

1. **New function `hex_connectivity_dask`** in `hex_analysis.py` that
   implements the full lazy pipeline.

2. **Keep `hex_connectivity`** for now, since it operates on pre-labelled
   DataArrays and is used by `hex_connectivity_power`. It serves a
   different use case (small, already-materialised data).

3. **Export both** from `__init__.py`.

### Why not a `lazy=True` kwarg

Adding `lazy=True` to `hex_connectivity` would require the function to
accept two fundamentally different input types (pre-labelled DataArray vs
raw Dataset with lon/lat) and return two different types (GeoDataFrame vs
dask DataFrame). This violates the principle of functions doing one thing.
Two separate functions with clear names are better.

### Naming

- `hex_connectivity_dask(ds, hp, ...)` — the lazy pipeline function.
- `hex_connectivity(hex_ids, ...)` — the existing eager function on
  pre-labelled DataArrays.

Alternatively, if the eager function is eventually removed:
- `hex_connectivity(ds, hp, ...)` replaces both.

### Module location

Both functions live in `src/hextraj/hex_analysis.py`. No new module needed.

---

## Notebook sketch

A demonstration notebook should show:

1. **Open a chunked dataset** — `xr.open_dataset(path, chunks={"traj": N})`
   or `xr.open_zarr(store)`.

2. **Call `hex_connectivity_dask`** — show the lazy return value, inspect
   `.visualize()` or `len(ddf.__dask_graph__())`.

3. **Compute and visualise** — `.compute()`, attach geometry with
   `hp.edges_geodataframe()`, plot with matplotlib.

4. **Weighted connectivity** — pass a weight column, show weighted sums.

5. **Grouped connectivity** — add `release_region` as a groupby key, show
   per-region OD matrices.

6. **Normalisation** — divide OD counts by marginal from-hex counts to get
   conditional probabilities.

7. **Timing comparison** — time the lazy pipeline vs the eager
   `hex_connectivity` on the same dataset.

The existing `hex_conn_dask.ipynb` already covers steps 1-3 manually.
The notebook should demonstrate the library function doing it in one call.

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

1. **Return type**: result is a dask DataFrame or Series (not yet computed).
2. **Correct counts after compute**: compare `.compute()` against the eager
   `hex_connectivity` on the same small dataset.
3. **INVALID_HEX_ID propagation**: NaN positions produce INVALID_HEX_ID
   in the result.
4. **Weight column**: weighted sum matches manual computation.
5. **Groupby dims**: extra groupby keys appear in the result index.
6. **Chunk independence**: result is identical regardless of chunk size
   (parametrize over `chunk_traj=10, 50, 100`).
7. **Single-chunk case**: works when the entire dataset fits in one chunk.

**Integration tests:**

8. **Round-trip with `hex_connectivity`**: for a small dataset, the lazy
   pipeline's `.compute()` result matches the eager function's output
   (same pairs, same counts).
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

## Open questions

1. **Obs-resolved connectivity**: should this be a separate function
   (`hex_connectivity_obs_resolved_dask`) or a mode of the same function?
   The output structure is different (triple-indexed), suggesting a
   separate function.

2. **`hp.label` on dask arrays directly**: should `HexProj.label` grow a
   dask-aware code path (detecting dask arrays and calling `map_blocks`
   internally)? This would simplify the pipeline but couples `HexProj` to
   dask. Current recommendation: keep `hp.label` numpy-only, use
   `da.map_blocks` externally.

3. **Filtering INVALID before groupby**: the notebook filters
   `ddf[(ddf.from_id != INVALID_HEX_ID) & ...]` before groupby. Should
   the library function do this by default, with an opt-in
   `include_invalid=True`? Current recommendation: filter by default
   (most users want valid-only connectivity), with `include_invalid=True`
   to keep the INVALID bucket.
