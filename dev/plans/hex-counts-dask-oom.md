# `hex_counts` OOMs on dask-backed DataArrays — deep-dive findings

Investigation notes for GitHub issue
[#32](https://github.com/willirath/hextraj/issues/32). This document records
the state of the problem, verified behaviour of the underlying libraries, and
the design tensions that a solution must navigate. It does **not** propose a
fix — that is deliberately left to a separate reasoning pass.

---

## 1. Problem statement and reproduction

`hextraj.hex_analysis.hex_counts` is documented to accept an `xr.DataArray` of
hex IDs that may be backed by a dask array, produce a sparse count
GeoDataFrame, and keep the aggregation pipeline lazy until the small result is
materialised. In practice the current implementation materialises the *input*
into the calling process, which OOMs on realistic trajectory datasets.

Reproducer from the issue (paraphrased):

```python
hex_ids = xr.DataArray(
    da.random.randint(0, 100_000, size=(500_000, 5000), chunks=(1000, 5000)),
    dims=("trajectory", "obs"),
)
gdf = hex_counts(hex_ids, reduce_dims=["trajectory", "obs"], hp=hp)
```

`(500_000, 5000)` int64 is a single 20 GB buffer once computed — the exact
amount the function tries to land in one process.

---

## 2. Where the eager computation happens

`src/hextraj/hex_analysis.py` has two independent eager paths.

**Full-reduction branch** (`hex_analysis.py:153-159`):

```python
if not keep_dims:
    hex_array = hex_ids.values.ravel()          # <-- eager .compute() of the whole array
    counts = pd.Series(hex_array).value_counts(sort=False)
    ...
```

`.values` on a dask-backed `xr.DataArray` always materialises to a numpy array
in the calling process. Verified: for `xr.DataArray(da.arange(100, chunks=10))`,
`type(x.data) == dask.array.Array` before `.values`, and `type(x.values) == numpy.ndarray`.

**Partial-reduction branch** (`hex_analysis.py:161-172`):

```python
for coords, group_data in hex_ids.groupby(keep_dims):
    hex_array = group_data.values.ravel()       # <-- eager per group
    counts = pd.Series(hex_array).value_counts(sort=False)
    for hex_id, count in counts.items():
        index_tuples.append(coords + (hex_id,))
        results.append(count)
```

`xr.DataArray.groupby(keep_dims)` itself is lazy — each `group_data` still has
`type(group_data.data) == dask.array.Array` (verified). The eagerness is
entirely in the inner `.values.ravel()`, and the construction accumulates
Python lists of tuples and counts which is then converted to a `MultiIndex`.
For large inputs this is slow even if each group fits in memory one at a time,
and for the pathological case (single keep-dim coordinate, all rows) it
degenerates to the full-reduction OOM.

---

## 3. Design intent vs. current implementation

From `dev/plans/hex-analysis-functions.md` (the function's design plan):

- **Sparse output**: only observed hex IDs (plus `INVALID_HEX_ID` when
  present) appear in the result. Even for billion-row inputs the unique-hex
  count is typically 10³ – 10⁵, so the returned GeoDataFrame is small.
- **Lazy pipeline**: *"Because `hex_ids` may be backed by a Dask array, the
  pipeline must remain lazy: use `dask.dataframe` groupby operations or
  compute per-chunk and reduce, rather than forcing `.values` on the full
  array before filtering."* The plan explicitly rules out the approach the
  implementation took.
- **Accepted input types**: `xr.DataArray`, `dask.dataframe.Series`, or
  `pd.Series`. The current implementation only handles `xr.DataArray` and
  `pd.Series`; the `dd.Series` path mentioned in the plan was never wired up.
- **`INVALID_HEX_ID` is an ordinary bucket** and must remain in the result.
  The issue's one-liner workaround drops it (`vc[vc.index >= 0]`), which is
  a behaviour change.
- **Output is a `GeoDataFrame`** — inherently eager. "Lazy throughout" cannot
  mean the return type is lazy; it can only mean the aggregation stays lazy
  until the small count table can be materialised cheaply.

---

## 4. Component-by-component limitations

### dask (array layer)

- `.values` always materialises to numpy in the calling process. There is no
  "lazy value_counts" on `dask.array` — routing through `dask.dataframe` is
  the only practical option.
- `dask.Array.ravel()` stays lazy and preserves chunking.

### xarray

- `DataArray.groupby(keys)` is lazy at construction. Iteration that calls
  `.values` per group is eager per group.
- `DataArray.to_dataset(name=...).to_dask_dataframe(dim_order=[...])` is the
  canonical bridge to dask DataFrame world. It promotes dim coordinates to
  explicit columns and maps dask chunks to DataFrame partitions.
- `dim_order` is load-bearing. It determines whether chunks map cleanly to
  partitions or require a shuffle — exactly the concern `hex_connectivity_dask`
  addresses by rechunking `obs` to `-1`. For a generic `hex_counts` with
  arbitrary user-named dims, there is no "traj/obs" hint to lean on.
- `xr.apply_ufunc(hp.label, ..., dask="parallelized", output_dtypes=[np.int64])`
  works (verified): an upstream `xr.Dataset` of lon/lat dask arrays flows
  through to a lazy `hex_ids` DataArray with matching chunks. This is the
  canonical way users arrive at a dask-backed input to `hex_counts`.

### dask.dataframe

- `Series.value_counts()` and `DataFrame.groupby([...]).size()` are parallel
  tree-reductions. Peak memory scales with unique-keys-per-partition, not
  total rows — this is exactly the scaling `hex_counts` needs.
- `dd.from_dask_array(flat)` returns a 1D `Series` directly (verified). The
  form used in the issue's suggested fix — `dd.from_dask_array(flat,
  columns="hex_id")["hex_id"]` — **fails** on the installed dask `2026.1.2`:

  ```
  TypeError: '<' not supported between instances of 'str' and 'int'
  ```

  The failure is in the `__getitem__` path (it routes through `.loc`, which
  compares the index label against integer divisions). This is worth calling
  out so any fix does not blindly mirror the issue's snippet.
- Partial reduction (`reduce_dims` is a strict subset of `hex_ids.dims`) can
  be expressed as
  `ds.to_dask_dataframe(dim_order=[...]).groupby([keep_dims + ["hex_id"]]).size()`
  and stays lazy end-to-end (verified on a small synthetic example).

### pyproj

- `pyproj.Transformer.from_crs(...)` and `HexProj` instances pickle cleanly
  (verified). Dask distributed (process-based workers) is safe.
- `pyproj.Transformer` is **not thread-safe** per upstream documentation:
  concurrent `.transform()` calls on a single transformer can corrupt internal
  state. Under the threaded scheduler (dask.dataframe's default for many ops)
  this is a latent risk, currently masked by the GIL and the fact that each
  task runs `hp.label` sequentially on its partition. Noted for completeness;
  not a blocker for #32.

### shapely

- Shapely 2.x provides vectorised `shapely.polygons(coords)` and
  `shapely.linestrings(coords)` constructors.
- `HexProj.to_geodataframe` (`hexproj.py:192-246`) already uses the batched
  path: one inverse projection for all corners, one `shapely.polygons` call.
- `_build_counts_geodataframe` (`hex_analysis.py:396-430`) duplicates the
  older per-hex logic: it calls `hp.hex_corners_lon_lat` in a Python loop
  (seven transformer calls per hex) and constructs polygons one at a time.
  For 10⁴ + unique hexes this is 10 – 100× slower than the batched path.
  This is a separate, latent performance bug that becomes visible the moment
  the counting OOM is fixed.
- `HexProj.edges_geodataframe` is the matching batched path for LineStrings.

### hextraj internals

- `encode_hex_id` / `decode_hex_id` (`src/hextraj/hex_id.py`) are already
  dask-compatible — they use `np.where` and arithmetic that flow through
  dask lazily. The module comment at line 43 explicitly says so.
- `HexProj.label` is **not** dask-aware: it calls `np.asarray(lon, dtype=float)`
  on its inputs. The intended pattern is
  `xr.apply_ufunc(hp.label, ..., dask="parallelized")`, which invokes `label`
  per chunk and preserves laziness.
- `hex_connectivity_dask` (`hex_analysis.py:16-110`) is the project's existing
  model for a dask-native analysis function. It returns a lazy `dd.DataFrame`,
  uses `to_dask_dataframe(dim_order=[...])` + `map_partitions(hp.label, ...)`,
  and rechunks `obs` to `-1` so each partition holds complete trajectories.
  `hex_counts` was planned to follow the same pattern (see
  `dev/plans/hex-analysis-functions.md`) but never did.

---

## 5. Why the "obvious" fix is not obvious

The issue's suggested fix — wrap one branch with `dask.is_dask_collection`
and use `dd.from_dask_array(flat).value_counts()` — patches only the full
reduction and only for the `xr.DataArray` input type. The deeper design
problem has several independent axes:

1. **Full vs partial reduction are structurally different.**
   Full reduction → 1D flatten → `Series.value_counts()`. Partial reduction
   needs the keep-dim coordinates as explicit columns, which means routing
   through `to_dask_dataframe(dim_order=...)` followed by
   `groupby([keep_dims, "hex_id"]).size()`. These are different pipelines
   with different chunk-alignment requirements.

2. **Input polymorphism.** The planned API admits `xr.DataArray`,
   `pd.Series`, and `dask.dataframe.Series`. Each wants a different path.
   The `dd.Series` route is the cleanest entry point for a user already in
   DataFrame-land — notably anyone coming from `hex_connectivity_dask`.

3. **Chunk-layout sensitivity.** `to_dask_dataframe` can force a shuffle
   when `dim_order` does not match the input chunking. For a user chunked on
   `traj` reducing `obs`, `dim_order=["traj", "obs"]` aligns partitions
   cleanly. For a user chunked on `obs` reducing `traj`, a shuffle is
   unavoidable. `hex_counts` has no fixed "traj/obs" semantics and cannot
   silently rechunk; behaviour will depend on user input shape.

4. **Lazy-vs-eager return type.** `GeoDataFrame` is eager, so the function
   must call `.compute()` on the small count table before constructing
   geometry. Users in a larger lazy pipeline may prefer a lazy `dd.DataFrame`
   intermediate (to `persist`, to chain further dask work). The existing
   plan `dev/plans/dask-native-connectivity.md` §6 faces the same tension and
   settles on "return lazy by default, offer a convenience wrapper to
   materialise" — whether `hex_counts` should follow suit, or stay eager and
   match its current signature, is an open design choice.

5. **Geometry construction is a second scaling knob.** Even after counts
   shrink to 10⁴ unique hexes, `_build_counts_geodataframe`'s per-hex Python
   loop is the next bottleneck. `HexProj.to_geodataframe` already solved this
   with a batched implementation; `hex_counts` should use it (directly, or
   via a refactor of `_build_counts_geodataframe`).

6. **`INVALID_HEX_ID` must stay a bucket.** Any `value_counts`/`groupby`
   path must preserve the `-1` entry. The issue's workaround filters it out,
   which would be a breaking change from the current documented contract.

7. **`pyproj` thread-safety — latent.** If any future fix dispatches
   labelling inside `map_partitions`, the same thread-safety caveat as
   `hex_connectivity_dask` applies. Not a blocker for #32 but worth
   remembering when the eventual solution is designed.

---

## 6. Test coverage gap

`tests/test_hex_analysis.py` exercises `hex_counts` only with numpy-backed
fixtures (`hex_ids`, `hex_ids_invalid`). There is no dask-backed fixture and
no assertion that the pipeline stays lazy. `tests/test_hex_connectivity_dask.py`
covers the connectivity function comprehensively (lazy graph, chunk
independence, custom dim names, groupby alignment) and is the right template
to extend to `hex_counts`.

---

## 7. Summary of findings

- The OOM is real and has two sources (full and partial reduction), not one.
- The issue's suggested one-liner fix is both incomplete (full reduction
  only) and syntactically broken on the installed dask version.
- The design plan for `hex_counts` already specified a lazy dask pipeline
  and accepted `dd.Series` input; the implementation shipped without these.
- The project already has a reference implementation of a dask-native
  analysis function (`hex_connectivity_dask`) whose patterns — `dim_order`,
  `map_partitions`, lazy `dd.DataFrame` return — apply directly.
- A complete fix must address: full-vs-partial dispatch, input-type
  polymorphism (`dd.Series` support), `dim_order` / chunk alignment, the
  lazy-vs-eager return contract, `INVALID_HEX_ID` preservation, and the
  slow per-hex geometry construction in `_build_counts_geodataframe`.
- Solution design is deliberately out of scope for this document.
