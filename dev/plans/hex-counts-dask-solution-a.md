# `hex_counts` dask OOM — solution design (A)

## 1. Problem recap

`hextraj.hex_analysis.hex_counts` OOMs on dask-backed `xr.DataArray` inputs
because both its branches call `.values.ravel()` on the full input before
aggregating. See
[`dev/plans/hex-counts-dask-oom.md`](./hex-counts-dask-oom.md) for the full
investigation; §5 of that document lists the seven tensions this design must
reconcile. This plan picks one coherent resolution.

---

## 2. Design goals and non-goals

**Goals**

1. Never materialise the input. All aggregation happens through
   `dask.dataframe` so peak memory scales with
   *unique-hex-IDs-per-partition*, not total rows.
2. Single `hex_counts` entry point that accepts `xr.DataArray` (numpy- or
   dask-backed), `pd.Series`, and `dask.dataframe.Series`.
3. Preserve the current return contract: an eager `gpd.GeoDataFrame` with the
   same index structure it produces today (single-level `hex_id` index for
   full reduction; `MultiIndex` with `keep_dims + ["hex_id"]` for partial
   reduction). `INVALID_HEX_ID` stays a regular row with `geometry=None`.
4. Batched geometry construction — stop calling `hp.hex_corners_lon_lat` in a
   Python loop inside `_build_counts_geodataframe`.
5. Follow the patterns `hex_connectivity_dask` already established
   (`to_dask_dataframe(dim_order=...)`, `groupby(...).size()`, `dd.DataFrame`
   internals).

**Non-goals**

- A lazy-`dd.DataFrame` return mode. The current signature returns a
  `GeoDataFrame`; the result is sparse (10³–10⁵ rows for realistic inputs),
  and the deep-dive §3 already notes *"Output is a GeoDataFrame — inherently
  eager."* Users who want a lazy intermediate can call the helper
  `hex_counts_lazy` described in §9; it is the implementation surface used
  internally and is exported as a public shortcut. But the primary surface
  stays eager.
- Automatic rechunking of user input. We document the chunk-alignment trap
  (§6) but do not paper over it. `hex_connectivity_dask` rechunks `obs=-1`
  because it has domain semantics; `hex_counts` has no fixed semantics and
  must stay user-driven.
- Fixing `HexProj.label`'s numpy-only signature. Orthogonal to #32.
- `pyproj` thread-safety fixes. Latent, same state as
  `hex_connectivity_dask`.

---

## 3. Proposed architecture (high level)

Three conceptual layers:

```
                          hex_counts(...)   <-- user-facing; returns GeoDataFrame
                               │
                               ▼
                 hex_counts_lazy(...)        <-- returns a lazy dd.Series of counts
                    (dispatch + pipeline)    indexed by (*keep_dims, hex_id)
                               │
                               ▼
       _counts_to_geodataframe(counts, hp)   <-- .compute() + batched geometry
                               │
                               ▼
              hp.to_geodataframe(hex_ids, count=...)   <-- existing batched builder
```

- `hex_counts_lazy` does dispatch and builds the lazy `dd.Series` of counts.
  Fully reduced case → 1D `dd.Series`; partial → `dd.Series` with a
  `MultiIndex` of `(*keep_dims, "hex_id")`.
- `_counts_to_geodataframe` is the eager step: `.compute()` + attach geometry
  via the batched path.
- `hex_counts` is a thin wrapper: `hex_counts_lazy` → `.compute()` →
  build GeoDataFrame.

Keeping the lazy half callable as a public helper satisfies power users who
want to `.persist()` intermediate counts or chain further dask work without
forcing a redesign of the primary API (see deep dive §5.4).

---

## 4. Dispatch and input handling

### 4.1 Detection

Use `dask.is_dask_collection` uniformly. Verified on 2026-04-22 with the
installed dask:

```
is_dask_collection(xr.DataArray(da.arange(10, chunks=5)))  -> True
is_dask_collection(xr.DataArray(np.arange(10)))            -> False
is_dask_collection(dd.from_pandas(pd.Series([1,2,3])...))  -> True
is_dask_collection(pd.Series([1,2,3]))                     -> False
```

This gives a single predicate that works across all four input types. No
`hasattr(x, "chunks")` heuristics; no `isinstance` on `da.Array`.

### 4.2 Routing table

| Input type               | `reduce_dims` used? | Lazy inside? | Path                            |
|--------------------------|---------------------|--------------|---------------------------------|
| `pd.Series`              | ignored             | no           | `Series.value_counts` → eager   |
| `dask.dataframe.Series`  | ignored             | yes          | `dd.Series.value_counts` → lazy |
| `xr.DataArray` (numpy)   | used                | no           | xarray→`dd` path, but the graph is eagerly computed at the end. We still route through `dd` to share one code path with the dask-backed case — no separate pandas branch — and avoid duplicating the partial-reduction logic. |
| `xr.DataArray` (dask)    | used                | yes          | xarray→`dd` path, lazy          |

Rationale for routing numpy `xr.DataArray` through the same `dd` path as the
dask one: §5 of this document folds full and partial reductions into a
single pipeline. Forking on backing type doubles the surface and invites
drift. A trivially chunked `dd.DataFrame` over a numpy DataArray is cheap
(one partition), and the final `.compute()` is a no-op convert.

Early branch for Series inputs, because they skip the whole xarray-shaped
bookkeeping:

```python
def hex_counts_lazy(hex_ids, reduce_dims=None):
    if isinstance(hex_ids, (pd.Series, dd.Series)):
        return hex_ids.value_counts(sort=False)
    # xr.DataArray path: build ddf and reduce
    ...
```

`pd.Series.value_counts` is fine to return directly; callers who asked for
the "lazy" function and passed a pandas Series get a pandas Series back —
that is the natural laziness for that input type. The eager wrapper
`hex_counts` treats both the same way (`.compute()` is a no-op on a pandas
Series if we guard on `is_dask_collection`).

### 4.3 `dask.dataframe.Series` input — concrete

The deep dive called out that this input type was in the original plan but
never wired. The path is one line:

```python
if isinstance(hex_ids, dd.Series):
    return hex_ids.value_counts(sort=False)   # lazy dd.Series
```

`reduce_dims` is ignored — a Series has no named dimensions. `INVALID_HEX_ID`
is preserved because `value_counts` keeps any value present in the data
(verified — `-1 in vc.compute().index.values`). This is the cleanest entry
point for a user already in DataFrame-land (e.g. downstream of
`hex_connectivity_dask`).

### 4.4 Why not `dd.DataFrame.groupby(hex_id).size()` instead of `value_counts`?

For the full-reduction case, `value_counts()` is semantically equivalent and
reads clearer. For the partial-reduction case we *do* use `groupby` because
we need multiple key columns (`keep_dims + ["hex_id"]`).

---

## 5. Full-reduction pipeline

The common case: `reduce_dims` covers every dim of `hex_ids`. The result is
a 1D `dd.Series` (or `pd.Series` for numpy input) indexed by `hex_id`.

### 5.1 Sketch

```python
def _full_reduction(hex_ids):                  # xr.DataArray, possibly dask-backed
    # Route through dd regardless of backing.
    ds = hex_ids.to_dataset(name="hex_id")
    # dim_order: use hex_ids.dims verbatim.  to_dask_dataframe is a graph rewire
    # for dask input and a pandas construction for numpy input.
    ddf = ds.to_dask_dataframe(dim_order=list(hex_ids.dims))
    counts = ddf["hex_id"].value_counts(sort=False)
    return counts            # dd.Series when dask-backed, pd.Series otherwise
```

Wait — `xr.Dataset.to_dask_dataframe()` returns a `dd.DataFrame` even for
numpy-backed input. That is fine; we eagerly compute downstream. There is no
need to branch on backing here.

### 5.2 Why not `dd.from_dask_array(hex_ids.data.ravel())`?

Two reasons:

1. Works for dask input but not numpy input. We would have to branch.
2. `.data` access is considered brittle elsewhere in the codebase (see
   `dev/plans/dask-native-connectivity.md` §2).

`to_dask_dataframe` is the canonical xarray→dd bridge. `hex_connectivity_dask`
uses it. Pattern-match.

### 5.3 `INVALID_HEX_ID` preservation

`dd.Series.value_counts(sort=False)` includes every value that appears in any
partition, including `-1`. Verified. No filtering anywhere in the pipeline.

### 5.4 Chunking

One chunk → one partition → tree-reduction over partitions. Peak memory is
(unique hex IDs per partition) × 8 bytes plus pandas overhead. For the
reproducer (`500_000 × 5000` int64, `(1000, 5000)` chunks), that is 1000
trajectories × 5000 obs = 5 million rows per partition; if the user input
really is uniform-random over 100 000 hex IDs, each partition sees ~50 000
unique IDs, which is a ~800 kB `pd.Series.value_counts` result per
partition. The tree reduction merges these. 500 partitions total. No
materialisation of a 20 GB array at any point.

---

## 6. Partial-reduction pipeline

`reduce_dims` is a strict subset of `hex_ids.dims`. Keep-dim coordinates
must reach the output index.

### 6.1 Sketch

```python
def _partial_reduction(hex_ids, reduce_dims):
    all_dims = list(hex_ids.dims)
    keep_dims = [d for d in all_dims if d not in reduce_dims]

    # dim_order: keep-dims first, reduce-dims last.  Reason: matches the
    # natural chunk ordering of inputs that come out of apply_ufunc(hp.label)
    # over a (traj, obs) Dataset chunked on traj; the groupby key then runs
    # in row-major order within each partition.
    dim_order = keep_dims + reduce_dims

    ds = hex_ids.to_dataset(name="hex_id")
    ddf = ds.to_dask_dataframe(dim_order=dim_order)
    counts = ddf.groupby(keep_dims + ["hex_id"]).size()
    return counts            # dd.Series with MultiIndex (*keep_dims, hex_id)
```

Verified on a small example: `ddf = ds.to_dask_dataframe(dim_order=['traj',
'obs'])` followed by `ddf.groupby(['traj', 'hex_id']).size()` produces the
right counts with `-1` preserved and stays lazy.

### 6.2 Chunk alignment

Two regimes:

- **Input chunking matches `dim_order`'s leading dims** (e.g. user chunked
  on `traj`, reducing `obs`). Each dask chunk maps to one dd partition.
  The groupby key's keep-dim values are contiguous within a partition, so
  the tree reduction is efficient and no shuffle happens.
- **Input chunking does not match** (e.g. user chunked on `obs`, reducing
  `traj`). `to_dask_dataframe` still works but partitions mix keep-dim
  values. `dd.groupby(...).size()` still produces correct results —
  shuffling is dask's concern — but the cost is a shuffle between the map
  and the reduce. We document this in the docstring and do not rechunk
  silently. A user who cares can `hex_ids = hex_ids.chunk({...})` before
  calling.

This is the principled position: `hex_counts` has no fixed
trajectory/observation semantics (unlike `hex_connectivity_dask` which
rechunks `obs=-1` by design). Silent rechunking of an unknown dim layout
would risk graph bloat and surprise.

### 6.3 Keep-dim *coordinate* values vs. index positions

`to_dask_dataframe` promotes a dim's *coordinate* values to a column when
one exists, otherwise the 0-based range. Our groupby uses those column
values, so the returned MultiIndex carries real coordinate values — which
is what the current numpy implementation also does (it iterates via
`hex_ids.groupby(keep_dims)`, which yields coordinate values).
Regression-safe.

### 6.4 Empty input

If `hex_ids.size == 0` the dd pipeline produces an empty Series with the
correct MultiIndex names. The eager wrapper converts that to an empty
GeoDataFrame with the right index metadata (matches current behaviour).

---

## 7. Geometry construction

Replace `_build_counts_geodataframe`'s per-hex loop with
`HexProj.to_geodataframe`, which already does the batched path
(`hexproj.py:192-246`): one inverse projection for all corners, one
`shapely.polygons` call. The signature already supports arbitrary value
columns via `**value_cols`.

### 7.1 New eager helper

```python
def _counts_to_geodataframe(counts, hp, keep_dims):
    """counts: pd.Series (after .compute() if needed).  Returns GeoDataFrame."""
    if is_dask_collection(counts):
        counts = counts.compute()

    if keep_dims:
        # MultiIndex (*keep_dims, hex_id).  Extract the hex_id level for geometry.
        hex_ids_arr = counts.index.get_level_values("hex_id").to_numpy()
        gdf = hp.to_geodataframe(hex_ids_arr, count=counts.values)
        gdf.index = counts.index          # restore full MultiIndex
    else:
        hex_ids_arr = counts.index.to_numpy()
        gdf = hp.to_geodataframe(hex_ids_arr, count=counts.values)
        gdf.index.name = "hex_id"
    return gdf
```

### 7.2 Remove `_build_counts_geodataframe`

The function is dead code after this refactor. Per `AGENTS.md` §Backwards
compatibility, delete it outright — no deprecation wrapper, no re-export
shim. Anyone importing the private `_build_counts_geodataframe` was
reaching into a private API.

### 7.3 `INVALID_HEX_ID` carry-through

`hp.to_geodataframe(hex_ids)` already handles `-1` → `geometry=None`
(verified in `hexproj.py:209-212`, the `invalid` mask path). No extra work
needed; the lazy pipeline never filters `-1`, the batched geometry builder
handles it uniformly.

---

## 8. Return type and laziness

### 8.1 `hex_counts` stays eager → GeoDataFrame

Reasons (restating the case, tightly):

- Matches the current documented contract. No silent API shift for callers
  already using it.
- The `GeoDataFrame` is small by construction (sparse over observed hex
  IDs). Materialising is cheap.
- The aggregation pipeline is lazy right up until `.compute()`, which
  resolves the one tension in §5 of the deep dive that mattered: eager
  *input* was the OOM, not eager *output*.

### 8.2 `hex_counts_lazy` is the public lazy variant

Exported from `hextraj.hex_analysis`. Returns a lazy `dd.Series` (or plain
`pd.Series` for numpy/`pd.Series` input — follows the laziness of the
input). Use cases:

- User is in a multi-step lazy pipeline and wants to `.persist()` the
  counts.
- User wants to reduce further (e.g. join, normalise) before materialising.
- Downstream of `hex_connectivity_dask` in a notebook where
  `.compute()` is explicit.

No `GeoDataFrame` comes out of `hex_counts_lazy`. That is a feature, not an
omission: the geometry-attaching eager step is exactly the boundary where
the lazy world ends.

### 8.3 Not a mode flag

I considered `hex_counts(..., lazy=False)` and rejected it. Return type
depending on a kwarg makes type checking, docs, and downstream code noisy.
Two functions with distinct types is cleaner.

---

## 9. Final API (signatures, types)

In `src/hextraj/hex_analysis.py`:

```python
def hex_counts(
    hex_ids: xr.DataArray | dd.Series | pd.Series,
    reduce_dims: str | list[str] | None = None,
    hp: HexProj | None = None,
) -> gpd.GeoDataFrame:
    """Count hex visits; return a GeoDataFrame with hex polygons.

    Aggregation is performed lazily via dask.dataframe when the input is
    dask-backed; the small result is materialised and decorated with hex
    polygon geometry via HexProj.to_geodataframe.
    """


def hex_counts_lazy(
    hex_ids: xr.DataArray | dd.Series | pd.Series,
    reduce_dims: str | list[str] | None = None,
) -> dd.Series | pd.Series:
    """Lazy count-only form of hex_counts.  No geometry attached."""
```

Exports in `src/hextraj/__init__.py` add `hex_counts_lazy` alongside the
existing `hex_counts`.

### 9.1 Signature changes vs. current

- `hex_ids` gains `dd.Series` in the type union. Runtime-equivalent; only a
  type-hint change.
- `reduce_dims` unchanged.
- `hp` unchanged.
- Return type unchanged for `hex_counts`.

### 9.2 Backwards compatibility

Per `AGENTS.md`: no backwards-compat shims. One deliberate break worth
calling out:

- `_build_counts_geodataframe` is deleted. Not exported, not expected to
  have external consumers. Anyone reaching into a private helper takes the
  hit.

No other user-visible change. The return GeoDataFrame shape (index names,
columns, CRS, `-1` row with `None` geometry) is unchanged.

---

## 10. Tests

### 10.1 New test file

`tests/test_hex_counts_dask.py` — mirrors
`tests/test_hex_connectivity_dask.py` in layout. Plain `pytest` functions,
`@pytest.mark.parametrize` for chunk sweeps, no test classes (per
`AGENTS.md`).

### 10.2 Fixtures

```python
@pytest.fixture
def hp():
    return HexProj(hex_size_meters=2_000_000)

@pytest.fixture(params=[(3, 4), (2, 6), (6, 2)])
def chunks(request):
    return request.param

@pytest.fixture
def hex_ids_dask(hp, chunks):
    """6 traj × 4 obs, dask-backed, with injected NaNs so INVALID appears."""
    rng = np.random.default_rng(0)
    lon_np = rng.uniform(-20, 20, size=(6, 4)).astype(np.float64)
    lat_np = rng.uniform(-10, 10, size=(6, 4)).astype(np.float64)
    lon_np[0, 1] = np.nan; lat_np[0, 1] = np.nan
    lon_np[3, 2] = np.nan; lat_np[3, 2] = np.nan
    lon = xr.DataArray(
        da.from_array(lon_np, chunks=chunks), dims=("traj", "obs")
    )
    lat = xr.DataArray(
        da.from_array(lat_np, chunks=chunks), dims=("traj", "obs")
    )
    ids = xr.apply_ufunc(
        hp.label, lon, lat, dask="parallelized", output_dtypes=[np.int64],
    )
    return ids

@pytest.fixture
def hex_ids_numpy(hex_ids_dask):
    """Eager ground truth: same data, numpy-backed."""
    return hex_ids_dask.compute()
```

### 10.3 Required tests

1. **`test_hex_counts_dask_returns_geodataframe`** — `isinstance(gdf,
   gpd.GeoDataFrame)` on a dask-backed input.
2. **`test_hex_counts_lazy_returns_dd_series`** — `hex_counts_lazy` on
   dask-backed input returns a `dd.Series`, and its task graph is
   non-empty (no eager compute happened inside the function).
3. **`test_hex_counts_lazy_graph_does_not_materialise_input`** — assert
   that the number of tasks in the graph is bounded (e.g. < 10 × number of
   partitions) — weak but catches "accidentally embedded the full array"
   regressions. Alternative: assert no `getitem` of the full array appears
   in the graph (brittle across dask versions — document the option, pick
   the bounded-tasks form).
4. **`test_hex_counts_dask_matches_numpy`** — parametrised over chunk
   layouts; for every chunk layout,
   `hex_counts(hex_ids_dask, reduce_dims=[...]) ==
    hex_counts(hex_ids_numpy, reduce_dims=[...])` by value (sort index,
   compare `count` column exactly). Covers full reduction.
5. **`test_hex_counts_dask_partial_reduction_matches_numpy`** — same as
   #4 but with `reduce_dims=["obs"]` and `reduce_dims=["traj"]`. Uses the
   MultiIndex comparison.
6. **`test_hex_counts_invalid_preserved_dask`** — dask-backed input with
   NaNs; assert `INVALID_HEX_ID in result.index` and
   `result.loc[INVALID_HEX_ID, "geometry"] is None`.
7. **`test_hex_counts_chunk_independence`** — parametrised over multiple
   chunk layouts; the computed GeoDataFrame is identical by value. Covers
   the chunk-layout-sensitivity concern (deep dive §5.3).
8. **`test_hex_counts_dd_series_input`** — pass a `dd.Series` directly
   (built via `dd.from_pandas(pd.Series([...]), npartitions=2)`); assert
   the output GeoDataFrame matches the one from the equivalent `pd.Series`
   input.
9. **`test_hex_counts_lazy_dd_series_stays_lazy`** — `hex_counts_lazy` on
   a `dd.Series` returns a `dd.Series`.
10. **`test_hex_counts_preserves_coord_values_in_multiindex`** — build a
    `DataArray` with non-trivial integer coordinate values (e.g.
    `traj=[100, 200, ...]`); assert the returned MultiIndex carries those
    values, not 0..N-1. Matches the `test_has_obs_column` pattern from the
    connectivity tests.

### 10.4 Kept / deleted existing tests

All current `tests/test_hex_analysis.py::test_hex_counts_*` tests pass
unchanged (they use numpy-backed fixtures and test the same output
contract). No edits needed.

---

## 11. Known pitfalls

1. **`dd.from_dask_array(x, columns="hex_id")["hex_id"]`** — fails on dask
   2026.1.2 with `TypeError: '<' not supported between instances of 'str'
   and 'int'` (verified by this agent, matches deep-dive §4). **Do not use
   this form.** The design above avoids it entirely by routing through
   `to_dask_dataframe` on the xarray side.

2. **`DataArray.groupby(keep_dims)` iteration is lazy *only* at
   construction** — the loop body's `.values.ravel()` is eager per group.
   If the implementation agent ever reaches for `DataArray.groupby` inside
   a Python loop, they are re-introducing the bug. The design here uses
   `dd.groupby`, not `xr.groupby`.

3. **`to_dask_dataframe` without `dim_order`** produces a non-deterministic
   axis ordering that can force shuffles even when the input is
   chunk-aligned. Always pass `dim_order=keep_dims + reduce_dims`
   explicitly. (Deep dive §4, connectivity-plan §3.)

4. **`HexProj.label` is not dask-aware** — it calls `np.asarray`. Users
   build a dask-backed `hex_ids` DataArray via
   `xr.apply_ufunc(hp.label, lon_da, lat_da, dask="parallelized",
   output_dtypes=[np.int64])`. This is unrelated to `hex_counts` itself
   (the input is *already* labelled); noted here only so test fixtures do
   the right thing.

5. **`pyproj.Transformer` is not thread-safe** — `hex_counts` does not call
   `hp.label` inside `map_partitions`, so #32 does not trip this. If a
   future variant does, carry a `HexProj` instance per partition or dispatch
   under `scheduler="processes"`.

6. **`dd.Series.value_counts(sort=True)` forces a shuffle** to produce the
   globally-sorted ordering. Use `sort=False`; we do not rely on ordering.

7. **`xr.Dataset.to_dask_dataframe` on numpy-backed input** still returns a
   `dd.DataFrame` (with one partition). This is fine for our unified code
   path but deserves mention: the type of the intermediate is always
   `dd.*` regardless of input backing.

8. **Empty-partition edge case** — in pathological chunking where a
   partition is empty, `dd.groupby(...).size()` is still correct but
   returns an empty intermediate for that partition. No special-case
   needed in the implementation; noted in case a test triggers it.

9. **`reduce_dims=[]` (explicit empty list)** — currently treated as "no
   reduction", producing one row per element with `count=1`. Under the new
   design, `keep_dims == hex_ids.dims` means the groupby key is every dim
   plus `hex_id`, each group has one element → count=1. Semantics
   preserved; covered by `test_hex_counts_empty_reduce_dims` in the
   existing test file.

10. **`reduce_dims=None` (default)** — the current implementation treats
    this identically to `reduce_dims=[]`. Keep that. Document it in the
    docstring.

---

## 12. Open questions

1. **`hex_counts_lazy` — public or underscore-prefixed?** I have proposed
   public. Rationale: users coming off `hex_connectivity_dask` will want a
   matching lazy counter. But if the project would rather expose one public
   eager function and keep the lazy half private, renaming to
   `_hex_counts_lazy` and deleting the `__init__` export is a one-line
   change. Flagging for user input.

2. **Should `hex_counts` reuse `hex_connectivity_dask`'s rechunk-`obs=-1`
   pattern when the user's dims happen to be named `"traj"` and `"obs"`?**
   My current answer is no — magic-naming behaviour is a trap — but the
   connectivity function sets a precedent. The user may want a matching
   hint here (e.g. a `rechunk_hint: dict | None = None` passthrough to
   `hex_ids.chunk(...)`). Not proposed, flagged.

3. **Dense output helper?** The plan document for `hex_counts` mentions
   `.reindex(all_hex_ids, fill_value=0)` as a user-side convenience.
   Should the library ship a `hex_counts_dense(..., hex_ids_universe=...)`
   that does it? Orthogonal to this fix; defer.

4. **Which dask version to test against in CI?** We verified against
   `2026.1.2`. The `columns=` failure mode is version-dependent. If CI
   covers older dask, re-verify the `value_counts` path behaves
   identically. No evidence of breakage on recent versions so far.

---

## 13. Summary

- One-line fix is impossible; the OOM has two sources (full and partial
  reduction) and the original plan's `dd.Series` input path was never
  implemented.
- Root cause is `.values.ravel()` on the input. Remove it everywhere.
- Design routes every non-Series input through
  `to_dask_dataframe(dim_order=...)` → `value_counts` (full) or
  `groupby(...).size()` (partial), staying lazy until a small count table
  is materialised.
- `hex_counts` keeps its eager `GeoDataFrame` return; a new
  `hex_counts_lazy` exposes the lazy `dd.Series` for pipeline users.
- `dd.Series` input path (the one missing from the original
  implementation) is a one-liner: `hex_ids.value_counts(sort=False)`.
- `INVALID_HEX_ID` is preserved through every stage (`value_counts` keeps
  it; `groupby.size` keeps it; `HexProj.to_geodataframe` already maps it
  to `geometry=None`).
- Geometry construction switches from the per-hex loop in
  `_build_counts_geodataframe` to the batched `HexProj.to_geodataframe`;
  the old helper is deleted outright.
- Dispatch uses `dask.is_dask_collection` uniformly — verified to work on
  `xr.DataArray`, `dd.Series`, `pd.Series`, numpy-backed DataArray.
- Tests mirror `tests/test_hex_connectivity_dask.py`: dask-backed fixture,
  lazy-graph assertion, chunk-independence parametrize, correctness
  against the numpy ground truth, `INVALID_HEX_ID` preservation,
  `dd.Series` input, coordinate-value preservation in the MultiIndex.
- Pitfalls: the broken `columns="hex_id"` form, `xr.groupby` iteration
  eagerness, missing `dim_order`, `HexProj.label` non-dask-awareness,
  `pyproj` thread-safety (latent), `value_counts(sort=True)` shuffle.
