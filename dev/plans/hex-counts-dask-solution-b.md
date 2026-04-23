# hex_counts dask OOM — solution design (B)

Companion document to `dev/plans/hex-counts-dask-oom.md`. This file proposes a
concrete fix; the deep-dive stays authoritative on verified behaviour.

---

## 1. Problem recap

`hextraj.hex_analysis.hex_counts` OOMs on dask-backed `xr.DataArray` inputs
because both the full-reduction branch (`hex_analysis.py:153-159`) and the
partial-reduction branch (`hex_analysis.py:161-172`) call `.values.ravel()` on
dask-backed data, materialising the entire input into the calling process.
The additional latent bottleneck is `_build_counts_geodataframe`
(`hex_analysis.py:396-430`), whose per-hex Python loop calls
`hp.hex_corners_lon_lat` once per hex (seven transformer round-trips each).
The full background, with verified library behaviour, is in
`dev/plans/hex-counts-dask-oom.md` §§1–5.

---

## 2. Design goals and non-goals

**Goals**

- No `.values` on dask-backed inputs in any code path. All aggregation stays
  inside dask until the (tiny) count table is ready.
- Input polymorphism: `xr.DataArray` (numpy and dask), `pd.Series`,
  `dask.dataframe.Series`. One function, four detected paths.
- `INVALID_HEX_ID` (`-1`) is always preserved as an ordinary bucket with
  `geometry=None`.
- Batched geometry: reuse `HexProj.to_geodataframe` and retire the per-hex
  loop.
- Follow the existing `hex_connectivity_dask` pattern (deep-dive §4):
  `to_dask_dataframe(dim_order=[...])` + dask-dataframe aggregation.

**Non-goals**

- A lazy return type for `hex_counts`. The deep dive §5 flags the
  lazy-vs-eager tension; section 8 below picks eager (`GeoDataFrame`) as the
  single return type, with a rationale.
- `hp.label` dask-awareness. The function accepts **already-labelled** hex
  IDs (per the existing contract); labelling upstream is the user's job and is
  already handled by `xr.apply_ufunc(hp.label, ..., dask="parallelized")`.
- `pyproj` thread-safety (deep-dive §4). Not touched by this fix; noted.
- Backwards compatibility. `AGENTS.md` is explicit: no deprecation shims.

---

## 3. Proposed architecture (high level)

```
hex_counts(hex_ids, reduce_dims=None, hp=None)
  ├── dispatch on input type ──────────────────────────┐
  │                                                    │
  │  xr.DataArray (numpy) ──► eager numpy path         │
  │  xr.DataArray (dask)  ──► dask-dataframe path ─┐   │
  │  dask.dataframe.Series ──► dask-dataframe path ┤   │
  │  pd.Series            ──► eager numpy path ────┘   │
  │                                                    │
  └── all paths converge on: ──────────────────────────┘
       (counts_series | counts_dataframe)
       .pipe(_attach_geometry)
       → GeoDataFrame
```

Every path produces a small pandas Series (full reduction) or pandas DataFrame
(partial reduction) whose index carries `hex_id` (and optional keep-dim
levels), and whose values are counts. The shared `_attach_geometry` helper
builds the `GeoDataFrame` in one batched geometry call.

Concretely, the module is organised as:

```
hex_counts                     # public entry point; dispatch only
_hex_counts_numpy(...)         # numpy xr.DataArray + pd.Series path
_hex_counts_dask(...)          # dask xr.DataArray + dd.Series path
_attach_geometry(counts, hp)   # shared; wraps HexProj.to_geodataframe
```

`_hex_counts_dask` internally normalises `xr.DataArray` → `dd.DataFrame` with
`hex_id` column and optional keep-dim columns, then hands off to a single
groupby/value_counts reducer.

---

## 4. Dispatch and input handling

### 4.1 Detection

Use duck-typing against `dask.is_dask_collection`, not chunk attributes. The
former is the public, stable predicate and is already imported in the
connectivity path. isinstance checks against `dd.Series` / `dd.DataFrame`
layer on top to separate the xarray-dask case from the dataframe-dask case.

```python
import dask
import dask.dataframe as dd

def hex_counts(hex_ids, reduce_dims=None, hp=None):
    if hp is None:
        hp = HexProj()

    # dask.dataframe.Series path
    if isinstance(hex_ids, dd.Series):
        return _hex_counts_dask_series(hex_ids, hp)

    # xr.DataArray path
    if isinstance(hex_ids, xr.DataArray):
        if dask.is_dask_collection(hex_ids.data):
            return _hex_counts_dask_dataarray(hex_ids, reduce_dims, hp)
        return _hex_counts_numpy_dataarray(hex_ids, reduce_dims, hp)

    # pd.Series path — fall through (no reduce_dims)
    if isinstance(hex_ids, pd.Series):
        return _hex_counts_numpy_series(hex_ids, hp)

    raise TypeError(
        f"hex_counts: unsupported input type {type(hex_ids).__name__}"
    )
```

Per `AGENTS.md` "no defensive error handling" — the only raise is the
dispatch fallthrough, which is unavoidable because silently accepting
unknown types would hide user bugs.

### 4.2 Why these four paths (and not a single unified path)

A single "convert everything to `dd.Series` and groupby" path would work
semantically, but:

1. It pays dask overhead (scheduler, task graph) for trivially small numpy
   inputs (the existing test fixtures are 5×2).
2. It loses the keep-dim column structure that `xr.Dataset.to_dask_dataframe`
   gives for free when the input is an `xr.DataArray`.

Keeping the numpy path eager and the dask paths dask-native matches what the
project already does for `hex_connectivity_dask` vs `hex_connectivity`.

### 4.3 What `reduce_dims` means per input type

| Input                         | `reduce_dims`                                   |
|-------------------------------|-------------------------------------------------|
| `xr.DataArray` (numpy/dask)   | str / list[str] / None; `None` → `[]` (no reduction) |
| `pd.Series`                   | Ignored (flat already)                          |
| `dask.dataframe.Series`       | Ignored (flat already)                          |

This matches the documented contract in `dev/plans/hex-analysis-functions.md`.

---

## 5. Full-reduction pipeline

**Trigger:** `reduce_dims` covers every dim of `hex_ids`, i.e. `keep_dims == []`.

### 5.1 Dask path

Convert the dask-backed `xr.DataArray` directly to a 1D `dd.Series` via the
dask-array layer. **Do not** route through `to_dask_dataframe` here — that
forces a coordinate promotion and a shuffle we don't need for a pure flatten.

```python
# hex_ids.data is dask.array.Array; .ravel() is lazy and preserves chunking.
flat = hex_ids.data.ravel()
series = dd.from_dask_array(flat)          # 1-D Series, no `columns` kwarg
series.name = "hex_id"
counts = series.value_counts(sort=False)   # lazy tree-reduction
counts_pd = counts.compute()               # small: unique hexes only
counts_pd.index.name = "hex_id"
```

**Key choices and rationale:**

- `dd.from_dask_array(flat)` returns a 1D `Series` directly. This is the
  verified form (deep-dive §4). The `columns="hex_id"` form from the issue
  snippet fails on dask `2026.1.2` — we do **not** use it.
- `value_counts(sort=False)` avoids a final global sort. The order is
  irrelevant; the caller receives a GeoDataFrame indexed by `hex_id` and will
  sort or filter as it likes.
- `INVALID_HEX_ID` is preserved because it is just another int64 value —
  `value_counts` does not special-case `-1`. Confirmed by deep-dive §4's
  "dask.dataframe value_counts is a parallel tree-reduction" statement.

### 5.2 Numpy path

Unchanged from today, minus the dead `keep_dims` branch (which is now
dispatched away upstream):

```python
flat = hex_ids.values.ravel()
counts_pd = pd.Series(flat).value_counts(sort=False)
counts_pd.index.name = "hex_id"
```

Both paths end at an identical object: a `pd.Series` named "count" (after a
rename inside `_attach_geometry`) indexed by `hex_id`.

---

## 6. Partial-reduction pipeline

**Trigger:** `reduce_dims` is a strict subset of `hex_ids.dims`.

### 6.1 Dask path — the core transformation

Route through `to_dask_dataframe(dim_order=[...])`. Assemble a mini Dataset
whose variables are (a) the hex ID array and (b) the keep-dim coordinate
arrays broadcast to the full shape. This is the same pattern
`hex_connectivity_dask` uses.

```python
all_dims = list(hex_ids.dims)
keep_dims = [d for d in all_dims if d not in reduce_dims]
dim_order = keep_dims + reduce_dims   # keep-dim-major ordering

# Build the mini Dataset: hex_id as the main variable, keep-dim coords
# promoted to data variables so they survive to_dask_dataframe as columns.
var_dict = {"hex_id": hex_ids.rename("hex_id")}
for d in keep_dims:
    coord = hex_ids.coords.get(d, xr.DataArray(
        np.arange(hex_ids.sizes[d]), dims=[d]
    ))
    var_dict[d] = coord.broadcast_like(hex_ids)

mini_ds = xr.Dataset(var_dict)
ddf = mini_ds.to_dask_dataframe(dim_order=dim_order)
# ddf columns: [*keep_dims, 'hex_id', ...index columns xarray adds]

counts = ddf.groupby(keep_dims + ["hex_id"]).size()
counts_pd = counts.compute()
counts_pd.name = "count"
counts_pd.index.names = keep_dims + ["hex_id"]
```

**Why `dim_order = keep_dims + reduce_dims`:** `to_dask_dataframe` flattens
with the first dim as outermost. If the user's dask chunking aligns with the
keep dims (e.g. user chunks on `"traj"` and reduces `"obs"`), partitions map
one-to-one with keep-dim blocks and no shuffle is needed. If the user's
chunking aligns with reduce dims instead, `to_dask_dataframe` will interleave
rows within partitions — the `groupby` still computes the correct answer
(dask groupby is shuffle-aware) but may pay a shuffle cost. This matches the
connectivity-path approach in `dev/plans/dask-native-connectivity.md` §3
("Always pass `dim_order` explicitly") but we do **not** silently `rechunk`.

### 6.2 Chunk-alignment policy

`hex_connectivity_dask` rechunks `obs` to `-1` because it has a
semantically-privileged `obs_dim` — it needs whole trajectories per partition
to set `from_id = hex_id[obs=0]`. `hex_counts` has no such requirement: a
groupby over `keep_dims + ["hex_id"]` is partition-agnostic. Therefore:

- **No rechunking.** Accept whatever chunking the user brought.
- If `dim_order` alignment is unfavourable, dask pays a shuffle. Document this
  in the docstring: *"Performance is best when the chunks of `hex_ids` align
  with `keep_dims`; misalignment triggers a dask shuffle during aggregation."*

This trade-off is the honest one: silently rechunking a 20 GB input is worse
than paying the shuffle.

### 6.3 Numpy path

The current per-group `.values.ravel()` loop is correct but slow. Replace
with a single numpy-level groupby via `pd.DataFrame`:

```python
flat = hex_ids.values.ravel()
# Build keep-dim coordinate meshgrid ravelled in the same order as hex_ids.
# xarray's default is C order over dims in their declared order, so we follow
# that explicitly with np.meshgrid(..., indexing='ij').
coord_grids = np.meshgrid(
    *[hex_ids.coords.get(d, xr.DataArray(np.arange(hex_ids.sizes[d]),
                                         dims=[d])).values
      for d in hex_ids.dims],
    indexing="ij",
)
df = pd.DataFrame({d: g.ravel() for d, g in zip(hex_ids.dims, coord_grids)})
df["hex_id"] = flat
counts_pd = df.groupby(keep_dims + ["hex_id"]).size()
counts_pd.name = "count"
```

Symmetric to the dask path, which makes testing the dask path against the
numpy path a natural correctness oracle.

### 6.4 Empty-input edge case

When every observation is `INVALID_HEX_ID`-filtered out or the input is
empty, `groupby().size()` returns an empty Series. `_attach_geometry` handles
this trivially — an empty geometry list is still a valid (empty)
`GeoDataFrame`.

---

## 7. Geometry construction

Retire `_build_counts_geodataframe` entirely and route through
`HexProj.to_geodataframe` (`hexproj.py:192-246`), which is already the
vectorised path: one batched inverse projection for all corners, one
`shapely.polygons` call, `INVALID_HEX_ID` → `None` geometry built in.

### 7.1 Full-reduction assembly

```python
def _attach_geometry(counts_pd, hp):
    # counts_pd: pd.Series with name None, index named "hex_id" (full-reduction)
    #            or pd.Series with MultiIndex ending in "hex_id" (partial-reduction)
    if isinstance(counts_pd.index, pd.MultiIndex):
        hex_id_values = counts_pd.index.get_level_values("hex_id").to_numpy()
    else:
        hex_id_values = counts_pd.index.to_numpy()

    # Unique hex IDs drive the geometry batch; attach back via reindex.
    unique_ids = np.unique(hex_id_values)
    geo = hp.to_geodataframe(unique_ids)   # index=unique_ids, Polygon/None

    # Broadcast geometries back to the (possibly repeated) hex_id axis.
    geometries = geo.geometry.reindex(hex_id_values).values

    gdf = gpd.GeoDataFrame(
        {"count": counts_pd.to_numpy(), "geometry": geometries},
        index=counts_pd.index,
        crs="EPSG:4326",
    )
    return gdf
```

**Why reindex-by-unique:** in the partial-reduction output, the same
`hex_id` appears across multiple keep-dim rows. Calling `to_geodataframe`
on the unique set avoids redundant projection work and then pandas' native
reindex broadcasts the small geometry table back to the full index.

### 7.2 Index naming

- Full reduction: `counts_pd.index.name = "hex_id"`.
- Partial reduction: `counts_pd.index.names = keep_dims + ["hex_id"]`.

Assembled inside each path before `_attach_geometry` is called. The output
`GeoDataFrame` inherits this index directly — matches the existing contract
exercised in `test_hex_counts_partial_dims_multiindex_*`.

### 7.3 INVALID_HEX_ID

`HexProj.to_geodataframe` already emits `None` for `INVALID_HEX_ID`
(`hexproj.py:209-212, 240-241`). No special-casing needed anywhere else.
Deep-dive §3 and §5 both call this out as a required contract — the fix
honours it by construction.

---

## 8. Return type and laziness

### 8.1 Decision: eager `GeoDataFrame` only

`hex_counts` returns `gpd.GeoDataFrame`, always. No lazy return, no mode
flag, no parallel `hex_counts_dask` function.

### 8.2 Rationale (against the deep-dive §5 tension)

1. **The output is small.** Sparse counts: O(unique hexes) rows. Even a
   billion-observation input yields a count table that fits in pandas
   memory. The dask graph is useful for the *reduction*, not the *result*.

2. **Geometry is eager.** `GeoDataFrame` cannot be lazy without a separate
   `dask-geopandas` dependency. Introducing it here is out of scope.

3. **Asymmetry with connectivity is acceptable.** `hex_connectivity_dask`
   returns a lazy `dd.DataFrame` because its result is *not* necessarily
   small (obs-resolved output has `n_traj × n_obs` rows). `hex_counts`'s
   result is always small — the laziness payoff is absent.

4. **One function, one contract.** The deep dive notes "whether `hex_counts`
   should follow [the lazy-by-default pattern], or stay eager and match its
   current signature, is an open design choice." We pick eager — the
   existing signature stands. Section 9 gives the final spelling.

### 8.3 What "lazy until the end" means here

The aggregation pipeline is lazy: `ravel → dd.from_dask_array →
value_counts` (or `to_dask_dataframe → groupby → size`) builds a dask graph.
The graph is only materialised by the single `counts.compute()` inside
`_hex_counts_dask`. After that, everything is small pandas/geopandas. This
matches the design intent in `dev/plans/hex-analysis-functions.md` ("the
pipeline must remain lazy… until the small count table can be materialised
cheaply").

---

## 9. Final API (signatures, types)

```python
def hex_counts(
    hex_ids: xr.DataArray | pd.Series | dd.Series,
    reduce_dims: str | list[str] | None = None,
    hp: HexProj | None = None,
) -> gpd.GeoDataFrame:
    """Count hex visits, optionally reducing over specified dimensions.

    Accepts an xr.DataArray of int64 hex IDs (numpy- or dask-backed), a
    pd.Series, or a dask.dataframe.Series. The aggregation is lazy for
    dask-backed inputs; the final GeoDataFrame is always eager.

    Args:
        hex_ids: Hex IDs as xr.DataArray (numpy- or dask-backed), pd.Series,
            or dask.dataframe.Series. int64 throughout; INVALID_HEX_ID (-1)
            is preserved as an ordinary bucket with geometry=None.
        reduce_dims: Dimension name(s) to sum over. None means no reduction
            (numpy path retains the full grid; Series paths ignore the arg).
            Required to be a subset of hex_ids.dims when hex_ids is a
            DataArray.
        hp: HexProj for geometry. Defaults to HexProj() if omitted.

    Returns:
        GeoDataFrame with:
          - Index: "hex_id" when all dims reduced (or Series input),
            otherwise a MultiIndex of (keep_dims..., "hex_id").
          - Columns: "count" (int64) and "geometry" (Polygon or None).

    Notes:
        Performance is best when the chunks of a dask-backed input align
        with keep_dims; misalignment triggers a dask shuffle during
        aggregation. No automatic rechunking is performed.
    """
```

### Internals (non-public)

```python
def _hex_counts_numpy_dataarray(hex_ids, reduce_dims, hp) -> GeoDataFrame
def _hex_counts_numpy_series(series, hp) -> GeoDataFrame
def _hex_counts_dask_dataarray(hex_ids, reduce_dims, hp) -> GeoDataFrame
def _hex_counts_dask_series(series, hp) -> GeoDataFrame
def _attach_geometry(counts_pd: pd.Series, hp) -> GeoDataFrame
```

Kept private (leading underscore) — they are implementation detail and may
be refactored freely.

### Backwards compatibility

The public signature of `hex_counts` is unchanged from today. Internals are
rewritten. `_build_counts_geodataframe` is **deleted** without a deprecation
stub, per `AGENTS.md` ("if something is gone, it's gone"). Any test that
imports `_build_counts_geodataframe` directly is rewritten to exercise the
public path.

Similarly, `_build_edge_geometries` (used only by
`hex_connectivity`/`hex_connectivity_power`) is a separate issue and not
touched by this change.

---

## 10. Tests

New test file: `tests/test_hex_counts_dask.py`, patterned on
`tests/test_hex_connectivity_dask.py`. The existing numpy tests in
`tests/test_hex_analysis.py` stay untouched — they guard the numpy path.

### 10.1 Fixtures

```python
@pytest.fixture
def hp():
    return HexProj(hex_size_meters=2_000_000)


@pytest.fixture(params=[(2, 3), (3, 3), (6, 1)])
def chunks(request):
    return request.param   # (chunk_traj, chunk_obs)


@pytest.fixture
def hex_ids_dask(hp, chunks):
    """6 traj × 3 obs dask-backed hex IDs with two NaN positions."""
    rng = np.random.default_rng(42)
    lon_np = rng.uniform(-10, 10, size=(6, 3)).astype(np.float64)
    lat_np = rng.uniform(-5, 5, size=(6, 3)).astype(np.float64)
    lon_np[0, 1] = np.nan
    lat_np[0, 1] = np.nan
    lon_np[3, 2] = np.nan
    lat_np[3, 2] = np.nan
    lon = da.from_array(lon_np, chunks=chunks)
    lat = da.from_array(lat_np, chunks=chunks)
    # Label lazily via apply_ufunc (the canonical path).
    hex_ids = xr.apply_ufunc(
        hp.label, lon, lat,
        dask="parallelized", output_dtypes=[np.int64],
    )
    return xr.DataArray(
        hex_ids, dims=("traj", "obs"),
        coords={"traj": np.arange(6), "obs": np.arange(3)},
    )


@pytest.fixture
def hex_ids_numpy(hp, hex_ids_dask):
    """Eager equivalent of hex_ids_dask — the numpy oracle."""
    return xr.DataArray(
        hex_ids_dask.values, dims=hex_ids_dask.dims, coords=hex_ids_dask.coords,
    )
```

### 10.2 Required tests

- **`test_returns_geodataframe_for_dask_input`** — result is a
  `gpd.GeoDataFrame` (not `dd.DataFrame`, not a pandas frame).
- **`test_lazy_before_compute`** — parametrized flag to assert that the
  intermediate `dd` object has a non-empty graph. Concretely, monkeypatch or
  introspect via a seam: expose `_hex_counts_dask_dataarray` for a direct
  call that returns the intermediate `dd.Series` / `dd.DataFrame` and
  assert `len(x.__dask_graph__()) > 0`. Alternatively: assert `hex_counts`
  does not call `.values` by patching `xr.DataArray.values` and letting any
  access raise. Prefer the latter — it is the real regression guard.
- **`test_matches_numpy_full_reduction`** — assert
  `hex_counts(hex_ids_dask, reduce_dims=["traj", "obs"])` equals
  `hex_counts(hex_ids_numpy, reduce_dims=["traj", "obs"])` on the `count`
  column, compared after aligning by `hex_id` (use `.sort_index()`).
- **`test_matches_numpy_partial_reduction_obs_kept`** — same oracle pattern
  for `reduce_dims=["traj"]`.
- **`test_matches_numpy_partial_reduction_traj_kept`** — symmetric,
  `reduce_dims=["obs"]`.
- **`test_invalid_hex_id_preserved`** — `INVALID_HEX_ID in result.index`
  (or in the `hex_id` level of the MultiIndex) and
  `result.loc[... INVALID_HEX_ID, "geometry"] is None`.
- **`test_dd_series_input`** — feed `dd.from_pandas(pd.Series([h1, h2, -1,
  h1]), npartitions=2)` and assert correct counts, correct index name,
  `-1` preserved.
- **`test_chunk_independence`** — parametrize over `chunks` fixture values;
  assert `result.sort_index()` is identical across chunk layouts.
- **`test_does_not_materialise_input`** — build a dask-backed input whose
  total size exceeds a deliberately tight memory budget (e.g. chunks of
  `(100, 100)` int64 with shape `(10_000, 100)` ≈ 8 MB; scale down/up to
  keep test time reasonable). Assert the result is computed without raising
  `MemoryError`; more sharply, assert `hex_ids.data` is still a dask array
  after the call (untouched). Secondary: use a `pytest.MonkeyPatch` to
  replace `xr.DataArray.values` on the test-local array with a sentinel
  that raises, and assert that still succeeds (full regression guard
  against `.values.ravel()` sneaking back in).
- **`test_custom_dim_names`** — build a DataArray with `dims=("particle",
  "time")`, reduce over both, reduce over one. Guards against any hardcoded
  `"traj"`/`"obs"` in the partial-reduction path.

### 10.3 Geometry regression tests

- **`test_geometry_batched_once_per_unique_hex`** — monkeypatch
  `HexProj.hex_corners_lon_lat` to raise. The test calls `hex_counts` and
  asserts no exception — the batched path in
  `HexProj.to_geodataframe` does not call `hex_corners_lon_lat`. This kills
  the per-hex loop regression.

### 10.4 Style

Follow `tests/test_hex_connectivity_dask.py`:
- plain `pytest` functions
- `@pytest.mark.parametrize` for chunk and dim-order sweeps
- no test classes
- every test imports `hex_counts` from `hextraj.hex_analysis`

---

## 11. Known pitfalls

Implementers must avoid the following, found during investigation:

1. **`dd.from_dask_array(flat, columns="hex_id")["hex_id"]`** — fails on
   dask `2026.1.2` with `TypeError: '<' not supported between instances of
   'str' and 'int'` (deep-dive §4). Use `dd.from_dask_array(flat)` and set
   `.name` afterwards.

2. **`.values` on any dask-backed DataArray.** Every path that currently
   does this is a latent OOM. Search-and-destroy across `hex_analysis.py`.

3. **`to_dask_dataframe()` without explicit `dim_order=`.** Produces
   nondeterministic row ordering and can trigger spurious shuffles
   (deep-dive §4, `dev/plans/dask-native-connectivity.md` §3).

4. **Silent `rechunk`.** Do not rechunk the input. A 20 GB input rechunked
   is still a 20 GB input — and the shuffle can be worse than the
   original partition layout. Let the user own chunking.

5. **`pd.MultiIndex.from_tuples` in the hot loop.** The current
   implementation accumulates a Python list of tuples and calls
   `from_tuples` at the end. For a partial reduction over 10⁵ keep-dim
   slices × 10³ unique hexes, that is 10⁸ tuples — slow and memory-heavy.
   The dask-dataframe `groupby(...).size()` path produces the MultiIndex
   natively; no Python loop needed.

6. **`pyproj.Transformer` thread-safety.** Not triggered by the
   reductions themselves (no per-chunk labelling inside `hex_counts` — the
   function receives already-labelled data). Flagged for completeness.

7. **`xr.DataArray.groupby(list_of_dims)` iteration.** Even without
   `.values`, iterating a grouped DataArray constructs a Python-level loop
   over groups. This is the shape of the current partial-reduction bug and
   should not reappear in any form — the new path delegates grouping to
   `dd.groupby`.

8. **Silent coordinate loss.** When a keep dim has no coordinate attached
   (xarray allows this), `to_dask_dataframe` drops the column. The design
   above falls back to
   `xr.DataArray(np.arange(sizes[d]), dims=[d])` to synthesise integer
   positions, matching what `hex_connectivity_dask` does for `obs_vals`.

9. **Empty-input groupby.** Both `value_counts` on an empty dask Series and
   `groupby().size()` on an empty dask DataFrame compute cleanly to an
   empty pandas object. `_attach_geometry` must handle an empty index —
   calling `hp.to_geodataframe(np.array([], dtype=np.int64))` already does
   (verified by reading `hexproj.py:212`: `geometries = [None] * 0 → []`).

---

## 12. Open questions

Items the user should confirm before implementation starts:

1. **`dd.Series` with non-int64 dtype.** Spec says the Series should hold
   int64 hex IDs. Should `hex_counts` coerce (`astype(np.int64)`) or let
   pandas/dask raise on downstream comparison with `INVALID_HEX_ID`?
   Preference: no coerce (Pythonic — let the real error surface).

2. **Should `pd.Series` input flow through the dask path when large?**
   Currently no — `pd.Series` always takes the eager path. A user with a
   10 GB pandas Series will OOM. This is acceptable (they brought an
   eager Series); flagging it.

3. **`reduce_dims=None` semantics for `xr.DataArray`.** Current behaviour:
   no reduction, so the index has every element of every dim plus
   `hex_id` — essentially the input flattened with hex_id appended. Is
   this genuinely useful or should it raise? The existing test
   `test_hex_counts_empty_reduce_dims` exercises `reduce_dims=[]` (count 1
   per element). `None` is a distinct thing — I would make it default to
   reducing *all dims* (the common case), not *no dims*. This is a
   user-facing contract question; current implementation treats
   `None`→`[]`.

4. **Keep-dim coordinates as floats.** If a coordinate is float-valued
   (e.g. `time` as `np.datetime64` or a float longitude grid), the
   `groupby` produces a float-level MultiIndex. Fine for groupby
   correctness but awkward for `.loc` lookups. Not specific to this fix;
   flagging as pre-existing behaviour.

5. **Do we want `dask-geopandas` on the dependency list?** Currently no.
   This design does not need it. If a future story wants a lazy
   `GeoDataFrame` return, that's where it would come in — but the
   decision belongs to a separate conversation.

---

## 13. Summary

The fix has two structural moves and one cleanup:

1. **Dispatch `hex_counts` on input type**: four paths (`xr.DataArray`
   numpy, `xr.DataArray` dask, `pd.Series`, `dd.Series`), detected via
   `isinstance` + `dask.is_dask_collection`.
2. **Route the dask branches through `dd.from_dask_array` (full reduction)
   or `to_dask_dataframe(dim_order=...)` + `groupby(...).size()` (partial
   reduction)**. No `.values` anywhere. No silent rechunking. `INVALID_HEX_ID`
   flows through naturally.
3. **Delete `_build_counts_geodataframe`** and assemble geometry through
   `HexProj.to_geodataframe`, once, on the unique hex IDs.

Return type stays eager `GeoDataFrame`. Signature of `hex_counts` stays
unchanged (bar the addition of `dd.Series` to the `hex_ids` type union).
New tests live in `tests/test_hex_counts_dask.py` modelled on
`tests/test_hex_connectivity_dask.py`, with a regression guard that bans
`.values` on the input.
