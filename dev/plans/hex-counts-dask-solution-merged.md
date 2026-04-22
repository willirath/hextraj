# `hex_counts` dask OOM — merged solution plan

Merges the architecture of
[`hex-counts-dask-solution-a.md`](./hex-counts-dask-solution-a.md) with the
implementation detail of
[`hex-counts-dask-solution-b.md`](./hex-counts-dask-solution-b.md).
Authoritative background is
[`hex-counts-dask-oom.md`](./hex-counts-dask-oom.md).

## 1. Scope

- Fix OOM on dask-backed inputs (issue #32).
- Add `dd.Series` input path (planned in `hex-analysis-functions.md`, never shipped).
- Replace `_build_counts_geodataframe`'s per-hex loop with the batched path in `HexProj.to_geodataframe`.
- Expose a lazy-returning `hex_counts_lazy` for the streaming-to-zarr/parquet case.
- **Redesign `reduce_dims` defaults.** Both `None` and `[]` mean "reduce all dims". The current `None`→`[]`→"one row per element, count=1" chain is degenerate and gets cut.
- **Re-audit `tests/test_hex_analysis.py`.** Any test that encodes the old defaults or reaches into `_build_counts_geodataframe` is rewritten or deleted — not preserved.

Out of scope: `pyproj` thread-safety, `HexProj.label` dask-awareness, lazy `GeoDataFrame` via `dask-geopandas`.

No backwards-compat shims. No `_deprecated` stubs. `_build_counts_geodataframe` is deleted.

## 2. Architecture

```
hex_counts(hex_ids, reduce_dims=None, hp=None) -> GeoDataFrame
    │
    ├── hex_counts_lazy(hex_ids, reduce_dims=None)
    │       dispatch → dd.Series (full) or dd.DataFrame (partial)
    │       or pd.Series/pd.DataFrame for eager inputs
    │
    └── _attach_geometry(counts, hp) -> GeoDataFrame
            .compute() if lazy → batched geometry via HexProj.to_geodataframe
```

`hex_counts_lazy` is a public function in its own right — the `GeoDataFrame`-building layer is the only eager step.

## 3. Dispatch

Single predicate for lazy vs eager: `dask.is_dask_collection`. `isinstance` narrows container type.

| Input                          | reduce_dims used | Path                                           |
|--------------------------------|------------------|------------------------------------------------|
| `pd.Series`                    | ignored          | `series.value_counts(sort=False)`              |
| `dd.Series`                    | ignored          | `series.value_counts(sort=False)` (lazy)       |
| `xr.DataArray` (numpy-backed)  | used             | route through `to_dask_dataframe` path         |
| `xr.DataArray` (dask-backed)   | used             | route through `to_dask_dataframe` path (lazy)  |

Unified `to_dask_dataframe` path for both `xr.DataArray` backings (A's choice). Numpy input becomes a trivially-chunked `dd.DataFrame` with one partition; `.compute()` at the end is a no-op. This eliminates a whole branch vs B's four-path design.

Unknown input type raises `TypeError` from the dispatch fallthrough — the only defensive raise in the whole function.

## 4. Full-reduction pipeline

Triggered when `reduce_dims` is `None`, `[]`, or covers every dim of `hex_ids`.

```python
ds = hex_ids.to_dataset(name="__hex_id__")
ddf = ds.to_dask_dataframe(dim_order=list(hex_ids.dims))
counts = ddf["__hex_id__"].value_counts(sort=False)    # dd.Series (or pd.Series)
counts.index.name = "hex_id"
```

Series inputs short-circuit:

```python
if isinstance(hex_ids, (pd.Series, dd.Series)):
    counts = hex_ids.value_counts(sort=False)
    counts.index.name = "hex_id"
```

- `INVALID_HEX_ID` is preserved; `value_counts` keeps every value present.
- `sort=False` avoids a global shuffle. Eager `hex_counts` sorts by `hex_id` at the end for reproducibility; `hex_counts_lazy` returns unsorted.
- Do **not** use `dd.from_dask_array(flat, columns="hex_id")["hex_id"]` — broken on dask 2026.1.2 (`TypeError`, deep-dive §4). Not needed anyway; `to_dask_dataframe` is the canonical bridge.
- Internal column name `__hex_id__` avoids collisions if the user's DataArray has a dim literally named `hex_id`. Renamed to `hex_id` on the way out.

## 5. Partial-reduction pipeline

Triggered when `reduce_dims` is a strict subset of `hex_ids.dims`.

```python
all_dims = list(hex_ids.dims)
keep_dims = [d for d in all_dims if d not in reduce_dims]
dim_order = keep_dims + reduce_dims

var_dict = {"__hex_id__": hex_ids}
for d in keep_dims:
    coord = hex_ids.coords.get(d, xr.DataArray(np.arange(hex_ids.sizes[d]), dims=[d]))
    var_dict[d] = coord.broadcast_like(hex_ids)

mini_ds = xr.Dataset(var_dict)
ddf = mini_ds.to_dask_dataframe(dim_order=dim_order)
counts = ddf.groupby(keep_dims + ["__hex_id__"]).size().rename("count")
```

From B: the broadcast-keep-coord pattern, coord-missing fallback, and explicit `dim_order`. `hex_connectivity_dask` uses the same shape.

### Lazy-path output shape

`hex_counts_lazy` returns:
- Full reduction: `dd.Series` (or `pd.Series`) indexed by `hex_id`.
- Partial reduction: **`dd.DataFrame`** (or `pd.DataFrame`) with columns `(*keep_dims, "hex_id", "count")` — directly parquet/zarr-writable without `.reset_index()` gymnastics.

Under the hood we call `.reset_index()` on the Series returned by `groupby(...).size()` before returning. Trivial.

### Chunk-alignment policy

**No automatic rechunking.** `hex_connectivity_dask` rechunks `obs=-1` because it has privileged `(traj, obs)` semantics. `hex_counts` has no fixed dim semantics. `groupby().size()` produces correct results under any chunking — misaligned chunking pays a dask shuffle but does not break correctness or bloat memory.

Docstring call-out: *"Performance is best when `hex_ids` is chunked along `keep_dims`. Misalignment triggers a dask shuffle during aggregation."*

## 6. Geometry

Delete `_build_counts_geodataframe` outright. Build geometry once per unique hex and broadcast via `reindex` (B's optimisation):

```python
def _attach_geometry(counts, hp):
    # counts: pd.Series (full-reduction, index=hex_id) or pd.DataFrame
    #         (partial-reduction, columns include hex_id + count)
    if is_dask_collection(counts):
        counts = counts.compute()
    hex_id_col = counts.index if isinstance(counts, pd.Series) else counts["hex_id"]
    unique_ids = np.unique(hex_id_col.to_numpy())
    geo = hp.to_geodataframe(unique_ids)
    geometries = geo.geometry.reindex(hex_id_col).values
    ...  # assemble GeoDataFrame with the right index
    return gdf
```

`HexProj.to_geodataframe` (`hexproj.py:192-246`) already handles `INVALID_HEX_ID` → `geometry=None` via its `invalid` mask. No extra filtering anywhere in the pipeline.

## 7. Final API

```python
def hex_counts(
    hex_ids: xr.DataArray | pd.Series | dd.Series,
    reduce_dims: str | list[str] | None = None,
    hp: HexProj | None = None,
) -> gpd.GeoDataFrame:
    """Count hex visits, optionally keeping a subset of dims as index levels.

    reduce_dims:
        - None (default) or []: reduce over all dims.
        - str or list[str]: the dims to sum over; remaining dims become
          leading index levels of the result.

    Aggregation is lazy for dask-backed inputs; the small count table is
    materialised and decorated with hex polygon geometry on return.
    """


def hex_counts_lazy(
    hex_ids: xr.DataArray | pd.Series | dd.Series,
    reduce_dims: str | list[str] | None = None,
) -> dd.Series | dd.DataFrame | pd.Series | pd.DataFrame:
    """Lazy count-only form of hex_counts.  No geometry.

    For the streaming-to-parquet/zarr case: full reduction returns a Series
    indexed by hex_id; partial reduction returns a DataFrame with columns
    (*keep_dims, "hex_id", "count") ready for .to_parquet()/.to_zarr().
    """
```

Both exported from `hextraj.hex_analysis` and re-exported from `hextraj/__init__.py`.

## 8. Defaults redesign — deliberate breaking changes

These are intentional behaviour changes from the current implementation.

| Surface | Current | New |
|---|---|---|
| `reduce_dims=None` | `[]` → "no reduction, count=1 per element" | Reduce all dims |
| `reduce_dims=[]` | "count=1 per element" | Reduce all dims (same as `None`) |
| Output row ordering | insertion order from `value_counts(sort=False)` | sorted by `hex_id` in eager path; unsorted in lazy path |
| `_build_counts_geodataframe` (private) | exists, used | deleted |

Rationale: `hex_counts(hex_ids)` with no args should do the obvious thing — count visits across everything. The old "count=1 per element" degeneracy had no real use case; anyone wanting per-element labels can stay in `hex_ids` directly.

## 9. Tests

New file: `tests/test_hex_counts_dask.py`, patterned on `test_hex_connectivity_dask.py`. Plain `pytest`, `@pytest.mark.parametrize`, no test classes.

### Fixtures

```python
@pytest.fixture
def hp():
    return HexProj(hex_size_meters=2_000_000)

@pytest.fixture(params=[(2, 3), (3, 3), (6, 1)])
def chunks(request):
    return request.param

@pytest.fixture
def hex_ids_dask(hp, chunks):
    # 6 traj × 3 obs, NaN-injected, labelled via apply_ufunc
    ...

@pytest.fixture
def hex_ids_numpy(hex_ids_dask):
    return hex_ids_dask.compute()
```

### Required tests

1. `test_returns_geodataframe_for_dask_input`
2. `test_matches_numpy_full_reduction` — dask vs numpy oracle, `.sort_index()` before compare
3. `test_matches_numpy_partial_reduction_obs_kept`
4. `test_matches_numpy_partial_reduction_traj_kept`
5. `test_invalid_hex_id_preserved` — `-1` in index, `geometry is None`
6. `test_dd_series_input` — `dd.from_pandas(pd.Series(...), npartitions=2)`
7. `test_chunk_independence` — parametrized over `chunks`; results identical
8. `test_custom_dim_names` — `dims=("particle", "time")`; reduce over each
9. `test_reduce_dims_none_reduces_all` — new default behaviour
10. `test_reduce_dims_empty_list_reduces_all` — symmetric with `None`
11. `test_hex_counts_lazy_returns_dd_series_full_reduction`
12. `test_hex_counts_lazy_returns_dd_dataframe_partial_reduction` — columns are `(*keep_dims, "hex_id", "count")`
13. `test_hex_counts_lazy_graph_nonempty` — `len(result.__dask_graph__()) > 0`
14. `test_preserves_coord_values_in_multiindex` — non-trivial coord values survive

### Regression guards (B's idea, kept)

15. `test_does_not_call_values_on_input` — `pytest.MonkeyPatch` replaces `xr.DataArray.values` with a property that raises. `hex_counts` must still succeed on the dask-backed fixture.
16. `test_geometry_batched_not_per_hex` — monkeypatch `HexProj.hex_corners_lon_lat` to raise. `hex_counts` must still succeed (batched path in `HexProj.to_geodataframe` never calls it).

### `tests/test_hex_analysis.py` re-audit

Delete: `test_hex_counts_empty_reduce_dims` (encodes the old `[]` → "count=1 per element" contract that we are removing).

Update or delete anything that reaches into `_build_counts_geodataframe`. Keep the numpy-path behavioural tests (returns GeoDataFrame, count column correct, INVALID handling, MultiIndex structure) — those are still correct after the redesign. Review them one by one; don't wholesale-preserve.

## 10. Pitfalls (short list)

1. `dd.from_dask_array(flat, columns=...)["..."]` — fails on dask 2026.1.2. Don't use it; we don't.
2. `.values` on a dask-backed `DataArray` — the original bug. The monkeypatch test above guards against it sneaking back in.
3. `to_dask_dataframe` without `dim_order` — nondeterministic row ordering, spurious shuffles. Always pass it explicitly.
4. `xr.DataArray.groupby(list)` iteration — lazy at construction but eager per group if you `.values` inside. We don't use xarray groupby; we use `dd.groupby`.
5. `pd.MultiIndex.from_tuples` in a Python loop — shape of the current partial-reduction bug. `dd.groupby().size()` emits a proper MultiIndex natively.
6. Silent coordinate loss on dims with no explicit coord — fall back to `np.arange(size)`.
7. `value_counts(sort=True)` forces a global shuffle — use `sort=False` and sort the eager result at the end if needed.
8. Internal column name collision — use `__hex_id__` internally, rename to `hex_id` on output.

## 11. Documentation updates

Surfaces that reference `hex_counts` and must not go stale:

- **`src/hextraj/hex_analysis.py` docstrings.** `hex_counts` and new `hex_counts_lazy` get docstrings covering: new `reduce_dims` default, lazy-path return shapes (Series for full, DataFrame for partial), chunk-alignment note, `INVALID_HEX_ID` handling, a one-line streaming example (`hex_counts_lazy(...).to_parquet(...)`). Sphinx autoapi regenerates `docs/_build/html/api/hextraj/hex_analysis/index.html` from these — no hand-written API narrative to update.
- **`notebooks/hex_analysis.ipynb`.** Fix only what the API change breaks. Current usage is `hex_counts(hex_ids, reduce_dims=["traj", "obs"], hp=hp)` — explicit list, unaffected. If nothing breaks, do not touch the notebook. Demonstrating the new `reduce_dims=None` default and `hex_counts_lazy` streaming path is **deferred to a separate GitHub issue** (coverage-gap follow-up, alongside issue #31).
- **`README.md` and `docs/index.md`.** Only passing mentions (one line each listing `hex_counts` as an analysis function). No content change needed.
- **`docs/_build/`.** Stale after any docstring or notebook change. Rebuild with `pixi run docs-build` at the end of the implementation PR; don't commit the old build.
- **`dev/plans/hex-analysis-functions.md`.** The original design plan. Add a note at the top pointing to this merged plan for the dask/defaults redesign — don't rewrite it; it's historical context.

## 12. Open questions

None. Decisions committed:

- `hex_counts_lazy` is public.
- Partial-reduction lazy return is `dd.DataFrame` (columns `(*keep_dims, "hex_id", "count")`), not `dd.Series` with MultiIndex — parquet/zarr-friendly.
- Both `reduce_dims=None` and `reduce_dims=[]` mean "reduce all dims".
- Eager output sorted by `hex_id`; lazy output unsorted.
- No auto-rechunking; documented as user responsibility.
- `dd.Series` dtype not coerced — if user passes non-int64, the real error surfaces (Pythonic, per `AGENTS.md`).
- `dask-geopandas` not added as a dependency. `hex_counts_lazy` returns count tables only; geometry is eager-only and attaches in `hex_counts`.
