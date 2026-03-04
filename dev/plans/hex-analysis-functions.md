# Hex analysis convenience functions

## Overview

This plan covers two families of analysis functions that sit above the raw
`hp.label()` output and provide statistically meaningful reductions. Both
operate on an `xr.DataArray` of int64 hex IDs — the natural output of calling
`hp.label()` and wrapping the result in xarray. Both treat `INVALID_HEX_ID` as
a regular bucket rather than filtering it out, so the user receives the full
picture and can decide how to handle it.

Neither function is a method on `HexProj`. They are standalone module-level
functions that accept labelled data from any `HexProj` instance.

Connectivity matrices and heatmaps are often sparse: a trajectory dataset may
touch only a few hundred hex IDs out of the large integer space of all possible
IDs. The design principle is: **`xr.DataArray` (or dask/pandas Series) in,
`GeoDataFrame` out**. Named dimensions and lazy Dask backing are preserved on
input; the output is a `GeoDataFrame` that integrates naturally with
downstream visualisation and geospatial tooling.

The two output shapes are:

- **Heatmaps** (`hex_counts`): a GeoDataFrame of hex polygons with a `count`
  column. Each row is a hex cell (including INVALID_HEX_ID, which has
  `geometry=None`), with the number of observations that fell in it.

- **Connectivity** (`hex_connectivity`): a GeoDataFrame of LineStrings with a
  `count` column, indexed by a `(from_id, to_id)` MultiIndex. This structure
  makes `from_id` the primary lookup key — the user can query all destinations
  reachable from hex X via `.loc[(X, slice(None))]`. INVALID_HEX_ID endpoints
  have `geometry=None`, consistent with existing `HexProj` methods.

Dask compatibility is preserved throughout: the computation pipeline remains
lazy until `.compute()` is called on the output.

---

## Input data model: hex_ids as an xr.DataArray

Throughout this plan, `hex_ids` is an `xr.DataArray` with dtype `int64`,
arbitrary named dimensions, and integer coordinate values. The Cape Verde
example dataset used throughout has shape and dims:

```
hex_ids.shape == (n_island, n_year, n_obs, n_traj)
hex_ids.dims  == ("island", "year", "obs", "traj")
```

Positions where the original lon/lat was NaN are filled with `INVALID_HEX_ID`
(`np.int64(-1)`), as produced by `HexProj.label()`. These positions are
**included** in all count and connectivity operations as a dedicated bucket.
Users who want to exclude them can filter `.drop(INVALID_HEX_ID)` on the
returned GeoDataFrame.

### Why positions become invalid

Invalid hex IDs arise from invalid *particle positions*. In a well-run
simulation there is essentially one cause: **domain exit** — the particle left
the region covered by circulation data and is now considered out of bounds.
This is meaningful directional information: it tells you which hexes export
particles out of the domain.

A secondary cause is **particle deletion** (biological death, numerical error,
intentional culling), which produces the same sentinel value but different
physical meaning. Both are captured in the INVALID_HEX_ID bucket. The
distinction is left to the user: the full count is always reported, and
normalisation choices (over total observations, total trajectories, or total
time steps) are the user's responsibility.

- **Heatmaps**: the raw count $C(k, h)$ includes INVALID_HEX_ID as a bucket.
  Users can normalise over all observations (including invalid) or over valid
  observations only, depending on intent.

- **Connectivity**: a trajectory that starts valid (obs=0) but becomes invalid
  before obs=-1 connects to INVALID_HEX_ID as its destination. The pair
  `(h, INVALID_HEX_ID)` in the result tells the user which starting hexes
  export particles out of the domain. This is preserved in the output by
  default.

The functions accept `xr.DataArray` as the primary input type because named
dimensions allow the user to pass dimension names as strings directly, avoiding
ambiguous positional axis arithmetic. `hex_counts` additionally accepts a
`dask.dataframe.Series` or `pd.Series` for cases where the user already holds
a flat column of hex IDs from a prior pipeline step.

---

## 1. Hex heatmaps: `hex_counts`

### Proposed signature

```python
def hex_counts(
    hex_ids: xr.DataArray | dask.dataframe.Series | pd.Series,
    reduce_dims: str | list[str] | None = None,
) -> gpd.GeoDataFrame:
    ...
```

- `hex_ids`: `xr.DataArray` of int64 hex IDs, **or** a `dask.dataframe.Series`
  / `pd.Series` of int64 hex IDs (already 1D; `reduce_dims` is not needed and
  is ignored when a Series is passed).
- `reduce_dims`: string or list of strings naming the dimensions to sum over.
  All other dimensions are retained as output index levels. Required when
  `hex_ids` is an `xr.DataArray`; ignored for Series input.
- Returns: `gpd.GeoDataFrame` with:
  - Index: `hex_id` (when all dims are reduced) or a `pd.MultiIndex` of
    `(unreduced_dim_0, ..., unreduced_dim_n, hex_id)` when unreduced
    dimensions remain.
  - Column `count`: non-negative integer visit count.
  - Column `geometry`: hex polygon for each valid hex ID, `None` for
    INVALID_HEX_ID (consistent with `HexProj` polygon methods).
  - All observed hex IDs appear, including INVALID_HEX_ID. Only hex IDs that
    were never observed are absent.

### What it does and why

A hex heatmap answers: *how many times was each hex visited (including
out-of-domain), collapsing over a chosen set of dimensions?* The result is a
count GeoDataFrame indexed by hex ID and any unreduced dimensions.

The practical approach is to flatten the chosen dimensions into a single axis
and apply a `groupby`-style count keyed by hex ID. INVALID_HEX_ID is treated
as an ordinary bucket. This produces a sparse result (only observed hex IDs
appear, but INVALID_HEX_ID is included when at least one observation maps to
it). Entries for unobserved hex IDs are absent rather than zero; users who want
dense output can call `.reindex(all_hex_ids, fill_value=0)`.

When unreduced dimensions remain (e.g. `island` and `year`), the operation is
applied independently within each slice defined by the Cartesian product of the
unreduced dimension coordinates. The output is assembled with a `pd.MultiIndex`
whose leading levels correspond to the unreduced dimensions and whose final
level is `hex_id`.

Because `hex_ids` may be backed by a Dask array, the pipeline must remain lazy:
use `dask.dataframe` groupby operations or compute per-chunk and reduce, rather
than forcing `.values` on the full array before filtering.

### The mathematics

Let $\mathbf{H}$ be the hex ID tensor with index set $\mathcal{I}$ (the full
Cartesian product of all dimension coordinates). Partition $\mathcal{I}$ into
"reduce" indices $\mathcal{R}$ and "keep" indices $\mathcal{K}$, so each
element is indexed by $(k, r)$ with $k \in \mathcal{K}$, $r \in \mathcal{R}$.

Define the hex indicator for hex $h$ (including $h = \texttt{INVALID\_HEX\_ID}$):

$$X_h(k, r) = \begin{cases} 1 & \text{if } H(k, r) = h \\ 0 & \text{otherwise} \end{cases}$$

The count for hex $h$ at keep-index $k$ is:

$$C(k, h) = \sum_{r \in \mathcal{R}} X_h(k, r)$$

In Einstein summation notation, treating $X$ as a rank-$N$ tensor and summing
over the axes in $\mathcal{R}$:

$$C_{k_1 \ldots k_m,\, h} = \sum_{r_1} \cdots \sum_{r_n} X_{k_1 \ldots k_m,\, r_1 \ldots r_n,\, h}$$

where $m = |\mathcal{K}|$, $n = |\mathcal{R}|$.

The implementation avoids materialising $X$ by flattening the $\mathcal{R}$
axes into a 1D event array and applying `pandas.value_counts` per slice $k$.
Results are stacked into a `GeoDataFrame` with a `pd.MultiIndex`.

### Concrete usage examples

**Example 1: Total visit count per hex (reduce over all dims)**

```python
counts = hex_counts(hex_ids, reduce_dims=["island", "year", "obs", "traj"])
# counts is a GeoDataFrame indexed by hex_id
# The INVALID_HEX_ID row (if present) has geometry=None
# Answer: how many total observations fell in each hex (including out-of-domain)?
```

**Example 2: Per-island-year visit count (reduce over obs and traj)**

```python
counts = hex_counts(hex_ids, reduce_dims=["obs", "traj"])
# counts is a GeoDataFrame with MultiIndex (island, year, hex_id)
# Answer: for each island/year, how many trajectory observations fell in each hex?
```

**Example 3: Per-trajectory visit count (reduce over obs only)**

```python
counts = hex_counts(hex_ids, reduce_dims=["obs"])
# counts is a GeoDataFrame with MultiIndex (island, year, traj, hex_id)
# Answer: for each individual trajectory, how many times did it visit each hex?
```

**Example 4: From a flat Series (e.g. from a prior pipeline step)**

```python
hex_col = some_dataframe["hex_id"]   # pd.Series or dask.dataframe.Series
counts = hex_counts(hex_col)
# reduce_dims is not needed; the Series is already 1D
```

### Edge cases

- **All positions in a slice map to INVALID_HEX_ID**: the slice contributes a
  single row with `hex_id=INVALID_HEX_ID` and the full count.
- **`reduce_dims` is all dims**: output is a GeoDataFrame indexed by `hex_id`
  only.
- **`reduce_dims` is empty**: each element maps to a count of 1; the result has
  one row per unique observed hex ID (including INVALID_HEX_ID). Degenerate
  but not an error.
- **Hex IDs that appear in some slices but not others**: each slice contains
  only its own observed hex IDs. There is no automatic re-indexing to a common
  set; users who need aligned slices should call `.reindex` explicitly after
  the fact.

---

## 2. Connectivity matrices: `hex_connectivity`

### Proposed signature

```python
def hex_connectivity(
    hex_ids: xr.DataArray,
    from_dim: str,
    from_idx: int,
    to_dim: str,
    to_idx: int,
    weight: xr.DataArray | None = None,
) -> gpd.GeoDataFrame:
    ...
```

- `hex_ids`: `xr.DataArray` of int64 hex IDs.
- `from_dim`: name of the dimension along which the origin position is selected.
- `from_idx`: integer index along `from_dim` for the origin (e.g. `0` for start).
- `to_dim`: name of the dimension along which the destination is selected.
  May be the same as `from_dim`.
- `to_idx`: integer index along `to_dim` for the destination (e.g. `-1` for end).
- `weight`: optional `xr.DataArray` of the same shape as `hex_ids`, dtype
  numeric. If provided, each pair contributes its weight value instead of 1.
  Pairs involving INVALID_HEX_ID are still counted; their weight is taken from
  the corresponding position.
- Returns: `gpd.GeoDataFrame` with:
  - Index: `pd.MultiIndex` of `(from_id, to_id)`.
  - Column `count`: non-negative count (or summed weights).
  - Column `geometry`: LineString from the centroid of `from_id` to the
    centroid of `to_id`; `None` when either endpoint is INVALID_HEX_ID
    (consistent with existing `hp.edges_geodataframe` behaviour).
  - All observed pairs appear, including those where `from_id` or `to_id` is
    INVALID_HEX_ID.

The `(from_id, to_id)` MultiIndex structure makes `from_id` the primary lookup
key. To query all destinations reachable from hex X:

```python
conn.loc[(X, slice(None))]
```

### What it does and why

A connectivity matrix answers: *how many trajectories went from hex $h_0$ to
hex $h_1$, where "from" and "to" are defined by selecting specific positions
along a named dimension?*

The canonical application is start-to-end connectivity: `from_dim="obs"`,
`from_idx=0`, `to_dim="obs"`, `to_idx=-1`. This extracts two arrays of hex IDs
(one per selected position), forms pairs, and counts how many times each pair
occurs across all remaining dimensions. INVALID_HEX_ID is treated as a regular
destination (or origin). A pair `(h, INVALID_HEX_ID)` tells the user that
trajectories starting in hex $h$ left the simulation domain.

The `weight` option replaces the implicit per-pair weight of 1 with a value
from a parallel data array — for example, weighting pairs by temperature at the
destination, or by transit duration.

### Connectivity as a conditional heatmap

Connectivity $C(h_0, h_1)$ is the joint count of (start hex, end hex) pairs,
including out-of-domain destinations. It can be read as:

$$C(h_0, h_1) = \text{count}\bigl(\text{start} = h_0,\ \text{end} = h_1\bigr)$$

Dividing by the marginal count of starts recovers the conditional distribution:

$$P(\text{end} = h_1 \mid \text{start} = h_0) = \frac{C(h_0, h_1)}{\sum_{h_1'} C(h_0, h_1')}$$

Because INVALID_HEX_ID is included in the sum over $h_1'$, the marginal is the
total number of trajectories that started in $h_0$, regardless of whether they
ended in domain or not. This guarantees that conditional probabilities sum to 1
and that normalisation is unambiguous.

### INVALID_HEX_ID as a domain-exit bucket

A trajectory that is valid at obs=0 but invalid at obs=-1 connects from its
starting hex to INVALID_HEX_ID. In a well-run simulation, becoming invalid
means leaving the domain covered by circulation data. This is meaningful: the
connectivity entry `(h, INVALID_HEX_ID)` quantifies domain export from hex $h$.

The INVALID_HEX_ID bucket is included in the result by default. Users who want
to exclude it can filter:

```python
conn = conn.drop(INVALID_HEX_ID, level="to_id", errors="ignore")
conn = conn.drop(INVALID_HEX_ID, level="from_id", errors="ignore")
```

### The mathematics

Select origin and destination slices:

$$F = H\big[\ldots,\ \text{from\_idx along from\_dim}\big]$$
$$T = H\big[\ldots,\ \text{to\_idx along to\_dim}\big]$$

After selection, both $F$ and $T$ have the input shape with `from_dim` and
`to_dim` collapsed to scalars. Let $\mathcal{K}$ be the remaining coordinate
space (all dims except `from_dim` and `to_dim`). The total event index is
$k \in \mathcal{K}$.

Define the pair indicator for hex pair $(h_0, h_1)$:

$$P_{h_0, h_1}(k) = \begin{cases} 1 & \text{if } F(k) = h_0 \text{ and } T(k) = h_1 \\ 0 & \text{otherwise} \end{cases}$$

where $h_0$ and $h_1$ range over all observed values including
$\texttt{INVALID\_HEX\_ID}$.

The unweighted connectivity is:

$$C(h_0, h_1) = \sum_{k \in \mathcal{K}} P_{h_0, h_1}(k)$$

With a weight array $W$ (evaluated at `to_idx` along `to_dim`):

$$C_W(h_0, h_1) = \sum_{k \in \mathcal{K}} P_{h_0, h_1}(k) \cdot W(k)$$

Dividing $C_W$ by $C$ gives the mean weight per connectivity pair, e.g. mean
arrival temperature.

### Design decision: unreduced dimensions

All dimensions other than `from_dim` and `to_dim` are always summed over.
Users who want per-year results should subset the input DataArray before
calling:

```python
conn_1993 = hex_connectivity(hex_ids.sel(year=1993), from_dim="obs", ...)
```

This keeps the function signature minimal and makes the reduction contract
unambiguous. A `reduce_dims` argument can be added later if needed.

### Weight evaluation convention

When `weight` is provided, the weight value for a pair is taken from the
`to_idx` position along `to_dim` — i.e. the value at the destination
observation. Users who want origin-side or mean weights should pre-compute
their weight array before calling.

### Concrete usage examples

**Example 1: Start-to-end connectivity matrix**

```python
conn = hex_connectivity(
    hex_ids,
    from_dim="obs", from_idx=0,
    to_dim="obs",   to_idx=-1,
)
# conn is a GeoDataFrame with MultiIndex (from_id, to_id)
# Rows with to_id=INVALID_HEX_ID show domain-exit counts per starting hex
# conn.loc[(h, slice(None))] → all destinations reached from hex h
```

**Example 2: Per-year connectivity matrix**

```python
conn_by_year = {
    year: hex_connectivity(
        hex_ids.sel(year=year),
        from_dim="obs", from_idx=0,
        to_dim="obs",   to_idx=-1,
    )
    for year in hex_ids.year.values
}
```

**Example 3: Temperature-weighted connectivity matrix**

```python
conn_temp = hex_connectivity(
    hex_ids,
    from_dim="obs", from_idx=0,
    to_dim="obs",   to_idx=-1,
    weight=temp_ids,  # xr.DataArray same dims as hex_ids
)
# conn_temp / conn_unweighted → mean arrival temperature per connectivity pair
```

### Edge cases

- **Either selected slice is all-INVALID_HEX_ID**: result contains only
  INVALID_HEX_ID-to-INVALID_HEX_ID pairs (or is empty if the other slice is
  also all-INVALID_HEX_ID).
- **`from_dim == to_dim`, `from_idx == to_idx`**: all pairs are self-loops
  $(h, h)$. Correct; diagonal-only connectivity is a legitimate result.
- **`from_idx` or `to_idx` out of range**: `isel` raises `IndexError` naturally.
  No defensive wrapping.
- **`weight` contains NaN for a pair**: NaN propagates into the sum. Users
  should pre-fill or mask weight arrays if they want to exclude specific values.

---

## Module location and exports

Both functions belong in a new module `src/hextraj/hex_analysis.py`. They
should be exported from `hextraj/__init__.py`:

```python
from .hex_analysis import hex_counts, hex_connectivity
```

The module imports: `numpy`, `pandas`, `xarray`, `geopandas`,
`INVALID_HEX_ID` from `.hex_id`. No pyproj directly (geometry is delegated to
the `HexProj` instance passed in, or to existing GeoDataFrame-building methods
on `HexProj`).

---

## Return type rationale

Both functions return a `GeoDataFrame`:

1. **Natural fit for spatial data**: a GeoDataFrame assigns geometry to each
   statistical object — hex polygons for heatmap cells, LineStrings for
   connectivity edges. This lets users visualise results directly without a
   separate join step.

2. **INVALID_HEX_ID geometry=None**: entries for INVALID_HEX_ID carry
   `geometry=None`, which is consistent with the existing `HexProj` polygon
   and edge methods. Users can filter or handle these rows the same way they
   would in any other GeoDataFrame.

3. **`(from_id, to_id)` MultiIndex for connectivity**: indexing the
   connectivity GeoDataFrame by a `(from_id, to_id)` MultiIndex makes
   `from_id` the primary lookup key. The query
   `.loc[(h, slice(None))]` returns all destinations reachable from hex `h`,
   directly supporting the "forward perspective" use case described in the
   overview.

4. **Sparsity**: only observed hex IDs (and INVALID_HEX_ID when present) appear
   in the index. An `xr.DataArray` indexed over all possible hex IDs would be
   mostly zeros and potentially enormous.

5. **Dask compatibility**: the computation pipeline remains lazy until
   `.compute()` is called. Dask DataFrames support the groupby/count operations
   needed internally; the final `.compute()` materialises the result before
   constructing the GeoDataFrame.

6. **Normalisation**: users who want conditional probabilities or visit
   fractions operate on the `count` column with standard pandas operations
   rather than navigating xarray dimension semantics.

The `hex_id` (or `from_id`/`to_id`) index levels contain the sorted set of all
observed hex IDs from the input, including INVALID_HEX_ID. Results from
different calls are aligned via standard pandas `reindex` or join operations.
