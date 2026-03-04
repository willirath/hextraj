# Plan: `hex_connectivity_power`

## Overview

`hex_connectivity_power(conn, n, hp)` takes the GeoDataFrame returned by
`hex_connectivity` and computes $n$-generation connectivity — that is, it
applies the underlying transfer matrix $n$ times and returns a result in the
same GeoDataFrame format.

The input `conn` holds raw pair counts (not yet probabilities). The function
row-normalises to conditional probabilities, raises the resulting transfer
matrix to the power $n$, then serialises back into a `(from_id, to_id)`
GeoDataFrame.

---

## 1. Input

`conn` is the direct output of `hex_connectivity`:

- `pd.MultiIndex` with levels `("from_id", "to_id")`.
- Column `count`: raw (integer or float) pair counts.
- Column `geometry`: LineString or `None`.

The function does not re-read the original trajectory data; it works entirely
from the summary GeoDataFrame.

---

## 2. INVALID_HEX_ID semantics

### Physical interpretation

INVALID_HEX_ID is the out-of-domain absorbing state. Probability mass flows
into it from hexes that export particles, but it has no outgoing transitions in
the original data (any trajectory already invalid at obs=0 contributes only to
the INVALID→INVALID pair, which is an artefact of the extraction step rather
than a physical transition).

### Decision: absorbing state with optional conditioning

The default behaviour keeps INVALID as a proper absorbing state:

- The INVALID row of $T$ is set to a **self-loop** (INVALID→INVALID = 1).
  Rationale: probability mass that has left the domain must go somewhere; a
  self-loop is the identity absorbing state. Setting the row to all-zero would
  make the matrix sub-stochastic and is harder to interpret.
- After enough generations all mass that ever reaches INVALID accumulates there.
  This is physically correct for an open domain.

An optional kwarg `condition_on_valid=False` (default `False`) allows the user
to condition on staying in-domain:

- When `True`: before computing $T^n$, zero the INVALID column **and**
  renormalise each row so that rows sum to 1. INVALID is removed from both
  index levels of the output. This answers "given that a particle stays in
  domain for $n$ steps, where does it end up?"
- When `False` (default): INVALID is kept as absorbing.

```python
def hex_connectivity_power(
    conn: gpd.GeoDataFrame,
    n: int,
    hp: HexProj,
    condition_on_valid: bool = False,
) -> gpd.GeoDataFrame:
    ...
```

---

## 3. Normalisation

### Row-normalisation to $T$

For each `from_id` row:

$$T_{ij} = \frac{C(h_i, h_j)}{\sum_{j'} C(h_i, h_{j'})}$$

The denominator is the total count departing from $h_i$, which already
includes transitions to INVALID. This guarantees rows sum to 1.

### INVALID row

After pivoting from the sparse `conn`, any `from_id` that is INVALID_HEX_ID
gets its row replaced with a self-loop (probability 1 in the INVALID column,
0 elsewhere). If INVALID_HEX_ID does not appear as a `from_id` at all, it is
added to the matrix index as a self-loop row so the matrix remains square over
all states that appear as `to_id`.

### `condition_on_valid=True`

Zero the INVALID column, then renormalise rows. Rows that had probability 1 in
the INVALID column (particles that always leave) will have all-zero rows after
zeroing — these `from_id` values are dropped from the output entirely (they
have no valid conditional distribution). Remaining rows are renormalised to sum
to 1.

---

## 4. Pivot and matrix representation

The set of unique hex IDs appearing in either level of the MultiIndex is
typically in the low hundreds to low thousands for real trajectory datasets.
Dense `numpy` is appropriate at this scale:

- Pivot `conn["count"]` to a dense `(N, N)` numpy array indexed by a sorted
  list of all unique IDs (the union of `from_id` and `to_id` values, including
  INVALID_HEX_ID).
- Row-normalise in-place.
- Raise to the $n$-th power with `numpy.linalg.matrix_power`.
- Re-serialise to the sparse GeoDataFrame format (drop zero entries).

`scipy.sparse` would be worth considering only if $N$ grows beyond ~5 000 or
if the matrix is extremely sparse (fill < 1%). At typical scales the dense
path is simpler, more readable, and fast enough.

---

## 5. Output format

The output is a GeoDataFrame in the same format as `hex_connectivity`:

| aspect | value |
|--------|-------|
| Index | `pd.MultiIndex` with levels `("from_id", "to_id")` |
| Column `probability` | float in [0, 1], the $(i,j)$ entry of $T^n$ |
| Column `geometry` | LineString from centroid of `from_id` to centroid of `to_id`; `None` if either endpoint is INVALID_HEX_ID |
| Zero entries | dropped (sparse representation) |

### Column naming: `probability` not `count`

The output column is named `probability` rather than `count` because the
values are now conditional probabilities, not raw counts. Keeping the name
`count` would be misleading.

To keep the output format maximally compatible with `hex_connectivity`
(e.g. for plotting code that addresses `gdf["count"]`), the column name
choice is documented clearly. If strict format compatibility is needed, the
caller can rename: `result.rename(columns={"probability": "count"})`.

The `geometry` column is reconstructed by the function from `hp` using the
same logic as `hex_connectivity` — centroids decoded from `from_id` and
`to_id`, LineString assembled from the pair, `None` for INVALID endpoints.

---

## 6. `hp` argument

`hp` is a `HexProj` instance, required for reconstructing LineString geometries
in the output. There is no meaningful default because the coordinate system is
experiment-specific. The function does not default to `HexProj()`.

Geometry reconstruction mirrors the existing `hex_connectivity` code:

```python
q_f, r_f = decode_hex_id(f_id)
hex_f = redblobhex.Hex(q_f, r_f, -q_f - r_f)
lon_f, lat_f = hp.hex_to_lon_lat_SoA(hex_f)
# ... same for t_id
geometries.append(LineString([(lon_f, lat_f), (lon_t, lat_t)]))
```

---

## 7. Algorithm sketch

```python
def hex_connectivity_power(conn, n, hp, condition_on_valid=False):
    # 1. Collect all unique IDs appearing in either index level
    all_ids = sorted(set(conn.index.get_level_values("from_id"))
                     | set(conn.index.get_level_values("to_id")))
    id_to_idx = {hid: i for i, hid in enumerate(all_ids)}
    N = len(all_ids)

    # 2. Pivot conn["count"] to dense NxN array
    T = np.zeros((N, N), dtype=float)
    for (f_id, t_id), row in conn.iterrows():
        T[id_to_idx[f_id], id_to_idx[t_id]] = row["count"]

    # 3. Ensure INVALID is present and set as self-loop
    if INVALID_HEX_ID in id_to_idx:
        inv_i = id_to_idx[INVALID_HEX_ID]
        T[inv_i, :] = 0.0
        T[inv_i, inv_i] = 1.0

    # 4. Row-normalise (for non-INVALID rows)
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid divide-by-zero; zero rows stay zero
    T /= row_sums

    # 5. Optionally condition on valid
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

    # 6. Raise to n-th power
    Tn = np.linalg.matrix_power(T, n)

    # 7. Serialise back to sparse GeoDataFrame
    from_ids_out, to_ids_out, probs_out = [], [], []
    for i, f_id in enumerate(all_ids):
        for j, t_id in enumerate(all_ids):
            p = Tn[i, j]
            if p > 0:
                from_ids_out.append(f_id)
                to_ids_out.append(t_id)
                probs_out.append(p)

    # 8. Build geometries (same logic as hex_connectivity)
    geometries = _build_edge_geometries(from_ids_out, to_ids_out, hp)

    multi_index = pd.MultiIndex.from_arrays(
        [np.array(from_ids_out, dtype=np.int64),
         np.array(to_ids_out, dtype=np.int64)],
        names=["from_id", "to_id"]
    )
    return gpd.GeoDataFrame(
        {"probability": probs_out, "geometry": geometries},
        index=multi_index,
        crs="EPSG:4326",
    )
```

A private helper `_build_edge_geometries(from_ids, to_ids, hp)` should be
extracted from the existing `hex_connectivity` body and shared by both
functions to avoid duplication.

---

## 8. Non-stationary extension (note only)

For a sequence of per-year connectivity matrices $T_1, T_2, \ldots, T_n$
(each from `hex_connectivity` called on a single year), the $n$-generation
product is $T_n \cdot T_{n-1} \cdots T_1$. A future function could accept:

```python
def hex_connectivity_product(conn_list, hp, condition_on_valid=False):
    ...
```

where `conn_list` is an ordered list of GeoDataFrames from `hex_connectivity`.
The pivot and normalisation steps are identical to `hex_connectivity_power`;
the only difference is that `matrix_power` is replaced by an ordered sequence
of `@` (matmul) operations:

```python
T_total = T_list[0]
for T_k in T_list[1:]:
    T_total = T_k @ T_total
```

The union of all ID sets across all matrices defines the shared index, with
missing entries filled as zero before multiplication.

---

## 9. Module location and exports

- Implementation in `src/hextraj/hex_analysis.py`, alongside `hex_counts` and
  `hex_connectivity`.
- Export from `hextraj/__init__.py`:

```python
from .hex_analysis import hex_counts, hex_connectivity, hex_connectivity_power
```

- The private helper `_build_edge_geometries` is an internal module function,
  not exported.

---

## 10. TDD notes

Tests should cover:

- `n=1` returns the same transition structure as row-normalised `conn` (up to
  floating-point tolerance).
- `n=2` on a simple 2-hex system matches hand-computed $T^2$.
- INVALID as absorbing state: after large $n$, all probability mass from any
  exporting hex drains to INVALID.
- `condition_on_valid=True`: INVALID is absent from the output index, rows sum
  to 1 over valid hexes only.
- `n=0` returns the identity (each hex maps to itself with probability 1).
- All-INVALID input: single self-loop entry.
- Zero-probability entries are dropped from the output (sparsity check).
