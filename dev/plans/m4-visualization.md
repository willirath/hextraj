# M4 — Visualization

## Scope

M4 is mostly free: `to_geodataframe` already returns a proper `GeoDataFrame` with EPSG:4326
`Polygon` geometries. Three of the four roadmap bullets are standard geopandas one-liners
that belong in the notebook, not the library:

- **Hex outlines:** `gdf.boundary.plot()`
- **Filled patches / choropleth:** `gdf.plot(column="count", cmap=...)`
- **Hex centers:** `gdf.set_geometry(gdf.centroid)`

**The only genuinely new code:** one new `HexProj` method for weighted edges.

---

## New method: `HexProj.edges_geodataframe`

```python
def edges_geodataframe(self, from_ids, to_ids, **value_cols):
    """Build a GeoDataFrame of LineString edges between hex centers.

    Parameters
    ----------
    from_ids : array-like
        1D int64 hex IDs for the origin end of each edge.
    to_ids : array-like
        1D int64 hex IDs for the destination end of each edge.
    **value_cols :
        Additional columns aligned with from_ids / to_ids (e.g. weight=...).

    Returns
    -------
    geopandas.GeoDataFrame
        MultiIndex (from_id, to_id), LineString geometry column, value_cols,
        CRS=EPSG:4326. Edges with an INVALID_HEX_ID endpoint have None geometry.
    """
```

### Internal sketch

1. `decode_hex_id(from_ids)` → `(q_from, r_from)`, same for `to_ids`
2. Build `invalid` mask where either endpoint is `INVALID_HEX_ID`
3. For valid rows: `hex_to_pixel` on both ends → `_transform_proj_to_lon_lat` → `shapely.linestrings(coords_Nx2x2)`
4. Assemble `GeoDataFrame` with `pd.MultiIndex.from_arrays([from_ids, to_ids], names=["from_id", "to_id"])`

Follows the exact same projection-then-inverse pattern as `to_geodataframe`.

---

## Tests: `tests/test_hexproj_edges.py`

11 plain `pytest` functions, no test classes:

| Test | What it checks |
|------|---------------|
| `test_edges_geodataframe_returns_geodataframe` | return type is `GeoDataFrame` |
| `test_edges_geodataframe_geometry_is_linestring` | all valid rows have `LineString` geometry |
| `test_edges_geodataframe_multiindex` | index is `pd.MultiIndex` named `["from_id", "to_id"]`, values match inputs |
| `test_edges_geodataframe_crs` | `crs.to_epsg() == 4326` |
| `test_edges_geodataframe_value_cols` | single kwarg appears as column with correct values |
| `test_edges_geodataframe_multiple_value_cols` | two kwargs, both present and correct |
| `test_edges_geodataframe_invalid_from_id` | `INVALID_HEX_ID` in `from_ids` → `None` geometry for that row |
| `test_edges_geodataframe_invalid_to_id` | same for `to_ids` |
| `test_edges_geodataframe_self_loop` | `from_id == to_id` → degenerate `LineString`, no exception |
| `test_edges_geodataframe_linestring_endpoints` | decode `(q,r)`, compute expected centers, assert `LineString` coords match *(key correctness test)* |
| `@pytest.mark.parametrize("n", [0, 1, 10, 100])` `test_edges_geodataframe_length` | `len(result) == n` |

---

## Notebook

Fold into existing `notebooks/hex_aggregation.ipynb`. Add four new sections at the end:

1. **Hex outlines** — `region_gdf.boundary.plot(...)`
2. **Hex centers** — `region_gdf.set_geometry(region_gdf.centroid).plot(...)`
3. **Weighted transition edges** — build `(from_id, to_id)` pairs from consecutive labels, groupby-count, call `hp.edges_geodataframe(..., weight=...)`
4. **Combined plot** — choropleth beneath, edges on top with `linewidth` scaled by weight

---

## Agent order

1. **Haiku test agent** — write `tests/test_hexproj_edges.py` (failing only, no implementation)
2. **Haiku implementation agent** — add `edges_geodataframe` to `hexproj.py`, make tests pass
3. **Main agent** — run full test suite, escalate to sonnet if haiku struggles on `shapely.linestrings`
4. **Haiku notebook agent** — append visualization sections to `hex_aggregation.ipynb`, execute notebook

---

## Key files

- `src/hextraj/hexproj.py` — only file that gets new code
- `tests/test_hexproj_grid.py` — pattern to follow for tests
- `notebooks/hex_aggregation.ipynb` — receives new visualization sections
- `src/hextraj/hex_id.py` — `decode_hex_id`, `INVALID_HEX_ID`
- `src/hextraj/redblobhex_array.py` — `hex_to_pixel`, `Hex`
