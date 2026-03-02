# Roadmap

## M1 — Packaging rewrite
_Foundation. Everything else builds on this._

- [x] Adopt `src/hextraj/` layout; move `data/` inside package
- [x] Migrate metadata from `setup.cfg` to `pyproject.toml` `[project]` table; delete `setup.py` and `setup.cfg`
- [x] Switch env management to pixi (`pixi.toml`, lockfile)
- [x] Centralise pyproj handling in `_proj.py`; fix deprecated `pyproj.Proj(init=...)` API
- [x] Promote `geopandas` / `shapely` to core dependencies
- [x] Pin Python floor at 3.10
- [x] Verify tests pass in new layout

Design doc: `dev/docs/packaging.md`

---

## M2 — Hex ID design
_Prerequisite for aggregation, visualization, and invalid handling._

- [x] Implement Cantor pairing of `(q, r)` → int64 as the public hex ID
- [x] Add vectorised `encode_hex_id(q, r)` / `decode_hex_id(hex_id)` module-level functions
- [x] Replace `INTNaN` sentinel with `INVALID_HEX_ID = np.int64(-1)`
- [x] Update tests

Design doc: `dev/docs/hex-id-design.md`

---

## M3 — GeoDataFrame output and grid construction
_Unlocks visualization and region-aware grid construction._

- [x] Add `HexProj.to_geodataframe(hex_ids, **value_cols)` → `GeoDataFrame` with hex `Polygon` geometries and int64 ID index
- [x] Fix and clean up `rectangle_of_hexes` (no mutation of `self`, correct `central_lat`)
- [x] Add `region_of_hexes(region_polygon)` using dense grid + `.intersects()`
- [x] Tests for all three

Depends on: M2

---

## M4 — Visualization
_With geopandas as the output format, most of this is thin convenience._

- [ ] Hex centers: `GeoDataFrame` of `Point` geometries (falls out of M3)
- [ ] Hex outlines: `GeoDataFrame.boundary.plot()`
- [ ] Filled patches: `GeoDataFrame.plot(column=...)` choropleth
- [ ] Weighted edges between centers: `GeoDataFrame` of `LineString` geometries
- [ ] Notebook section demonstrating all four

Depends on: M3

---

## M5 — Connectivity and aggregation (deferred)
_Bring in recipes once ready. Dask array / Dask DataFrame heavy._

- [ ] Revisit connectivity computation scaling (O(N_traj × N_hex²))
- [ ] Update tutorial with improved Dask-native recipes
- [ ] Scaling benchmarks notebook

Depends on: M2, M3

---

## Backlog

- M2 leftover: Remove or fold `hex_AoS_to_string` into the new ID scheme.
- Docstrings to tensorflow / google style.
- CI workflows: low development volume, not worth the overhead for now
- `xarray.align` docs: document in notebook prose when redoing tutorial
