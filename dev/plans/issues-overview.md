# Issues overview

All known issues gathered from GitHub issues, PR comments, branch inspection, and code review.
Grouped by topic, ranked low → high complexity within each group.

---

## Infrastructure / packaging

| ID | Source | Item | Complexity |
|----|--------|------|------------|
| I1 | Code | Deprecated `pyproj.Proj(init="epsg:4326")` in `hexproj.py:53` — crashes on pyproj ≥3 | Low |
| I2 | Code | `setup.cfg` + `setup.py` → migrate to modern `pyproject.toml` `[project]` table | Low |
| I3 | Code | Python classifiers stuck at 3.6/3.7/3.8; CI envs exist for 3.9–3.11 | Low |
| I4 | Code | No CI workflow files in `.github/workflows/` — CI is fully broken/missing | Medium |
| I5 | PR #19 | New visualization code has near-zero test coverage | Medium |

feedback:

Let's build packaging from the ground up. src/hextraj/ structure. Use uv or pixi for env management. Which one? (If we want to build docs as well, we go for pixi?)

Not sure about CI. Let's defer. There's not going to be a lot of development by not many people anyway. So may be overkill.

---

## Visualization (GH #16, PR #19)

| ID | Source | Item | Complexity |
|----|--------|------|------------|
| V1 | GH #16 | Plot hex centers | Low |
| V2 | GH #16, PR #19 | Plot hex outlines | Low–Medium |
| V3 | GH #16, PR #19 | Filled hex patches (color-mapped) via `hex_corners_lon_lat_*` + `PatchCollection` | Medium |
| V4 | GH #16 | Weighted lines between hex centers (connectivity visualization) | Medium–High |

feedback:

Visualization is a key todo. Let's defer, however, the discussion a little. I want to discuss vis w/ vis-friendly refactoring in mind.

---

## Grid construction (GH #15, branches `rect_of_hexes`, `add-vis`)

| ID | Source | Item | Complexity |
|----|--------|------|------------|
| G1 | `rect_of_hexes` branch | `rectangle_of_hexes(lon1, lon2, lat1, lat2)` — partially implemented; mutates `self.lon_origin`/`lat_origin`, `central_lat` formula wrong | Low–Medium |
| G2 | GH #15 | General shaped-region hex construction (non-rectangular masks, e.g. land/ocean masks) | High |

feedback: 

G2 complexity is high if we construct bottom up. But hexes are cheap to track. And just filtering a bunch of hexes for whether they intersect with another polygon is easy.

---

## Connectivity / scoring (PR #20, merged `poc-scoring`)

| ID | Source | Item | Complexity |
|----|--------|------|------------|
| C1 | PR #20 | Connectivity computation noted to scale as O(N_traj × N_hex²) — no fix yet | Medium |
| C2 | PR #20 | Add scaling example / benchmarks to tutorial notebook | Medium |
| C3 | GH #16, PR #20 | Weighted lines between centers as connectivity visualization | Medium–High |
| C4 | Tutorial | Connectivity graph using centers misleading near coasts — no solution yet | High |

feedback:

This is a dask array, dask dataframe heavy topic. Let's defer. I'll bring in recipes once we get there. We learned a lot since I wrote this package.

---

## Data aggregation / API

| ID | Source | Item | Complexity |
|----|--------|------|------------|
| A1 | Tutorial | Only `.mean()` shown for hex aggregation — flexible stats not exposed | Low |
| A2 | Tutorial | `xarray.align` behavior for hex-aggregated arithmetic implicit — needs docs/tests | Low |
| A3 | Code | `hex_corners_lon_lat_xarray` raises `NotImplementedError` for Dask arrays | Medium |
| A4 | Code | `hex_AoS_to_string` in `aux.py` is unused and untested | Low |

feedback:

Needs a lot of discussion about how to id hexes. (tuple label not ideal for numpy based data structures etc.)

---

## Invalid / edge-case handling (GH #14, merged PR #17)

| ID | Source | Item | Complexity |
|----|--------|------|------------|
| E1 | Code | `INTNaN` sentinel (`np.nan.astype(int)`) is platform-dependent; behavior on non-x86 unclear | Medium |
| E2 | GH #14 | NaN lon/lat → `INTNaN` hex label behavior is undocumented in the public API | Low |

feedback:

Needs to be discussed together with the topic above.