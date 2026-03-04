# API Refactor Plan: Helper Function Review Findings

## Executive Summary

A comprehensive review of helper functions in `hex_analysis.py` and related modules identified 12 issues spanning API boundaries, redundant NaN handling, private/public exposure of internal types, scalar Python loops, inconsistent type handling, and dead code. This plan addresses each issue with clear rationale and recommended actions. The changes improve composability with dask pipelines, eliminate runtime warnings, reduce scalar loop overhead, and sharpen the public API boundary.

---

## Issue 1: Private NaN handling in `label`

**Description:**
The `label` function contains a pre-flight `np.isnan` check that is redundant with downstream `INTNaN` propagation in `hex_round` and `hex_to_pixel`. This redundant check is the source of the `RuntimeWarning: invalid value encountered in cast` warnings seen in user code.

**Current behavior:**
- `label` checks `np.isnan(lon), np.isnan(lat)` upfront
- Downstream functions still propagate `INTNaN` values through hex operations
- Finally, `encode_hex_id` maps `INTNaN` to `INVALID_HEX_ID`
- The upfront check triggers a warning but is overridden by downstream logic

**Recommended action:**
Remove the pre-flight `np.isnan` check in `label`. Allow NaN values to propagate naturally through `hex_round` and `hex_to_pixel` as `INTNaN`, then map to `INVALID_HEX_ID` exactly once in `encode_hex_id`. This eliminates the warning and simplifies the code path.

---

## Issue 2: Make `lon_lat_to_hex_SoA` and `hex_to_lon_lat_SoA` private

**Description:**
These functions are internal structure converters exposed at the module level. Only internal callers use them; users should work with the public `Hex` namedtuple type directly.

**Current behavior:**
- Both functions are public (no `_` prefix)
- Used internally by other helper functions
- Not documented in docstrings or examples

**Recommended action:**
Rename to `_lon_lat_to_hex_SoA` and `_hex_to_lon_lat_SoA` (add `_` prefix to mark as private). If users need to work with hex coordinates, expose the internal `Hex` namedtuple type in the public API (e.g., `HexProj.Hex` or document it in `__init__.py`).

---

## Issue 3: `hex_corners_lon_lat` — wrong boundary and wrong type

**Description:**
The function name suggests it accepts hex IDs, but it actually accepts internal `Hex` namedtuples. Its only current use is inside `_build_counts_geodataframe`, a scalar Python loop.

**Current behavior:**
- Signature: `hex_corners_lon_lat(hex_tuple)` — expects a `Hex` namedtuple
- Used only in `_build_counts_geodataframe` (scalar context)
- Not used in notebooks or by users

**Recommended action:**
Make this function private (`_hex_corners_lon_lat`) or remove it after resolving issue 5 (which will eliminate its only caller). If retained, consider accepting int64 hex IDs instead of `Hex` tuples to align with public API expectations.

---

## Issue 4: `hex_of_hexes` — yields internal types, unused

**Description:**
This function yields internal `Hex` namedtuples, not int64 hex IDs. It is unused in notebooks and user code.

**Current behavior:**
- Yields `Hex` namedtuples instead of int64 IDs
- Undefined use case; no examples or tests
- Not called anywhere in the codebase

**Recommended action:**
Remove entirely, or reconsider and redefine to yield int64 hex IDs with clear use case documentation. If retained, update to be consistent with the public API (return hex IDs, not `Hex` tuples).

---

## Issue 5: Replace `_build_counts_geodataframe` scalar loop with vectorized call

**Description:**
This function is a scalar Python loop calling `hp.hex_corners_lon_lat` one hex at a time. This is inefficient and ties geometry building to a specific caller.

**Current behavior:**
- Called only by `hex_counts`
- Loops over unique hex IDs, calling `hex_corners_lon_lat` for each one
- Builds a GeoDataFrame with hex polygons

**Recommended action:**
Delete `_build_counts_geodataframe`. Replace its call in `hex_counts` with a direct call to `hp.to_geodataframe(hex_ids_array, ...)` (a vectorized method that operates on all hex IDs at once). This eliminates the scalar loop and decouples geometry building from the aggregation logic.

---

## Issue 6: Replace `_build_edge_geometries` scalar loop with vectorized call

**Description:**
This function is a scalar Python loop calling `hp.hex_to_lon_lat_SoA` one edge pair at a time. This is inefficient and couples geometry building to specific callers.

**Current behavior:**
- Called by `hex_connectivity` and `hex_connectivity_power`
- Loops over edge pairs, calling `hex_to_lon_lat_SoA` for each one
- Builds a GeoDataFrame with edge geometries (lines)

**Recommended action:**
Delete `_build_edge_geometries`. Replace its calls in `hex_connectivity` and `hex_connectivity_power` with direct calls to `hp.edges_geodataframe(from_ids_array, to_ids_array, ...)` (a vectorized method that operates on all edges at once). This eliminates the scalar loop and decouples geometry building from connectivity logic.

---

## Issue 7: `hex_counts` input type — xarray coupling is complex and buggy

**Description:**
The function accepts both `xr.DataArray` and `pd.Series`, with an `isinstance` branch that silently ignores `reduce_dims` for Series input. This is a design smell — the xarray-specific logic is not worth the complexity, and the Series branch is broken.

**Current behavior:**
- `hex_counts(ds, reduce_dims=None, hp=None)` accepts `xr.DataArray | pd.Series`
- `reduce_dims` parameter only works for DataArray; ignored for Series (silent bug)
- Increases code complexity without clear user benefit

**Recommended action:**
Reconsider the API. Either:
1. Remove Series support entirely and require xarray input (sharper boundary, clearer intent).
2. Or, make the function accept plain 1D ID arrays (numpy or pandas) and return plain counts, with a separate `to_geodataframe` call for geometry. This is simpler and more composable.

Recommend option 2: simplify to `hex_counts(hex_ids, ...)` (array-like) and let users handle the xarray wrapping/unwrapping.

---

## Issue 8: `hex_connectivity` slicing boundary — move responsibility to caller

**Description:**
The function currently accepts a 2D DataArray and slices it internally via `from_dim`, `from_idx`, `to_dim`, `to_idx`. This makes the signature complex and incompatible with dask pipelines that bypass it entirely because of the xarray requirement.

**Current behavior:**
- Signature: `hex_connectivity(ds, from_dim, from_idx, to_dim, to_idx, ...)`
- Performs internal slicing: `ds.isel({from_dim: from_idx, to_dim: to_idx})`
- Couples slicing logic to the function, making it inflexible

**Recommended action:**
Simplify to: `hex_connectivity(from_ids, to_ids, weight=None, hp=None)`. Move the `isel` responsibility to the caller. The caller pre-slices the dataset to the obs range of interest before extracting ID arrays. This:
- Simplifies the signature (1D arrays in, 1D counts out)
- Makes the function composable with dask pipelines
- Lets users control the slicing logic without tying it to connectivity computation

---

## Issue 9: `hex_connectivity_power` — make `hp` optional

**Description:**
The `hp` parameter is required but only used for geometry building (`hp.edges_geodataframe`). Users who only want probability values should not be forced to construct a `HexProj`.

**Current behavior:**
- `hp` is a required positional or keyword argument
- Only used to build geometries (optional enrichment)
- Raises an error if not provided, even if user only wants counts/probabilities

**Recommended action:**
Make `hp` optional with a default value of `None`. When `None`, return counts/probabilities without geometry. When provided, attach geometry. This is consistent with `hex_connectivity` post-hoc enrichment pattern and gives users control over when geometry is needed.

---

## Issue 10: Inconsistent `INVALID_HEX_ID` detection

**Description:**
Two different checks exist across the codebase:
- `== INVALID_HEX_ID` in `hex_analysis.py`
- `INTNaN` check after decode in `hexproj.py` (e.g., in `to_geodataframe`, `edges_geodataframe`)

This inconsistency can lead to bugs if one path is updated without the other.

**Current behavior:**
- `hex_analysis.py` uses `== INVALID_HEX_ID`
- `hexproj.py` uses `INTNaN` checks or relies on `encode_hex_id` to produce `INVALID_HEX_ID`
- Scattered checks make the logic hard to follow

**Recommended action:**
Standardize on `== INVALID_HEX_ID` everywhere. Ensure that:
1. Invalid positions always decode to exactly `INVALID_HEX_ID` (enforce in `encode_hex_id`)
2. All validity checks use `== INVALID_HEX_ID` (no `INTNaN` checks in user-facing code)
3. Add a helper function if validation is repeated: `def is_valid_hex_id(hid): return hid != INVALID_HEX_ID`

---

## Issue 11: `make_transformer` returns unused `self.proj`

**Description:**
The `make_transformer` function returns a tuple `(proj, transformer)`, but `self.proj` is never used after construction.

**Current behavior:**
- `make_transformer` returns both `proj` and `transformer`
- Only `transformer` is used; `proj` is stored in `self.proj` but never accessed

**Recommended action:**
Simplify `make_transformer` to return only `transformer`. Remove `self.proj` from the constructor. If debugging or inspection of the projection is needed later, it can be reconstructed from the CRS information embedded in the transformer or added as a separate property.

---

## Issue 12: Dead `name` fields in NamedTuples

**Description:**
The NamedTuples `Hex`, `Point`, `Layout`, and `Orientation` in `redblobhex_array.py` all have a `name` field leftover from debugging. This field makes tuple length 4 instead of 3, and could break positional unpacking in user code.

**Current behavior:**
- Example: `Hex = namedtuple("Hex", ["q", "r", "name"])` — should be `["q", "r"]`
- Length is 3 instead of 2, breaking `q, r = hex_tuple`-style unpacking
- Similar issue in `Point`, `Layout`, `Orientation`

**Recommended action:**
Remove the `name` field from all NamedTuples. If debugging output is needed, add a `__str__` or `__repr__` method instead. This restores the tuple to its intended structure and prevents unpacking bugs.

---

## Summary Table

| Issue | Type | Severity | Action | Location |
|-------|------|----------|--------|----------|
| 1 | Redundant NaN check | High | Remove pre-flight check in `label`; let INTNaN propagate | `hex_analysis.py:label` |
| 2 | Private/public boundary | Medium | Rename to `_lon_lat_to_hex_SoA`, `_hex_to_lon_lat_SoA` | `hex_analysis.py` |
| 3 | Wrong type, unused | Medium | Make private or remove; fix type if retained | `hex_analysis.py:hex_corners_lon_lat` |
| 4 | Unused, wrong type | Low | Remove or redefine to return int64 IDs | `hex_analysis.py:hex_of_hexes` |
| 5 | Scalar loop | Medium | Delete; use `hp.to_geodataframe(hex_ids_array)` | `hex_analysis.py:_build_counts_geodataframe` |
| 6 | Scalar loop | Medium | Delete; use `hp.edges_geodataframe(from_ids, to_ids)` | `hex_analysis.py:_build_edge_geometries` |
| 7 | Over-engineered API | High | Simplify to plain arrays; remove xarray branch | `hex_analysis.py:hex_counts` |
| 8 | Inflexible boundary | High | Simplify to `(from_ids, to_ids)`; move slicing to caller | `hex_analysis.py:hex_connectivity` |
| 9 | Forced dependency | Low | Make `hp` optional in `hex_connectivity_power` | `hex_analysis.py:hex_connectivity_power` |
| 10 | Inconsistent validation | High | Standardize on `== INVALID_HEX_ID` everywhere | `hex_analysis.py`, `hexproj.py` |
| 11 | Unused return value | Low | Remove `self.proj` from `HexProj`; simplify `make_transformer` | `hexproj.py:make_transformer` |
| 12 | Dead code in types | Medium | Remove `name` fields from NamedTuples | `redblobhex_array.py` |

---

## Notes

- **Backwards compatibility**: This project makes no backwards-compatibility guarantees (see `AGENTS.md`). Changes should proceed without deprecation wrappers or removals stubs.
- **OD (Origin-Destination)**: Throughout this plan, OD refers to directed pairs of hex cells (origin cell, destination cell), commonly used to represent flow or connectivity between regions.
- **Dask composability**: Several changes (issues 5, 6, 8) aim to improve composability with dask pipelines by eliminating scalar loops and simplifying array boundaries.
- **Geometry decoupling**: Issues 5 and 6 separate geometry building from aggregation logic, allowing users to build counts/connectivity without forcing geometry attachment.
