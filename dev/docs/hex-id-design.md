# Hex ID design

## The problem

A hex is defined by cube coordinates `(q, r, s)` with `q + r + s = 0`, so `s` is
redundant — the minimal key is `(q, r)`. The current code carries all three through
the stack as a `Hex` namedtuple or as a numpy structured dtype. This creates friction:

- Structured dtypes are awkward as DataFrame/DataArray indices
- `(q, r, s)` tuples don't serialise cleanly to parquet/zarr/netCDF
- `groupby` on tuple-valued arrays requires workarounds
- geopandas needs a hashable, 1D index

---

## Chosen approach: Cantor pairing of `(q, r)` → int64

Map each coordinate from ℤ to ℕ₀ via:

```
z(n) = 2n      if n ≥ 0
z(n) = -2n - 1 if n < 0
```

Then apply the Cantor pairing function:

```
cantor(a, b) = (a + b)(a + b + 1) / 2 + b
hex_id = cantor(z(q), z(r))
```

Fully reversible. `s` is dropped from the ID (`s = -q - r` is one arithmetic op away).
Internally `Hex(q, r, s)` namedtuples are unchanged — the ID is only the public scalar
label.

### Resolution budget

For `max(|q|, |r|) = M`, the pairing grows as `≈ 8M²`. int64 cap ≈ 9.2 × 10¹⁸:

```
8M² ≲ 9.2 × 10¹⁸  →  M ≲ 2³⁰ ≈ 1.07 × 10⁹
```

Worst-case global coordinate magnitude for a given `hex_size`:

```
max_r ≈ 20,037,000 m / (√3/2 × hex_size_m)
```

| hex_size | global max_coord | cantor ID at max | fits int64? |
|----------|-----------------|------------------|-------------|
| 1 m      | ~23,100,000     | ~4 × 10¹⁵        | yes         |
| 1 cm     | ~2,310,000,000  | ~4 × 10¹⁹        | no          |

Global floor: **~2 cm** (`M ≈ 10⁹`). For all practical scientific use (hex_size ≥ 1 m
globally, or finer regionally) int64 is sufficient.

feedback:

### Sentinel / invalid value

`cantor` always returns a value ≥ 0, and `cantor(0, 0) = 0` maps to the origin hex
`(q=0, r=0)`. Reserve `INVALID_HEX_ID = np.int64(-1)` — negative values are outside
the valid range by construction. No platform-dependent casting, no `numpy.ma`.

```python
INVALID_HEX_ID = np.int64(-1)
```

feedback:

---

## Public API shape

```python
# module-level, vectorised over numpy arrays
encode_hex_id(q, r)   -> np.int64 array   # invalid inputs → INVALID_HEX_ID
decode_hex_id(hex_id) -> (q, r)           # INVALID_HEX_ID → (NaN, NaN)
```

`s` is always recoverable as `s = -q - r`. Where the full tuple is needed (e.g.
neighbor lookup, internal geometry) it is reconstructed on the fly.

- Internal arrays: `Hex(q, r, s)` namedtuple — no change to `redblobhex_array.py`
- External scalar label: int64 Cantor ID
- GeoDataFrame index: int64 hex ID
- xarray coordinate: int64 hex ID (multidim per-hex data stays in xarray;
  GeoDataFrame is for 2D spatial aggregation and visualization — complementary)

feedback:

---

## GeoDataFrame as the aggregation output format

After labelling trajectories, the natural aggregation pipeline is:

```
lon/lat arrays
    → HexProj.label(lon, lat)                  # int64 hex ID arrays, Dask-friendly
    → groupby(hex_id).agg(...)                 # plain int64 groupby
    → HexProj.to_geodataframe(hex_ids, values) # decodes IDs, builds Polygons
    → GeoDataFrame.plot(column=...)            # choropleth
```

`to_geodataframe` decodes IDs to `(q, r)`, recovers `s = -q - r`, constructs
`shapely.Polygon` objects from hex corner coordinates, and returns a `GeoDataFrame`
with:
- index: int64 hex ID
- geometry: hex `Polygon` in WGS84
- any passed value columns

For multidimensional per-hex data (time series, depth profiles, metadata), the int64
hex ID serves as an xarray coordinate. GeoDataFrame and xarray are complementary.

This format also feeds grid construction (M3): generate a dense grid as a GeoDataFrame,
filter by `.intersects(region_polygon)`.

feedback:

---

## Invalid / NaN handling

Current: `INTNaN = np.array(np.nan).astype(int)` — platform-dependent, undefined
outside x86-64.

Replacement: `INVALID_HEX_ID = np.int64(-1)`. Negative values are unreachable by
the Cantor encoding, so this is a portable, unambiguous sentinel. No `numpy.ma`,
no masked arrays, no performance overhead.

feedback:
