# hextraj

Hex-labelling of trajectory data.

Maps lon/lat positions to a projected hexagonal grid and provides tools for aggregation and connectivity analysis.

- `hex_counts` — heat maps and density aggregation
- `hex_connectivity` — origin-destination matrices
- `hex_connectivity_power` — multi-generation transport probabilities
- `hex_connectivity_dask` — lazy dask-native connectivity for large datasets
- Full dask support throughout

## Installation

```shell
pip install git+https://github.com/willirath/hextraj.git@main
```

For dask, scipy, and cartopy support:

```shell
pip install "hextraj[full] @ git+https://github.com/willirath/hextraj.git@main"
```

## Quick example

```python
from hextraj import HexProj

hp = HexProj(lon_origin=-3.0, lat_origin=54.0, hex_size_meters=50_000)

# Label positions -> int64 hex IDs
hex_ids = hp.label(lon, lat)

# Build a GeoDataFrame with Polygon geometries
gdf = hp.to_geodataframe(hp.region_of_hexes(region_polygon))
gdf["count"] = counts.reindex(gdf.index).fillna(0)
gdf.plot(column="count", cmap="YlOrRd")
```

```{toctree}
:maxdepth: 2
:caption: Notebooks

notebooks/hex_grid_construction
notebooks/hex_aggregation
notebooks/hex_conn
notebooks/hex_analysis
notebooks/hex_conn_dask
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```
