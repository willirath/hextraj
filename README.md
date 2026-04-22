# hextraj

[![PyPI](https://img.shields.io/pypi/v/hextraj)](https://pypi.org/project/hextraj/)
[![License:MIT](https://img.shields.io/badge/License-MIT-lightgray.svg?style=flt-square)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://willirath.github.io/hextraj/)

Hex labelling of trajectory data.

| Hex region | OD connectivity |
|:---:|:---:|
| ![Hex region example](figures/hex_region_example.png) | ![OD connectivity example](figures/hex_conn_dask_example.png) |
| [hex_aggregation.ipynb](notebooks/hex_aggregation.ipynb) | [hex_conn_dask.ipynb](notebooks/hex_conn_dask.ipynb) |

Maps lon/lat positions to a projected hexagonal grid and provides tools for aggregation and connectivity analysis.

- `hex_counts` — heat maps and density aggregation
- `hex_connectivity` — origin-destination matrices
- `hex_connectivity_power` — multi-generation transport probabilities
- `hex_connectivity_dask` — lazy dask-native connectivity for large datasets
- Full dask support throughout

## Getting started

Explore the notebooks:

- [`hex_conn.ipynb`](notebooks/hex_conn.ipynb) — Label NW Shelf trajectories, compute OD connectivity, visualise choropleth + edge overlay.
- [`hex_aggregation.ipynb`](notebooks/hex_aggregation.ipynb) — Grid construction, choropleth aggregation, and weighted edges.
- [`hex_grid_construction.ipynb`](notebooks/hex_grid_construction.ipynb) — Rectangle and region hex grids.
- [`hex_analysis.ipynb`](notebooks/hex_analysis.ipynb) — Analysis functions (`hex_counts`, `hex_connectivity`, `hex_connectivity_power`).
- [`hex_conn_dask.ipynb`](notebooks/hex_conn_dask.ipynb) — Dask-native connectivity for large datasets.

## Installation

```shell
pip install hextraj
```

With dask, scipy, and cartopy:

```shell
pip install hextraj[full]
```

Or from source:

```shell
pip install git+https://github.com/willirath/hextraj.git@main
```

## Quick example

```python
from hextraj import HexProj

hp = HexProj(lon_origin=-3.0, lat_origin=54.0, hex_size_meters=50_000)

# Label positions → int64 hex IDs
hex_ids = hp.label(lon, lat)

# Build a GeoDataFrame with Polygon geometries
gdf = hp.to_geodataframe(hp.region_of_hexes(region_polygon))
gdf["count"] = counts.reindex(gdf.index).fillna(0)
gdf.plot(column="count", cmap="YlOrRd")
```
