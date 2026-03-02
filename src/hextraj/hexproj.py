import numpy as np
import pyproj
import xarray as xr

from numpy.typing import NDArray

from . import redblobhex_array as redblobhex
from ._proj import make_transformer
from .hex_id import encode_hex_id, decode_hex_id, INVALID_HEX_ID


class HexProj:
    def __init__(
        self,
        projection_name: str = "laea",
        lon_origin: float = 0.0,
        lat_origin: float = 0.0,
        hex_size_meters: float = 100_000,
        hex_orientation: str = "flat",
    ):
        """HexProj Labeller.

        Parameters
        ----------
        projection_name: str
            Defaults to: "laea"
        lon_origin: float
            Defaults to: 0.0
        lat_origin: float
            Defaults to: 0.0
        hex_size_meters: float
            Defaults to: 100000.0
        hex_orientation: str
            Can be "flat" or "pointy". Defaults to: "flat"

        """
        self.projection_name = projection_name
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.hex_size_meters = hex_size_meters
        self.hex_orientation = hex_orientation

        self._set_up_projection()
        self._set_up_hex_layout()

    def _set_up_projection(self):
        """Initialize projection."""
        self.proj, self.transformer_relto_wgs = make_transformer(
            self.projection_name, self.lat_origin, self.lon_origin
        )

    def _set_up_hex_layout(self):
        """Set up hex layout (in projected space!)."""
        if self.hex_orientation == "flat":
            _orientation = redblobhex.orientation_flat
        elif self.hex_orientation == "pointy":
            _orientation = redblobhex.orientation_pointy
        else:
            raise ValueError("Only 'flat' and 'pointy' orientation is supported.")

        self.hex_layout_projected = redblobhex.Layout(
            orientation=_orientation,
            size=redblobhex.Point(self.hex_size_meters, self.hex_size_meters),
            origin=redblobhex.Point(0, 0),  # always at center of projected space
        )

        corner_offsets = tuple(
            redblobhex.hex_corner_offset(self.hex_layout_projected, c) for c in range(7)
        )
        self.corner_offsets_projected = corner_offsets
        self.corner_offsets_x = np.array([p.x for p in corner_offsets])
        self.corner_offsets_y = np.array([p.y for p in corner_offsets])

    def _transform_lon_lat_to_proj(self, lon=None, lat=None):
        return redblobhex.Point(
            *self.transformer_relto_wgs.transform(
                lon, lat, direction=pyproj.enums.TransformDirection.FORWARD
            )
        )

    def _transform_proj_to_lon_lat(self, x=None, y=None):
        return self.transformer_relto_wgs.transform(
            x, y, direction=pyproj.enums.TransformDirection.INVERSE
        )

    def lon_lat_to_hex_SoA(self, lon=None, lat=None) -> redblobhex.Hex:
        """Point in lon lat to hex label (tuple of arrays).

        This is the internal representation which makes best use of the
        array capabilities of pyproj.

        Parameters
        ----------
        lon: float
           Longitude.
        lat: float
           Latitude.

        Returns
        -------
        tuple
            Tuple of arrays.

        """
        xy_projected = self._transform_lon_lat_to_proj(lon=lon, lat=lat)
        hex_tuple = redblobhex.hex_round(
            redblobhex.pixel_to_hex(self.hex_layout_projected, xy_projected)
        )
        return hex_tuple

    def label(self, lon, lat):
        """Convenience wrapper to get hex IDs from lon/lat coordinates.

        Parameters
        ----------
        lon: float or array-like
            Longitude(s).
        lat: float or array-like
            Latitude(s).

        Returns
        -------
        int64 or ndarray
            Hex ID(s) encoded from q and r coordinates.
        """
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        nan_mask = np.isnan(lon) | np.isnan(lat)
        hex_soa = self.lon_lat_to_hex_SoA(lon=lon, lat=lat)
        result = encode_hex_id(hex_soa.q, hex_soa.r)
        if np.any(nan_mask):
            result = np.where(nan_mask, INVALID_HEX_ID, result)
        # Handle scalar case: if input was scalar (0-d), return np.int64
        if result.ndim == 0:
            return np.int64(result)
        return result

    def hex_to_lon_lat_SoA(self, hex_tuple=None):
        """Hex tuple to lon, lat (from hex tuple of arrays).

        This is the internal representation which makes best use of the
        array capabilities of pyproj.

        Parameters
        ----------
        hex_tuple: tuple
            Hex tuple (q, r) or (q, r, s). If only (q, r) provided, s is computed.

        Returns
        -------
        tuple
            lon, lat
        """
        if len(hex_tuple) == 2:
            q, r = hex_tuple
            s = -q - r
            hex_tuple = (q, r, s)
        hex_center_projected = redblobhex.hex_to_pixel(
            self.hex_layout_projected, redblobhex.Hex(*hex_tuple)
        )
        return self._transform_proj_to_lon_lat(
            hex_center_projected.x, hex_center_projected.y
        )

    def hex_corners_lon_lat(self, hex_tuple=None):
        """Hex tuple to corner lon, lat.

        Parameters
        ----------
        hex_tuple: tuple
            Hex tuple.

        Returns
        -------
        list
            List of (lon, lat) tuples of corners.
        """
        hex_center_projected = redblobhex.hex_to_pixel(
            self.hex_layout_projected, hex_tuple
        )
        corners_projected = tuple(
            redblobhex.Point(
                hex_center_projected.x + cop.x, hex_center_projected.y + cop.y
            )
            for cop in self.corner_offsets_projected
        )
        corners_lon_lat = [
            self._transform_proj_to_lon_lat(c.x, c.y) for c in corners_projected
        ]
        return corners_lon_lat

    def hex_of_hexes(self, map_radius: int = 2):
        """Generate collection of hexes which fill a hex centered about (0, 0, 0).

        Parameters
        ----------
        map_radius: int
           Defaults to: 2

        Returns
        -------
        generator
           List of hex tuples.
        """
        for q in range(-map_radius, map_radius + 1):
            r1 = max(-map_radius, -q - map_radius)
            r2 = min(map_radius, -q + map_radius)
            for r in range(r1, r2 + 1):
                yield redblobhex.Hex(q, r, -q - r)

    def to_geodataframe(self, hex_ids, **value_cols):
        """Convert hex IDs to a GeoDataFrame with Polygon geometries.

        Parameters
        ----------
        hex_ids: array-like
            1D array of int64 hex IDs (may include INVALID_HEX_ID)
        **value_cols: dict
            Additional columns to include in the GeoDataFrame

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with index=hex_ids, geometry column, and value_cols
        """
        import geopandas
        import shapely

        hex_ids = np.asarray(hex_ids, dtype=np.int64)
        q_coords, r_coords = decode_hex_id(hex_ids)

        invalid = (q_coords == redblobhex.INTNaN) | (r_coords == redblobhex.INTNaN)
        valid = ~invalid

        geometries = [None] * len(hex_ids)

        if valid.any():
            q_valid = q_coords[valid]
            r_valid = r_coords[valid]
            hex_soa = redblobhex.Hex(q_valid, r_valid, -q_valid - r_valid)
            center = redblobhex.hex_to_pixel(self.hex_layout_projected, hex_soa)

            # shape (7, N_valid)
            corners_x = center.x[np.newaxis, :] + self.corner_offsets_x[:, np.newaxis]
            corners_y = center.y[np.newaxis, :] + self.corner_offsets_y[:, np.newaxis]

            # one batched inverse projection call
            corner_lons, corner_lats = self._transform_proj_to_lon_lat(
                corners_x.ravel(), corners_y.ravel()
            )

            n_valid = valid.sum()
            # reshape to (7, N_valid), transpose to (N_valid, 7)
            corner_lons = corner_lons.reshape(7, n_valid).T
            corner_lats = corner_lats.reshape(7, n_valid).T

            # (N_valid, 7, 2) coordinate array for shapely.polygons
            coords = np.stack([corner_lons, corner_lats], axis=-1)
            polygons = shapely.polygons(coords)

            valid_indices = np.where(valid)[0]
            for i, poly in zip(valid_indices, polygons):
                geometries[i] = poly

        return geopandas.GeoDataFrame(
            {**value_cols, "geometry": geometries},
            index=hex_ids,
            crs="EPSG:4326"
        )

    def rectangle_of_hexes(self, lon_min, lon_max, lat_min, lat_max):
        """Generate all hex IDs covering a bounding box.

        Parameters
        ----------
        lon_min: float
            Minimum longitude
        lon_max: float
            Maximum longitude
        lat_min: float
            Minimum latitude
        lat_max: float
            Maximum latitude

        Returns
        -------
        np.ndarray
            1D array of int64 hex IDs
        """
        from shapely.geometry import box as shapely_box

        hex_corners = []
        for lon, lat in [
            (lon_min, lat_min),
            (lon_max, lat_min),
            (lon_max, lat_max),
            (lon_min, lat_max),
        ]:
            hex_SoA = self.lon_lat_to_hex_SoA(lon=lon, lat=lat)
            hex_corners.append((hex_SoA.q, hex_SoA.r))

        q_coords = np.array([h[0] for h in hex_corners], dtype=np.int64)
        r_coords = np.array([h[1] for h in hex_corners], dtype=np.int64)

        q_min = q_coords.min() - 1
        q_max = q_coords.max() + 1
        r_min = r_coords.min() - 1
        r_max = r_coords.max() + 1

        q_range = np.arange(q_min, q_max + 1, dtype=np.int64)
        r_range = np.arange(r_min, r_max + 1, dtype=np.int64)
        q_mesh, r_mesh = np.meshgrid(q_range, r_range, indexing='ij')
        q_flat = q_mesh.ravel()
        r_flat = r_mesh.ravel()
        s_flat = -q_flat - r_flat

        hex_ids = encode_hex_id(q_flat, r_flat)

        # Build bbox as shapely Polygon and filter by intersects
        bbox_polygon = shapely_box(lon_min, lat_min, lon_max, lat_max)
        gdf = self.to_geodataframe(hex_ids)
        mask = gdf.geometry.intersects(bbox_polygon)

        return hex_ids[mask.values]

    def region_of_hexes(self, region_polygon):
        """Generate all hex IDs whose polygons intersect a region polygon.

        Parameters
        ----------
        region_polygon: shapely.geometry.Polygon
            Polygon in WGS84 lon/lat coordinates

        Returns
        -------
        np.ndarray
            1D array of int64 hex IDs
        """
        bounds = region_polygon.bounds
        lon_min, lat_min, lon_max, lat_max = bounds

        candidate_hex_ids = self.rectangle_of_hexes(
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max
        )

        gdf = self.to_geodataframe(candidate_hex_ids)
        mask = gdf.geometry.intersects(region_polygon)

        return candidate_hex_ids[mask.values]

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
        import pandas as pd
        import geopandas
        import shapely

        from_ids = np.asarray(from_ids, dtype=np.int64)
        to_ids = np.asarray(to_ids, dtype=np.int64)

        # Decode both endpoints
        q_from, r_from = decode_hex_id(from_ids)
        q_to, r_to = decode_hex_id(to_ids)

        # Build invalid mask where either endpoint is INVALID_HEX_ID
        invalid = (q_from == redblobhex.INTNaN) | (r_from == redblobhex.INTNaN) | \
                  (q_to == redblobhex.INTNaN) | (r_to == redblobhex.INTNaN)
        valid = ~invalid

        geometries = [None] * len(from_ids)

        if valid.any():
            # For valid rows: hex_to_pixel on both ends
            q_from_valid = q_from[valid]
            r_from_valid = r_from[valid]
            q_to_valid = q_to[valid]
            r_to_valid = r_to[valid]

            hex_from = redblobhex.Hex(q_from_valid, r_from_valid, -q_from_valid - r_from_valid)
            hex_to = redblobhex.Hex(q_to_valid, r_to_valid, -q_to_valid - r_to_valid)

            center_from = redblobhex.hex_to_pixel(self.hex_layout_projected, hex_from)
            center_to = redblobhex.hex_to_pixel(self.hex_layout_projected, hex_to)

            # Transform from projected space to lon/lat
            lon_from, lat_from = self._transform_proj_to_lon_lat(center_from.x, center_from.y)
            lon_to, lat_to = self._transform_proj_to_lon_lat(center_to.x, center_to.y)

            # Build coordinate arrays for LineStrings: shape (N_valid, 2, 2)
            n_valid = valid.sum()
            coords = np.stack([
                np.stack([lon_from, lat_from], axis=-1),  # (N_valid, 2) - first endpoint
                np.stack([lon_to, lat_to], axis=-1)       # (N_valid, 2) - second endpoint
            ], axis=1)  # (N_valid, 2, 2)

            linestrings = shapely.linestrings(coords)

            valid_indices = np.where(valid)[0]
            for i, linestring in zip(valid_indices, linestrings):
                geometries[i] = linestring

        return geopandas.GeoDataFrame(
            {**value_cols, "geometry": geometries},
            index=pd.MultiIndex.from_arrays([from_ids, to_ids], names=["from_id", "to_id"]),
            crs="EPSG:4326"
        )

    def __repr__(self):
        """Repr."""
        return (
            "HexProj("
            f"projection_name={repr(self.projection_name)}, "
            f"lon_origin={repr(self.lon_origin)}, "
            f"lat_origin={repr(self.lat_origin)}, "
            f"hex_size_meters={repr(self.hex_size_meters)}, "
            f"hex_orientation={repr(self.hex_orientation)}, "
            ")"
        )
