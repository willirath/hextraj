import pyproj

from . import redblobhex_array as redblobhex

from numpy.typing import NDArray
import xarray as xr


class HexProj(object):
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
        self.proj = pyproj.Proj(
            "+proj={projection_name} +lat_0={lat_origin} +lon_0={lon_origin} +datum=WGS84 +units=m".format(
                projection_name=self.projection_name,
                lat_origin=self.lat_origin,
                lon_origin=self.lon_origin,
            )
        )
        proj_wgs = pyproj.Proj(init="epsg:4326")
        self.transformer_relto_wgs = pyproj.Transformer.from_proj(proj_wgs, self.proj)

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

        self.corner_offsets_projected = tuple(
            redblobhex.hex_corner_offset(self.hex_layout_projected, c) for c in range(7)
        )

    def _transform_lon_lat_to_proj(self, lon: float = None, lat: float = None):
        return redblobhex.Point(
            *self.transformer_relto_wgs.transform(
                lon, lat, direction=pyproj.enums.TransformDirection.FORWARD
            )
        )

    def _transform_proj_to_lon_lat(self, x: float = None, y: float = None):
        return self.transformer_relto_wgs.transform(
            x, y, direction=pyproj.enums.TransformDirection.INVERSE
        )

    def lon_lat_to_hex_SoA(self, lon: float = None, lat: float = None) -> redblobhex.Hex:
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

    def lon_lat_to_hex_AoS(self, lon: float = None, lat: float = None) -> NDArray:
        """Point in lon lat to hex label (array of tuples).

        This is a representation which allows for handling hex labels as
        categoricals.

        Parameters
        ----------
        lon: float
           Longitude.
        lat: float
           Latitude.

        Returns
        -------
        array
            Array of tuples.

        """
        hex_tuple_SoA = self.lon_lat_to_hex_SoA(lon=lon, lat=lat)
        hex_tuple_AoS = (
            hex_tuple_SoA.q.astype([("q", int)]).astype(tuple)
            + hex_tuple_SoA.r.astype([("r", int)]).astype(tuple)
            + hex_tuple_SoA.s.astype([("s", int)]).astype(tuple)
        )
        return hex_tuple_AoS

    def hex_to_lon_lat_SoA(self, hex_tuple: redblobhex.Hex = None):
        """Hex tuple to lon, lat (from hex tuple of arrays).

        This is the internal representation which makes best use of the 
        array capabilities of pyproj.

        Parameters
        ----------
        hex_tuple: tuple
            Hex tuple.

        Returns
        -------
        tuple
            lon, lat
        """
        hex_center_projected = redblobhex.hex_to_pixel(
            self.hex_layout_projected, hex_tuple
        )
        return self._transform_proj_to_lon_lat(
            hex_center_projected.x, hex_center_projected.y
        )

    def hex_corners_lon_lat(self, hex_tuple: redblobhex.Hex = None):
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
