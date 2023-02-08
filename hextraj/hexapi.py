import pyproj
from . import redblobhex


class HexGrid(object):
    def __init__(
        self,
        projection_name: str = "laea",
        lon_origin: float = 0.0,
        lat_origin: float = 0.0,
        hex_size_meters: float = 100_000,
        hex_orientation: str = "flat"
    ):
        """HexGrid Labeller.

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
                projection_name=self.projection_name, lat_origin=self.lat_origin, lon_origin=self.lon_origin,
            )
        )
        proj_wgs = pyproj.Proj(init='epsg:4326')
        self.transformer_relto_wgs = pyproj.Transformer.from_proj(proj_wgs, self.proj)

    def _set_up_hex_layout(self):
        """Set up hex layout (in projected space!)."""
        if self.hex_orientation == "flat":
            _orientation = redblobhex.layout_flat
        elif self.hex_orientation == "pointy":
            _orientation = redblobhex.layout_pointy
        else:
            raise ValueError("Only 'flat' and 'pointy' orientation is supported.")

        self.hex_layout_projected = redblobhex.Layout(
            orientation=_orientation,
            size=redblobhex.Point(self.hex_size_meters, self.hex_size_meters),
            origin=redblobhex.Point(0, 0),  # always at center of projected space
        )

    def _transform_lon_lat_to_proj(self, lon: float = None, lat: float = None):
        return redblobhex.Point(*self.transform_lon_lat_to_proj(lon, lat, pyproj.enums.TransformDirection.FORWARD))

    def _transform_proj_to_lon_lat(self, x: float = None, y: float = None):
        return self.transform_lon_lat_to_proj(x, y, pyproj.enums.TransformDirection.INVERSE)

    def lon_lat_to_hex(self, lon: float = None, lat: float = None) -> redblobhex._Hex:
        """Point in lon lat to hex.
        
        Parameters
        ----------
        lon: float
           Longitude.
        lat: float
           Latitude.

        Returns
        -------
        Hex tuple.
        """
        xy_projected = self._transform_lon_lat_to_proj(lon=lon, lat=lat)
        hex_tuple = redblobhex.hex_round(redblobhex.pixel_to_hex(self.hex_layout_projected, xy_projected))
        return hex_tuple
