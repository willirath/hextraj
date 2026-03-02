import pyproj


def make_transformer(projection_name, lat_origin, lon_origin):
    """Build a pyproj.Proj and a WGS84 <-> projected Transformer.

    Parameters
    ----------
    projection_name: str
        Proj projection name, e.g. "laea".
    lat_origin: float
        Latitude of the projection origin.
    lon_origin: float
        Longitude of the projection origin.

    Returns
    -------
    proj: pyproj.Proj
    transformer: pyproj.Transformer
        Transforms in the forward direction from WGS84 (lon, lat) to the
        projected CRS. Use ``direction=pyproj.enums.TransformDirection.INVERSE``
        for the reverse.
    """
    proj = pyproj.Proj(
        f"+proj={projection_name} +lat_0={lat_origin} +lon_0={lon_origin} "
        "+datum=WGS84 +units=m"
    )
    transformer = pyproj.Transformer.from_crs(
        "epsg:4326", proj.crs, always_xy=True
    )
    return proj, transformer
