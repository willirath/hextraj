import pytest

from hextraj.hexapi import HexGrid


@pytest.mark.parametrize("orientation", ["flat", "pointy"])
def test_origin_projected_to_zero(orientation):
    hexgrid = HexGrid(
        lon_origin=0, lat_origin=0, hex_size_meters=100, hex_orientation=orientation
    )
    hex = hexgrid.lon_lat_to_hex(lon=0.0, lat=0.0)
    assert hex == (0, 0, 0)
