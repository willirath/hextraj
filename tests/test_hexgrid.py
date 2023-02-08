import pytest

from hextraj.hexgrid import HexGrid


@pytest.mark.parametrize("orientation", ["flat", "pointy"])
def test_origin_projected_to_zero(orientation):
    hexgrid = HexGrid(
        lon_origin=0, lat_origin=0, hex_size_meters=100, hex_orientation=orientation
    )
    hex = hexgrid.lon_lat_to_hex(lon=0.0, lat=0.0)
    assert hex == (0, 0, 0)


@pytest.mark.parametrize("orientation", ["flat", "pointy"])
def test_hex_of_hexes(orientation):
    hexgrid = HexGrid(
        lon_origin=0, lat_origin=0, hex_size_meters=100, hex_orientation=orientation
    )
    assert 1 == len(list(hexgrid.hex_of_hexes(map_radius=0)))
    assert 7 == len(list(hexgrid.hex_of_hexes(map_radius=1)))
    assert 19 == len(list(hexgrid.hex_of_hexes(map_radius=2)))
