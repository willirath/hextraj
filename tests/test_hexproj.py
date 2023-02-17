import numpy as np
import pytest

from hextraj.hexproj import HexProj
from hextraj.redblobhex_array import Hex


@pytest.mark.parametrize("orientation", ["flat", "pointy"])
@pytest.mark.parametrize("point_dtype", ["scalar", "array"])
def test_origin_projected_to_zero(orientation, point_dtype):
    hex_proj = HexProj(
        lon_origin=0, lat_origin=0, hex_size_meters=100, hex_orientation=orientation
    )
    if point_dtype == "scalar":
        hex = hex_proj.lon_lat_to_hex_SoA(lon=0.0, lat=0.0)
        assert hex == Hex(0, 0, 0)
    if point_dtype == "array":
        hex = hex_proj.lon_lat_to_hex_SoA(lon=np.zeros((2, 3)), lat=np.zeros((2, 3)))
        assert hex == Hex(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)))


@pytest.mark.parametrize("orientation", ["flat", "pointy"])
def test_hex_of_hexes(orientation):
    hex_proj = HexProj(
        lon_origin=0, lat_origin=0, hex_size_meters=100, hex_orientation=orientation
    )
    assert 1 == len(list(hex_proj.hex_of_hexes(map_radius=0)))
    assert 7 == len(list(hex_proj.hex_of_hexes(map_radius=1)))
    assert 19 == len(list(hex_proj.hex_of_hexes(map_radius=2)))


def test_check_orientations_available():
    hp_flat = HexProj(hex_orientation="flat")
    hp_pointy = HexProj(hex_orientation="pointy")
    with pytest.raises(ValueError, match="Only 'flat' and 'pointy'"):
        hp_nonexistent = HexProj(hex_orientation="nonexistent")
