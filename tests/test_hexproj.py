import numpy as np
import pytest

from hextraj.hexproj import HexProj
from hextraj.redblobhex_array import Hex


def test_hexproj_repr():
    """Make sure the repr makes sense."""
    hex_proj = HexProj(
        projection_name="laea",
        lon_origin=0,
        lat_origin=0,
        hex_size_meters=100,
        hex_orientation="flat",
    )
    hp_repr = repr(hex_proj)
    assert "laea" in hp_repr
    assert "lon_origin=0" in hp_repr
    assert "lat_origin=0" in hp_repr
    assert "hex_size_meters=100" in hp_repr
    assert "flat" in hp_repr


def test_hexproj_evaluates():
    """Make sure the repr is evaluable."""
    hex_proj = HexProj(
        projection_name="laea",
        lon_origin=0,
        lat_origin=0,
        hex_size_meters=100,
        hex_orientation="flat",
    )
    hp_repr = repr(hex_proj)
    hp2 = eval(hp_repr)

    assert hp_repr == repr(hp2)


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


@pytest.mark.parametrize("orientation", ["flat", "pointy"])
def test_hex_corners_lon_lat_returns_closed_ring_of_seven(orientation):
    """The seventh corner repeats the first so the polygon closes."""
    hex_proj = HexProj(
        lon_origin=0,
        lat_origin=0,
        hex_size_meters=100_000,
        hex_orientation=orientation,
    )
    corners = hex_proj.hex_corners_lon_lat(Hex(0, 0, 0))

    assert len(corners) == 7
    assert corners[0] == corners[-1]


@pytest.mark.parametrize("orientation", ["flat", "pointy"])
def test_hex_corners_lon_lat_surround_the_origin(orientation):
    """The origin hex is centred on (lon_origin, lat_origin)."""
    hex_proj = HexProj(
        lon_origin=0,
        lat_origin=0,
        hex_size_meters=100_000,
        hex_orientation=orientation,
    )
    lons, lats = np.array(hex_proj.hex_corners_lon_lat(Hex(0, 0, 0))).T

    assert lons.min() < 0.0 < lons.max()
    assert lats.min() < 0.0 < lats.max()


@pytest.mark.parametrize("orientation", ["flat", "pointy"])
def test_hex_corner_offsets_close_the_ring_exactly(orientation):
    """The closing offset must be bit-identical to the first, not recomputed."""
    hex_proj = HexProj(
        lon_origin=0,
        lat_origin=0,
        hex_size_meters=100_000,
        hex_orientation=orientation,
    )

    assert hex_proj.corner_offsets_x[0] == hex_proj.corner_offsets_x[-1]
    assert hex_proj.corner_offsets_y[0] == hex_proj.corner_offsets_y[-1]


@pytest.mark.parametrize("orientation", ["flat", "pointy"])
def test_to_geodataframe_polygons_have_no_degenerate_vertex(orientation):
    """A closed hexagon is seven coordinates; an eighth means a degenerate edge.

    Shapely appends a closing vertex when the last coordinate does not
    equal the first exactly, which leaves a zero-length edge behind.
    """
    hex_proj = HexProj(
        lon_origin=0,
        lat_origin=0,
        hex_size_meters=100_000,
        hex_orientation=orientation,
    )
    hex_ids = hex_proj.label(
        np.array([0.0, 5.0, -12.0, 30.0]), np.array([0.0, 40.0, -33.0, 60.0])
    )

    for geom in hex_proj.to_geodataframe(hex_ids).geometry:
        coords = np.asarray(geom.exterior.coords)
        assert len(coords) == 7
        assert np.array_equal(coords[0], coords[-1])
