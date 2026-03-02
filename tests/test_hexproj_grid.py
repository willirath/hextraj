"""Tests for HexProj grid and GeoDataFrame methods: to_geodataframe, rectangle_of_hexes, region_of_hexes."""

import numpy as np
import pytest

from hextraj.hexproj import HexProj
from hextraj.hex_id import encode_hex_id, decode_hex_id, INVALID_HEX_ID


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def hex_proj():
    """Create a standard HexProj instance for testing."""
    return HexProj(hex_size_meters=500_000)


@pytest.fixture
def sample_hex_ids(hex_proj):
    """Create a few sample hex IDs by encoding nearby (q, r) coordinates."""
    q_coords = np.array([0, 1, 2], dtype=np.int64)
    r_coords = np.array([0, -1, 1], dtype=np.int64)
    return encode_hex_id(q_coords, r_coords)


# ============================================================================
# Feature 1: HexProj.to_geodataframe(hex_ids, **value_cols)
# ============================================================================


def test_to_geodataframe_returns_geodataframe(hex_proj, sample_hex_ids):
    """Call to_geodataframe with a few hex IDs, check return type is GeoDataFrame."""
    gdf = hex_proj.to_geodataframe(sample_hex_ids)

    # Should be able to import and check type
    import geopandas
    assert isinstance(gdf, geopandas.GeoDataFrame)


def test_to_geodataframe_index_is_hex_ids(hex_proj, sample_hex_ids):
    """Check the GeoDataFrame index equals the input hex IDs."""
    gdf = hex_proj.to_geodataframe(sample_hex_ids)

    # Index should match the input hex IDs
    np.testing.assert_array_equal(gdf.index.values, sample_hex_ids)


def test_to_geodataframe_geometry_is_polygon(hex_proj, sample_hex_ids):
    """Check all geometries are shapely Polygons."""
    from shapely.geometry import Polygon

    gdf = hex_proj.to_geodataframe(sample_hex_ids)

    # All geometries should be Polygons (not None)
    assert "geometry" in gdf.columns
    for geom in gdf.geometry:
        assert isinstance(geom, Polygon), f"Expected Polygon but got {type(geom)}"


def test_to_geodataframe_value_cols(hex_proj, sample_hex_ids):
    """Pass a value array as keyword arg, check it appears as a column."""
    value_array = np.array([10.0, 20.0, 30.0])

    gdf = hex_proj.to_geodataframe(sample_hex_ids, my_values=value_array)

    # Should have the value column
    assert "my_values" in gdf.columns
    np.testing.assert_array_equal(gdf["my_values"].values, value_array)


def test_to_geodataframe_multiple_value_cols(hex_proj, sample_hex_ids):
    """Pass multiple value columns as keyword args."""
    vals1 = np.array([1, 2, 3])
    vals2 = np.array([10, 20, 30])

    gdf = hex_proj.to_geodataframe(
        sample_hex_ids,
        column_a=vals1,
        column_b=vals2
    )

    # Should have both columns
    assert "column_a" in gdf.columns
    assert "column_b" in gdf.columns
    np.testing.assert_array_equal(gdf["column_a"].values, vals1)
    np.testing.assert_array_equal(gdf["column_b"].values, vals2)


def test_to_geodataframe_invalid_hex_id_has_nan_geometry(hex_proj):
    """Pass INVALID_HEX_ID, check geometry is None or NaN (not a Polygon)."""
    from shapely.geometry import Polygon

    # Create array with one valid and one invalid hex ID
    q_valid = np.array([0], dtype=np.int64)
    r_valid = np.array([0], dtype=np.int64)
    valid_id = encode_hex_id(q_valid, r_valid)[0]

    hex_ids = np.array([valid_id, INVALID_HEX_ID], dtype=np.int64)
    gdf = hex_proj.to_geodataframe(hex_ids)

    # First geometry should be a Polygon
    assert isinstance(gdf.geometry.iloc[0], Polygon)

    # Second geometry should be None or NaN (not a Polygon)
    second_geom = gdf.geometry.iloc[1]
    assert second_geom is None


# ============================================================================
# Feature 2: HexProj.rectangle_of_hexes(lon_min, lon_max, lat_min, lat_max)
# ============================================================================


def test_rectangle_of_hexes_returns_int64_array(hex_proj):
    """Check that rectangle_of_hexes returns an int64 numpy array."""
    result = hex_proj.rectangle_of_hexes(
        lon_min=-10.0,
        lon_max=10.0,
        lat_min=-10.0,
        lat_max=10.0
    )

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64


def test_rectangle_of_hexes_nonempty(hex_proj):
    """Non-trivial bounding box yields at least one hex."""
    result = hex_proj.rectangle_of_hexes(
        lon_min=-10.0,
        lon_max=10.0,
        lat_min=-10.0,
        lat_max=10.0
    )

    # Should have at least one hex
    assert len(result) > 0


def test_rectangle_of_hexes_nonempty_large_bbox(hex_proj):
    """Larger bounding box yields multiple hexes."""
    result = hex_proj.rectangle_of_hexes(
        lon_min=-30.0,
        lon_max=30.0,
        lat_min=-30.0,
        lat_max=30.0
    )

    # Should have multiple hexes for a larger region
    assert len(result) > 1


def test_rectangle_of_hexes_coverage(hex_proj):
    """Encode the center lon/lat of the bbox, check that hex ID appears in result."""
    lon_min, lon_max = -5.0, 5.0
    lat_min, lat_max = -5.0, 5.0

    # Center of the bounding box
    center_lon = (lon_min + lon_max) / 2.0
    center_lat = (lat_min + lat_max) / 2.0

    # Get the hex ID of the center point
    hex_SoA = hex_proj.lon_lat_to_hex_SoA(lon=center_lon, lat=center_lat)
    center_hex_id = encode_hex_id(hex_SoA.q, hex_SoA.r)

    # Get all hex IDs in the rectangle
    rect_hex_ids = hex_proj.rectangle_of_hexes(
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max
    )

    assert center_hex_id in rect_hex_ids


def test_rectangle_of_hexes_small_bbox(hex_proj):
    """Small bounding box around origin should yield at least one hex."""
    result = hex_proj.rectangle_of_hexes(
        lon_min=-0.5,
        lon_max=0.5,
        lat_min=-0.5,
        lat_max=0.5
    )

    assert len(result) > 0


# ============================================================================
# Feature 3: HexProj.region_of_hexes(region_polygon)
# ============================================================================


def test_region_of_hexes_returns_int64_array(hex_proj):
    """Check that region_of_hexes returns an int64 numpy array."""
    from shapely.geometry import Polygon

    # Create a simple square polygon around the origin
    coords = [(-5, -5), (5, -5), (5, 5), (-5, 5), (-5, -5)]
    polygon = Polygon(coords)

    result = hex_proj.region_of_hexes(polygon)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64


def test_region_of_hexes_nonempty(hex_proj):
    """Polygon covering a few hexes yields results."""
    from shapely.geometry import Polygon

    # Create a polygon that covers a reasonable area
    coords = [(-10, -10), (10, -10), (10, 10), (-10, 10), (-10, -10)]
    polygon = Polygon(coords)

    result = hex_proj.region_of_hexes(polygon)

    assert len(result) > 0


def test_region_of_hexes_nonempty_large_polygon(hex_proj):
    """Larger polygon yields multiple hexes."""
    from shapely.geometry import Polygon

    # Create a larger polygon
    coords = [(-20, -20), (20, -20), (20, 20), (-20, 20), (-20, -20)]
    polygon = Polygon(coords)

    result = hex_proj.region_of_hexes(polygon)

    # Should have multiple hexes
    assert len(result) > 1


def test_region_of_hexes_intersects_polygon(hex_proj):
    """Decode result IDs to (q,r), build GeoDataFrame, assert all hex geometries intersect polygon."""
    from shapely.geometry import Polygon

    # Create a simple square polygon
    lon_min, lon_max = -10.0, 10.0
    lat_min, lat_max = -10.0, 10.0
    coords = [(lon_min, lat_min), (lon_max, lat_min), (lon_max, lat_max), (lon_min, lat_max), (lon_min, lat_min)]
    polygon = Polygon(coords)

    result_hex_ids = hex_proj.region_of_hexes(polygon)

    # Build GeoDataFrame from returned hex IDs
    gdf = hex_proj.to_geodataframe(result_hex_ids)

    # All geometries should intersect the polygon
    assert (gdf.geometry.intersects(polygon)).all(), \
        "Not all returned hex geometries intersect the polygon"


def test_region_of_hexes_origin_inside(hex_proj):
    """Polygon containing origin should yield hexes."""
    from shapely.geometry import Polygon

    # Create a polygon around the origin
    coords = [(-5, -5), (5, -5), (5, 5), (-5, 5), (-5, -5)]
    polygon = Polygon(coords)

    result = hex_proj.region_of_hexes(polygon)

    # Should have at least the hex at origin
    assert len(result) > 0


def test_region_of_hexes_hexagon_polygon(hex_proj):
    """Use a hexagonal polygon region and check results are valid."""
    from shapely.geometry import Polygon

    # Create a simple hexagonal polygon
    angles = np.linspace(0, 2 * np.pi, 7)
    radius = 15.0
    coords = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in angles]
    hexagon = Polygon(coords)

    result = hex_proj.region_of_hexes(hexagon)

    # Should have at least one hex
    assert len(result) > 0

    # All results should be valid int64 hex IDs
    assert all(h != INVALID_HEX_ID for h in result)


def test_region_of_hexes_small_polygon(hex_proj):
    """Very small polygon might yield one or zero hexes."""
    from shapely.geometry import Polygon

    # Create a small polygon around origin
    coords = [(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1)]
    polygon = Polygon(coords)

    result = hex_proj.region_of_hexes(polygon)

    # Should be a valid int64 array
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64


def test_region_of_hexes_is_subset_of_rectangle(hex_proj):
    """Every ID returned by region_of_hexes must also appear in rectangle_of_hexes for the same bounds."""
    from shapely.geometry import box as shapely_box

    lon_min, lon_max, lat_min, lat_max = -10.0, 10.0, -10.0, 10.0
    polygon = shapely_box(lon_min, lat_min, lon_max, lat_max)

    region_ids = hex_proj.region_of_hexes(polygon)
    rect_ids = hex_proj.rectangle_of_hexes(lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)

    assert set(region_ids).issubset(set(rect_ids))


def test_to_geodataframe_crs(hex_proj, sample_hex_ids):
    """Test that GeoDataFrame has CRS set to EPSG:4326."""
    gdf = hex_proj.to_geodataframe(sample_hex_ids)
    assert gdf.crs.to_epsg() == 4326


def test_to_geodataframe_batch_polygon_count(hex_proj):
    """Call to_geodataframe with 10 hex IDs and check length and geometry types."""
    from shapely.geometry import Polygon

    q_coords = np.arange(10, dtype=np.int64)
    r_coords = np.zeros(10, dtype=np.int64)
    hex_ids = encode_hex_id(q_coords, r_coords)

    gdf = hex_proj.to_geodataframe(hex_ids)

    assert len(gdf) == 10
    for geom in gdf.geometry:
        assert isinstance(geom, Polygon)
