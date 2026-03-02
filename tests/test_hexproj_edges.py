"""Tests for HexProj.edges_geodataframe method: building GeoDataFrame of LineString edges between hex centers."""

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


# ============================================================================
# Test: edges_geodataframe returns GeoDataFrame
# ============================================================================


def test_edges_geodataframe_returns_geodataframe(hex_proj):
    """Call edges_geodataframe with hex IDs, check return type is GeoDataFrame."""
    from_ids = np.array([0, 1], dtype=np.int64)
    to_ids = np.array([1, 2], dtype=np.int64)

    # Encode some valid hex IDs
    from_q = np.array([0, 1], dtype=np.int64)
    from_r = np.array([0, -1], dtype=np.int64)
    to_q = np.array([1, 2], dtype=np.int64)
    to_r = np.array([-1, 1], dtype=np.int64)

    from_ids = encode_hex_id(from_q, from_r)
    to_ids = encode_hex_id(to_q, to_r)

    gdf = hex_proj.edges_geodataframe(from_ids, to_ids)

    import geopandas
    assert isinstance(gdf, geopandas.GeoDataFrame)


# ============================================================================
# Test: edges_geodataframe geometry is LineString
# ============================================================================


def test_edges_geodataframe_geometry_is_linestring(hex_proj):
    """Check all valid rows have LineString geometry."""
    from shapely.geometry import LineString

    # Create two edges between valid hex IDs
    from_q = np.array([0, 1], dtype=np.int64)
    from_r = np.array([0, -1], dtype=np.int64)
    to_q = np.array([1, 2], dtype=np.int64)
    to_r = np.array([-1, 1], dtype=np.int64)

    from_ids = encode_hex_id(from_q, from_r)
    to_ids = encode_hex_id(to_q, to_r)

    gdf = hex_proj.edges_geodataframe(from_ids, to_ids)

    # All geometries in valid rows should be LineStrings
    for geom in gdf.geometry:
        assert isinstance(geom, LineString), f"Expected LineString but got {type(geom)}"


# ============================================================================
# Test: edges_geodataframe MultiIndex
# ============================================================================


def test_edges_geodataframe_multiindex(hex_proj):
    """Check index is MultiIndex named ['from_id', 'to_id']."""
    import pandas as pd

    from_q = np.array([0, 1], dtype=np.int64)
    from_r = np.array([0, -1], dtype=np.int64)
    to_q = np.array([1, 2], dtype=np.int64)
    to_r = np.array([-1, 1], dtype=np.int64)

    from_ids = encode_hex_id(from_q, from_r)
    to_ids = encode_hex_id(to_q, to_r)

    gdf = hex_proj.edges_geodataframe(from_ids, to_ids)

    # Index should be MultiIndex
    assert isinstance(gdf.index, pd.MultiIndex)

    # Index names should match
    assert gdf.index.names == ["from_id", "to_id"]

    # Index values should match inputs
    assert np.array_equal(gdf.index.get_level_values("from_id"), from_ids)
    assert np.array_equal(gdf.index.get_level_values("to_id"), to_ids)


# ============================================================================
# Test: edges_geodataframe CRS
# ============================================================================


def test_edges_geodataframe_crs(hex_proj):
    """Test that GeoDataFrame has CRS set to EPSG:4326."""
    from_q = np.array([0], dtype=np.int64)
    from_r = np.array([0], dtype=np.int64)
    to_q = np.array([1], dtype=np.int64)
    to_r = np.array([-1], dtype=np.int64)

    from_ids = encode_hex_id(from_q, from_r)
    to_ids = encode_hex_id(to_q, to_r)

    gdf = hex_proj.edges_geodataframe(from_ids, to_ids)

    assert gdf.crs.to_epsg() == 4326


# ============================================================================
# Test: edges_geodataframe single value column
# ============================================================================


def test_edges_geodataframe_value_cols(hex_proj):
    """Pass a single value array as keyword arg, check it appears as a column."""
    from_q = np.array([0, 1], dtype=np.int64)
    from_r = np.array([0, -1], dtype=np.int64)
    to_q = np.array([1, 2], dtype=np.int64)
    to_r = np.array([-1, 1], dtype=np.int64)

    from_ids = encode_hex_id(from_q, from_r)
    to_ids = encode_hex_id(to_q, to_r)

    weight_array = np.array([10.0, 20.0])

    gdf = hex_proj.edges_geodataframe(from_ids, to_ids, weight=weight_array)

    # Should have the weight column
    assert "weight" in gdf.columns
    np.testing.assert_array_equal(gdf["weight"].values, weight_array)


# ============================================================================
# Test: edges_geodataframe multiple value columns
# ============================================================================


def test_edges_geodataframe_multiple_value_cols(hex_proj):
    """Pass multiple value columns as keyword args."""
    from_q = np.array([0, 1], dtype=np.int64)
    from_r = np.array([0, -1], dtype=np.int64)
    to_q = np.array([1, 2], dtype=np.int64)
    to_r = np.array([-1, 1], dtype=np.int64)

    from_ids = encode_hex_id(from_q, from_r)
    to_ids = encode_hex_id(to_q, to_r)

    weight_array = np.array([10.0, 20.0])
    count_array = np.array([5, 15])

    gdf = hex_proj.edges_geodataframe(
        from_ids, to_ids,
        weight=weight_array,
        count=count_array
    )

    # Should have both columns
    assert "weight" in gdf.columns
    assert "count" in gdf.columns
    np.testing.assert_array_equal(gdf["weight"].values, weight_array)
    np.testing.assert_array_equal(gdf["count"].values, count_array)


# ============================================================================
# Test: edges_geodataframe invalid from_id
# ============================================================================


def test_edges_geodataframe_invalid_from_id(hex_proj):
    """INVALID_HEX_ID in from_ids should result in None geometry."""
    from_q = np.array([0, 0], dtype=np.int64)
    from_r = np.array([0, 0], dtype=np.int64)
    to_q = np.array([1, 1], dtype=np.int64)
    to_r = np.array([-1, -1], dtype=np.int64)

    valid_from_ids = encode_hex_id(from_q[:1], from_r[:1])
    valid_to_ids = encode_hex_id(to_q[:1], to_r[:1])

    # Create mixed array with one valid and one invalid from_id
    from_ids = np.array([valid_from_ids[0], INVALID_HEX_ID], dtype=np.int64)
    to_ids = np.array([valid_to_ids[0], valid_to_ids[0]], dtype=np.int64)

    gdf = hex_proj.edges_geodataframe(from_ids, to_ids)

    # First row should have geometry
    assert gdf.geometry.iloc[0] is not None

    # Second row should have None geometry
    assert gdf.geometry.iloc[1] is None


# ============================================================================
# Test: edges_geodataframe invalid to_id
# ============================================================================


def test_edges_geodataframe_invalid_to_id(hex_proj):
    """INVALID_HEX_ID in to_ids should result in None geometry."""
    from_q = np.array([0, 0], dtype=np.int64)
    from_r = np.array([0, 0], dtype=np.int64)
    to_q = np.array([1, 1], dtype=np.int64)
    to_r = np.array([-1, -1], dtype=np.int64)

    valid_from_ids = encode_hex_id(from_q[:1], from_r[:1])
    valid_to_ids = encode_hex_id(to_q[:1], to_r[:1])

    # Create mixed array with one valid and one invalid to_id
    from_ids = np.array([valid_from_ids[0], valid_from_ids[0]], dtype=np.int64)
    to_ids = np.array([valid_to_ids[0], INVALID_HEX_ID], dtype=np.int64)

    gdf = hex_proj.edges_geodataframe(from_ids, to_ids)

    # First row should have geometry
    assert gdf.geometry.iloc[0] is not None

    # Second row should have None geometry
    assert gdf.geometry.iloc[1] is None


# ============================================================================
# Test: edges_geodataframe self-loop
# ============================================================================


def test_edges_geodataframe_self_loop(hex_proj):
    """from_id == to_id should create a degenerate LineString, no exception."""
    from_q = np.array([0], dtype=np.int64)
    from_r = np.array([0], dtype=np.int64)

    hex_id = encode_hex_id(from_q, from_r)

    # Create a self-loop (from_id == to_id)
    from_ids = np.array([hex_id[0]], dtype=np.int64)
    to_ids = np.array([hex_id[0]], dtype=np.int64)

    gdf = hex_proj.edges_geodataframe(from_ids, to_ids)

    # Should not raise an exception
    # Geometry should exist (could be degenerate LineString with same start/end)
    assert gdf.geometry.iloc[0] is not None


# ============================================================================
# Test: edges_geodataframe LineString endpoints
# ============================================================================


def test_edges_geodataframe_linestring_endpoints(hex_proj):
    """Verify LineString coordinates match decoded hex centers."""
    from_q = np.array([0], dtype=np.int64)
    from_r = np.array([0], dtype=np.int64)
    to_q = np.array([1], dtype=np.int64)
    to_r = np.array([-1], dtype=np.int64)

    from_ids = encode_hex_id(from_q, from_r)
    to_ids = encode_hex_id(to_q, to_r)

    gdf = hex_proj.edges_geodataframe(from_ids, to_ids)

    # Get the LineString from the result
    linestring = gdf.geometry.iloc[0]

    # Decode hex IDs to get expected centers
    q_from, r_from = decode_hex_id(from_ids[0])
    q_to, r_to = decode_hex_id(to_ids[0])

    # Compute expected center coordinates via hex_to_lon_lat_SoA
    expected_lon_from, expected_lat_from = hex_proj.hex_to_lon_lat_SoA((q_from, r_from))
    expected_lon_to, expected_lat_to = hex_proj.hex_to_lon_lat_SoA((q_to, r_to))

    # Extract LineString coordinates
    coords = np.array(linestring.coords)

    # Should be a LineString with 2 points (from and to)
    assert coords.shape[0] == 2

    # Check endpoints (allowing for small floating point differences)
    np.testing.assert_allclose(coords[0], [expected_lon_from, expected_lat_from], rtol=1e-5)
    np.testing.assert_allclose(coords[1], [expected_lon_to, expected_lat_to], rtol=1e-5)


# ============================================================================
# Test: edges_geodataframe length with parametrize
# ============================================================================


@pytest.mark.parametrize("n", [0, 1, 10, 100])
def test_edges_geodataframe_length(hex_proj, n):
    """Test that result length matches input length for various sizes."""
    if n == 0:
        # Empty arrays
        from_ids = np.array([], dtype=np.int64)
        to_ids = np.array([], dtype=np.int64)
    else:
        # Generate n edges: from hex (i, 0) to hex (i+1, -1)
        from_q = np.arange(n, dtype=np.int64)
        from_r = np.zeros(n, dtype=np.int64)
        to_q = np.arange(1, n + 1, dtype=np.int64)
        to_r = -np.ones(n, dtype=np.int64)

        from_ids = encode_hex_id(from_q, from_r)
        to_ids = encode_hex_id(to_q, to_r)

    gdf = hex_proj.edges_geodataframe(from_ids, to_ids)

    # Result length should match input length
    assert len(gdf) == n
