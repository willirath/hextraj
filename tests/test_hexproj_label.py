import numpy as np
import pytest

from hextraj.hexproj import HexProj
from hextraj.hex_id import encode_hex_id, decode_hex_id, INVALID_HEX_ID
from hextraj.redblobhex_array import INTNaN


def test_label_returns_int64_array():
    """Call hp.label(lon, lat) with numpy arrays, check dtype is int64 and result is ndarray."""
    hp = HexProj(hex_size_meters=500_000)
    lon = np.array([0.0, 10.0, 20.0])
    lat = np.array([0.0, 10.0, 20.0])
    result = hp.label(lon, lat)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.int64


def test_label_scalar():
    """Call with scalar lon/lat, check result is np.int64."""
    hp = HexProj(hex_size_meters=500_000)
    lon = 0.0
    lat = 0.0
    result = hp.label(lon, lat)
    assert isinstance(result, np.int64)


def test_label_roundtrip():
    """Label then decode, check decoded (q, r) match what lon_lat_to_hex_SoA gives."""
    hp = HexProj(hex_size_meters=500_000)
    lon = np.array([0.0, 10.0, 20.0])
    lat = np.array([0.0, 10.0, 20.0])

    # Get labeled hex IDs
    hex_ids = hp.label(lon, lat)

    # Get hex coords from lon_lat_to_hex_SoA
    hex_soa = hp.lon_lat_to_hex_SoA(lon=lon, lat=lat)
    expected_hex_ids = encode_hex_id(hex_soa.q, hex_soa.r)

    # They should match
    np.testing.assert_array_equal(hex_ids, expected_hex_ids)


def test_label_same_position_same_id():
    """Two identical positions get the same hex ID."""
    hp = HexProj(hex_size_meters=500_000)
    lon = np.array([5.0, 5.0, 10.0])
    lat = np.array([5.0, 5.0, 15.0])

    result = hp.label(lon, lat)

    # First two should be identical
    assert result[0] == result[1]
    # Third should be different
    assert result[0] != result[2]


def test_label_preserves_shape():
    """2D input arrays, result has same shape."""
    hp = HexProj(hex_size_meters=500_000)
    lon = np.array([[0.0, 10.0], [20.0, 30.0]])
    lat = np.array([[0.0, 10.0], [20.0, 30.0]])

    result = hp.label(lon, lat)

    assert result.shape == lon.shape
    assert result.shape == (2, 2)


def test_label_invalid_produces_invalid_hex_id():
    """Pass NaN lon/lat (which pyproj maps to INTNaN coords), result should be INVALID_HEX_ID."""
    hp = HexProj(hex_size_meters=500_000)
    lon = np.array([0.0, float("nan"), 20.0])
    lat = np.array([0.0, float("nan"), 20.0])

    result = hp.label(lon, lat)

    assert result.dtype == np.int64
    assert result[1] == INVALID_HEX_ID
    # Valid positions should not be INVALID_HEX_ID
    assert result[0] != INVALID_HEX_ID
    assert result[2] != INVALID_HEX_ID


