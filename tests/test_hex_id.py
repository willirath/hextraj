import numpy as np
import pytest

from hextraj.hex_id import INVALID_HEX_ID, decode_hex_id, encode_hex_id
from hextraj.redblobhex_array import INTNaN


# (q, r, expected_hex_id) — hand-computed from Cantor(z(q), z(r))
KNOWN_VALUES = [
    (0,  0,  0),   # z(0)=0,  z(0)=0  -> cantor(0,0)=0
    (1,  0,  3),   # z(1)=2,  z(0)=0  -> cantor(2,0)=3
    (0,  1,  5),   # z(0)=0,  z(1)=2  -> cantor(0,2)=5
    (-1, 0,  1),   # z(-1)=1, z(0)=0  -> cantor(1,0)=1
    (0,  -1, 2),   # z(0)=0,  z(-1)=1 -> cantor(0,1)=2
    (10, 0, 210),  # z(10)=20, z(0)=0 -> cantor(20,0)=20*21/2+0=210
    (0, 10, 230),  # z(0)=0, z(10)=20 -> cantor(0,20)=20*21/2+20=230
    (-10, 0, 190), # z(-10)=19, z(0)=0 -> cantor(19,0)=19*20/2+0=190
    (0, -10, 209), # z(0)=0, z(-10)=19 -> cantor(0,19)=19*20/2+19=209
    (10, 10, 840), # z(10)=20, z(10)=20 -> cantor(20,20)=40*41/2+20=840
    (-10, -10, 760), # z(-10)=19, z(-10)=19 -> cantor(19,19)=38*39/2+19=760
    (100, 50, 45250), # z(100)=200, z(50)=100 -> cantor(200,100)=300*301/2+100=45250
    (-37, 42, 12487), # z(-37)=73, z(42)=84 -> cantor(73,84)=157*158/2+84=12487
]


@pytest.mark.parametrize("q,r,expected", KNOWN_VALUES)
def test_encode_known_values(q, r, expected):
    assert encode_hex_id(q, r) == np.int64(expected)


@pytest.mark.parametrize("q,r,_", KNOWN_VALUES)
def test_roundtrip(q, r, _):
    q_out, r_out = decode_hex_id(encode_hex_id(q, r))
    assert q_out == q
    assert r_out == r


@pytest.mark.parametrize("q,r,hex_id", KNOWN_VALUES)
def test_decode_known_values(q, r, hex_id):
    q_out, r_out = decode_hex_id(np.int64(hex_id))
    assert q_out == q
    assert r_out == r


def test_encode_returns_int64_scalar():
    assert encode_hex_id(0, 0).dtype == np.int64


def test_encode_returns_int64_array():
    q = np.array([0, 1, -1])
    r = np.array([0, 0,  0])
    result = encode_hex_id(q, r)
    assert result.dtype == np.int64
    np.testing.assert_array_equal(result, [0, 3, 1])


def test_encode_preserves_shape():
    q = np.arange(6).reshape(2, 3)
    r = np.zeros((2, 3), dtype=int)
    assert encode_hex_id(q, r).shape == (2, 3)


def test_decode_array():
    hex_ids = np.array([0, 3, 5, 1, 2], dtype=np.int64)
    q, r = decode_hex_id(hex_ids)
    np.testing.assert_array_equal(q, [0,  1, 0, -1,  0])
    np.testing.assert_array_equal(r, [0,  0, 1,  0, -1])


def test_roundtrip_array():
    q = np.array([-5, -1, 0, 1, 5], dtype=np.int64)
    r = np.array([-3,  0, 0, 0, 3], dtype=np.int64)
    q_out, r_out = decode_hex_id(encode_hex_id(q, r))
    np.testing.assert_array_equal(q_out, q)
    np.testing.assert_array_equal(r_out, r)


def test_invalid_hex_id_constant():
    assert INVALID_HEX_ID == np.int64(-1)
    assert isinstance(INVALID_HEX_ID, np.int64)


@pytest.mark.parametrize("q,r", [(INTNaN, 0), (0, INTNaN), (INTNaN, INTNaN)])
def test_encode_invalid_input(q, r):
    assert encode_hex_id(q, r) == INVALID_HEX_ID


def test_encode_array_with_invalid():
    q = np.array([0, INTNaN, 1], dtype=np.int64)
    r = np.array([0, 0,      1], dtype=np.int64)
    result = encode_hex_id(q, r)
    assert result[0] != INVALID_HEX_ID
    assert result[1] == INVALID_HEX_ID
    assert result[2] != INVALID_HEX_ID


def test_decode_invalid_hex_id():
    q, r = decode_hex_id(INVALID_HEX_ID)
    assert q == INTNaN
    assert r == INTNaN


def test_decode_array_with_invalid():
    hex_ids = np.array([0, INVALID_HEX_ID, 3], dtype=np.int64)
    q, r = decode_hex_id(hex_ids)
    assert q[0] != INTNaN
    assert q[1] == INTNaN
    assert r[1] == INTNaN


# Tests for numpy SoA (structure of arrays) format
def test_roundtrip_numpy_soa():
    """Verify hex_id encoding/decoding works with plain numpy SoA arrays."""
    q = np.array([0, 1, -1, 10, -10, 100], dtype=np.int64)
    r = np.array([0, 0,  0,  0,   0,  50], dtype=np.int64)

    hex_ids = encode_hex_id(q, r)
    q_out, r_out = decode_hex_id(hex_ids)

    np.testing.assert_array_equal(q_out, q)
    np.testing.assert_array_equal(r_out, r)


def test_roundtrip_numpy_soa_2d():
    """Verify hex_id encoding/decoding preserves 2D shape with numpy arrays."""
    q = np.array([[0, 1, -1], [10, -10, 100]], dtype=np.int64)
    r = np.array([[0, 0, 0], [0, 0, 50]], dtype=np.int64)

    hex_ids = encode_hex_id(q, r)
    q_out, r_out = decode_hex_id(hex_ids)

    np.testing.assert_array_equal(q_out, q)
    np.testing.assert_array_equal(r_out, r)
    assert q_out.shape == (2, 3)
    assert r_out.shape == (2, 3)


# Tests for dask arrays
def test_roundtrip_dask_arrays():
    """Verify hex_id encoding/decoding works with dask arrays."""
    da = pytest.importorskip("dask.array")

    q = da.from_array(np.array([0, 1, -1, 10, -10, 100], dtype=np.int64))
    r = da.from_array(np.array([0, 0,  0,  0,   0,  50], dtype=np.int64))

    q_out, r_out = decode_hex_id(encode_hex_id(q, r))

    np.testing.assert_array_equal(q_out, [0, 1, -1, 10, -10, 100])
    np.testing.assert_array_equal(r_out, [0, 0,  0,  0,   0,  50])


def test_roundtrip_dask_arrays_lazy():
    """Verify hex_id encode/decode preserves lazy dask arrays without materializing."""
    da = pytest.importorskip("dask.array")

    q = da.from_array(np.array([0, 1, -1, 10, -10, 100], dtype=np.int64))
    r = da.from_array(np.array([0, 0,  0,  0,   0,  50], dtype=np.int64))

    # encode should return a dask array
    hex_ids = encode_hex_id(q, r)
    assert isinstance(hex_ids, da.Array), "encode_hex_id should return dask array for dask input"

    # decode should return dask arrays (not materialized numpy arrays)
    q_out, r_out = decode_hex_id(hex_ids)
    assert isinstance(q_out, da.Array), "decode_hex_id should return dask array for q"
    assert isinstance(r_out, da.Array), "decode_hex_id should return dask array for r"

    # only after compute() should we verify values
    np.testing.assert_array_equal(q_out.compute(), [0, 1, -1, 10, -10, 100])
    np.testing.assert_array_equal(r_out.compute(), [0, 0,  0,  0,   0,  50])
