from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .redblobhex_array import INTNaN


INVALID_HEX_ID = np.int64(-1)


def _to_int64(x) -> np.ndarray:
    """Cast x to int64, staying lazy if x is a dask array."""
    if hasattr(x, "astype"):
        return x.astype(np.int64)
    return np.asarray(x, dtype=np.int64)


def _z(n):
    return np.where(n >= 0, 2 * n, -2 * n - 1)


def _z_inv(n):
    return np.where(n % 2 == 0, n // 2, -(n + 1) // 2)


def encode_hex_id(q: ArrayLike, r: ArrayLike) -> np.int64 | NDArray[np.int64]:
    """Encode (q, r) hex coordinates to a single int64 via Cantor pairing.

    Args:
        q: Axial q coordinate(s). int64 scalar, ndarray, or dask array.
        r: Axial r coordinate(s). int64 scalar, ndarray, or dask array.

    Returns:
        int64 scalar or array of hex IDs. Inputs where q or r equal INTNaN
        map to INVALID_HEX_ID.
    """
    q = _to_int64(q)
    r = _to_int64(r)
    invalid = (q == INTNaN) | (r == INTNaN)
    a, b = _z(q), _z(r)
    s = a + b
    result = np.where(invalid, INVALID_HEX_ID, s * (s + 1) // 2 + b)
    # numpy scalar path: ndim==0 only occurs for numpy arrays, not dask
    if hasattr(result, "ndim") and result.ndim == 0:
        return np.int64(result)
    return result.astype(np.int64)


def decode_hex_id(
    hex_id: ArrayLike,
) -> tuple[np.int64, np.int64] | tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Decode a Cantor int64 hex ID back to (q, r) axial coordinates.

    Args:
        hex_id: int64 scalar, ndarray, or dask array of hex IDs.

    Returns:
        Tuple (q, r) of same type as input. INVALID_HEX_ID maps to (INTNaN, INTNaN).
    """
    hex_id = _to_int64(hex_id)
    # scalar path: ndim==0 only occurs for numpy scalars, not dask arrays
    scalar = hex_id.ndim == 0

    invalid = hex_id == INVALID_HEX_ID

    safe = np.where(invalid, np.int64(0), hex_id)
    w = np.floor((np.sqrt(8 * safe.astype(float) + 1) - 1) / 2).astype(np.int64)
    t = w * (w + 1) // 2
    b = hex_id - t
    a = w - b

    q = _z_inv(a).astype(np.int64)
    r = _z_inv(b).astype(np.int64)

    if scalar:
        return (np.int64(INTNaN), np.int64(INTNaN)) if invalid else (np.int64(q), np.int64(r))

    # Use np.where instead of item assignment so dask arrays stay lazy
    q = np.where(invalid, INTNaN, q)
    r = np.where(invalid, INTNaN, r)
    return q, r
