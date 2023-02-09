from hextraj.redblobhex_array import Hex, Layout, orientation_flat, Point, hex_corner_offset

import numpy as np
import xarray as xr
from dask import array as darr
import pandas as pd


def test_validity_check_scalars():
    assert Hex(1, 2, -3)._check_valid()
    assert not Hex(1, 2, 3)._check_valid()


def test_validity_check_numpy_arrays():
    assert Hex(np.array([1, ]), np.array([2, ]), np.array([- 3, ]))._check_valid()
    assert not Hex(np.array([1, ]), np.array([2, ]), np.array([3, ]))._check_valid()


def test_validity_check_dask_arrays():
    assert Hex(darr.array([1, ]), darr.array([2, ]), darr.array([- 3, ]))._check_valid()
    assert not Hex(darr.array([1, ]), darr.array([2, ]), darr.array([3, ]))._check_valid()


def test_validity_check_pandas_series():
    q = pd.Series([1, 1, 1], index=[1, 2, 3])
    r = pd.Series([2, 2, 2], index=[1, 2, 3])
    s = pd.Series([-3, -3, -3], index=[1, 2, 3])
    assert Hex(q, r, s)._check_valid()
    assert not Hex(q, r, -s)._check_valid()


def test_validity_check_xr_dataarrays():
    # scalars
    q, r, s = xr.DataArray(1), xr.DataArray(2), xr.DataArray(-3)
    assert Hex(q, r, s)._check_valid()
    assert not Hex(q, r, -s)._check_valid()

    # arrays
    q, r, s = xr.DataArray([1, ]), xr.DataArray([2, ]), xr.DataArray([-3, ])
    assert Hex(q, r, s)._check_valid()
    assert not Hex(q, r, -s)._check_valid()

    # dask arrays
    q = xr.DataArray(1 * darr.ones((1, 2)), dims=("x", "y"))
    r = xr.DataArray(2 * darr.ones((1, 2)), dims=("x", "y"))
    s = xr.DataArray(-3 * darr.ones((1, 2)), dims=("x", "y"))
    assert Hex(q, r, s)._check_valid().compute()
    assert not Hex(q, r, -s)._check_valid().compute()


def test_hex_corner_offset_flat():
    """Check that in a flat layout, corners 0 & 3 are on x-axis."""
    layout = Layout(
        orientation=orientation_flat,
        size=Point(1, 1),
        origin=Point(0, 0),
    )
    corner_0_offset = hex_corner_offset(layout, corner=0)
    np.testing.assert_array_almost_equal(corner_0_offset.x, 1.0)
    np.testing.assert_array_almost_equal(corner_0_offset.y, 0.0)

    corner_3_offset = hex_corner_offset(layout, corner=3)
    np.testing.assert_array_almost_equal(corner_3_offset.x, -1.0)
    np.testing.assert_array_almost_equal(corner_3_offset.y, 0.0)

    corners = hex_corner_offset(layout, corner=np.array([0, 1, 2, 3, 4, 5, 6]))
    np.testing.assert_array_almost_equal(
        corners.x,
        [1, 0.5, -0.5, -1, -0.5, 0.5, 1],
    )
    np.testing.assert_array_almost_equal(
        corners.y,
        [0, -np.sin(np.deg2rad(60)), -np.sin(np.deg2rad(60)), 0, np.sin(np.deg2rad(60)), np.sin(np.deg2rad(60)), 0],
    )