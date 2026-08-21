"""Pin the numpy hex math to the vendored scalar reference.

``redblobhex.py`` is generated code from redblobgames.com and is never
imported by hextraj. ``redblobhex_array.py`` is a hand-written numpy port
of it, and every module uses that port instead. These tests keep the two
in agreement, so a change to the port that drifts from the reference
fails here.

Each test feeds one vectorised call and the equivalent loop of scalar
calls the same inputs, which checks the port and its vectorisation
together.
"""

import numpy as np
import pytest

from hextraj import redblobhex as scalar
from hextraj import redblobhex_array as arr


ORIENTATIONS = [
    ("pointy", scalar.layout_pointy, arr.orientation_pointy),
    ("flat", scalar.layout_flat, arr.orientation_flat),
]

# Axial coordinates covering both signs, an origin hex, and a long jump.
QR = [(0, 0), (1, 0), (0, 1), (-1, 2), (3, -2), (5, 5), (-4, -4), (12, -7)]

# Fractional hexes for rounding. The last three sit on a half-way tie,
# where Python's round() and numpy's round() must agree on round-half-even.
QR_FRACTIONAL = [
    (0.2, -0.1),
    (1.4, 0.3),
    (-2.6, 1.1),
    (3.49, -1.51),
    (0.5, 0.25),
    (1.5, -0.5),
    (2.5, 0.5),
]

SIZE = (50_000.0, 50_000.0)
ORIGIN = (0.0, 0.0)


def _layouts(scalar_orientation, array_orientation):
    """Build the matching scalar and array layout for one orientation."""
    return (
        scalar.Layout(scalar_orientation, scalar.Point(*SIZE), scalar.Point(*ORIGIN)),
        arr.Layout(array_orientation, arr.Point(*SIZE), arr.Point(*ORIGIN)),
    )


@pytest.mark.parametrize("name,scalar_orientation,array_orientation", ORIENTATIONS)
def test_orientation_constants_match(name, scalar_orientation, array_orientation):
    fields = ["f0", "f1", "f2", "f3", "b0", "b1", "b2", "b3", "start_angle"]
    for field in fields:
        assert getattr(array_orientation, field) == pytest.approx(
            getattr(scalar_orientation, field)
        ), f"{name}.{field} drifted from the reference"


@pytest.mark.parametrize("name,scalar_orientation,array_orientation", ORIENTATIONS)
def test_hex_to_pixel_matches_scalar(name, scalar_orientation, array_orientation):
    scalar_layout, array_layout = _layouts(scalar_orientation, array_orientation)
    q = np.array([qq for qq, _ in QR], dtype=float)
    r = np.array([rr for _, rr in QR], dtype=float)

    got = arr.hex_to_pixel(array_layout, arr.Hex(q, r, -q - r))
    expected = [
        scalar.hex_to_pixel(scalar_layout, scalar.Hex(float(qq), float(rr), float(-qq - rr)))
        for qq, rr in QR
    ]

    np.testing.assert_allclose(got.x, [p.x for p in expected])
    np.testing.assert_allclose(got.y, [p.y for p in expected])


@pytest.mark.parametrize("name,scalar_orientation,array_orientation", ORIENTATIONS)
def test_pixel_to_hex_matches_scalar(name, scalar_orientation, array_orientation):
    scalar_layout, array_layout = _layouts(scalar_orientation, array_orientation)
    # Offset off the hex centres so the fractional coordinates are non-trivial.
    xs = np.array([qq * 13_000.0 + 700.0 for qq, _ in QR])
    ys = np.array([rr * 11_000.0 - 400.0 for _, rr in QR])

    got = arr.pixel_to_hex(array_layout, arr.Point(xs, ys))
    expected = [
        scalar.pixel_to_hex(scalar_layout, scalar.Point(float(x), float(y)))
        for x, y in zip(xs, ys)
    ]

    np.testing.assert_allclose(got.q, [h.q for h in expected])
    np.testing.assert_allclose(got.r, [h.r for h in expected])
    np.testing.assert_allclose(got.s, [h.s for h in expected])


@pytest.mark.parametrize("name,scalar_orientation,array_orientation", ORIENTATIONS)
@pytest.mark.parametrize("corner", range(7))
def test_hex_corner_offset_matches_scalar(
    name, scalar_orientation, array_orientation, corner
):
    scalar_layout, array_layout = _layouts(scalar_orientation, array_orientation)

    got = arr.hex_corner_offset(array_layout, corner)
    expected = scalar.hex_corner_offset(scalar_layout, corner)

    assert got.x == pytest.approx(expected.x)
    assert got.y == pytest.approx(expected.y)


def test_hex_round_matches_scalar():
    q = np.array([qq for qq, _ in QR_FRACTIONAL], dtype=float)
    r = np.array([rr for _, rr in QR_FRACTIONAL], dtype=float)

    got = arr.hex_round(arr.Hex(q, r, -q - r))
    expected = [
        scalar.hex_round(scalar.Hex(float(qq), float(rr), float(-qq - rr)))
        for qq, rr in QR_FRACTIONAL
    ]

    np.testing.assert_array_equal(got.q, [h.q for h in expected])
    np.testing.assert_array_equal(got.r, [h.r for h in expected])
    np.testing.assert_array_equal(got.s, [h.s for h in expected])


def test_hex_round_result_stays_on_the_cube_plane():
    q = np.array([qq for qq, _ in QR_FRACTIONAL], dtype=float)
    r = np.array([rr for _, rr in QR_FRACTIONAL], dtype=float)

    got = arr.hex_round(arr.Hex(q, r, -q - r))

    np.testing.assert_array_equal(got.q + got.r + got.s, np.zeros_like(got.q))


@pytest.mark.parametrize("name,scalar_orientation,array_orientation", ORIENTATIONS)
def test_pixel_to_hex_round_trips_through_hex_to_pixel(
    name, scalar_orientation, array_orientation
):
    _, array_layout = _layouts(scalar_orientation, array_orientation)
    q = np.array([qq for qq, _ in QR], dtype=float)
    r = np.array([rr for _, rr in QR], dtype=float)

    pixels = arr.hex_to_pixel(array_layout, arr.Hex(q, r, -q - r))
    back = arr.hex_round(arr.pixel_to_hex(array_layout, pixels))

    np.testing.assert_array_equal(back.q, q.astype(int))
    np.testing.assert_array_equal(back.r, r.astype(int))
