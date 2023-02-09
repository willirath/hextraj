from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray

class Hex(NamedTuple):
    q: NDArray  # all of the same shape
    r: NDArray  # all of the same shape
    s: NDArray  # all of the same shape
    name: str = "Hex"

    def _check_valid(self):
        """Apply all validity checks."""
        return self._check_valid_coords() & self._check_valid_shapes()

    def _check_valid_coords(self):
        """Check if sum of coords is zero."""
        return np.all(np.round(self.q + self.r + self.s) == 0)

    def _check_valid_shapes(self):
        """Check if all coords have the same shape."""
        return (
            (np.array(self.q).shape == np.array(self.r).shape)
            & (np.array(self.q).shape == np.array(self.s).shape)
        )

    def __eq__(self, other):
        return (
            np.array_equal(self.q, other.q)
            & np.array_equal(self.r, other.r)
            & np.array_equal(self.s, other.s)
        )

class Point(NamedTuple):
    x: float
    y: float
    name: str = "Point"


class Orientation(NamedTuple):
    f0: float
    f1: float
    f2: float
    f3: float
    b0: float
    b1: float
    b2: float
    b3: float
    start_angle: float
    name: str = "Orientation"


orientation_pointy = Orientation(
    np.sqrt(3.0),
    np.sqrt(3.0) / 2.0,
    0.0,
    3.0 / 2.0,
    np.sqrt(3.0) / 3.0,
    -1.0 / 3.0,
    0.0,
    2.0 / 3.0,
    0.5,
)

orientation_flat = Orientation(
    3.0 / 2.0,
    0.0,
    np.sqrt(3.0) / 2.0,
    np.sqrt(3.0),
    2.0 / 3.0,
    0.0,
    -1.0 / 3.0,
    np.sqrt(3.0) / 3.0,
    0.0,
)
 

class Layout(NamedTuple):
    orientation: Orientation
    size: Point
    origin: Point
    name: str = "Layout"


def hex_corner_offset(layout, corner):
    M = layout.orientation
    size = layout.size
    angle = 2.0 * np.pi * (M.start_angle - corner) / 6.0
    return Point(size.x * np.cos(angle), size.y * np.sin(angle))


def hex_round(hex: Hex):
    qi = np.int_(np.round(hex.q))
    ri = np.int_(np.round(hex.r))
    si = np.int_(np.round(hex.s))
    q_diff = abs(qi - hex.q)
    r_diff = abs(ri - hex.r)
    s_diff = abs(si - hex.s)
    qi, ri, si = (
        np.where(
            (q_diff > r_diff) & (q_diff > s_diff),
            -ri - si,
            qi,
        ), 
        np.where(
            ~((q_diff > r_diff) & (q_diff > s_diff)) & (r_diff > s_diff),
            -qi - si,
            ri,
        ),
        np.where(
            ~((q_diff > r_diff) & (q_diff > s_diff)) & ~(r_diff > s_diff),
            -qi - ri,
            si,
        )
    )
    qi, ri, si = qi[()], ri[()], si[()]
    return Hex(qi, ri, si)


def hex_to_pixel(layout, h):
    M = layout.orientation
    size = layout.size
    origin = layout.origin
    x = (M.f0 * h.q + M.f1 * h.r) * size.x
    y = (M.f2 * h.q + M.f3 * h.r) * size.y
    return Point(x + origin.x, y + origin.y)


def pixel_to_hex(layout, p):
    M = layout.orientation
    size = layout.size
    origin = layout.origin
    pt = Point((p.x - origin.x) / size.x, (p.y - origin.y) / size.y)
    q = M.b0 * pt.x + M.b1 * pt.y
    r = M.b2 * pt.x + M.b3 * pt.y
    return Hex(q, r, -q - r)