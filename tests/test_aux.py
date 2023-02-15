from hextraj.redblobhex_array import Hex
from hextraj.aux import hex_AoS_to_SoA, hex_SoA_to_AoS


import numpy as np

def test_hex_SoA_AoS_roundtrip():
    hex_SoA = Hex(
        q = np.ones((1, 2, 3)),
        r = 2 * np.ones((1, 2, 3)),
        s = -3 * np.ones((1, 2, 3)),
    )
    hex_AoS = hex_SoA_to_AoS(hex_SoA)
    hex_SoA_2nd = hex_AoS_to_SoA(hex_AoS)
    hex_AoS_2nd = hex_SoA_to_AoS(hex_SoA_2nd)

    np.testing.assert_equal(hex_SoA, hex_SoA_2nd)
    np.testing.assert_equal(hex_AoS, hex_AoS_2nd)