from .redblobhex_array import Hex

import numpy as np


def hex_SoA_to_AoS(hex_tuple_SoA):
    """Transform a hex tuple of arrays to an array of tuples.
    
    Parameters
    ----------
    hex_tuple_SoA: Hex
        Namedtuple with attributes q, r, s.

    Returns
    -------
    array
        Array of type Hex. Same shape as q, r, s in the original array.
    """
    hex_tuple_AoS = (
        hex_tuple_SoA.q.astype([("q", int)]).astype(Hex)
        + hex_tuple_SoA.r.astype([("r", int)]).astype(Hex)
        + hex_tuple_SoA.s.astype([("s", int)]).astype(Hex)
    )
    return hex_tuple_AoS


def hex_AoS_to_SoA(hex_tuple_AoS):
    """Transform an array of hex tuples to a tuple of arrays.

    Parameters
    ----------
    hex_tuple_AoS: array
        Array with dtype Hex.

    Returns
    -------
    Hex
        Namedtuple with elements q, r, s which are arrays of the
        same shape as hex_tuple_AoS.
    """
    _shape = hex_tuple_AoS.shape

    _q, _r, _s = zip(*hex_tuple_AoS.reshape((-1, )))

    hex_tuple_SoA = Hex(
        q=np.array(_q).reshape(_shape),
        r=np.array(_r).reshape(_shape),
        s=np.array(_s).reshape(_shape),
    )

    return hex_tuple_SoA


def hex_AoS_to_string(hex_AoS):
    hex_AoS_string = np.array(list(map(str, list(hex_AoS.reshape(-1, ))))).reshape(hex_AoS.shape)
    return hex_AoS_string
