"""
RF channel propagation and MCS spectral efficiency lookup functions.
"""

import math

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

from .constants import LINK_CONST_DB, SINR_TH, EFF


@njit(cache=True)
def sinr_from_dist(dist: float) -> float:
    """Compute received SINR [dB] from slant range distance [m] under FSPL."""
    return LINK_CONST_DB - 20.0 * math.log10(dist / 1000.0)


@njit(cache=True)
def get_se(sinr: float) -> float:
    """Map received SINR [dB] to spectral efficiency [bps/Hz] using the MCS table."""
    if sinr < SINR_TH[0]:
        return 1e-6
    for i in range(len(SINR_TH)):
        if sinr < SINR_TH[i]:
            return EFF[i - 1]
    return EFF[-1]
