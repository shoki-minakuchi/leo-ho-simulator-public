"""
Spatial traffic and Poisson user arrival generator.
"""

import numpy as np

from .constants import LAMBDA_ARRIVAL, AREAS
from .orbit import lla_to_ecef


def generate_arrivals(scale: float, max_duration: float, seed: int):
    """
    Generate Poisson UE arrival timestamps, ECEF positions, and target-area flags
    weighted across ground regions.
    """
    rng = np.random.default_rng(seed)
    sys_lambda = (2800 * scale) * LAMBDA_ARRIVAL
    weights = [a["weight"] for a in AREAS]
    arr_t, arr_pos, arr_tgt = [], [], []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / sys_lambda)
        if t > max_duration:
            break
        info = AREAS[rng.choice(len(AREAS), p=weights)]
        lat = rng.normal(info["pos"][0], 0.3)
        lon = rng.normal(info["pos"][1], 0.3)
        arr_t.append(t)
        arr_pos.append(lla_to_ecef(lat, lon))
        arr_tgt.append(info["is_target"])
    return (
        np.array(arr_t, dtype=np.float64),
        np.array(arr_pos, dtype=np.float64),
        np.array(arr_tgt, dtype=np.bool_),
    )
