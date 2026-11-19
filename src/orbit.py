"""
Orbital mechanics, ECEF geometry, and visibility pruning grid precomputation.
"""

import math
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

from .constants import (
    R_E, R_ORB, MEAN_MOTION, COS_INCL, SIN_INCL, EARTH_OMEGA,
    NUM_PLANES, SATS_PER_PLANE, GRID_DT, CENTER_LATLON, VIS_ELEV_CENTER,
)
from .channel import sinr_from_dist


def lla_to_ecef(lat: float, lon: float, alt: float = 0.0) -> list:
    """Convert geodetic coordinates (lat, lon, alt [m]) to ECEF [m]."""
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    r = R_E + alt
    return [
        r * np.cos(lat_r) * np.cos(lon_r),
        r * np.cos(lat_r) * np.sin(lon_r),
        r * np.sin(lat_r),
    ]


@njit(cache=True)
def sat_elev_dist(raan0: float, phase0: float, t: float,
                  ux: float, uy: float, uz: float, ue_norm: float):
    """Compute elevation angle [deg] and slant range [m] for one satellite at time t."""
    m_anom = phase0 + MEAN_MOTION * t
    ra = raan0 - EARTH_OMEGA * t
    xo, yo = R_ORB * math.cos(m_anom), R_ORB * math.sin(m_anom)
    cr, sr = math.cos(ra), math.sin(ra)
    x = xo * cr - yo * COS_INCL * sr
    y = xo * sr + yo * COS_INCL * cr
    z = yo * SIN_INCL
    vx, vy, vz = x - ux, y - uy, z - uz
    dist = math.sqrt(vx * vx + vy * vy + vz * vz)
    cos_th = (vx * ux + vy * uy + vz * uz) / (dist * ue_norm)
    cos_th = max(-1.0, min(1.0, cos_th))
    return 90.0 - math.degrees(math.acos(cos_th)), dist


@njit(cache=True)
def grid_index(t: float, n_grid: int) -> int:
    """Map continuous simulation time t to a discrete visibility grid index."""
    return int(min(max(t, 0.0) / GRID_DT, n_grid - 1))


@njit(cache=True)
def compute_geo(raan0, phase0, vis_idx, vis_cnt, ue_pos, t, extra_s, elevs, sinrs):
    """
    Evaluate elevation and SINR for pruned visible satellites at time t.
    Returns the number of candidate satellites checked from the visibility grid.
    """
    g = grid_index(t, vis_cnt.shape[0])
    ux, uy, uz = ue_pos[0], ue_pos[1], ue_pos[2]
    ue_norm = math.sqrt(ux * ux + uy * uy + uz * uz)
    elevs[:] = -90.0
    sinrs[:] = -1e9
    n = vis_cnt[g]
    for k in range(n):
        s = vis_idx[g, k]
        e, d = sat_elev_dist(raan0[s], phase0[s], t, ux, uy, uz, ue_norm)
        elevs[s], sinrs[s] = e, sinr_from_dist(d)
    if extra_s >= 0 and elevs[extra_s] <= -90.0:
        e, d = sat_elev_dist(raan0[extra_s], phase0[extra_s], t, ux, uy, uz, ue_norm)
        elevs[extra_s], sinrs[extra_s] = e, sinr_from_dist(d)
    return n


def precompute_constellation(max_time: float, verbose: bool = True):
    """
    Precompute Walker-Delta satellite orbits and a time-indexed visibility pruning grid
    around the target region center.
    """
    if verbose:
        print("Building constellation & visibility pruning index...", end="", flush=True)
    planes = np.repeat(np.arange(NUM_PLANES), SATS_PER_PLANE)
    sats = np.tile(np.arange(SATS_PER_PLANE), NUM_PLANES)
    raan0_all = np.radians(planes * (360.0 / NUM_PLANES))
    phase0_all = np.radians(sats * (360.0 / SATS_PER_PLANE)) + (planes % 2) * np.radians(180.0 / SATS_PER_PLANE)

    t_grid = np.arange(0.0, max_time + 3 * GRID_DT, GRID_DT)
    c = np.array(lla_to_ecef(*CENTER_LATLON))
    m_anom = phase0_all[:, None] + MEAN_MOTION * t_grid[None, :]
    ra = raan0_all[:, None] - EARTH_OMEGA * t_grid[None, :]
    xo, yo = R_ORB * np.cos(m_anom), R_ORB * np.sin(m_anom)
    x = xo * np.cos(ra) - yo * COS_INCL * np.sin(ra)
    y = xo * np.sin(ra) + yo * COS_INCL * np.cos(ra)
    z = yo * SIN_INCL
    vx, vy, vz = x - c[0], y - c[1], z - c[2]
    d = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    cos_th = (vx * c[0] + vy * c[1] + vz * c[2]) / (d * np.linalg.norm(c))
    elev_c = 90.0 - np.degrees(np.arccos(np.clip(cos_th, -1.0, 1.0)))

    valid = np.where(elev_c.max(axis=1) >= VIS_ELEV_CENTER)[0]
    raan0 = raan0_all[valid].astype(np.float64)
    phase0 = phase0_all[valid].astype(np.float64)
    vis = (elev_c[valid] >= VIS_ELEV_CENTER).T
    vis[:-1] |= vis[1:]
    vis_cnt = vis.sum(axis=1).astype(np.int32)
    vis_idx = np.full((vis.shape[0], int(vis_cnt.max())), -1, dtype=np.int32)
    for t_idx in range(vis.shape[0]):
        idx = np.where(vis[t_idx])[0]
        vis_idx[t_idx, :len(idx)] = idx

    if verbose:
        print(f" done ({len(valid)} active sats, avg {vis_cnt.mean():.1f} visible/step)")
    return raan0, phase0, vis_idx, vis_cnt
