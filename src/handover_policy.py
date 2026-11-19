"""
Baseline handover decision policies and binary-search next-event root finder.
"""

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

from .constants import (
    MIN_ELEV, SAFE_ELEV, TOTAL_PRB, SCAN_DT, TIME_TOL,
    M_BHO, M_CHO, M_LOAD_MIN, M_LOAD_STD,
)
from .channel import get_se
from .orbit import compute_geo, grid_index


@njit(cache=True)
def req_elev_of(method_id: int) -> float:
    """Return the minimum operational elevation angle [deg] for a policy."""
    return MIN_ELEV if method_id == M_LOAD_MIN else SAFE_ELEV


@njit(cache=True)
def ho_margin_of(method_id: int) -> float:
    """Return the fixed hysteresis margin for a baseline policy."""
    if method_id == M_CHO:
        return 3.0
    if method_id == M_LOAD_MIN or method_id == M_LOAD_STD:
        return 0.05
    return 0.0


@njit(cache=True)
def ho_metric(method_id: int, sinr: float, count_after: int) -> float:
    """
    Evaluate handover candidate score:
    - M_LOAD_MIN / M_LOAD_STD: negative normalized user load
    - M_BHO / M_CHO: received SINR [dB]
    """
    count_after = max(1, count_after)
    if method_id == M_LOAD_MIN or method_id == M_LOAD_STD:
        return -(count_after / TOTAL_PRB)
    return sinr


@njit(cache=True)
def is_load_aware(method_id: int) -> bool:
    """Return True if the policy consumes periodic broadcast satellite load info."""
    return method_id in (M_LOAD_MIN, M_LOAD_STD)


@njit(cache=True)
def trigger(u: int, t: float, method_id: int, srv: int, req_elev: float, margin: float,
            cur_se: float, thr_elev: float, raan0, phase0, vis_idx, vis_cnt, ue_pos,
            sat_count_cached, ue_ho_block, elevs, sinrs) -> bool:
    """
    Check whether a handover trigger or discrete physical state transition
    (elevation threshold crossing or MCS step change) occurs at time t.
    """
    n = compute_geo(raan0, phase0, vis_idx, vis_cnt, ue_pos, t, srv, elevs, sinrs)
    if elevs[srv] < thr_elev:
        return True
    if get_se(sinrs[srv]) != cur_se:
        return True

    critical = elevs[srv] < req_elev
    if (not critical) and t < ue_ho_block[u]:
        return False

    old_met = ho_metric(method_id, sinrs[srv], sat_count_cached[srv])
    g = grid_index(t, vis_cnt.shape[0])

    for k in range(n):
        i = vis_idx[g, k]
        if i == srv or elevs[i] < req_elev:
            continue
        if critical:
            return True
        if ho_metric(method_id, sinrs[i], sat_count_cached[i] + 1) > old_met + margin:
            return True
    return False


@njit(cache=True)
def scan_window(u: int, now: float, max_duration: float, method_id: int, srv: int,
                req_elev: float, margin: float, cur_se: float, thr_elev: float,
                raan0, phase0, vis_idx, vis_cnt, ue_pos, sat_count_cached,
                ue_ho_block, elevs, sinrs) -> float:
    """
    Scan forward in time and locate the next asynchronous trigger timestamp
    via coarse stepping followed by binary-search refinement.
    """
    a = now
    k = 1
    while True:
        b = min(now + k * SCAN_DT, max_duration)
        hit = trigger(
            u, b, method_id, srv, req_elev, margin, cur_se, thr_elev,
            raan0, phase0, vis_idx, vis_cnt, ue_pos, sat_count_cached,
            ue_ho_block, elevs, sinrs,
        )
        if hit:
            lo, hi = a, b
            while hi - lo > TIME_TOL:
                mid = 0.5 * (lo + hi)
                if trigger(
                    u, mid, method_id, srv, req_elev, margin, cur_se, thr_elev,
                    raan0, phase0, vis_idx, vis_cnt, ue_pos, sat_count_cached,
                    ue_ho_block, elevs, sinrs,
                ):
                    hi = mid
                else:
                    lo = mid
            return max(hi, now + TIME_TOL)
        if b >= max_duration:
            return -1.0
        a = b
        k += 1
