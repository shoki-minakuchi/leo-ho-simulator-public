"""
Satellite PRB resource allocation and user download progress tracking.
"""

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

from .constants import (
    TOTAL_PRB, PRB_BW_MHZ, MAX_UE_THPT_MBPS, FINISH_EPS, EVENT_SAT_CHECK,
)
from .event_queue import heap_push


@njit(cache=True)
def flush_sat(s: int, now: float, sat_head, ue_next, ue_remain, ue_thpt, ue_last_calc):
    """Advance remaining download data for all active UEs on satellite s up to time now."""
    curr = sat_head[s]
    while curr != -1:
        dt = now - ue_last_calc[curr]
        if dt > 0.0:
            ue_remain[curr] = max(0.0, ue_remain[curr] - (ue_thpt[curr] / 8.0) * dt)
            ue_last_calc[curr] = now
        curr = ue_next[curr]


@njit(cache=True)
def recalc_sat(s: int, now: float, max_duration: float, sat_head, ue_next, sat_count,
               ue_remain, ue_thpt, ue_last_calc, ue_se, sat_gen,
               hp_t, hp_type, hp_id, hp_gen, hp_sz: int) -> int:
    """
    Recompute equal-PRB per-UE throughput on satellite s and schedule the earliest
    completion check event using a generation counter for lazy invalidation.
    """
    flush_sat(s, now, sat_head, ue_next, ue_remain, ue_thpt, ue_last_calc)
    c_users = sat_count[s]
    if c_users == 0:
        return hp_sz

    prb_alloc = TOTAL_PRB / c_users
    next_t = 1e30
    curr = sat_head[s]
    while curr != -1:
        if ue_remain[curr] <= FINISH_EPS:
            ue_thpt[curr] = 0.0
            next_t = min(next_t, now)
        elif ue_last_calc[curr] > now:
            ue_thpt[curr] = 0.0
            next_t = min(next_t, ue_last_calc[curr])
        else:
            spd = prb_alloc * PRB_BW_MHZ * ue_se[curr]
            ue_thpt[curr] = min(spd, MAX_UE_THPT_MBPS)
            if ue_thpt[curr] > 0.0:
                next_t = min(next_t, now + (ue_remain[curr] * 8.0) / ue_thpt[curr])
        curr = ue_next[curr]

    sat_gen[s] += 1
    if next_t <= max_duration:
        hp_sz = heap_push(hp_t, hp_type, hp_id, hp_gen, hp_sz, next_t, EVENT_SAT_CHECK, s, sat_gen[s])
    return hp_sz
