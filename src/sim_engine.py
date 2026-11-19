"""
Core discrete-event-driven simulation engine (@njit accelerated).
"""

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

from .constants import (
    FIXED_FILE_SIZE_MB, WARMUP_TIME, MIN_ELEV, TOTAL_PRB, FINISH_EPS,
    HOF_PENALTY, RLF_PENALTY, HOF_BACKOFF, SIB_INTERVAL, RACH_DELAY_MAX,
    TIME_TO_STAY, HEAP_INIT_PER_USER, HEAP_HEADROOM,
    EVENT_ARRIVAL, EVENT_SAT_CHECK, EVENT_PHYSICS, EVENT_SIB_UPDATE,
)
from .channel import get_se
from .orbit import compute_geo, grid_index
from .event_queue import heap_push, heap_pop, link_ue, unlink_ue
from .resource_manager import flush_sat, recalc_sat
from .handover_policy import (
    req_elev_of, ho_margin_of, ho_metric, is_load_aware, scan_window,
)


@njit(cache=True)
def run_sim(method_id: int, max_duration: float, raan0, phase0, vis_idx, vis_cnt,
            arr_t, arr_pos, arr_is_target, time_to_stay: float = TIME_TO_STAY,
            sib_interval: float = SIB_INTERVAL,
            custom_margin: float = -1.0):
    """
    Execute an event-driven LEO satellite network simulation and return:
    (avg_mbps, edge10_mbps, ho, hof, rlf, drop, block, unfinished, heap_peak, evals, max_ev_per_ue).
    """
    n_users = len(arr_t)
    n_sats = raan0.shape[0]
    n_grid = vis_cnt.shape[0]

    ue_sat = np.full(n_users, -1, dtype=np.int32)
    ue_remain = np.full(n_users, FIXED_FILE_SIZE_MB, dtype=np.float64)
    ue_thpt = np.zeros(n_users, dtype=np.float64)
    ue_last_calc = arr_t.astype(np.float64).copy()
    ue_se = np.zeros(n_users, dtype=np.float64)
    ue_gen_phy = np.zeros(n_users, dtype=np.int32)
    ue_active = np.zeros(n_users, dtype=np.bool_)
    ue_prev = np.full(n_users, -1, dtype=np.int32)
    ue_next = np.full(n_users, -1, dtype=np.int32)
    ue_ho_block = np.zeros(n_users, dtype=np.float64)

    sat_count = np.zeros(n_sats, dtype=np.int32)
    sat_count_cached = np.zeros(n_sats, dtype=np.int32)
    sat_head = np.full(n_sats, -1, dtype=np.int32)
    sat_gen = np.zeros(n_sats, dtype=np.int32)

    stat_thpt = np.zeros(n_users, dtype=np.float64)
    n_thpt = 0
    stat_rlf = stat_hof = stat_ho = stat_drop = stat_block = 0
    n_eval = 0
    ue_evals = np.zeros(n_users, dtype=np.int32)

    elevs = np.zeros(n_sats, dtype=np.float64)
    sinrs = np.zeros(n_sats, dtype=np.float64)
    finish_buf = np.zeros(n_users, dtype=np.int32)

    cap = max(1024, n_users * HEAP_INIT_PER_USER)
    hp_t = np.zeros(cap, dtype=np.float64)
    hp_type = np.zeros(cap, dtype=np.int32)
    hp_id = np.zeros(cap, dtype=np.int32)
    hp_gen = np.zeros(cap, dtype=np.int32)
    hp_sz = 0
    heap_peak = 0

    for i in range(n_users):
        if arr_t[i] <= max_duration:
            hp_sz = heap_push(hp_t, hp_type, hp_id, hp_gen, hp_sz, float(arr_t[i]), EVENT_ARRIVAL, i, 0)

    # Schedule initial periodic broadcast load update
    hp_sz = heap_push(hp_t, hp_type, hp_id, hp_gen, hp_sz, 0.0, EVENT_SIB_UPDATE, 0, 0)

    req_elev = req_elev_of(method_id)
    margin = custom_margin if custom_margin >= 0.0 else ho_margin_of(method_id)
    load_aware = is_load_aware(method_id)

    while hp_sz > 0:
        while hp_sz + HEAP_HEADROOM + n_users > cap:
            cap *= 2
            nt = np.zeros(cap, dtype=np.float64); nt[:hp_sz] = hp_t[:hp_sz]; hp_t = nt
            ny = np.zeros(cap, dtype=np.int32);   ny[:hp_sz] = hp_type[:hp_sz]; hp_type = ny
            ni = np.zeros(cap, dtype=np.int32);   ni[:hp_sz] = hp_id[:hp_sz]; hp_id = ni
            ng = np.zeros(cap, dtype=np.int32);   ng[:hp_sz] = hp_gen[:hp_sz]; hp_gen = ng
        heap_peak = max(heap_peak, hp_sz)

        ev_t, ev_type, ev_id, ev_gen, hp_sz = heap_pop(hp_t, hp_type, hp_id, hp_gen, hp_sz)
        if ev_t > max_duration:
            break

        if ev_type == EVENT_SIB_UPDATE:
            for s in range(n_sats):
                sat_count_cached[s] = sat_count[s]

            next_sib = ev_t + sib_interval
            if next_sib <= max_duration:
                hp_sz = heap_push(hp_t, hp_type, hp_id, hp_gen, hp_sz, next_sib, EVENT_SIB_UPDATE, 0, 0)

            if load_aware:
                for u in range(n_users):
                    if ue_active[u]:
                        ue_gen_phy[u] += 1
                        delay = np.random.uniform(0.0, RACH_DELAY_MAX)
                        hp_sz = heap_push(hp_t, hp_type, hp_id, hp_gen, hp_sz, ev_t + delay, EVENT_PHYSICS, u, ue_gen_phy[u])
            continue

        elif ev_type == EVENT_ARRIVAL:
            u = ev_id
            counted = arr_is_target[u] and arr_t[u] >= WARMUP_TIME
            n_vis = compute_geo(raan0, phase0, vis_idx, vis_cnt, arr_pos[u], ev_t, -1, elevs, sinrs)
            g = grid_index(ev_t, n_grid)
            best_s, best_sinr = -1, -1e9
            for k in range(n_vis):
                i = vis_idx[g, k]
                if elevs[i] >= req_elev and sinrs[i] > best_sinr:
                    best_s, best_sinr = i, sinrs[i]
            if best_s == -1:
                for k in range(n_vis):
                    i = vis_idx[g, k]
                    if elevs[i] >= MIN_ELEV and sinrs[i] > best_sinr:
                        best_s, best_sinr = i, sinrs[i]
            if best_s == -1:
                if counted:
                    stat_block += 1
                continue

            ue_active[u] = True
            ue_last_calc[u] = ev_t
            ue_se[u] = get_se(best_sinr)
            link_ue(u, best_s, ue_sat, sat_head, ue_prev, ue_next, sat_count)
            hp_sz = recalc_sat(
                best_s, ev_t, max_duration, sat_head, ue_next, sat_count,
                ue_remain, ue_thpt, ue_last_calc, ue_se, sat_gen,
                hp_t, hp_type, hp_id, hp_gen, hp_sz,
            )
            ue_gen_phy[u] += 1
            hp_sz = heap_push(hp_t, hp_type, hp_id, hp_gen, hp_sz, ev_t, EVENT_PHYSICS, u, ue_gen_phy[u])

        elif ev_type == EVENT_SAT_CHECK:
            s = ev_id
            if sat_gen[s] != ev_gen:
                continue
            flush_sat(s, ev_t, sat_head, ue_next, ue_remain, ue_thpt, ue_last_calc)
            n_fin = 0
            curr = sat_head[s]
            while curr != -1:
                if ue_remain[curr] <= FINISH_EPS:
                    finish_buf[n_fin] = curr
                    n_fin += 1
                curr = ue_next[curr]
            for k in range(n_fin):
                u = finish_buf[k]
                if arr_is_target[u] and arr_t[u] >= WARMUP_TIME:
                    dur = ev_t - arr_t[u]
                    if dur > 0.0:
                        stat_thpt[n_thpt] = FIXED_FILE_SIZE_MB * 8.0 / dur
                        n_thpt += 1
                unlink_ue(u, s, ue_sat, sat_head, ue_prev, ue_next, sat_count)
                ue_active[u] = False
                ue_thpt[u] = 0.0
                ue_gen_phy[u] += 1
            hp_sz = recalc_sat(
                s, ev_t, max_duration, sat_head, ue_next, sat_count,
                ue_remain, ue_thpt, ue_last_calc, ue_se, sat_gen,
                hp_t, hp_type, hp_id, hp_gen, hp_sz,
            )

        elif ev_type == EVENT_PHYSICS:
            u = ev_id
            if (not ue_active[u]) or ue_gen_phy[u] != ev_gen:
                continue
            n_eval += 1
            ue_evals[u] += 1
            counted = arr_is_target[u] and arr_t[u] >= WARMUP_TIME
            old_s = ue_sat[u]
            n_vis = compute_geo(raan0, phase0, vis_idx, vis_cnt, arr_pos[u], ev_t, old_s, elevs, sinrs)
            g = grid_index(ev_t, n_grid)

            if elevs[old_s] < MIN_ELEV:
                if counted:
                    stat_rlf += 1
                best_new, best_new_sinr = -1, -1e9
                for k in range(n_vis):
                    i = vis_idx[g, k]
                    if i != old_s and elevs[i] >= MIN_ELEV and sinrs[i] > best_new_sinr:
                        best_new, best_new_sinr = i, sinrs[i]
                flush_sat(old_s, ev_t, sat_head, ue_next, ue_remain, ue_thpt, ue_last_calc)
                unlink_ue(u, old_s, ue_sat, sat_head, ue_prev, ue_next, sat_count)
                hp_sz = recalc_sat(
                    old_s, ev_t, max_duration, sat_head, ue_next, sat_count,
                    ue_remain, ue_thpt, ue_last_calc, ue_se, sat_gen,
                    hp_t, hp_type, hp_id, hp_gen, hp_sz,
                )
                if best_new == -1:
                    if counted:
                        stat_drop += 1
                        dl = FIXED_FILE_SIZE_MB - ue_remain[u]
                        dur = ev_t - arr_t[u]
                        if dl > 0.0 and dur > 0.0:
                            stat_thpt[n_thpt] = dl * 8.0 / dur
                            n_thpt += 1
                    ue_active[u] = False
                    ue_thpt[u] = 0.0
                    continue
                link_ue(u, best_new, ue_sat, sat_head, ue_prev, ue_next, sat_count)
                ue_se[u] = get_se(best_new_sinr)
                ue_last_calc[u] = ev_t + RLF_PENALTY
                hp_sz = recalc_sat(
                    best_new, ev_t, max_duration, sat_head, ue_next, sat_count,
                    ue_remain, ue_thpt, ue_last_calc, ue_se, sat_gen,
                    hp_t, hp_type, hp_id, hp_gen, hp_sz,
                )
                old_s = best_new
            else:
                is_critical = elevs[old_s] < req_elev
                old_met = ho_metric(method_id, sinrs[old_s], sat_count_cached[old_s])
                best_s, best_met, best_net = -1, -1e9, -1e9

                for k in range(n_vis):
                    i = vis_idx[g, k]
                    if i == old_s or elevs[i] < req_elev:
                        continue
                    m = ho_metric(method_id, sinrs[i], sat_count_cached[i] + 1)
                    if is_critical:
                        if m > best_met:
                            best_s, best_met = i, m
                    else:
                        net = m - (old_met + margin)
                        if net > 0.0 and m > best_met:
                            best_s, best_met, best_net = i, m, net

                do_ho = best_s != -1 and (is_critical or best_net > 0.0)
                if (not is_critical) and ev_t < ue_ho_block[u]:
                    do_ho = False

                if do_ho and sat_count[best_s] + 1 > TOTAL_PRB:
                    if counted:
                        stat_hof += 1
                    ue_ho_block[u] = ev_t + HOF_BACKOFF
                    flush_sat(old_s, ev_t, sat_head, ue_next, ue_remain, ue_thpt, ue_last_calc)
                    ue_se[u] = get_se(sinrs[old_s])
                    ue_last_calc[u] = ev_t + HOF_PENALTY
                    hp_sz = recalc_sat(
                        old_s, ev_t, max_duration, sat_head, ue_next, sat_count,
                        ue_remain, ue_thpt, ue_last_calc, ue_se, sat_gen,
                        hp_t, hp_type, hp_id, hp_gen, hp_sz,
                    )
                    if is_critical:
                        ue_gen_phy[u] += 1
                        hp_sz = heap_push(hp_t, hp_type, hp_id, hp_gen, hp_sz, ev_t + HOF_PENALTY, EVENT_PHYSICS, u, ue_gen_phy[u])
                        continue
                elif do_ho:
                    if counted:
                        stat_ho += 1
                    flush_sat(old_s, ev_t, sat_head, ue_next, ue_remain, ue_thpt, ue_last_calc)
                    unlink_ue(u, old_s, ue_sat, sat_head, ue_prev, ue_next, sat_count)
                    hp_sz = recalc_sat(
                        old_s, ev_t, max_duration, sat_head, ue_next, sat_count,
                        ue_remain, ue_thpt, ue_last_calc, ue_se, sat_gen,
                        hp_t, hp_type, hp_id, hp_gen, hp_sz,
                    )

                    link_ue(u, best_s, ue_sat, sat_head, ue_prev, ue_next, sat_count)
                    ue_se[u] = get_se(sinrs[best_s])
                    hp_sz = recalc_sat(
                        best_s, ev_t, max_duration, sat_head, ue_next, sat_count,
                        ue_remain, ue_thpt, ue_last_calc, ue_se, sat_gen,
                        hp_t, hp_type, hp_id, hp_gen, hp_sz,
                    )
                    old_s = best_s
                    if not is_critical:
                        ue_ho_block[u] = ev_t + time_to_stay
                else:
                    new_se = get_se(sinrs[old_s])
                    if new_se != ue_se[u]:
                        ue_se[u] = new_se
                        hp_sz = recalc_sat(
                            old_s, ev_t, max_duration, sat_head, ue_next, sat_count,
                            ue_remain, ue_thpt, ue_last_calc, ue_se, sat_gen,
                            hp_t, hp_type, hp_id, hp_gen, hp_sz,
                        )

            srv = ue_sat[u]
            cur_se = ue_se[u]
            thr_elev = req_elev if elevs[srv] >= req_elev else MIN_ELEV

            t_next = scan_window(
                u, ev_t, max_duration, method_id, srv, req_elev, margin, cur_se, thr_elev,
                raan0, phase0, vis_idx, vis_cnt, arr_pos[u], sat_count_cached,
                ue_ho_block, elevs, sinrs,
            )
            if t_next > 0.0:
                ue_gen_phy[u] += 1
                hp_sz = heap_push(hp_t, hp_type, hp_id, hp_gen, hp_sz, t_next, EVENT_PHYSICS, u, ue_gen_phy[u])

    for s in range(n_sats):
        flush_sat(s, max_duration, sat_head, ue_next, ue_remain, ue_thpt, ue_last_calc)
    n_censored = 0
    for u in range(n_users):
        if ue_active[u] and arr_is_target[u] and arr_t[u] >= WARMUP_TIME:
            n_censored += 1
            dur = max_duration - arr_t[u]
            dl = FIXED_FILE_SIZE_MB - ue_remain[u]
            if dl > 0.0 and dur > 0.0:
                stat_thpt[n_thpt] = dl * 8.0 / dur
                n_thpt += 1

    avg, edge = 0.0, 0.0
    if n_thpt > 0:
        v = np.sort(stat_thpt[:n_thpt])
        avg = float(np.mean(v))
        edge = float(v[min(int(n_thpt * 0.1), n_thpt - 1)])
    return avg, edge, stat_ho, stat_hof, stat_rlf, stat_drop, stat_block, n_censored, heap_peak, n_eval, int(ue_evals.max())
