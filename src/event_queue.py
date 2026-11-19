"""
Discrete-event priority queue (array-based min-heap) and O(1) doubly linked
list structures for managing user-to-satellite associations.
"""

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


@njit(cache=True)
def ev_before(t1: float, ty1: int, id1: int, t2: float, ty2: int, id2: int) -> bool:
    """Compare two events lexicographically by (timestamp, event_type, entity_id)."""
    if t1 != t2:
        return t1 < t2
    if ty1 != ty2:
        return ty1 < ty2
    return id1 < id2


@njit(cache=True)
def heap_push(hp_t, hp_type, hp_id, hp_gen, hp_sz: int,
              t: float, ev_type: int, ev_id: int, gen: int) -> int:
    """Push an event onto the array-based binary min-heap and return the new size."""
    idx = hp_sz
    while idx > 0:
        p = (idx - 1) // 2
        if not ev_before(t, ev_type, ev_id, hp_t[p], hp_type[p], hp_id[p]):
            break
        hp_t[idx] = hp_t[p]
        hp_type[idx] = hp_type[p]
        hp_id[idx] = hp_id[p]
        hp_gen[idx] = hp_gen[p]
        idx = p
    hp_t[idx] = t
    hp_type[idx] = ev_type
    hp_id[idx] = ev_id
    hp_gen[idx] = gen
    return hp_sz + 1


@njit(cache=True)
def heap_pop(hp_t, hp_type, hp_id, hp_gen, hp_sz: int):
    """Pop the earliest event from the min-heap and return (t, type, id, gen, new_size)."""
    rt, rtype, rid, rgen = hp_t[0], hp_type[0], hp_id[0], hp_gen[0]
    hp_sz -= 1
    if hp_sz > 0:
        t, ev_type, ev_id, gen = hp_t[hp_sz], hp_type[hp_sz], hp_id[hp_sz], hp_gen[hp_sz]
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= hp_sz:
                break
            right = left + 1
            small = left
            if right < hp_sz and ev_before(
                hp_t[right], hp_type[right], hp_id[right],
                hp_t[left], hp_type[left], hp_id[left],
            ):
                small = right
            if not ev_before(hp_t[small], hp_type[small], hp_id[small], t, ev_type, ev_id):
                break
            hp_t[idx] = hp_t[small]
            hp_type[idx] = hp_type[small]
            hp_id[idx] = hp_id[small]
            hp_gen[idx] = hp_gen[small]
            idx = small
        hp_t[idx] = t
        hp_type[idx] = ev_type
        hp_id[idx] = ev_id
        hp_gen[idx] = gen
    return rt, rtype, rid, rgen, hp_sz


@njit(cache=True)
def link_ue(u: int, s: int, ue_sat, sat_head, ue_prev, ue_next, sat_count):
    """Attach user u to satellite s in O(1) using an intrusive doubly linked list."""
    ue_sat[u] = s
    old_head = sat_head[s]
    ue_next[u] = old_head
    ue_prev[u] = -1
    if old_head != -1:
        ue_prev[old_head] = u
    sat_head[s] = u
    sat_count[s] += 1


@njit(cache=True)
def unlink_ue(u: int, s: int, ue_sat, sat_head, ue_prev, ue_next, sat_count):
    """Detach user u from satellite s in O(1)."""
    p, n = ue_prev[u], ue_next[u]
    if p != -1:
        ue_next[p] = n
    else:
        sat_head[s] = n
    if n != -1:
        ue_prev[n] = p
    ue_prev[u] = -1
    ue_next[u] = -1
    sat_count[s] -= 1
    ue_sat[u] = -1
