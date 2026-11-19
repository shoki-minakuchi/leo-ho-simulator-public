"""Unit tests for the array-based min-heap and O(1) doubly linked list."""

import unittest
import numpy as np

from src.event_queue import heap_push, heap_pop, link_ue, unlink_ue


class TestEventQueue(unittest.TestCase):
    def test_heap_orders_by_time_type_and_id(self):
        cap = 16
        hp_t = np.zeros(cap, dtype=np.float64)
        hp_type = np.zeros(cap, dtype=np.int32)
        hp_id = np.zeros(cap, dtype=np.int32)
        hp_gen = np.zeros(cap, dtype=np.int32)
        sz = 0

        events = [
            (5.0, 2, 10, 1),
            (1.5, 3, 2, 1),
            (1.5, 1, 5, 1),
            (1.5, 1, 2, 1),
            (0.0, 4, 0, 0),
        ]
        for t, ty, eid, gen in events:
            sz = heap_push(hp_t, hp_type, hp_id, hp_gen, sz, t, ty, eid, gen)

        self.assertEqual(sz, len(events))

        popped = []
        while sz > 0:
            t, ty, eid, gen, sz = heap_pop(hp_t, hp_type, hp_id, hp_gen, sz)
            popped.append((t, ty, eid, gen))

        self.assertEqual(popped, sorted(events))

    def test_doubly_linked_list_attach_and_detach(self):
        n_users = 4
        n_sats = 2
        ue_sat = np.full(n_users, -1, dtype=np.int32)
        sat_head = np.full(n_sats, -1, dtype=np.int32)
        ue_prev = np.full(n_users, -1, dtype=np.int32)
        ue_next = np.full(n_users, -1, dtype=np.int32)
        sat_count = np.zeros(n_sats, dtype=np.int32)

        for u in (0, 1, 2):
            link_ue(u, 0, ue_sat, sat_head, ue_prev, ue_next, sat_count)
        self.assertEqual(sat_count[0], 3)

        # Unlink middle node (1), head node (2), and tail node (0)
        unlink_ue(1, 0, ue_sat, sat_head, ue_prev, ue_next, sat_count)
        self.assertEqual(sat_count[0], 2)
        self.assertEqual(ue_sat[1], -1)

        # Traverse remaining nodes from head
        visited = []
        curr = sat_head[0]
        while curr != -1:
            visited.append(int(curr))
            curr = ue_next[curr]
        self.assertEqual(visited, [2, 0])

        unlink_ue(2, 0, ue_sat, sat_head, ue_prev, ue_next, sat_count)
        unlink_ue(0, 0, ue_sat, sat_head, ue_prev, ue_next, sat_count)
        self.assertEqual(sat_count[0], 0)
        self.assertEqual(sat_head[0], -1)


if __name__ == "__main__":
    unittest.main()
