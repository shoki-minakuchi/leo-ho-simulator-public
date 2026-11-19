"""End-to-end smoke test across all baseline handover policies."""

import unittest

from src.constants import BASELINE_METHODS
from src.orbit import precompute_constellation
from src.traffic import generate_arrivals
from src.sim_engine import run_sim


class TestSimulatorSmoke(unittest.TestCase):
    def test_short_simulation_all_baseline_methods(self):
        max_duration = 60.0
        raan0, phase0, vis_idx, vis_cnt = precompute_constellation(max_duration, verbose=False)
        arr_t, arr_pos, arr_tgt = generate_arrivals(scale=0.25, max_duration=max_duration, seed=42)

        self.assertGreater(len(arr_t), 0)
        self.assertGreater(len(raan0), 0)

        for method_id in BASELINE_METHODS:
            result = run_sim(
                method_id, max_duration, raan0, phase0, vis_idx, vis_cnt,
                arr_t, arr_pos, arr_tgt,
            )
            avg_mbps, edge_mbps, ho, hof, rlf, drop, block, unfin, heap_peak, evals, max_ev = result
            self.assertGreaterEqual(avg_mbps, 0.0)
            self.assertGreaterEqual(edge_mbps, 0.0)
            self.assertGreaterEqual(heap_peak, 1)
            self.assertGreaterEqual(evals, 0)
            self.assertGreaterEqual(ho, 0)
            self.assertEqual(hof, 0)
            self.assertEqual(rlf, 0)


if __name__ == "__main__":
    unittest.main()
