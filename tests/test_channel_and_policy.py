"""Unit tests for RF channel lookup and baseline handover policies."""

import unittest

from src.constants import (
    MIN_ELEV, SAFE_ELEV, M_BHO, M_CHO, M_LOAD_MIN, M_LOAD_STD,
)
from src.channel import get_se, sinr_from_dist
from src.handover_policy import req_elev_of, ho_margin_of, ho_metric, is_load_aware


class TestChannelAndPolicy(unittest.TestCase):
    def test_mcs_table_monotonicity(self):
        self.assertAlmostEqual(get_se(-10.0), 1e-6)
        self.assertAlmostEqual(get_se(-7.0), 0.15)
        self.assertAlmostEqual(get_se(9.0), 2.40)
        self.assertAlmostEqual(get_se(25.0), 5.55)

    def test_fspl_sinr_decreases_with_distance(self):
        self.assertGreater(sinr_from_dist(550e3), sinr_from_dist(1100e3))

    def test_baseline_policy_properties(self):
        self.assertEqual(req_elev_of(M_LOAD_MIN), MIN_ELEV)
        self.assertEqual(req_elev_of(M_CHO), SAFE_ELEV)

        self.assertEqual(ho_margin_of(M_BHO), 0.0)
        self.assertEqual(ho_margin_of(M_CHO), 3.0)
        self.assertEqual(ho_margin_of(M_LOAD_STD), 0.05)

        self.assertFalse(is_load_aware(M_BHO))
        self.assertFalse(is_load_aware(M_CHO))
        self.assertTrue(is_load_aware(M_LOAD_MIN))
        self.assertTrue(is_load_aware(M_LOAD_STD))

    def test_load_metric_prefers_lower_load(self):
        m_low_load = ho_metric(M_LOAD_STD, 9.0, 2)
        m_high_load = ho_metric(M_LOAD_STD, 9.0, 20)
        self.assertGreater(m_low_load, m_high_load)


if __name__ == "__main__":
    unittest.main()
