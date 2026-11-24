"""Unit tests for Bounded Divergence Persistence Gating (BDPG)."""
import unittest
import numpy as np

from src.tracking.bdpg import (
    DEFAULT_BDPG_CONFIG,
    build_candidate_graph,
    prune_spurious_cytokinesis_forks,
)
from src.tracking.contracts import Lineage


class BDPGTests(unittest.TestCase):
    def test_bdpg_protects_drift_tilted_persistent_division(self):
        """Genuine drift-tilted cytokinesis where both daughters persist (len >= 5) is preserved."""
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [0.0, 20.0, 10.0, 10.0],
            [1.0, 20.0, 14.0, 10.0],
            [2.0, 20.0, 18.0, 10.0],
            [3.0, 20.0, 22.0, 10.0],
            [4.0, 20.0, 26.0, 10.0],
            [5.0, 20.0, 30.0, 10.0],
            [6.0, 20.0, 34.0, 10.0],
            [1.0, 18.0, 10.5, 10.0],
            [2.0, 16.0, 11.0, 10.0],
            [3.0, 14.0, 11.5, 10.0],
            [4.0, 12.0, 12.0, 10.0],
            [5.0, 10.0, 12.5, 10.0],
            [6.0,  8.0, 13.0, 10.0],
        ], dtype=np.float64)
        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
            [0, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
        ], dtype=np.int64)
        anchor = Lineage(coords, edges)
        cand, audit = build_candidate_graph(anchor, scale, variant_cfg=DEFAULT_BDPG_CONFIG)
        self.assertEqual(audit["fork_pruning"]["forks_pruned"], 0)
        self.assertEqual(len(cand.edges), len(edges))

    def test_ddks_spi_prunes_acute_crossover(self):
        """Acute parallel jump with extreme spatial asymmetry is pruned via DDKS-SPI."""
        scale = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
        coords = np.array([
            [0.0, 10.0, 10.0, 10.0],
            [1.0, 10.0, 11.0, 10.0],  # 0.406 um (close keeper)
            [1.0, 10.0, 22.0, 10.0],  # 4.875 um (far parallel jump)
        ], dtype=np.float64)
        edges = np.array([[0, 1], [0, 2]], dtype=np.int64)
        surviving, audit = prune_spurious_cytokinesis_forks(
            coords, edges, scale, None, DEFAULT_BDPG_CONFIG["fork_pruning"]
        )
        self.assertEqual(audit["forks_pruned"], 1)
        self.assertEqual(audit["spi_triggered"], 1)
        np.testing.assert_array_equal(surviving, [[0, 1]])


if __name__ == "__main__":
    unittest.main()
