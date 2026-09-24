import unittest

import numpy as np

from medialoop.gp import expected_improvement, predict
from medialoop.loop import inside_bounds, random_search, run
from medialoop.surfaces import cardiac, neural


class LoopTests(unittest.TestCase):
    def test_gp_fits_a_smooth_bump(self):
        xs = np.linspace(0, 12, 8)
        # other factors held at the cardiac-friendly corner: IWP 4, SB 0, LDN 0
        x = np.column_stack([
            xs,
            np.full(8, 4.0),
            np.zeros(8),
            np.zeros(8),
        ])
        y = np.array([
            cardiac({
                "CHIR99021_uM": float(v),
                "IWP2_uM": 4.0,
                "SB431542_uM": 0.0,
                "LDN193189_nM": 0.0,
            })
            for v in xs
        ])
        probe = np.array([[8.0, 4.0, 0.0, 0.0]])
        mu, sigma = predict(x, y, probe, noise=1e-3)
        self.assertAlmostEqual(float(mu[0]), float(y.max()), delta=0.15)
        self.assertGreater(float(sigma[0]), 0.0)
        ei = expected_improvement(mu, sigma, float(y.max()) - 0.2)
        self.assertTrue(np.isfinite(ei).all())

    def test_expected_improvement_beats_random_on_the_cartoon(self):
        ei = [run(cardiac, 8, 12, seed=s)["best_true"] for s in range(10)]
        rnd = [random_search(cardiac, 20, seed=1000 + s)["best_true"] for s in range(10)]
        self.assertGreater(sum(ei) / len(ei), sum(rnd) / len(rnd) + 0.05)
        neural_ei = [run(neural, 8, 12, seed=s)["best_true"] for s in range(8)]
        neural_rnd = [random_search(neural, 20, seed=2000 + s)["best_true"] for s in range(8)]
        self.assertGreater(sum(neural_ei) / len(neural_ei), sum(neural_rnd) / len(neural_rnd))

    def test_proposals_stay_inside_published_windows(self):
        result = run(cardiac, 4, 3, seed=1)
        self.assertTrue(inside_bounds(result["x"]))
        self.assertEqual(len(result["true"]), 7)


if __name__ == "__main__":
    unittest.main()
