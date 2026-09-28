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

    def test_batch_acquire_strategies(self):
        from medialoop.loop import batch_acquire
        pool = np.random.default_rng(42).uniform(0, 5, size=(20, 4))
        x_init = pool[:3]
        y_init = np.array([0.2, 0.5, 0.4])
        for strat in ["kriging_believer", "constant_liar_min", "constant_liar_max", "constant_liar_mean"]:
            picks = batch_acquire(x_init, y_init, pool[3:], batch_size=4, strategy=strat, noise=0.02)
            self.assertEqual(len(picks), 4)
            self.assertEqual(len(set(picks)), 4)

        with self.assertRaises(ValueError):
            batch_acquire(x_init, y_init, pool[3:], batch_size=0)
        with self.assertRaises(ValueError):
            batch_acquire(x_init, y_init, pool[3:], batch_size=100)

    def test_batch_run_simulation(self):
        result = run(cardiac, 4, 3, seed=1, batch_size=3, batch_strategy="kriging_believer")
        self.assertTrue(inside_bounds(result["x"]))
        self.assertEqual(len(result["true"]), 4 + 3 * 3)
        self.assertEqual(result["x"].shape[0], 13)

    def test_cli_batch_bakeoff(self):
        from medialoop.cli import bakeoff, main
        res = bakeoff("cardiac", seeds=2, batch_size=2)
        self.assertEqual(res["batch_size"], 2)
        self.assertEqual(res["budget"], 8 + 12 * 2)
        self.assertIn("mean_best_expected_improvement", res)
        # Test CLI invocation
        code = main(["--objective", "cardiac", "--seeds", "1", "--batch-size", "2"])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()

