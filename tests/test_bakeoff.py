import io
import json
import unittest
from unittest.mock import patch

import numpy as np

from medialoop.bakeoff import PENALTY_LENGTH, POLICIES, batch_penalty, campaign, candidate_grid, main as bakeoff_main
from medialoop.loop import inside_bounds
from medialoop.surfaces import OBJECTIVES, cardiac, cardiac_shifted, neural_shifted


class BakeoffTests(unittest.TestCase):
    def test_grid_contains_the_cardiac_cartoon_peak_and_stays_in_bounds(self):
        grid = candidate_grid()
        self.assertTrue(inside_bounds(grid))
        self.assertIn((8.0, 4.0, 0.0, 0.0), {tuple(row) for row in grid.tolist()})

    def test_guarded_policies_do_not_repeat_or_leave_the_box(self):
        for policy in ("random", "expected_improvement", "ucb", "thompson", "penalized_expected_improvement"):
            result = campaign(cardiac, policy, seed=1, n_init=4, rounds=3, batch_size=2)
            self.assertEqual(result["violations"], 0, policy)
            self.assertEqual(result["n_unique"], result["n_evaluations"], policy)
            self.assertGreaterEqual(result["final_simple_regret"], -1e-9)

    def test_repeat_policy_records_batch_violations(self):
        rounds, batch = 3, 2
        result = campaign(cardiac, "repeat_expected_improvement", seed=0, n_init=4, rounds=rounds, batch_size=batch)
        self.assertEqual(result["violations"], (batch - 1) * rounds)
        self.assertLess(result["n_unique"], result["n_evaluations"])

    def test_expected_improvement_regrets_less_than_random_on_the_cartoon(self):
        ei = [campaign(cardiac, "expected_improvement", seed=s, rounds=5, batch_size=2)["final_simple_regret"] for s in range(6)]
        rnd = [campaign(cardiac, "random", seed=s, rounds=5, batch_size=2)["final_simple_regret"] for s in range(6)]
        self.assertLess(sum(ei) / len(ei), sum(rnd) / len(rnd) - 0.05)

    def test_batch_penalty_is_zero_at_a_pick_and_rises_toward_one_with_distance(self):
        grid = candidate_grid()
        picked = grid[[0]]
        penalty = batch_penalty(grid, picked)
        self.assertEqual(penalty[0], 0.0)
        self.assertTrue(np.all((penalty >= 0.0) & (penalty <= 1.0)))
        order = np.argsort(np.linalg.norm((grid - grid.min(0)) / (grid.max(0) - grid.min(0))
                                          - (picked - grid.min(0)) / (grid.max(0) - grid.min(0)), axis=1))
        self.assertLess(penalty[order[1]], penalty[order[-1]])
        self.assertTrue(np.all(batch_penalty(grid, grid[:0]) == 1.0))
        self.assertEqual(PENALTY_LENGTH, 0.25)

    def test_the_original_four_policies_and_the_negative_control_are_all_still_present(self):
        for policy in ("random", "expected_improvement", "ucb", "thompson", "repeat_expected_improvement"):
            self.assertIn(policy, POLICIES)

    def test_unknown_policy_is_rejected(self):
        self.assertNotIn("dqn", POLICIES)
        with self.assertRaises(ValueError):
            campaign(cardiac, "dqn", seed=0)

    def test_shifted_objectives_move_the_peak_and_stay_on_the_grid(self):
        grid = candidate_grid()
        self.assertTrue(inside_bounds(grid))
        points = {tuple(row) for row in grid.tolist()}
        # The shifted peaks are deliberately on the lattice so regret stays
        # interpretable, and they are not the original cartoon peaks.
        self.assertIn((6.0, 2.0, 0.0, 0.0), points)
        self.assertIn((0.0, 0.0, 5.0, 250.0), points)
        self.assertAlmostEqual(
            cardiac_shifted({"CHIR99021_uM": 6.0, "IWP2_uM": 2.0,
                             "SB431542_uM": 0.0, "LDN193189_nM": 0.0}), 1.0
        )
        self.assertAlmostEqual(
            neural_shifted({"CHIR99021_uM": 0.0, "IWP2_uM": 0.0,
                            "SB431542_uM": 5.0, "LDN193189_nM": 250.0}), 1.0
        )
        # The shifted surfaces are genuinely different functions.
        self.assertNotAlmostEqual(
            cardiac_shifted({"CHIR99021_uM": 8.0, "IWP2_uM": 4.0,
                             "SB431542_uM": 0.0, "LDN193189_nM": 0.0}),
            cardiac({"CHIR99021_uM": 8.0, "IWP2_uM": 4.0,
                     "SB431542_uM": 0.0, "LDN193189_nM": 0.0}),
        )
        self.assertIn("cardiac_shifted", OBJECTIVES)
        self.assertIn("neural_shifted", OBJECTIVES)

    def test_expected_improvement_follows_a_shifted_surface(self):
        ei = [campaign(cardiac_shifted, "expected_improvement", seed=s,
                       rounds=5, batch_size=2)["final_simple_regret"] for s in range(6)]
        rnd = [campaign(cardiac_shifted, "random", seed=s,
                        rounds=5, batch_size=2)["final_simple_regret"] for s in range(6)]
        self.assertLess(sum(ei) / len(ei), sum(rnd) / len(rnd) - 0.05)

    def test_noise_is_reported_and_invalid_noise_is_rejected(self):
        result = campaign(cardiac, "expected_improvement", seed=0, noise=0.5)
        self.assertAlmostEqual(result["best_true"], result["best_true"])  # smoke
        summary = None
        with patch("sys.stdout", io.StringIO()) as buffer:
            self.assertEqual(
                bakeoff_main(["--objective", "cardiac", "--seeds", "2", "--noise", "0.5"]),
                0,
            )
            summary = json.loads(buffer.getvalue())
        self.assertEqual(summary["readout_noise"], 0.5)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            campaign(cardiac, "expected_improvement", seed=0, noise=-0.1)


if __name__ == "__main__":
    unittest.main()
