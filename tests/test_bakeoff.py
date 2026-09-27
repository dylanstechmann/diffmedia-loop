import unittest

from medialoop.bakeoff import POLICIES, campaign, candidate_grid
from medialoop.loop import inside_bounds
from medialoop.surfaces import cardiac


class BakeoffTests(unittest.TestCase):
    def test_grid_contains_the_cardiac_cartoon_peak_and_stays_in_bounds(self):
        grid = candidate_grid()
        self.assertTrue(inside_bounds(grid))
        self.assertIn((8.0, 4.0, 0.0, 0.0), {tuple(row) for row in grid.tolist()})

    def test_guarded_policies_do_not_repeat_or_leave_the_box(self):
        for policy in ("random", "expected_improvement", "ucb", "thompson"):
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

    def test_unknown_policy_is_rejected(self):
        self.assertNotIn("dqn", POLICIES)
        with self.assertRaises(ValueError):
            campaign(cardiac, "dqn", seed=0)


if __name__ == "__main__":
    unittest.main()
