import tempfile
import unittest
from pathlib import Path

import numpy as np

from medialoop.gp import predict
from medialoop.loop import inside_bounds
from medialoop.planner import propose
from medialoop.space import NAMES


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.candidates = self.root / "candidates.csv"
        self.candidates.write_text("candidate_id," + ",".join(NAMES) + "\na,0,0,0,0\nb,3,1,0,0\nc,6,2,0,0\nd,9,3,0,0\n")
        self.observed = self.root / "observations.csv"
        self.observed.write_text("candidate_id,response\na,0.1\nb,0.3\n")
        self.pending = self.root / "pending.csv"
        self.pending.write_text("candidate_id\nc\n")

    def test_excludes_observed_and_pending_and_records_hashes(self):
        result = propose(self.candidates, self.observed, self.pending)
        self.assertEqual(result["candidate_id"], "d")
        self.assertEqual(result["method"], "expected_improvement")
        self.assertEqual(len(result["input_sha256"]["observations"]), 64)
        self.assertEqual(result, propose(self.candidates, self.observed, self.pending))

    def test_exhaustion_and_ambiguous_history_are_errors(self):
        self.pending.write_text("candidate_id\nc\nd\n")
        with self.assertRaisesRegex(ValueError, "no unevaluated"):
            propose(self.candidates, self.observed, self.pending)
        self.pending.write_text("candidate_id\na\n")
        with self.assertRaisesRegex(ValueError, "both observed"):
            propose(self.candidates, self.observed, self.pending)
        self.observed.write_text("candidate_id,response\nunknown,0.1\n")
        with self.assertRaisesRegex(ValueError, "unknown"):
            propose(self.candidates, self.observed)

    def test_nonfinite_readout_and_bounds_are_rejected(self):
        self.observed.write_text("candidate_id,response\na,nan\n")
        with self.assertRaisesRegex(ValueError, "finite"):
            propose(self.candidates, self.observed)
        self.assertFalse(inside_bounds(np.array([[np.nan, 0, 0, 0]])))

    def test_flat_readouts_preserve_unmeasured_uncertainty(self):
        x = np.array([[0, 0, 0, 0], [0.1, 0, 0, 0]])
        mu, sigma = predict(x, np.array([0.5, 0.5]), np.array([[0, 0, 0, 0], [12, 5, 10, 250]]))
        self.assertGreater(sigma[1], sigma[0])
        np.testing.assert_allclose(mu, [0.5, 0.5])

    def test_initial_choice_is_reproducible(self):
        result = propose(self.candidates, seed=10)
        self.assertEqual(result, propose(self.candidates, seed=10))
        self.assertEqual(result["method"], "random_initialization")

    def test_batch_propose_and_cli(self):
        from medialoop.planner import main
        # 2 available: c and d (a and b observed)
        res_batch = propose(self.candidates, self.observed, batch_size=2, batch_strategy="kriging_believer")
        self.assertEqual(res_batch["batch_size"], 2)
        self.assertEqual(len(res_batch["proposals"]), 2)
        self.assertEqual(set(res_batch["candidates"]), {"c", "d"})

        # Constant liar min
        res_cl = propose(self.candidates, self.observed, batch_size=2, batch_strategy="constant_liar_min")
        self.assertEqual(len(res_cl["proposals"]), 2)

        # Batch size exceeding available candidates
        with self.assertRaises(ValueError):
            propose(self.candidates, self.observed, batch_size=3)

        # CLI test
        out_json = self.root / "batch_out.json"
        code = main([
            "--candidates", str(self.candidates),
            "--observations", str(self.observed),
            "--batch-size", "2",
            "--out", str(out_json),
        ])
        self.assertEqual(code, 0)
        self.assertTrue(out_json.exists())

