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
