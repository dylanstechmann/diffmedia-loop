import json
import tempfile
import unittest
from pathlib import Path

from medialoop.constraints import (
    check_candidates,
    enforce_constraints,
    factor_agreement,
    load_bundle,
)
from medialoop.planner import main as planner_main
from medialoop.planner import propose
from medialoop.space import NAMES

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
BUNDLE = EXAMPLES / "planner-constraints.json"


def _write_csv(path, header, rows):
    lines = [",".join(header)] + [",".join(str(value) for value in row) for row in rows]
    Path(path).write_text("\n".join(lines) + "\n")


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_checked_in_bundle_loads_and_agrees_with_builtin_box(self):
        bundle = load_bundle(BUNDLE)
        self.assertEqual(
            [item["protocol_id"] for item in bundle["protocols"]],
            ["giwi_cardiac", "dual_smad_neural"],
        )
        self.assertEqual(factor_agreement(bundle, require_complete=True), [])
        self.assertEqual(len(bundle["source_sha256"]), 64)
        for name in NAMES:
            self.assertIn(name, bundle["factors"])
        # The exclusivity rule travels with the bundle.
        self.assertEqual(len(bundle["exclusivity"]), 1)
        self.assertEqual(
            set(bundle["exclusivity"][0]["parameters"]),
            {"Noggin_ng_per_mL", "LDN193189_nM"},
        )

    def test_out_of_window_candidates_are_rejected(self):
        bundle = load_bundle(BUNDLE)
        rows = [
            {"candidate_id": "ok", "CHIR99021_uM": "6", "IWP2_uM": "4",
             "SB431542_uM": "5", "Noggin_ng_per_mL": "200", "LDN193189_nM": "0"},
            {"candidate_id": "high", "CHIR99021_uM": "40", "IWP2_uM": "4",
             "SB431542_uM": "5", "Noggin_ng_per_mL": "200", "LDN193189_nM": "0"},
        ]
        violations = check_candidates(rows, bundle)
        self.assertEqual(len(violations), 1)
        self.assertIn("outside the published window", violations[0])
        self.assertIn("40", violations[0])
        self.assertIn("giwi_cardiac", violations[0])
        self.assertEqual(check_candidates([rows[0]], bundle), [])

    def test_zero_dose_outside_a_published_window_is_rejected(self):
        # The simulation box includes zero; the published GiWi windows do not.
        bundle = load_bundle(BUNDLE)
        rows = [{"candidate_id": "zero", "CHIR99021_uM": "0", "IWP2_uM": "4",
                 "SB431542_uM": "5", "Noggin_ng_per_mL": "200", "LDN193189_nM": "0"}]
        violations = check_candidates(rows, bundle)
        self.assertEqual(len(violations), 1)
        self.assertIn("outside the published window", violations[0])

    def test_required_factor_columns_must_be_present(self):
        # A candidate row that omits a bundle factor column is rejected; the
        # omitted alternative cannot silently bypass the exported windows.
        bundle = load_bundle(BUNDLE)
        rows = [{"candidate_id": "omits", "CHIR99021_uM": "6", "IWP2_uM": "4",
                 "SB431542_uM": "5", "LDN193189_nM": "100"}]
        violations = check_candidates(rows, bundle)
        self.assertEqual(len(violations), 1)
        self.assertIn("missing required parameter Noggin_ng_per_mL", violations[0])

    def test_mutually_exclusive_alternatives_are_rejected(self):
        bundle = load_bundle(BUNDLE)
        both = [{"candidate_id": "both", "CHIR99021_uM": "6", "IWP2_uM": "4",
                 "SB431542_uM": "5", "Noggin_ng_per_mL": "200", "LDN193189_nM": "100"}]
        violations = check_candidates(both, bundle)
        self.assertEqual(len(violations), 1)
        self.assertIn("mutually exclusive", violations[0])
        alone = [{"candidate_id": "one", "CHIR99021_uM": "6", "IWP2_uM": "4",
                  "SB431542_uM": "5", "Noggin_ng_per_mL": "200", "LDN193189_nM": "0"}]
        self.assertEqual(check_candidates(alone, bundle), [])
        substituted = [{"candidate_id": "ldn", "CHIR99021_uM": "6", "IWP2_uM": "4",
                        "SB431542_uM": "5", "Noggin_ng_per_mL": "0", "LDN193189_nM": "100"}]
        self.assertEqual(check_candidates(substituted, bundle), [])
        neither = [{"candidate_id": "neither", "CHIR99021_uM": "6", "IWP2_uM": "4",
                    "SB431542_uM": "5", "Noggin_ng_per_mL": "0", "LDN193189_nM": "0"}]
        violations = check_candidates(neither, bundle)
        self.assertEqual(len(violations), 1)
        self.assertIn("must be positive", violations[0])

    def test_malformed_bundles_are_rejected(self):
        cases = {
            "not-json": "{oops",
            "empty-list": "[]",
            "bad-version": json.dumps({"schema_version": 99, "protocol_id": "x",
                                       "parameters": [{"name": "a", "low": 0, "high": 1}]}),
            "unordered-window": json.dumps({"schema_version": 1, "protocol_id": "x",
                                            "parameters": [{"name": "a", "low": 5, "high": 1}]}),
            "no-parameters": json.dumps({"schema_version": 1, "protocol_id": "x", "parameters": []}),
            "unknown-rule-members": json.dumps({"schema_version": 1, "protocol_id": "x",
                                                "parameters": [{"name": "a", "low": 0, "high": 1}],
                                                "exclusivity": [{"parameters": ["a", "nope"],
                                                                  "rule": "at_most_one_positive"}]}),
        }
        for name, text in cases.items():
            with self.subTest(case=name):
                path = self.root / f"{name}.json"
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_bundle(path)

    def test_factor_agreement_reports_drift(self):
        bundle = load_bundle(BUNDLE)
        drifted = json.loads(json.dumps(bundle))
        drifted["factors"]["CHIR99021_uM"]["high"] = 99.0
        discrepancies = factor_agreement(drifted)
        self.assertTrue(any("CHIR99021_uM" in item for item in discrepancies))

        incomplete = json.loads(json.dumps(bundle))
        del incomplete["factors"]["IWP2_uM"]
        discrepancies = factor_agreement(incomplete, require_complete=True)
        self.assertTrue(any("IWP2_uM" in item for item in discrepancies))


class PlannerConstraintTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Values inside the exported published windows (CHIR 2-12, IWP2 2-5,
        # SB431542 5-10, LDN 0-250, Noggin 100-300 with 0 as the explicit
        # inactive alternative). Every bundle factor column is present.
        _write_csv(self.root / "candidates.csv",
                   ["candidate_id", *NAMES, "Noggin_ng_per_mL"],
                   [["a", 6, 4, 5, 0, 200], ["b", 9, 3, 7, 0, 200],
                    ["c", 4, 5, 10, 0, 200], ["d", 12, 2, 6, 100, 0]])
        self.candidates = self.root / "candidates.csv"
        self.observed = self.root / "observations.csv"
        self.observed.write_text("candidate_id,response\na,0.1\nb,0.3\n")
        self.pending = self.root / "pending.csv"
        self.pending.write_text("candidate_id\nc\n")

    def test_propose_with_constraints_records_provenance(self):
        result = propose(self.candidates, self.observed, self.pending,
                         constraints_path=BUNDLE)
        self.assertEqual(result["candidate_id"], "d")
        provenance = result["constraints"]
        self.assertEqual(len(provenance["sha256"]), 64)
        self.assertEqual([item["protocol_id"] for item in provenance["protocols"]],
                         ["giwi_cardiac", "dual_smad_neural"])

    def test_propose_rejects_out_of_window_candidates(self):
        # Zero CHIR is inside the simulation box but outside the GiWi window.
        self.candidates.write_text(
            "candidate_id," + ",".join(NAMES) + ",Noggin_ng_per_mL\n"
            "a,0,4,5,0,200\nb,9,3,7,0,200\n"
        )
        self.observed.write_text("candidate_id,response\na,0.1\n")
        self.pending.write_text("candidate_id\n")
        with self.assertRaisesRegex(ValueError, "outside the published window"):
            propose(self.candidates, constraints_path=BUNDLE)

    def test_cli_constraints_flag(self):
        out = self.root / "proposal.json"
        code = planner_main([
            "--candidates", str(self.candidates),
            "--observations", str(self.observed),
            "--pending", str(self.pending),
            "--constraints", str(BUNDLE),
            "--out", str(out),
        ])
        self.assertEqual(code, 0)
        result = json.loads(out.read_text())
        self.assertIn("constraints", result)
        self.assertEqual(len(result["constraints"]["sha256"]), 64)

    def test_reserve_path_enforces_constraints(self):
        from medialoop.reservations import propose_and_reserve
        self.candidates.write_text(
            "candidate_id," + ",".join(NAMES) + ",Noggin_ng_per_mL\n"
            "a,0,4,5,0,200\nb,9,3,7,0,200\n"
        )
        self.observed.write_text("candidate_id,response\na,0.1\n")
        with self.assertRaisesRegex(ValueError, "outside the published window"):
            propose_and_reserve(self.candidates, self.observed,
                                ledger_path=self.root / "ledger.sqlite3",
                                request_id="bad-window", constraints_path=BUNDLE)

    def test_enforce_constraints_returns_bundle_provenance(self):
        import csv
        with open(self.candidates, newline="") as handle:
            rows = list(csv.DictReader(handle))
        provenance = enforce_constraints(BUNDLE, rows)
        self.assertEqual(len(provenance["sha256"]), 64)
        self.assertEqual(len(provenance["protocols"]), 2)


if __name__ == "__main__":
    unittest.main()
