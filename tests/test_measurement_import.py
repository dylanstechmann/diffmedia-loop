import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from medialoop.measurement_import import aggregate_measurements, import_to_directory
from medialoop.measurement_import import main as import_main
from medialoop.planner import propose
from medialoop.space import NAMES

MEASUREMENT_HEADER = [
    "candidate_id", "plate_id", "well_id", "biological_unit_id",
    "technical_replicate_id", "batch_id", "assay_id", "endpoint", "unit",
    "value", "well_measurement_standard_uncertainty", "status", "failure_reason",
]

CANDIDATES = "candidate_id," + ",".join(NAMES) + "\na,0,0,0,0\nb,3,1,0,0\nc,6,2,0,0\nd,9,3,0,0\n"


def measured_row(candidate, plate, well, unit, tech, value, uncertainty="0.01"):
    return [candidate, plate, well, unit, tech, "batch-01", "ASSAY-1",
            "viability", "fraction", value, uncertainty, "measured", ""]


def failed_row(candidate, plate, well, unit, tech, reason):
    return [candidate, plate, well, unit, tech, "batch-01", "ASSAY-1",
            "viability", "fraction", "", "", "failed", reason]


class MeasurementImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.candidates = self.root / "candidates.csv"
        self.candidates.write_text(CANDIDATES)
        self.measurements = self.root / "measurements.csv"

    def write_measurements(self, rows):
        with self.measurements.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(MEASUREMENT_HEADER)
            writer.writerows(rows)

    def default_rows(self):
        return [
            measured_row("a", "p1", "A01", "u1", "t1", "0.20"),
            measured_row("a", "p1", "A02", "u1", "t2", "0.22"),
            measured_row("a", "p1", "A03", "u2", "t1", "0.18"),
            measured_row("a", "p1", "A04", "u2", "t2", "0.21"),
            measured_row("b", "p1", "B01", "u3", "t1", "0.31"),
            measured_row("b", "p1", "B02", "u3", "t2", "0.29"),
        ]

    def import_observations(self, name="imported", rows=None):
        self.write_measurements(self.default_rows() if rows is None else rows)
        out = self.root / name
        import_to_directory(self.candidates, self.measurements, out, direction="maximize")
        return out / "observations.csv"

    def rewrite_observations(self, path, mutate_rows):
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        mutate_rows(rows)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def test_aggregates_technical_replicates_within_biological_units(self):
        self.write_measurements(self.default_rows())
        rows, report = aggregate_measurements(self.candidates, self.measurements, direction="maximize")
        by_id = {row["candidate_id"]: row for row in rows}
        # unit means for a: mean(0.20, 0.22) = 0.21 and mean(0.18, 0.21) = 0.195
        self.assertAlmostEqual(by_id["a"]["raw_endpoint_mean"], (0.21 + 0.195) / 2)
        self.assertEqual(by_id["a"]["n_biological_units"], 2)
        self.assertEqual(by_id["a"]["n_measured_wells"], 4)
        self.assertEqual(by_id["a"]["n_failed_wells"], 0)
        unit_sd = np.std([0.21, 0.195], ddof=1)
        self.assertAlmostEqual(float(by_id["a"]["biological_unit_mean_sd"]), unit_sd)
        self.assertAlmostEqual(float(by_id["a"]["biological_unit_mean_sem"]), unit_sd / math.sqrt(2))
        # technical replicate uncertainties propagate: sqrt(0.01^2 + 0.01^2)/2 per unit
        unit_u = math.sqrt(0.01 ** 2 + 0.01 ** 2) / 2
        expected_assay_u = math.sqrt(unit_u ** 2 + unit_u ** 2) / 2
        self.assertEqual(by_id["a"]["assay_measurement_uncertainty_status"], "measured")
        self.assertAlmostEqual(float(by_id["a"]["assay_measurement_standard_uncertainty_of_mean"]), expected_assay_u)
        self.assertEqual(report["aggregation_method"],
                         "mean technical replicates within biological_unit_id, then unweighted mean across biological units")
        self.assertEqual(report["assay_id"], "ASSAY-1")
        self.assertEqual(report["endpoint"], "viability")
        self.assertEqual(report["unit"], "fraction")

    def test_minimize_direction_negates_the_response(self):
        self.write_measurements(self.default_rows())
        rows, _ = aggregate_measurements(self.candidates, self.measurements, direction="minimize")
        by_id = {row["candidate_id"]: row for row in rows}
        self.assertAlmostEqual(by_id["a"]["raw_endpoint_mean"], 0.2025)
        self.assertAlmostEqual(float(by_id["a"]["response"]), -0.2025)
        self.assertEqual(by_id["a"]["objective_direction"], "minimize")
        self.assertEqual(by_id["a"]["response_transform"], "negated")
        with self.assertRaisesRegex(ValueError, "direction"):
            aggregate_measurements(self.candidates, self.measurements, direction="sideways")

    def test_failed_wells_are_retained_and_never_imputed(self):
        rows = self.default_rows() + [
            failed_row("a", "p1", "A05", "u2", "t3", "reader interruption"),
            failed_row("c", "p1", "C01", "u4", "t1", "well not seeded"),
        ]
        self.write_measurements(rows)
        rows_out, report = aggregate_measurements(self.candidates, self.measurements, direction="maximize")
        by_id = {row["candidate_id"]: row for row in rows_out}
        self.assertEqual(by_id["a"]["n_failed_wells"], 1)
        self.assertAlmostEqual(by_id["a"]["raw_endpoint_mean"], 0.2025)
        failed_condition = report["conditions"]["c"]
        self.assertEqual(failed_condition["status"], "no_measured_wells")
        self.assertEqual(failed_condition["n_failed_wells"], 1)
        self.assertEqual(failed_condition["n_biological_units_attempted"], 1)
        self.assertEqual(report["failed_reasons"], {"reader interruption": 1, "well not seeded": 1})
        self.assertEqual(report["conditions"]["a"]["failed_reasons"], {"reader interruption": 1})
        self.assertIn("never imputed", " ".join(report["notes"]))

    def test_shared_biological_units_across_conditions_are_flagged(self):
        rows = self.default_rows() + [measured_row("c", "p1", "C01", "u1", "t3", "0.5")]
        self.write_measurements(rows)
        _, report = aggregate_measurements(self.candidates, self.measurements, direction="maximize")
        self.assertTrue(report["shared_biological_units_across_conditions"])
        self.write_measurements(self.default_rows())
        _, report = aggregate_measurements(self.candidates, self.measurements, direction="maximize")
        self.assertFalse(report["shared_biological_units_across_conditions"])

    def test_rejects_ragged_duplicate_or_unknown_rows(self):
        self.measurements.write_text(
            ",".join(MEASUREMENT_HEADER) + "\n"
            "a,p1,A01,u1,t1,batch-01,ASSAY-1,viability,fraction,0.2,0.01,measured\n"
        )
        with self.assertRaisesRegex(ValueError, "ragged"):
            aggregate_measurements(self.candidates, self.measurements, direction="maximize")
        self.write_measurements(self.default_rows() + [measured_row("a", "p1", "A01", "u1", "t1", "0.2")])
        with self.assertRaisesRegex(ValueError, "duplicate plate_id/well_id"):
            aggregate_measurements(self.candidates, self.measurements, direction="maximize")
        self.write_measurements(self.default_rows() + [measured_row("zz", "p1", "Z01", "u9", "t1", "0.2")])
        with self.assertRaisesRegex(ValueError, "unknown candidate_id"):
            aggregate_measurements(self.candidates, self.measurements, direction="maximize")
        self.write_measurements(self.default_rows() + [measured_row("a", "p1", "A05", "u2", "t1", "0.2")])
        with self.assertRaisesRegex(ValueError, "duplicate technical replicate identity"):
            aggregate_measurements(self.candidates, self.measurements, direction="maximize")

    def test_rejects_invalid_well_statuses_and_uncertainty(self):
        self.write_measurements(self.default_rows() + [measured_row("b", "p1", "B03", "u3", "t3", "0.30", "")])
        with self.assertRaisesRegex(ValueError, "every measured well or for none"):
            aggregate_measurements(self.candidates, self.measurements, direction="maximize")
        reason_row = ["a", "p1", "A05", "u2", "t3", "batch-01", "ASSAY-1", "viability",
                      "fraction", "0.2", "0.01", "measured", "has a reason"]
        self.write_measurements(self.default_rows() + [reason_row])
        with self.assertRaisesRegex(ValueError, "measured wells must have blank failure_reason"):
            aggregate_measurements(self.candidates, self.measurements, direction="maximize")
        failed_with_value = ["a", "p1", "A05", "u2", "t3", "batch-01", "ASSAY-1", "viability",
                             "fraction", "0.2", "", "failed", "reader interruption"]
        self.write_measurements(self.default_rows() + [failed_with_value])
        with self.assertRaisesRegex(ValueError, "failed wells must have blank value"):
            aggregate_measurements(self.candidates, self.measurements, direction="maximize")
        self.write_measurements(self.default_rows() + [failed_row("a", "p1", "A05", "u2", "t3", "")])
        with self.assertRaisesRegex(ValueError, "failed wells need a failure_reason"):
            aggregate_measurements(self.candidates, self.measurements, direction="maximize")
        pending_row = ["a", "p1", "A05", "u2", "t3", "batch-01", "ASSAY-1", "viability",
                       "fraction", "", "", "pending", ""]
        self.write_measurements(self.default_rows() + [pending_row])
        with self.assertRaisesRegex(ValueError, "status must be measured or failed"):
            aggregate_measurements(self.candidates, self.measurements, direction="maximize")
        self.write_measurements([measured_row("a", "p1", "A01", "u1", "t1", "not-a-number")])
        with self.assertRaisesRegex(ValueError, "numeric value"):
            aggregate_measurements(self.candidates, self.measurements, direction="maximize")
        self.write_measurements([measured_row("a", "p1", "A01", "u1", "t1", "-0.2", "-0.01")])
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            aggregate_measurements(self.candidates, self.measurements, direction="maximize")

    def test_requires_exactly_one_assay_endpoint_and_unit(self):
        percent_row = measured_row("b", "p1", "B05", "u5", "t1", "0.35")
        percent_row[8] = "percent"
        self.write_measurements(self.default_rows() + [percent_row])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            aggregate_measurements(self.candidates, self.measurements, direction="maximize")
        blank_assay = [[value if column != "assay_id" else ""
                        for column, value in zip(MEASUREMENT_HEADER, row)]
                       for row in self.default_rows()]
        self.write_measurements(blank_assay)
        with self.assertRaisesRegex(ValueError, "nonblank"):
            aggregate_measurements(self.candidates, self.measurements, direction="maximize")
        two_endpoints = [[value if column != "endpoint" or row_index == 0 else "other_endpoint"
                          for column, value in zip(MEASUREMENT_HEADER, row)]
                         for row_index, row in enumerate(self.default_rows())]
        self.write_measurements(two_endpoints)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            aggregate_measurements(self.candidates, self.measurements, direction="maximize")

    def test_candidate_table_validation(self):
        self.write_measurements(self.default_rows())
        self.candidates.write_text(CANDIDATES + "e,0,0,0,0\n")
        with self.assertRaisesRegex(ValueError, "duplicate factor combinations"):
            aggregate_measurements(self.candidates, self.measurements, direction="maximize")
        self.candidates.write_text(CANDIDATES.replace("a,0,0,0,0", "a,0,0,0,99999"))
        with self.assertRaisesRegex(ValueError, "outside planner bounds"):
            aggregate_measurements(self.candidates, self.measurements, direction="maximize")
        self.candidates.write_text(CANDIDATES.replace("a,0,0,0,0", "a,zero,0,0,0"))
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            aggregate_measurements(self.candidates, self.measurements, direction="maximize")
        self.candidates.write_text(CANDIDATES + "a,1,0,0,0\n")
        with self.assertRaisesRegex(ValueError, "unique candidate_id"):
            aggregate_measurements(self.candidates, self.measurements, direction="maximize")
        self.candidates.write_text(CANDIDATES.replace("a,0,0,0,0", ",0,0,0,0"))
        with self.assertRaisesRegex(ValueError, "nonblank, unique candidate_id"):
            aggregate_measurements(self.candidates, self.measurements, direction="maximize")

    def test_import_retains_exact_parsed_sources_even_if_originals_change(self):
        from unittest.mock import patch
        import medialoop.measurement_import as importer
        self.write_measurements(self.default_rows())
        originals = {"source_candidates.csv": self.candidates.read_bytes(),
                     "source_measurements.csv": self.measurements.read_bytes()}
        render = importer._render_observations
        def replace_originals(rows):
            self.candidates.write_text("changed")
            self.measurements.write_text("changed")
            return render(rows)
        out = self.root / "snapshot-import"
        with patch.object(importer, "_render_observations", side_effect=replace_originals):
            report = import_to_directory(self.candidates, self.measurements, out, direction="maximize")
        for name, raw in originals.items():
            self.assertEqual((out / name).read_bytes(), raw)
            self.assertEqual(report["source_snapshots"][name]["sha256"],
                             hashlib.sha256(raw).hexdigest())

    def test_planner_rejects_changed_or_missing_raw_source_snapshot(self):
        observations = self.import_observations()
        source = observations.with_name("source_measurements.csv")
        raw = source.read_bytes()
        source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source snapshot integrity failed"):
            propose(self.candidates, observations)
        source.write_bytes(raw)
        source.unlink()
        with self.assertRaisesRegex(ValueError, "missing measurement source snapshot"):
            propose(self.candidates, observations)

    def test_explicit_null_source_snapshot_metadata_is_not_legacy(self):
        observations = self.import_observations()
        path = observations.with_name("measurement_report.json")
        report = json.loads(path.read_text())
        report["source_snapshots"] = None
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "source snapshots require both"):
            propose(self.candidates, observations)

    def test_import_to_directory_writes_provenance_and_refuses_overwrite(self):
        self.write_measurements(self.default_rows())
        out = self.root / "imported"
        report = import_to_directory(self.candidates, self.measurements, out, direction="maximize")
        observations_bytes = (out / "observations.csv").read_bytes()
        self.assertEqual(report["aggregated_observations_sha256"], hashlib.sha256(observations_bytes).hexdigest())
        self.assertEqual(report["source_candidates_sha256"],
                         hashlib.sha256(self.candidates.read_bytes()).hexdigest())
        self.assertEqual(report["source_measurements_sha256"],
                         hashlib.sha256(self.measurements.read_bytes()).hexdigest())
        saved = json.loads((out / "measurement_report.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["aggregated_observations_sha256"], report["aggregated_observations_sha256"])
        with self.assertRaises(FileExistsError):
            import_to_directory(self.candidates, self.measurements, out, direction="maximize")

    def test_cli_import_measurements(self):
        self.write_measurements(self.default_rows())
        out = self.root / "cli-import"
        import_main(["--candidates", str(self.candidates), "--measurements", str(self.measurements),
                     "--direction", "maximize", "--out-dir", str(out)])
        self.assertTrue((out / "observations.csv").exists())
        self.assertTrue((out / "measurement_report.json").exists())
        with self.assertRaises(SystemExit):
            import_main(["--candidates", str(self.candidates), "--measurements", str(self.measurements),
                         "--direction", "maximize", "--out-dir", str(out)])

    def test_planner_accepts_imported_observations_and_records_context(self):
        observations = self.import_observations()
        result = propose(self.candidates, observations)
        context = result["measurement_context"]
        self.assertEqual(context["assay_id"], "ASSAY-1")
        self.assertEqual(context["endpoint"], "viability")
        self.assertEqual(context["objective_direction"], "maximize")
        self.assertEqual(context["response_transform"], "identity")
        self.assertEqual(context["source_measurements_sha256"],
                         hashlib.sha256(self.measurements.read_bytes()).hexdigest())
        self.assertEqual(context["conditions"]["a"]["n_biological_units"], 2)
        self.assertEqual(context["conditions"]["b"]["n_biological_units"], 1)
        self.assertIn("Set --noise explicitly", context["note"])
        self.assertEqual(result, propose(self.candidates, observations))

    def test_planner_accepts_candidate_with_only_failed_wells(self):
        rows = self.default_rows() + [
            failed_row("c", "p1", "C01", "u4", "t1", "reader interruption")
        ]
        observations = self.import_observations(name="failed-condition", rows=rows)
        with observations.open(newline="") as handle:
            self.assertNotIn("c", {row["candidate_id"] for row in csv.DictReader(handle)})
        proposal = propose(self.candidates, observations)
        self.assertEqual(proposal["measurement_context"]["conditions"]["a"]["n_biological_units"], 2)

    def test_planner_requires_sibling_measurement_report(self):
        observations = self.import_observations()
        observations.with_name("measurement_report.json").unlink()
        with self.assertRaisesRegex(ValueError, "require their sibling measurement_report.json"):
            propose(self.candidates, observations)

    def test_planner_rejects_observations_made_for_a_different_candidate_table(self):
        observations = self.import_observations()
        self.candidates.write_text(CANDIDATES + "e,12,5,10,250\n")
        with self.assertRaisesRegex(ValueError, "different candidate table"):
            propose(self.candidates, observations)

    def test_planner_rejects_tampered_imported_observations(self):
        tamper_expectations = [
            ("response", lambda row: str(-float(row["raw_endpoint_mean"])),
             "does not match the declared endpoint direction"),
            ("objective_direction", lambda row: "minimize",
             "objective direction and response transform disagree"),
            ("measurement_schema_version", lambda row: "2",
             "unsupported measurement-import schema version"),
            ("unit", lambda row: "percent", "disagree on unit"),
        ]
        for index, (column, edit, expected) in enumerate(tamper_expectations):
            observations = self.import_observations(name=f"tampered-{index}")
            self.rewrite_observations(observations, lambda rows: [row.update({column: edit(row)}) for row in rows])
            with self.assertRaisesRegex(ValueError, expected):
                propose(self.candidates, observations)

        observations = self.import_observations()
        self.rewrite_observations(observations, lambda rows: [row.update(
            {"biological_unit_mean_sem": row["biological_unit_mean_sd"]}) for row in rows])
        with self.assertRaisesRegex(ValueError, "SEM must equal SD divided by sqrt"):
            propose(self.candidates, observations)

    def test_checked_in_example_imports_and_plans(self):
        root = Path(__file__).resolve().parents[1]
        candidates = root / "examples" / "candidates.example.csv"
        measurements = root / "examples" / "measurements.example.csv"
        out = self.root / "example-import"
        report = import_to_directory(candidates, measurements, out, direction="maximize")
        self.assertEqual(report["n_measured_wells"], 7)
        self.assertEqual(report["n_failed_wells"], 1)
        result = propose(candidates, out / "observations.csv")
        self.assertEqual(result["measurement_context"]["endpoint"], "illustrative_endpoint")
        self.assertEqual(result["measurement_context"]["conditions"]["synthetic-a"]["n_biological_units"], 2)


if __name__ == "__main__":
    unittest.main()