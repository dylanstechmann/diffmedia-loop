"""Validate and aggregate measured assay rows for the external planner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np

from medialoop.space import FACTORS, NAMES


MEASUREMENT_COLUMNS = {
    "candidate_id", "plate_id", "well_id", "biological_unit_id",
    "technical_replicate_id", "batch_id", "assay_id", "endpoint", "value",
    "unit", "well_measurement_standard_uncertainty", "status", "failure_reason",
}
AGGREGATION_METHOD = (
    "mean technical replicates within biological_unit_id, then unweighted mean "
    "across biological units"
)


def _read_csv_snapshot(path, required):
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: CSV must be UTF-8") from exc
    with io.StringIO(text, newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if len(header) != len(set(header)) or not set(required).issubset(header):
            raise ValueError(f"{path}: require unique columns including {','.join(required)}")
        rows = []
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}: row {line} is ragged")
            rows.append({key: value.strip() for key, value in row.items()})
    return rows, raw, hashlib.sha256(raw).hexdigest()


def _candidate_ids(path):
    rows, raw, digest = _read_csv_snapshot(path, ["candidate_id", *NAMES])
    ids = [row["candidate_id"] for row in rows]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("candidate table needs nonblank, unique candidate_id values")
    seen_factors = set()
    factors_by_id = {}
    bounds = {item["name"]: (item["low"], item["high"]) for item in FACTORS}
    for row in rows:
        try:
            values = tuple(float(row[name]) for name in NAMES)
        except ValueError as exc:
            raise ValueError(f"{row['candidate_id']}: candidate factors must be numeric") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{row['candidate_id']}: candidate factors must be finite")
        if any(not bounds[name][0] <= value <= bounds[name][1]
               for name, value in zip(NAMES, values)):
            raise ValueError(f"{row['candidate_id']}: factor outside planner bounds")
        if values in seen_factors:
            raise ValueError("candidate table contains duplicate factor combinations")
        seen_factors.add(values)
        factors_by_id[row["candidate_id"]] = dict(zip(NAMES, values))
    return factors_by_id, raw, digest


def aggregate_measurements(candidates_path, measurements_path, *, direction):
    """Return one explicit-direction response per measured condition plus audit data."""
    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be 'maximize' or 'minimize'")
    candidates, _, candidate_hash = _candidate_ids(candidates_path)
    rows, _, measurement_hash = _read_csv_snapshot(
        measurements_path, MEASUREMENT_COLUMNS
    )
    if not rows:
        raise ValueError("measurement CSV contains no rows")

    assay_ids = {row["assay_id"] for row in rows}
    endpoints = {row["endpoint"] for row in rows}
    units = {row["unit"] for row in rows}
    if any(not value for value in assay_ids | endpoints | units):
        raise ValueError("assay_id, endpoint, and unit must be nonblank on every row")
    if len(assay_ids) != 1 or len(endpoints) != 1 or len(units) != 1:
        raise ValueError("one import must contain exactly one assay_id, endpoint, and unit")

    seen_wells = set()
    seen_replicates = set()
    by_condition = {}
    uncertainty_statuses = set()
    units_by_condition = {}
    condition_by_unit = {}
    batches = set()
    failed_reasons = {}
    failed_by_condition = {}
    measured_by_condition = {}
    for line, row in enumerate(rows, 2):
        for field in (
            "candidate_id", "plate_id", "well_id", "biological_unit_id",
            "technical_replicate_id", "batch_id",
        ):
            if not row[field]:
                raise ValueError(f"measurement row {line}: {field} must be nonblank")
        candidate_id = row["candidate_id"]
        if candidate_id not in candidates:
            raise ValueError(f"measurement row {line}: unknown candidate_id {candidate_id!r}")
        well_key = (row["plate_id"], row["well_id"])
        if well_key in seen_wells:
            raise ValueError(f"measurement row {line}: duplicate plate_id/well_id")
        seen_wells.add(well_key)
        replicate_key = (candidate_id, row["biological_unit_id"], row["technical_replicate_id"])
        if replicate_key in seen_replicates:
            raise ValueError(f"measurement row {line}: duplicate technical replicate identity")
        seen_replicates.add(replicate_key)
        batches.add(row["batch_id"])
        units_by_condition.setdefault(candidate_id, set()).add(row["biological_unit_id"])
        condition_by_unit.setdefault(row["biological_unit_id"], set()).add(candidate_id)

        status = row["status"]
        reason = row["failure_reason"]
        if status == "measured":
            if reason:
                raise ValueError(f"measurement row {line}: measured wells must have blank failure_reason")
            try:
                value = float(row["value"])
            except ValueError as exc:
                raise ValueError(f"measurement row {line}: measured well needs a numeric value") from exc
            if not math.isfinite(value):
                raise ValueError(f"measurement row {line}: measured well value must be finite")
            uncertainty_text = row["well_measurement_standard_uncertainty"]
            if uncertainty_text:
                try:
                    well_uncertainty = float(uncertainty_text)
                except ValueError as exc:
                    raise ValueError(f"measurement row {line}: well uncertainty must be numeric or blank") from exc
                if not math.isfinite(well_uncertainty) or well_uncertainty < 0:
                    raise ValueError(f"measurement row {line}: well uncertainty must be finite and nonnegative")
                uncertainty_statuses.add("measured")
            else:
                well_uncertainty = None
                uncertainty_statuses.add("not_available")
            by_condition.setdefault(candidate_id, {}).setdefault(
                row["biological_unit_id"], []
            ).append((value, well_uncertainty))
            measured_by_condition[candidate_id] = measured_by_condition.get(candidate_id, 0) + 1
        elif status == "failed":
            if row["value"] or row["well_measurement_standard_uncertainty"]:
                raise ValueError(f"measurement row {line}: failed wells must have blank value and uncertainty")
            if not reason:
                raise ValueError(f"measurement row {line}: failed wells need a failure_reason")
            failed_by_condition[candidate_id] = failed_by_condition.get(candidate_id, 0) + 1
            failed_reasons[reason] = failed_reasons.get(reason, 0) + 1
        else:
            raise ValueError(f"measurement row {line}: status must be measured or failed")

    shared_units = any(len(condition_ids) > 1 for condition_ids in condition_by_unit.values())
    if len(uncertainty_statuses) > 1:
        raise ValueError("well uncertainty must be supplied for every measured well or for none")
    assay_uncertainty_available = uncertainty_statuses == {"measured"}
    condition_summary = {}
    aggregate_rows = []
    for candidate_id in candidates:
        unit_values = list(by_condition.get(candidate_id, {}).values())
        unit_means = [float(np.mean([value for value, _ in technical_values]))
                      for technical_values in unit_values]
        unit_uncertainties = [
            math.sqrt(sum(uncertainty ** 2 for _, uncertainty in technical_values))
            / len(technical_values)
            for technical_values in unit_values
        ] if assay_uncertainty_available else None
        n_units = len(unit_means)
        if not n_units:
            condition_summary[candidate_id] = {
                "status": "no_measured_wells",
                "n_biological_units_attempted": len(units_by_condition.get(candidate_id, set())),
                "n_biological_units_measured": 0,
                "n_measured_wells": 0,
                "n_failed_wells": failed_by_condition.get(candidate_id, 0),
                "assay_measurement_standard_uncertainty_of_mean": None,
                "failed_reasons": {},
            }
            continue
        raw_mean = float(np.mean(unit_means))
        biological_sd = None if n_units < 2 else float(np.std(unit_means, ddof=1))
        biological_sem = None if n_units < 2 else biological_sd / math.sqrt(n_units)
        assay_measurement_uncertainty = (
            math.sqrt(sum(value ** 2 for value in unit_uncertainties)) / n_units
            if unit_uncertainties is not None else None
        )
        response = raw_mean if direction == "maximize" else -raw_mean
        stats = {
            "status": "aggregated",
            "raw_endpoint_mean": raw_mean,
            "planner_response": response,
            "n_biological_units_attempted": len(units_by_condition.get(candidate_id, set())),
            "n_biological_units_measured": n_units,
            "n_measured_wells": measured_by_condition.get(candidate_id, 0),
            "n_failed_wells": failed_by_condition.get(candidate_id, 0),
            "biological_unit_mean_sd": biological_sd,
            "biological_unit_mean_sem": biological_sem,
            "assay_measurement_standard_uncertainty_of_mean": assay_measurement_uncertainty,
            "failed_reasons": {},
        }
        condition_summary[candidate_id] = stats
        aggregate_rows.append({
            "candidate_id": candidate_id,
            "response": response,
            "raw_endpoint_mean": raw_mean,
            "measurement_schema_version": "1",
            "assay_id": next(iter(assay_ids)),
            "endpoint": next(iter(endpoints)),
            "unit": next(iter(units)),
            "objective_direction": direction,
            "response_transform": "identity" if direction == "maximize" else "negated",
            "assay_measurement_uncertainty_status": (
                "measured" if assay_uncertainty_available else "not_available"
            ),
            "n_biological_units": n_units,
            "n_measured_wells": measured_by_condition.get(candidate_id, 0),
            "n_failed_wells": failed_by_condition.get(candidate_id, 0),
            "assay_measurement_standard_uncertainty_of_mean": (
                "" if assay_measurement_uncertainty is None else assay_measurement_uncertainty
            ),
            "biological_unit_mean_sd": "" if biological_sd is None else biological_sd,
            "biological_unit_mean_sem": "" if biological_sem is None else biological_sem,
            "shared_biological_units_across_conditions": str(shared_units).lower(),
            "aggregation_method": AGGREGATION_METHOD,
            "source_measurements_sha256": measurement_hash,
            "source_candidates_sha256": candidate_hash,
        })

    # Attribute failed wells by condition and reason, including conditions with no usable data.
    failed_by_reason = {}
    for row in rows:
        if row["status"] == "failed":
            reasons = failed_by_reason.setdefault(row["candidate_id"], {})
            reason = row["failure_reason"]
            reasons[reason] = reasons.get(reason, 0) + 1
    for candidate_id in candidates:
        condition_summary[candidate_id]["failed_reasons"] = failed_by_reason.get(candidate_id, {})
    return aggregate_rows, {
        "schema_version": 1,
        "source_measurements_sha256": measurement_hash,
        "source_candidates_sha256": candidate_hash,
        "assay_id": next(iter(assay_ids)),
        "endpoint": next(iter(endpoints)),
        "unit": next(iter(units)),
        "objective_direction": direction,
        "response_transform": "identity" if direction == "maximize" else "negated",
        "assay_measurement_uncertainty_status": (
            "measured" if assay_uncertainty_available else "not_available"
        ),
        "aggregation_method": AGGREGATION_METHOD,
        "shared_biological_units_across_conditions": shared_units,
        "batches_present": sorted(batches),
        "n_rows": len(rows),
        "n_measured_wells": sum(row["status"] == "measured" for row in rows),
        "n_failed_wells": sum(row["status"] == "failed" for row in rows),
        "failed_reasons": failed_reasons,
        "conditions": condition_summary,
        "notes": [
            "Technical replicates are averaged within biological_unit_id; biological units receive equal weight.",
            "Biological-unit SD and SEM describe the imported replicates and are not used by the fixed-noise GP.",
            "When provided, well measurement standard uncertainties are propagated separately through technical and biological means under independence; shared calibration uncertainty is not included.",
            "The planner maximizes one declared endpoint and does not model shared-unit, batch, or assay covariance.",
            "Failed wells are retained in this report and excluded from numeric aggregation; they are never imputed.",
            "A proposal is not a validated protocol or evidence of rejuvenation.",
        ],
    }


def _render_observations(rows):
    columns = [
        "candidate_id", "response", "raw_endpoint_mean", "measurement_schema_version", "assay_id",
        "endpoint", "unit", "objective_direction", "response_transform",
        "assay_measurement_uncertainty_status",
        "n_biological_units", "n_measured_wells", "n_failed_wells",
        "biological_unit_mean_sd", "biological_unit_mean_sem",
        "assay_measurement_standard_uncertainty_of_mean",
        "shared_biological_units_across_conditions", "aggregation_method",
        "source_measurements_sha256", "source_candidates_sha256",
    ]
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def import_to_directory(candidates_path, measurements_path, output_dir, *, direction):
    observations, report = aggregate_measurements(
        candidates_path, measurements_path, direction=direction
    )
    observations_text = _render_observations(observations)
    observation_bytes = observations_text.encode("utf-8")
    report["aggregated_observations_sha256"] = hashlib.sha256(observation_bytes).hexdigest()
    report_text = json.dumps(report, indent=2, allow_nan=False) + "\n"

    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staged = Path(temporary)
        (staged / "observations.csv").write_bytes(observation_bytes)
        (staged / "measurement_report.json").write_text(report_text, encoding="utf-8")
        output.mkdir()
        try:
            for name in ("observations.csv", "measurement_report.json"):
                (staged / name).rename(output / name)
        except BaseException:
            shutil.rmtree(output)
            raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate raw assay rows and aggregate technical replicates within biological units"
    )
    parser.add_argument("--candidates", required=True, help="candidate table used for this screen")
    parser.add_argument("--measurements", required=True, help="raw well-level CSV with explicit status and units")
    parser.add_argument("--direction", choices=["maximize", "minimize"], required=True,
                        help="predeclared direction for the single endpoint")
    parser.add_argument("--out-dir", required=True, help="new directory for observations.csv and measurement_report.json")
    args = parser.parse_args(argv)
    try:
        report = import_to_directory(
            args.candidates,
            args.measurements,
            args.out_dir,
            direction=args.direction,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "output": str(Path(args.out_dir)),
        "n_aggregated_conditions": sum(
            value["status"] == "aggregated" for value in report["conditions"].values()
        ),
        "n_failed_wells": report["n_failed_wells"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
