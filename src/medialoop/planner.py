"""Propose from a collaborator-supplied candidate list with an explicit ledger.

No surface oracle is called. Readouts and pending IDs come from the user.
The GP maximizes the supplied response and assumes a common noise scale.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from medialoop.constraints import enforce_constraints
from medialoop.gp import expected_improvement, predict
from medialoop.loop import inside_bounds
from medialoop.space import NAMES


def _read(path, required):
    # Parse and hash the same snapshot, including when another process replaces
    # an observations CSV while this request is being prepared.
    raw = Path(path).read_bytes()
    with io.StringIO(raw.decode("utf-8-sig"), newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if len(set(header)) != len(header) or not set(required).issubset(header):
            raise ValueError(f"{path}: require unique columns {','.join(required)}")
        rows = []
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"{path}: ragged CSV")
            row = {k: v.strip() for k, v in row.items()}
            if any(not row[key] for key in required):
                raise ValueError(f"{path}: missing value")
            rows.append(row)
    ids = [row["candidate_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate candidate_id (aggregate replicates explicitly)")
    return rows, hashlib.sha256(raw).hexdigest()


def _inputs(candidates_path, observations_path, pending_path):
    rows, candidates_hash = _read(candidates_path, ["candidate_id", *NAMES])
    observations, observations_hash = _read(observations_path, ["candidate_id", "response"]) if observations_path else ([], None)
    pending, pending_hash = _read(pending_path, ["candidate_id"]) if pending_path else ([], None)
    return rows, observations, pending, {"candidates": candidates_hash,
                                        "observations": observations_hash, "pending": pending_hash}


_REPORT_FIELDS = (
    "assay_id", "endpoint", "unit", "objective_direction", "response_transform",
    "assay_measurement_uncertainty_status", "aggregation_method",
    "shared_biological_units_across_conditions", "source_measurements_sha256",
    "source_candidates_sha256",
)


def _sibling_import_report(observations_path):
    """Read the measurement-report.json published beside an imported observations CSV."""
    if observations_path is None:
        return None
    report_path = Path(observations_path).with_name("measurement_report.json")
    if not report_path.exists():
        raise ValueError(
            "measurement-import observations require their sibling measurement_report.json"
        )
    try:
        report = json.loads(report_path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("planner could not read the sibling measurement_report.json") from exc
    if not isinstance(report, dict):
        raise ValueError("sibling measurement_report.json must be a JSON object")
    snapshots = report.get("source_snapshots")
    if "source_snapshots" in report:
        names = {"source_candidates.csv": "source_candidates_sha256",
                 "source_measurements.csv": "source_measurements_sha256"}
        if not isinstance(snapshots, dict) or set(snapshots) != set(names):
            raise ValueError("measurement source snapshots require both source tables")
        for name, digest_field in names.items():
            metadata = snapshots[name]
            path = report_path.with_name(name)
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise ValueError(f"missing measurement source snapshot: {name}") from exc
            digest = hashlib.sha256(raw).hexdigest()
            if (not isinstance(metadata, dict) or digest != metadata.get("sha256")
                    or len(raw) != metadata.get("size_bytes")
                    or digest != report.get(digest_field)):
                raise ValueError(f"measurement source snapshot integrity failed: {name}")
    return report


def _disagree(field, extra=""):
    raise ValueError(
        f"planner observations disagree on {field} with measurement_report.json{extra}; "
        "the observations table may have been edited independently of its source records"
    )


def _cross_check_import_report_fields(observations_path, first):
    """Fail when observation metadata no longer matches its sibling report."""
    report = _sibling_import_report(observations_path)
    if report is None:
        return
    for field in _REPORT_FIELDS:
        if field not in report:
            continue
        row_value = first[field]
        if row_value in ("true", "false") and isinstance(report[field], bool):
            row_value = row_value == "true"
        if report[field] != row_value:
            _disagree(field)


def _cross_check_import_report_totals(observations_path, observations, by_candidate, hashes):
    """Fail when observation counts or bytes no longer match its sibling report."""
    report = _sibling_import_report(observations_path)
    if report is None:
        return
    conditions = report.get("conditions")
    if isinstance(conditions, dict):
        for field in ("n_measured_wells", "n_failed_wells"):
            if field in report:
                total = sum(int(value.get(field, 0)) for value in conditions.values()
                            if isinstance(value, dict))
                if report[field] != total:
                    _disagree(field)
        for candidate_id, parsed in by_candidate.items():
            declared = conditions.get(candidate_id)
            if not isinstance(declared, dict):
                continue
            for field in ("n_measured_wells", "n_failed_wells"):
                if field in declared and declared[field] != parsed[field]:
                    _disagree(field, f" for {candidate_id}")
            if "n_biological_units_measured" in declared and declared["n_biological_units_measured"] != parsed["n_biological_units"]:
                _disagree("n_biological_units_measured", f" for {candidate_id}")
    declared_digest = report.get("aggregated_observations_sha256")
    if isinstance(declared_digest, str) and declared_digest:
        if hashes.get("observations") != declared_digest:
            raise ValueError(
                "planner observation bytes do not match the sibling measurement_report.json "
                "aggregated_observations_sha256; the observations table may have been edited "
                "independently of its source records"
            )


def _measurement_context(observations, hashes, observations_path=None):
    if not observations or "measurement_schema_version" not in observations[0]:
        return None
    required = {
        "candidate_id", "response", "raw_endpoint_mean", "measurement_schema_version",
        "assay_id", "endpoint", "unit", "objective_direction", "response_transform",
        "assay_measurement_uncertainty_status", "n_biological_units", "n_measured_wells", "n_failed_wells",
        "biological_unit_mean_sd", "biological_unit_mean_sem",
        "assay_measurement_standard_uncertainty_of_mean",
        "shared_biological_units_across_conditions", "aggregation_method",
        "source_measurements_sha256", "source_candidates_sha256",
    }
    for row in observations:
        if not required.issubset(row):
            raise ValueError("measurement-import observations are missing required metadata columns")
        if row["measurement_schema_version"] != "1":
            raise ValueError("unsupported measurement-import schema version")

    common_fields = (
        "assay_id", "endpoint", "unit", "objective_direction", "response_transform",
        "assay_measurement_uncertainty_status",
        "shared_biological_units_across_conditions", "aggregation_method",
        "source_measurements_sha256", "source_candidates_sha256",
    )
    for field in common_fields:
        values = {row[field] for row in observations}
        if len(values) != 1 or not next(iter(values)):
            raise ValueError(f"measurement-import observations disagree on {field}")
    first = observations[0]
    direction = first["objective_direction"]
    expected_transform = "identity" if direction == "maximize" else "negated" if direction == "minimize" else None
    if expected_transform is None or first["response_transform"] != expected_transform:
        raise ValueError("measurement-import objective direction and response transform disagree")
    if first["shared_biological_units_across_conditions"] not in {"true", "false"}:
        raise ValueError("shared_biological_units_across_conditions must be true or false")
    digest_fields = ("source_measurements_sha256", "source_candidates_sha256")
    for field in digest_fields:
        digest = first[field]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise ValueError(f"measurement-import {field} must be a SHA-256 digest")
    if first["source_candidates_sha256"] != hashes["candidates"]:
        raise ValueError("measurement-import rows were created for a different candidate table")
    _cross_check_import_report_fields(observations_path, first)

    by_candidate = {}
    for row in observations:
        try:
            response = float(row["response"])
            raw_mean = float(row["raw_endpoint_mean"])
            n_units = int(row["n_biological_units"])
            n_measured = int(row["n_measured_wells"])
            n_failed = int(row["n_failed_wells"])
        except ValueError as exc:
            raise ValueError("measurement-import summaries need numeric means and replicate counts") from exc
        if not np.isfinite([response, raw_mean]).all() or n_units < 1 or n_measured < n_units or n_failed < 0:
            raise ValueError("measurement-import summaries contain invalid means or replicate counts")
        expected_response = raw_mean if direction == "maximize" else -raw_mean
        if not np.isclose(response, expected_response, rtol=1e-12, atol=1e-15):
            raise ValueError("planner response does not match the declared endpoint direction")
        sd_text = row["biological_unit_mean_sd"]
        sem_text = row["biological_unit_mean_sem"]
        if n_units < 2:
            if sd_text or sem_text:
                raise ValueError("biological-unit SD/SEM must be blank with fewer than two units")
            sd = sem = None
        else:
            try:
                sd = float(sd_text)
                sem = float(sem_text)
            except ValueError as exc:
                raise ValueError("biological-unit SD/SEM must be numeric when at least two units exist") from exc
            if (not np.isfinite([sd, sem]).all() or sd < 0 or sem < 0
                    or not np.isclose(sem, sd / np.sqrt(n_units), rtol=1e-9, atol=1e-12)):
                raise ValueError("biological-unit SEM must equal SD divided by sqrt(n_units)")
        assay_u_text = row["assay_measurement_standard_uncertainty_of_mean"]
        if first["assay_measurement_uncertainty_status"] == "measured":
            try:
                assay_u = float(assay_u_text)
            except ValueError as exc:
                raise ValueError("measured assay uncertainty needs a numeric mean uncertainty") from exc
            if not np.isfinite(assay_u) or assay_u < 0:
                raise ValueError("assay measurement standard uncertainty must be finite and nonnegative")
        elif first["assay_measurement_uncertainty_status"] == "not_available":
            if assay_u_text:
                raise ValueError("assay uncertainty must be blank when marked not_available")
            assay_u = None
        else:
            raise ValueError("assay_measurement_uncertainty_status must be measured or not_available")
        by_candidate[row["candidate_id"]] = {
            "n_biological_units": n_units,
            "n_measured_wells": n_measured,
            "n_failed_wells": n_failed,
            "biological_unit_mean_sd": sd,
            "biological_unit_mean_sem": sem,
            "assay_measurement_standard_uncertainty_of_mean": assay_u,
        }
    _cross_check_import_report_totals(observations_path, observations, by_candidate, hashes)
    return {
        "schema_version": 1,
        "assay_id": first["assay_id"],
        "endpoint": first["endpoint"],
        "unit": first["unit"],
        "objective_direction": direction,
        "response_transform": first["response_transform"],
        "assay_measurement_uncertainty_status": first["assay_measurement_uncertainty_status"],
        "aggregation_method": first["aggregation_method"],
        "source_measurements_sha256": first["source_measurements_sha256"],
        "source_candidates_sha256": first["source_candidates_sha256"],
        "aggregated_observations_sha256": hashes["observations"],
        "shared_biological_units_across_conditions": first["shared_biological_units_across_conditions"] == "true",
        "conditions": by_candidate,
        "note": (
            "The fixed-noise GP maximizes this one endpoint. It does not use per-condition biological SEM "
            "or model batch/shared-unit covariance. Set --noise explicitly for the planner."
        ),
    }


def propose(candidates_path, observations_path=None, pending_path=None, *, seed=0, noise=0.02, batch_size=1, batch_strategy="kriging_believer", constraints_path=None):
    """Preview candidate(s) without reserving or modifying input files.

    When constraints_path names a constraint bundle exported by
    cell-protocol-compiler, the candidate table is checked against those
    published windows before a proposal is made; violations are errors.
    """
    rows, observations, pending, hashes = _inputs(candidates_path, observations_path, pending_path)
    provenance = enforce_constraints(constraints_path, rows) if constraints_path else None
    result = _propose(rows, observations, pending, hashes, seed=seed, noise=noise, batch_size=batch_size, batch_strategy=batch_strategy, observations_path=observations_path)
    if provenance is not None:
        result["constraints"] = provenance
    return result


def _propose(rows, observations, pending, hashes, *, seed, noise, reserved_ids=(), batch_size=1, batch_strategy="kriging_believer", observations_path=None):
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not rows:
        raise ValueError("empty candidate list")
    x = np.array([[float(row[name]) for name in NAMES] for row in rows])
    if not inside_bounds(x):
        raise ValueError("candidate factors must be finite and inside the simulation's encoded bounds")
    if len({tuple(row) for row in x}) != len(x):
        raise ValueError("duplicate factor combinations; give each condition one candidate_id")
    ids = [row["candidate_id"] for row in rows]
    lookup = {value: i for i, value in enumerate(ids)}
    observed_ids = {row["candidate_id"] for row in observations}
    pending_ids = {row["candidate_id"] for row in pending}
    if (observed_ids | pending_ids) - set(ids):
        raise ValueError("ledger contains an unknown candidate_id")
    if observed_ids & pending_ids:
        raise ValueError("candidate cannot be both observed and pending")
    pending_ids |= set(reserved_ids) - observed_ids
    y = np.array([float(row["response"]) for row in observations])
    if not np.isfinite(y).all():
        raise ValueError("responses must be finite")
    measurement_context = _measurement_context(observations, hashes, observations_path)
    available = [i for i, name in enumerate(ids) if name not in observed_ids | pending_ids]
    if not available:
        raise ValueError("no unevaluated, nonpending candidates remain")
    if batch_size > len(available):
        raise ValueError(f"requested batch_size {batch_size} exceeds {len(available)} available candidates")
    rng = np.random.default_rng(seed)
    model_ready = len(observations) >= 2
    if batch_size > 1 and model_ready and batch_strategy not in {
        "kriging_believer", "constant_liar_min", "constant_liar_max", "constant_liar_mean"
    }:
        raise ValueError(f"unknown batch strategy: {batch_strategy!r}")

    if batch_size == 1:
        mu = sigma = acquisition = None
        if len(observations) < 2:
            chosen = int(rng.choice(available))
            method = "random_initialization"
        else:
            measured = [lookup[row["candidate_id"]] for row in observations]
            means, stds = predict(x[measured], y, x[available], noise=noise)
            ei = expected_improvement(means, stds, float(y.max()))
            # Resolve ties reproducibly without systematically selecting the first CSV row.
            best = np.flatnonzero(np.isclose(ei, ei.max(), rtol=1e-10, atol=1e-15))
            local = int(rng.choice(best))
            chosen = available[local]
            mu, sigma, acquisition = float(means[local]), float(stds[local]), float(ei[local])
            method = "expected_improvement"
        result = {"schema_version": 1, "candidate_id": ids[chosen], "factors": dict(zip(NAMES, x[chosen].tolist())),
                  "method": method, "predicted_mean": mu, "predicted_std": sigma,
                  "expected_improvement": acquisition, "seed": seed, "noise": noise,
                  "n_observed": len(observations), "n_pending": len(pending_ids), "input_sha256": hashes,
                  "note": "Maximizes the supplied response. A proposal is not an executable protocol. Record it as pending before asking again."}
        if measurement_context is not None:
            result["measurement_context"] = measurement_context
        return result

    # Batch acquisition (batch_size > 1) via Kriging Believer or Constant Liar
    cur_measured = [lookup[row["candidate_id"]] for row in observations]
    cur_x = x[cur_measured] if cur_measured else np.empty((0, len(NAMES)))
    cur_y = np.copy(y)
    cur_available = list(available)
    proposals = []

    for b in range(batch_size):
        if len(cur_y) < 2 or len(cur_x) < 2:
            chosen = int(rng.choice(cur_available))
            method = "random_initialization"
            mu = sigma = acquisition = None
        else:
            avail_cand_x = x[cur_available]
            means, stds = predict(cur_x, cur_y, avail_cand_x, noise=noise)
            ei = expected_improvement(means, stds, float(cur_y.max()))
            best = np.flatnonzero(np.isclose(ei, ei.max(), rtol=1e-10, atol=1e-15))
            local = int(rng.choice(best))
            chosen = cur_available[local]
            mu, sigma, acquisition = float(means[local]), float(stds[local]), float(ei[local])
            method = "expected_improvement"

        proposals.append({
            "candidate_id": ids[chosen],
            "factors": dict(zip(NAMES, x[chosen].tolist())),
            "method": method,
            "predicted_mean": mu,
            "predicted_std": sigma,
            "expected_improvement": acquisition,
        })
        cur_available.remove(chosen)

        if b < batch_size - 1 and model_ready:
            pick_x = x[chosen:chosen + 1]
            if len(cur_y) >= 2 and len(cur_x) >= 2:
                if batch_strategy == "kriging_believer":
                    imputed_y = mu if mu is not None else float(cur_y.mean())
                elif batch_strategy == "constant_liar_min":
                    imputed_y = float(cur_y.min())
                elif batch_strategy == "constant_liar_max":
                    imputed_y = float(cur_y.max())
                elif batch_strategy == "constant_liar_mean":
                    imputed_y = float(cur_y.mean())
                else:
                    raise ValueError(f"unknown batch strategy: {batch_strategy!r}")
            cur_x = np.vstack([cur_x, pick_x]) if len(cur_x) else pick_x
            cur_y = np.append(cur_y, imputed_y)

    result = {
        "schema_version": 1,
        "batch_size": batch_size,
        "batch_strategy": batch_strategy if model_ready else "random_initialization_no_model",
        "candidates": [p["candidate_id"] for p in proposals],
        "proposals": proposals,
        "candidate_id": proposals[0]["candidate_id"],
        "factors": proposals[0]["factors"],
        "method": proposals[0]["method"],
        "predicted_mean": proposals[0]["predicted_mean"],
        "predicted_std": proposals[0]["predicted_std"],
        "expected_improvement": proposals[0]["expected_improvement"],
        "seed": seed,
        "noise": noise,
        "n_observed": len(observations),
        "n_pending": len(pending_ids),
        "input_sha256": hashes,
        "note": ("Batch uses a GP acquisition with explicit fantasy values; fantasies are not observed readouts. Record proposals as pending before asking again."
                 if model_ready else "Insufficient observed readouts for a GP fit; every batch member was selected by seeded random initialization."),
    }
    if measurement_context is not None:
        result["measurement_context"] = measurement_context
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Propose unmeasured conditions from a reviewed candidate table")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--observations")
    parser.add_argument("--pending")
    parser.add_argument("--constraints", help="constraint bundle exported by cell-protocol-compiler; candidates are checked against its published windows")
    parser.add_argument("--batch-size", type=int, default=1, help="number of candidates to propose in batch")
    parser.add_argument(
        "--batch-strategy",
        choices=["kriging_believer", "constant_liar_min", "constant_liar_max", "constant_liar_mean"],
        default="kriging_believer",
        help="heuristic for batch acquisition",
    )
    parser.add_argument("--reserve-ledger", help="atomically reserve in this local SQLite ledger")
    parser.add_argument("--request-id", help="unique request key; reuse it to recover the same reservation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if bool(args.reserve_ledger) != bool(args.request_id):
        parser.error("--reserve-ledger and --request-id must be supplied together")
    if args.reserve_ledger and args.batch_size > 1:
        parser.error("batch_size > 1 with SQLite reservation ledger is not supported; use preview mode or reserve sequentially")
    try:
        if args.reserve_ledger:
            from medialoop.reservations import propose_and_reserve
            result = propose_and_reserve(args.candidates, args.observations, args.pending,
                                         ledger_path=args.reserve_ledger, request_id=args.request_id,
                                         seed=args.seed, noise=args.noise,
                                         constraints_path=args.constraints)
        else:
            result = propose(args.candidates, args.observations, args.pending, seed=args.seed, noise=args.noise,
                             batch_size=args.batch_size, batch_strategy=args.batch_strategy,
                             constraints_path=args.constraints)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        try:
            with out.open("x", encoding="utf-8") as handle:
                handle.write(text)
        except FileExistsError:
            # A successful reservation retry may safely reuse its exact export.
            # Partial/different exports still require a fresh destination.
            if not args.reserve_ledger or out.read_text(encoding="utf-8") != text:
                raise
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
