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


def propose(candidates_path, observations_path=None, pending_path=None, *, seed=0, noise=0.02):
    """Preview one candidate without reserving it or modifying input files."""
    return _propose(*_inputs(candidates_path, observations_path, pending_path), seed=seed, noise=noise)


def _propose(rows, observations, pending, hashes, *, seed, noise, reserved_ids=()):
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
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
    available = [i for i, name in enumerate(ids) if name not in observed_ids | pending_ids]
    if not available:
        raise ValueError("no unevaluated, nonpending candidates remain")
    rng = np.random.default_rng(seed)
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
    return {"schema_version": 1, "candidate_id": ids[chosen], "factors": dict(zip(NAMES, x[chosen].tolist())),
            "method": method, "predicted_mean": mu, "predicted_std": sigma,
            "expected_improvement": acquisition, "seed": seed, "noise": noise,
            "n_observed": len(observations), "n_pending": len(pending_ids), "input_sha256": hashes,
            "note": "Maximizes the supplied response. A proposal is not an executable protocol. Record it as pending before asking again."}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Propose one unmeasured condition from a reviewed candidate table")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--observations")
    parser.add_argument("--pending")
    parser.add_argument("--reserve-ledger", help="atomically reserve in this local SQLite ledger")
    parser.add_argument("--request-id", help="unique request key; reuse it to recover the same reservation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if bool(args.reserve_ledger) != bool(args.request_id):
        parser.error("--reserve-ledger and --request-id must be supplied together")
    try:
        if args.reserve_ledger:
            from medialoop.reservations import propose_and_reserve
            result = propose_and_reserve(args.candidates, args.observations, args.pending,
                                         ledger_path=args.reserve_ledger, request_id=args.request_id,
                                         seed=args.seed, noise=args.noise)
        else:
            result = propose(args.candidates, args.observations, args.pending, seed=args.seed, noise=args.noise)
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
