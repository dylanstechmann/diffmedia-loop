"""Propose from a collaborator-supplied candidate list with an explicit ledger.

No surface oracle is called. Readouts and pending IDs come from the user.
The GP maximizes the supplied response and assumes a common noise scale.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from medialoop.gp import expected_improvement, predict
from medialoop.loop import inside_bounds
from medialoop.space import NAMES


def _read(path, required):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
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
    return rows


def propose(candidates_path, observations_path=None, pending_path=None, *, seed=0, noise=0.02):
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    rows = _read(candidates_path, ["candidate_id", *NAMES])
    if not rows:
        raise ValueError("empty candidate list")
    x = np.array([[float(row[name]) for name in NAMES] for row in rows])
    if not inside_bounds(x):
        raise ValueError("candidate factors must be finite and inside the simulation's encoded bounds")
    if len({tuple(row) for row in x}) != len(x):
        raise ValueError("duplicate factor combinations; give each condition one candidate_id")
    ids = [row["candidate_id"] for row in rows]
    lookup = {value: i for i, value in enumerate(ids)}
    observations = _read(observations_path, ["candidate_id", "response"]) if observations_path else []
    pending = _read(pending_path, ["candidate_id"]) if pending_path else []
    observed_ids = {row["candidate_id"] for row in observations}
    pending_ids = {row["candidate_id"] for row in pending}
    if (observed_ids | pending_ids) - set(ids):
        raise ValueError("ledger contains an unknown candidate_id")
    if observed_ids & pending_ids:
        raise ValueError("candidate cannot be both observed and pending")
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
    hashes = {name: None if path is None else hashlib.sha256(Path(path).read_bytes()).hexdigest()
              for name, path in [("candidates", candidates_path), ("observations", observations_path),
                                 ("pending", pending_path)]}
    return {"schema_version": 1, "candidate_id": ids[chosen], "factors": dict(zip(NAMES, x[chosen].tolist())),
            "method": method, "predicted_mean": mu, "predicted_std": sigma,
            "expected_improvement": acquisition, "seed": seed, "noise": noise,
            "n_observed": len(observations), "n_pending": len(pending), "input_sha256": hashes,
            "note": "Maximizes the supplied response. A proposal is not an executable protocol. Record it as pending before asking again."}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Propose one unmeasured condition from a reviewed candidate table")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--observations")
    parser.add_argument("--pending")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        result = propose(args.candidates, args.observations, args.pending, seed=args.seed, noise=args.noise)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
