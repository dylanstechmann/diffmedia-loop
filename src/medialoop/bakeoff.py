"""Fixed-grid acquisition bake-off on the synthetic media surface.

The current planner is expected improvement. This module compares it with
random search, fixed-kappa UCB, and one-draw Thompson sampling. A repeat
policy deliberately re-proposes the same point inside a batch so constraint
violations are countable. None of these scores are cell readouts.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from medialoop.gp import expected_improvement, predict, thompson_draw, upper_confidence_bound
from medialoop.loop import _row, inside_bounds
from medialoop.space import FACTORS
from medialoop.surfaces import OBJECTIVES

POLICIES = (
    "random",
    "expected_improvement",
    "ucb",
    "thompson",
    "repeat_expected_improvement",
    "penalized_expected_improvement",
)

# Batch-aware policy constants, fixed before the first run: a pick multiplies each later candidate's
# expected improvement by 1 - exp(-d^2 / (2 * LENGTH^2)), with d the Euclidean distance in
# window-scaled (0 to 1) coordinates. LENGTH is a stated choice, not tuned on the bake-off.
PENALTY_LENGTH = 0.25


def candidate_grid() -> np.ndarray:
    """Published-window lattice. The cardiac cartoon peak (8 µM, 4 µM, 0, 0) is on it."""
    chir = (0.0, 3.0, 6.0, 8.0, 12.0)
    iwp = (0.0, 2.0, 4.0, 5.0)
    sb = (0.0, 5.0, 10.0)
    ldn = (0.0, 100.0, 250.0)
    rows = [(a, b, c, d) for a in chir for b in iwp for c in sb for d in ldn]
    return np.asarray(rows, dtype=float)


def batch_penalty(grid_pool: np.ndarray, picked: np.ndarray) -> np.ndarray:
    """Multiplicative penalty in [0, 1] that is zero at an already picked point and rises with distance."""
    if len(picked) == 0:
        return np.ones(len(grid_pool))
    lows = np.array([item["low"] for item in FACTORS], dtype=float)
    spans = np.array([item["high"] - item["low"] for item in FACTORS], dtype=float)
    scaled_pool = (grid_pool - lows) / spans
    scaled_picked = (picked - lows) / spans
    distance_sq = ((scaled_pool[:, None, :] - scaled_picked[None, :, :]) ** 2).sum(axis=2)
    return np.prod(1.0 - np.exp(-distance_sq / (2.0 * PENALTY_LENGTH ** 2)), axis=1)


def _scores(policy, mu, sigma, fantasy_y, rng, kappa):
    if policy in ("expected_improvement", "repeat_expected_improvement", "penalized_expected_improvement"):
        return expected_improvement(mu, sigma, float(np.max(fantasy_y)))
    if policy == "ucb":
        return upper_confidence_bound(mu, sigma, kappa)
    if policy == "thompson":
        return thompson_draw(mu, sigma, rng)
    if policy == "random":
        return rng.random(len(mu))
    raise ValueError(f"unknown policy: {policy}")


def campaign(
    objective,
    policy: str,
    *,
    seed: int,
    n_init: int = 4,
    rounds: int = 4,
    batch_size: int = 2,
    noise: float = 0.02,
    kappa: float = 1.5,
) -> dict:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    if n_init < 2 or rounds < 1 or batch_size < 1:
        raise ValueError("need n_init >= 2, rounds >= 1, batch_size >= 1")
    if not np.isfinite(noise) or noise < 0 or not np.isfinite(kappa) or kappa < 0:
        raise ValueError("noise and kappa must be finite and nonnegative")
    grid = candidate_grid()
    if len(grid) <= n_init + rounds:
        raise ValueError("grid is smaller than the evaluation budget")
    truth = np.array([objective(_row(row)) for row in grid])
    oracle = float(truth.max())
    rng = np.random.default_rng(seed)
    observed = list(map(int, rng.choice(len(grid), size=n_init, replace=False)))
    y_obs = truth[observed] + rng.normal(0.0, noise, size=n_init)
    violations = 0
    regrets = []

    def mark() -> None:
        best = float(truth[list(dict.fromkeys(observed))].max())
        regrets.append(oracle - best)

    mark()
    for _ in range(rounds):
        picks: list[int] = []
        fantasy_x = grid[observed]
        fantasy_y = np.array(y_obs, dtype=float)
        for _step in range(batch_size):
            if policy == "repeat_expected_improvement":
                pool = [i for i in range(len(grid)) if i not in observed]
            else:
                pool = [i for i in range(len(grid)) if i not in observed and i not in picks]
            if not pool:
                break
            if policy == "random":
                choice = int(rng.choice(pool))
            else:
                mu, sigma = predict(fantasy_x, fantasy_y, grid[pool], noise=noise)
                score = _scores(policy, mu, sigma, fantasy_y, rng, kappa)
                if policy == "penalized_expected_improvement":
                    score = score * batch_penalty(grid[pool], grid[picks])
                tied = np.flatnonzero(np.isclose(score, score.max(), rtol=1e-10, atol=1e-15))
                # The repeat policy always re-selects the lowest index so a violation is forced.
                local = int(tied[0] if policy == "repeat_expected_improvement" else rng.choice(tied))
                choice = pool[local]
            if choice in observed or choice in picks:
                violations += 1
            picks.append(choice)
            if policy not in ("repeat_expected_improvement", "penalized_expected_improvement"):
                mu_c, _sigma_c = predict(fantasy_x, fantasy_y, grid[[choice]], noise=noise)
                fantasy_y = np.append(fantasy_y, float(mu_c[0]))
                fantasy_x = np.vstack([fantasy_x, grid[choice]])
        for choice in picks:
            y_obs = np.append(y_obs, truth[choice] + float(rng.normal(0.0, noise)))
            observed.append(choice)
            mark()
    chosen = grid[observed]
    if not inside_bounds(chosen):
        violations += int(len(observed))
    unique = list(dict.fromkeys(observed))
    return {
        "policy": policy,
        "best_true": float(truth[unique].max()),
        "oracle_max_on_grid": oracle,
        "final_simple_regret": regrets[-1],
        "cumulative_regret": float(sum(regrets)),
        "simple_regret": [float(value) for value in regrets],
        "violations": int(violations),
        "n_evaluations": len(observed),
        "n_unique": len(unique),
        "seed": seed,
        "batch_size": batch_size,
        "rounds": rounds,
        "n_init": n_init,
        "note": (
            "Synthetic surface. Simple regret is the gap to the best point on this fixed grid, "
            "not a biological optimum. Repeat expected improvement is a negative control that "
            "re-proposes the same candidate inside a batch."
        ),
    }


def summarize(objective: str, seeds: int, **kwargs) -> dict:
    fn = OBJECTIVES[objective]
    by_policy = {}
    for policy in POLICIES:
        runs = [campaign(fn, policy, seed=seed, **kwargs) for seed in range(seeds)]
        by_policy[policy] = {
            "mean_best_true": round(sum(item["best_true"] for item in runs) / seeds, 4),
            "mean_final_simple_regret": round(sum(item["final_simple_regret"] for item in runs) / seeds, 4),
            "mean_cumulative_regret": round(sum(item["cumulative_regret"] for item in runs) / seeds, 4),
            "total_violations": int(sum(item["violations"] for item in runs)),
            "mean_unique": round(sum(item["n_unique"] for item in runs) / seeds, 2),
        }
    return {
        "objective": objective,
        "seeds": seeds,
        "grid_size": int(len(candidate_grid())),
        "readout_noise": float(kwargs.get("noise", 0.02)),
        "policies": by_policy,
        "current_planner": "expected_improvement",
        "note": (
            "Scores are on a synthetic response surface, not cells. "
            "Bounds are published in-vitro windows, not doses for a person. "
            "Do not plate a proposal from this bake-off."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Acquisition bake-off on the synthetic media surface")
    parser.add_argument("--objective", choices=sorted(OBJECTIVES), default="cardiac")
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--init", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.02,
                        help="readout noise scale added to the synthetic objective")
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("seeds must be positive")
    json.dump(
        summarize(args.objective, args.seeds, n_init=args.init, rounds=args.rounds,
                  batch_size=args.batch, noise=args.noise),
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
