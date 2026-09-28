"""Run the simulation bake-off and print mean best scores."""

from __future__ import annotations

import argparse
import json
import sys

from medialoop.loop import random_search, run
from medialoop.surfaces import OBJECTIVES


def bakeoff(
    objective: str,
    seeds: int,
    batch_size: int = 1,
    batch_strategy: str = "kriging_believer",
) -> dict:
    fn = OBJECTIVES[objective]
    ei = []
    rnd = []
    budget_init, steps = 8, 12
    total_evals = budget_init + steps * batch_size
    for seed in range(seeds):
        ei.append(run(fn, budget_init, steps, seed=seed, batch_size=batch_size, batch_strategy=batch_strategy)["best_true"])
        rnd.append(random_search(fn, total_evals, seed=10_000 + seed)["best_true"])
    return {
        "objective": objective,
        "seeds": seeds,
        "batch_size": batch_size,
        "batch_strategy": batch_strategy,
        "budget": total_evals,
        "mean_best_expected_improvement": round(sum(ei) / len(ei), 4),
        "mean_best_random": round(sum(rnd) / len(rnd), 4),
        "note": (
            "Scores are on a synthetic response surface, not cells. "
            "Replace the surface with assay readouts before proposing a real plate. "
            "Bounds are published in-vitro windows, not doses for a person."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Closed-loop media search bake-off")
    parser.add_argument("--objective", choices=sorted(OBJECTIVES), default="cardiac")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1, help="batch acquisition size per round (default: 1)")
    parser.add_argument(
        "--batch-strategy",
        choices=["kriging_believer", "constant_liar_min", "constant_liar_max", "constant_liar_mean"],
        default="kriging_believer",
        help="heuristic for multi-point batch acquisition",
    )
    args = parser.parse_args(argv)
    json.dump(bakeoff(args.objective, args.seeds, batch_size=args.batch_size, batch_strategy=args.batch_strategy), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
