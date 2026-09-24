"""Run the simulation bake-off and print mean best scores."""

from __future__ import annotations

import argparse
import json
import sys

from medialoop.loop import random_search, run
from medialoop.surfaces import OBJECTIVES


def bakeoff(objective: str, seeds: int) -> dict:
    fn = OBJECTIVES[objective]
    ei = []
    rnd = []
    budget_init, steps = 8, 12
    for seed in range(seeds):
        ei.append(run(fn, budget_init, steps, seed=seed)["best_true"])
        rnd.append(random_search(fn, budget_init + steps, seed=10_000 + seed)["best_true"])
    return {
        "objective": objective,
        "seeds": seeds,
        "budget": budget_init + steps,
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
    args = parser.parse_args(argv)
    json.dump(bakeoff(args.objective, args.seeds), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
