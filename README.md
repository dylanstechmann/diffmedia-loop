# Diffmedia loop

A closed-loop planner for **in-vitro** differentiation-media factors.

It searches a box taken from published windows (CHIR99021, IWP2, SB431542, LDN-193189), fits a small Gaussian process, and proposes the next point by expected improvement. The response it optimizes in this repository is a cartoon: Wnt on then Wnt off scores as "cardiac"; dual SMAD inhibition scores as "neural"; each punishes the other. That is not a differentiation dataset. It is a bake-off of the planner against random search at the same budget, which is the thing you can prove with no hood.

On 12 seeds and a budget of 20 evaluations, mean best score was about **0.74 vs 0.24** (cardiac cartoon) and **0.74 vs 0.16** (neural cartoon). Replace `observe` with a real column (qPCR, a troponin fraction, a score from [brightfield-colony-qc](https://github.com/dylanstechmann/brightfield-colony-qc)) before anyone plates the suggestion.

## Non-goals

- Doses for a person, an animal, or a "cycle"
- A claim that 8 µM CHIR and 4 µM IWP2 is the right cardiac protocol. Those are the peaks of the **synthetic** surface, on purpose, so the test can see whether search finds them. [cell-protocol-compiler](https://github.com/dylanstechmann/cell-protocol-compiler) is where the published GiWi and dual-SMAD checklists live, including the warning that CHIR is line-dependent.
- A foundation model of a cell. Four factors and a GP.

## Run

```bash
make test
PYTHONPATH=src python3 -m medialoop.cli --objective cardiac --seeds 12
PYTHONPATH=src python3 -m medialoop.cli --objective neural --seeds 12
```

Python 3.10+ and numpy.

## Hooking a real assay later

`medialoop.loop.run` takes any function from a factor dict to a float in roughly \[0, 1\]. A wet-lab round is the same function with a human in it: the planner prints a point inside the published box, somebody runs that well, the number comes back, the GP updates. The box is a safety rail for the cartoon, not a substitute for a protocol range check.

## License

MIT.

## Plan from recorded measurements (v0.2)

`medialoop-plan` reads a collaborator-reviewed candidate table, completed
observations and pending IDs. It proposes one unused candidate, records input
hashes and never calls the synthetic response function. Install with
`python -m pip install -e .`.

```bash
medialoop-plan --candidates examples/candidates.csv \
  --observations examples/observations.csv --pending examples/pending.csv \
  --seed 0 --out artifacts/proposal.json
```

These example IDs, factor combinations and readouts are **synthetic software
fixtures**, not an experimental design for a cell line. A paper reporting each
factor separately does not validate their Cartesian product or timing.

- Candidates: unique `candidate_id` plus all four factor columns in `space.py`.
- Observations: `candidate_id,response`; the finite response is maximized.
- Pending: `candidate_id`; completed and pending IDs cannot overlap.
- A candidate can appear once in observations. Aggregate replicates explicitly
  and retain their raw data elsewhere; this GP assumes a common noise scale.
- With fewer than two observations the selection is random and reproducible.
  Otherwise it uses expected improvement with fixed GP hyperparameters.
- Record the proposal in the pending table **before** asking again. The planner
  is stateless and does not reserve conditions or coordinate concurrent users.
- On completion, remove the pending ID and add its observed response.

Unknown IDs, duplicate conditions, nonfinite values, out-of-box factors and an
exhausted candidate set are rejected. Output files are created exclusively.
`--noise` is an assumed response standard deviation, not an estimated noise
model. The fixed GP is a baseline and its uncertainty is not calibrated on cells.
