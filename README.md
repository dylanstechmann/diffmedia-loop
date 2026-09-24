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
