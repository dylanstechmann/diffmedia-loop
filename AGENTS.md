# Agent instructions — diffmedia-loop

Work only in this repository. The cardiac and neural scores are cartoons on a
fixed box (CHIR99021, IWP2, SB431542, LDN-193189). Expected improvement beating
random on that cartoon is the result you can prove with no hood.

## Do not

- Move the synthetic peaks and then claim the search “found biology.”
- Propose factor values outside the encoded published windows.
- Turn a proposal into a dose for a person or animal.
- Delete the repeat-EI policy that is supposed to show batch-uniqueness violations.
- Edit the candidate CSV from inside a reservation transaction.

## First commands

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
PYTHONPATH=src python3 -m medialoop.cli --objective cardiac --seeds 4
PYTHONPATH=src python3 -m medialoop.bakeoff --objective cardiac --seeds 2
```

Use fewer seeds than the README for a smoke run. Do not replace the README's 12-seed numbers unless you re-ran 12 seeds.

## Improve, in this order

1. If you change the GP or acquisition, re-run the bake-off and report simple regret against the best **grid** point, plus uniqueness violations.
2. Allowed: one new policy behind the same columns (regret, violations). Do not remove EI, UCB, Thompson, or random.
3. A real assay hook must keep `medialoop-plan` from calling the synthetic `observe`. The planner prints a point; it does not plate a well.
4. Cross-check any suggested micromolar value against `cell-protocol-compiler` windows before writing it into a fixture. If they disagree, stop and report the disagreement.

## Done when

Tests pass and the README still says the surface is synthetic.
