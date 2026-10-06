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

# Run simulation with batch acquisition
PYTHONPATH=src python3 -m medialoop.cli --objective cardiac --batch-size 4 --batch-strategy kriging_believer
```

Python 3.10+ and numpy.

### Batch acquisition (Kriging Believer and Constant Liar)

To suggest multiple conditions per round, use `--batch-size N` and
`--batch-strategy` with `kriging_believer`, `constant_liar_min`,
`constant_liar_max`, or `constant_liar_mean` on `medialoop` or
`medialoop-plan`. Kriging Believer uses the GP posterior mean as a temporary
readout for each intermediate candidate. Constant Liar uses the minimum,
maximum, or mean of observed responses as that temporary readout.

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
  and retain their raw data elsewhere. Ordinary observations use a common noise
  scale; validated measurement imports can provide per-observation assay noise.
- With fewer than two observations the selection is random and reproducible.
  Otherwise it uses expected improvement with fixed GP hyperparameters.
- Without a reservation ledger, record the proposal in the pending table
  **before** asking again. This preview mode does not reserve conditions or
  coordinate concurrent users.
- On completion, remove the pending ID and add its observed response.

Unknown IDs, duplicate conditions, nonfinite values, out-of-box factors and an
exhausted candidate set are rejected. Output files are created exclusively.
`--noise` is an assumed response standard deviation, not an estimated noise
model. For imported observations with measured assay uncertainty, the GP uses
the larger of that uncertainty and `--noise` for each observation. Otherwise it
uses the common `--noise` scale. The fixed GP is a baseline and its uncertainty
is not calibrated on cells.

### Import well-level assay measurements

Use `medialoop-import-measurements` to validate a raw well table and produce a
planner-compatible mean table plus an audit report. It does not call the
synthetic surface or treat a proposed candidate as an executable protocol.

```bash
medialoop-import-measurements \
  --candidates examples/candidates.csv \
  --measurements examples/measurements.example.csv \
  --direction maximize \
  --out-dir artifacts/measurement-import

medialoop-plan --candidates examples/candidates.csv \
  --observations artifacts/measurement-import/observations.csv \
  --noise 0.05 --out artifacts/proposal.json
```

The checked-in measurement file is an explicitly synthetic format example.
Raw tables require `candidate_id`, `plate_id`, `well_id`, `batch_id`,
`biological_unit_id`, `technical_replicate_id`, `assay_id`, `endpoint`, `unit`,
`value`, `well_measurement_standard_uncertainty`, `status`, and `failure_reason`.
Each import accepts exactly one assay ID, endpoint, and unit. Measured rows need
a finite value and blank failure reason. Failed rows need a reason and blank
value; they remain in `measurement_report.json` and are not imputed. The
well-level standard uncertainty is either supplied for every measured well in
the import or left blank for all of them. When supplied, the importer propagates
it through technical and biological means under an independence assumption;
shared calibration uncertainty is not included.

Technical replicates are averaged within each `biological_unit_id`, then
biological units receive equal weight. The report keeps biological-unit SD/SEM
separate from the propagated assay measurement uncertainty. Set `--direction`
to the endpoint's predeclared objective: the planner maximizes a `maximize`
value as recorded and negates a `minimize` value, with both raw mean and
transform recorded. Use a separate import for each endpoint; this planner does
not define a combined rejuvenation score.

The proposal carries the assay, endpoint, units, raw-measurement hash, candidate
hash, aggregation rule, failed-well counts, replicate summaries, and imported
uncertainty. When assay measurement uncertainty is available, the GP uses the
larger of that value and the assumed `--noise` floor for each observed
condition. The floor is not combined in quadrature because its sources are not
decomposed. Biological-unit SEM remains provenance only; reused biological
units across conditions, batch effects, shared calibration uncertainty, and
assay covariance are not modeled. A reported SEM is not a calibrated GP noise
value. Review those limits before using proposals to plan another experiment.

## Reserve a candidate atomically

Use one shared **local** SQLite ledger for all callers that need reservations:

```bash
medialoop-plan --candidates examples/candidates.csv \
  --observations examples/observations.csv --pending examples/pending.csv \
  --reserve-ledger artifacts/reservations.sqlite3 --request-id round-001 \
  --seed 0 --out artifacts/round-001.json
```

Selection and reservation commit in a single transaction before the proposal
is returned. `--batch-size` reserves every selected condition atomically under
the same request ID; if any insert fails, the whole batch rolls back.
Concurrent processes using that ledger cannot reserve the same condition under
different request IDs. Existing observed/pending CSV entries are also
excluded. The command reads those CSVs and writes the ledger and proposal JSON;
it never edits the CSVs. Input hashes describe the exact bytes parsed for the
original proposal.

Choose a new `--request-id` for each new request. Retry with the **same** ID,
candidate table, seed, noise, batch size and strategy to recover its original
proposal, even after observations change. This also recovers a committed reservation after output
export fails or the caller loses its connection. Retry with a new output path
if the previous export is partial or contains different text; an existing exact
export is accepted. A retry returns a historical proposal and does not reserve
another condition. The Python equivalent is
`medialoop.reservations.propose_and_reserve(..., ledger_path=..., request_id=...)`;
`medialoop.planner.propose(...)` remains a read-only preview.

The first successful reservation binds the ledger to the candidate CSV's exact
SHA-256. Keep that table unchanged. Reservations remain recorded after their
responses are added to the observations CSV, so completed conditions cannot
be issued again. There is intentionally no release/reissue command. Do not
delete, replace or copy the ledger to start another round of the same campaign;
doing so loses coordination. If a condition was also listed in a manual pending
CSV, remove that entry when adding its response to observations.

SQLite releases locks and rolls back uncommitted writes when a process exits.
Callers wait up to 30 seconds for a writer before returning an error; retry with
the same request ID. Use a local filesystem with reliable SQLite locking, not
network shares or cloud-synced copies. All reserving callers must use this API
and ledger. Manual CSV updates are outside the transaction: pause submissions
while updating observations/pending files, then resume with new request IDs.
Preview mode can still display already-reserved conditions because it does not
read the ledger. Store the ledger alongside its recovery journal files outside
Git and use SQLite-aware backups while it is active.

## Shared published-window constraints (v0.4)

The published factor windows are not retyped here. `examples/planner-constraints.json`
is a bundle exported by
[cell-protocol-compiler](https://github.com/dylanstechmann/cell-protocol-compiler)
(`protocolcompiler <id> --constraints`) for `giwi_cardiac` and `dual_smad_neural`,
carrying each source record's `protocol_sha256`. Pass it to any planning run:

```bash
medialoop-plan --candidates examples/candidates-constrained.csv \
  --observations examples/observations-constrained.csv \
  --pending examples/pending-constrained.csv \
  --constraints examples/planner-constraints.json \
  --seed 0 --out artifacts/proposal.json
```

- `examples/candidates-constrained.csv` holds conditions inside the
  exported windows; the older `candidates.csv` zero-dose synthetic fixtures
  are deliberately rejected by this flag.
- Every candidate factor column named in the bundle is checked against its
  published window; a value outside it is an error, not a clip. The
  simulation box may be wider than a published window (it includes
  zero-dose corners), so this catches conditions the box alone would allow.
- Mutually exclusive alternatives are enforced as data: a candidate with
  both `Noggin_ng_per_mL` and `LDN193189_nM` positive is rejected, matching
  the compiler's dual-SMAD rule, and at least one must be positive. Required
  parameter columns must be present; an omitted alternative no longer bypasses
  validation. Zero is an explicit inactive value for either alternative.
- Proposals record the bundle's SHA-256 and the source protocol hashes under
  `constraints`, so any result names the exact windows that bounded its
  search. The SQLite reservation path enforces the same bundle.
- The unit suite loads the checked-in bundle and compares it with the
  built-in box: no simulation factor may lack a published window, and no
  published window may exceed the box. Regenerate the bundle after any
  window changes in the compiler; the sync test fails first.

The bundle states limits a search may use. It is not an optimum, and it is
not a dose for a person or a cell line you have not tested.
The checker validates factor fields only. It does not validate culture timing,
cell-line suitability, required QC gates, or the full protocol narrative. The
bundled cartoon axes do not describe a combined wet-lab protocol.

## Acquisition bake-off (v0.3)

`medialoop-bakeoff` compares the current planner (expected improvement) with
random search, fixed-kappa UCB, and one-draw Thompson sampling on a **fixed
grid** inside the published windows. A fifth policy, repeat expected
improvement, re-proposes the same candidate inside a batch so violations are
visible. Guarded policies fantasize at the posterior mean within a batch
(kriging believer) and call the cartoon only once per chosen grid point.

```bash
PYTHONPATH=src python3 -m medialoop.bakeoff --objective cardiac --seeds 6
```

Report simple regret against the best point **on that grid**, cumulative
regret, and batch-uniqueness violations. This is still the synthetic surface.
It is not a dose, and it is not a reason to plate a well. A neural policy is
intentionally absent: a course project can add one behind the same regret
and violation columns without replacing the expected-improvement baseline.

Two shifted objectives, `--objective cardiac_shifted` and `neural_shifted`,
move the cartoon peak to a different on-grid location (6/2 µM CHIR/IWP2 and
5 µM / 250 nM SB431542/LDN). They check that the planner follows the surface
rather than the original peak's location. `--noise` sets the readout noise
scale and is reported as `readout_noise`; the previous fixed 0.02 remains the
default.


Measurement imports now retain `source_candidates.csv` and
`source_measurements.csv` as the exact byte snapshots that were parsed, including
failed wells. Their hashes and byte counts travel in the aggregation report.
The planner verifies these snapshots for new imports, so changed or missing raw
inputs cannot silently retain an apparently intact provenance trail. Existing
reports without the additive snapshot metadata remain readable.
