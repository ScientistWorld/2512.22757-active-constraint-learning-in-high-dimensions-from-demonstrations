# Evaluation

A submission is a JSON command manifest at `outputs/<name>.json`:

```json
{"protocol":"acl-v2","command":["python","my_solver.py"]}
```

Run `bash scripts/evaluate.sh` for development feedback and `bash scripts/evaluate_test.sh` for the private test. Both discover every manifest in `outputs/`; the identifier is its filename. The test requires no arguments and does not select methods through train scores. `random_baseline` is reserved for the evaluator's immutable control. The packaged `scripts/method.sh` registers the reproduced reference.

Each command handles one instance using newline-delimited JSON on stdin/stdout. It first receives bounds, dimension, horizon, time step, a fixed random seed, four initial demonstrations, a total limit of twelve demonstrations, and a limit of forty additional requests. Each demonstration contains positions, velocities and controls. Request another demonstration with `{"op":"query","start":[x,y],"goal":[x,y]}`. The response contains `demo` (or null for an infeasible/failed request) and `requests_remaining`. The evaluator records every request; only it generates demonstrations. Requesting more than the allowed number fails the run.

Send `{"op":"finish"}` when acquisition is complete. The evaluator then sends `{"points":[[x,y],...]}` in shuffled order. Return `{"predictions":[0,1,...]}`, where 1 means safe and 0 means unsafe, then exit. The number of predictions must match exactly. No more oracle queries are allowed after receiving these points. Truth, settings, flags, and written scores are never accepted as evidence. An optional `n_demos` declaration must equal the evaluator's ledger; it earns no reward.

Two experiments each contain twelve independent obstacles. Test and development draw different obstacles and different coordinates from the same distributions. Every method in a split sees the same initial trajectories, obstacles and query coordinates. Private parameters, answer generators, raw reference records and baseline records live in `eval/test/`. Development owns independent implementations in `eval/train/`. `data/constraints.py` supplies legacy toy examples and is not a test dependency.

Accuracy is the fraction correct on 4,096 uniformly sampled states per obstacle. Balanced accuracy, safe recall and unsafe recall use 8,192 independent states within 0.10 coordinate units of each boundary. Near-boundary balanced accuracy restricts that band to distance at most 0.035. Coordinates are expressed in units of five paper position units. Test coordinates are never presented in signed-distance or label order.

A protocol violation, nonbinary/nonfinite/wrong-sized output, excess query budget, or safe/unsafe boundary recall below 0.60 on either experiment fails the whole run. Failed runs receive zero scored metrics and are ordered below all successful runs; raw diagnostics explain failures. Demonstration count is a hard constraint, not a self-reported reward.

For successful methods, rank is the sum over metrics of experiment weight × coefficient magnitude × improvement over the frozen reference in the metric's better direction, divided by that metric's population standard deviation across the successful attempts being ranked and the reference. Zero-variance terms contribute zero. The evaluator publishes the scales and failures with its ranking. Rescoring another cohort recomputes the scales. Baseline and reference comparisons are recomputed from fixed raw outputs, never from candidate-written scores.

The test performs no offline training and never retrains the baseline. It runs each submitted active learner on unseen instances; the learner's per-instance adaptation is part of solving the task. Oracle trajectories and raw method responses are cached privately, with source version checks, so evaluator-only checks can rescore them. `scripts/reproduce.sh` reruns both reference and random control end to end before freezing their records. Do not invoke its packager-only bootstrap operation when submitting an alternative.
