# Evaluation

## Submitting a method

A submission is a JSON command manifest at `outputs/<name>.json`:

```json
{"protocol":"acl-v2","command":["python","my_solver.py"]}
```

- `bash scripts/evaluate.sh` gives development feedback.
- `bash scripts/evaluate_test.sh` runs the private test. It takes no arguments.
- Both discover every manifest in `outputs/`; a method's identifier is its filename.
- The test does not select methods through train scores.
- `random_baseline` is reserved for the evaluator's immutable control.
- The packaged `scripts/method.sh` registers the reproduced reference.

## Protocol

Each command handles one instance, using newline-delimited JSON on stdin/stdout.

**1. The first message** contains:
- `system`: `double_integrator`, `unicycle` or `quadcopter`;
- `dimension` (2 or 3), `state_dimension` and `control_dimension`;
- `model` (time step; quadrotor mass, inertia and gravity);
- `bounds`, `horizon`, `dt` and a fixed `seed`;
- four initial demonstrations (`initial_demos`);
- the total demonstration `budget` (twelve) and the additional-request `query_limit` (forty).

**2. Demonstrations.** Each demonstration has:
- `positions` (horizon × dimension);
- `controls` (horizon−1 rows);
- either `velocities` (double integrator) or full `states` (unicycle, quadrotor).

The dynamics are listed in `briefing/problem.md`; states and controls satisfy them exactly.

**3. Requests.** Request a demonstration with `{"op":"query","start":[...],"goal":[...]}`, giving positions of length `dimension` inside the bounds.
- The response contains `demo` (null for an infeasible or failed request) and `requests_remaining`.
- The evaluator records every request, and only it generates demonstrations.
- Exceeding the demonstration budget or the request limit fails the run.

**4. Finishing.** Send `{"op":"finish"}` when acquisition is complete.
- The evaluator then sends `{"points":[...]}` in shuffled order.
- Return `{"predictions":[0,1,...]}`, where 1 means safe and 0 means unsafe, then exit. The number of predictions must match exactly.
- No more requests are allowed after the points arrive.
- Only `predictions` and an optional `n_demos` are allowed in the final response. Truth, settings, comparison fields, flags and written scores are rejected.
- An optional `n_demos` must equal the evaluator's ledger; it earns no reward.

## Instances

Hidden test instances per experiment:

| Experiment | Hidden test | Development |
|---|---:|---:|
| `double_integrator_g3` | 72 | 24 |
| `complex_boundary_stress` | 72 | 24 |
| `unicycle_paper_constraints` | 72 | 24 |
| `quadcopter_hourglass` | 96 | 24 |

- Test and development draw different obstacles and coordinates from distributions of matching difficulty.
- Obstacle shapes carry smooth per-instance deformations, and the deformation family differs between development and test (matched in size). Neither the exact form of the development generators nor a published obstacle formula answers test instances.
- Every method in a split sees the same initial trajectories, obstacles and evaluation states.
- Private parameters, answer generators, raw reference records and baseline records live in `eval/test/`. Development owns independent implementations in `eval/train/`.
- `data/constraints.py` supplies legacy toy examples and is not a test dependency.

## Metrics

**Accuracy** is the fraction of uniformly sampled states in the workspace classified correctly:
- 4,096 states per 2D instance;
- 8,192 states per 3D instance.

**The boundary band** has 8,192 independent states per instance. They are placed within 0.5 position units of the boundary (2D) or 0.35 units (3D). The band is scored by:
- **balanced accuracy**;
- **safe recall** and **unsafe recall**;
- **near-boundary balanced accuracy**, which restricts the band to distance at most 0.175 (2D) or 0.125 (3D).

Test coordinates are never presented in signed-distance or label order.

## Failure

Any of these fails the whole run:
- a protocol violation;
- nonbinary, nonfinite or wrong-sized output;
- an excess request or demonstration;
- on any experiment, mean boundary balanced accuracy below 0.55, or a mean class recall below 0.10.

A failed run receives zero in every scored metric and is ordered below all successful runs. Raw diagnostics explain failures.

## Ranking

For successful methods, rank is a sum over experiments and metrics. Each term is:
- the experiment weight;
- times the metric coefficient's magnitude;
- times the improvement over the frozen reference in the metric's better direction;
- divided by that metric's population standard deviation across the successful attempts being ranked.

Zero-variance terms contribute zero. Experiment weights are: unicycle 0.35, quadrotor 0.25, double-integrator polynomial obstacle 0.2, complex boundaries 0.2. Metric coefficients are: accuracy 1.0, balanced accuracy 0.5, near-boundary balanced accuracy 0.6, safe recall 0.25, unsafe recall 0.25.

## What the test runs

- The test performs no offline training and never retrains the baseline. It runs each submitted active learner on unseen instances; the learner's per-instance adaptation is part of solving the task.
- Oracle trajectories, evaluation states and raw method responses are cached privately, with source version checks.
- `scripts/reproduce.sh` reruns both the reference and the random control end to end before freezing their records. Do not invoke its packager-only bootstrap operation when submitting an alternative.
