# Hardening

Final iteration. `REVIEW.md` found that only the 2D double integrator was scored. It asked for scored experiments on the paper's 4D unicycle constraints (App. B.1) and the 12D quadcopter, with the evaluator solving expert demonstrations on hidden instances. It also asked for a check that the reference beats random beyond paired noise there, and to say plainly where it does not.

Both experiments were added. The reference and the random control were rerun end to end from the final code on all four experiments; `scoring/scores_test.json` from that run is the new baseline.

- Comparisons are paired: every method sees the same instances, initial demonstrations and evaluation states.
- Intervals are 95% paired bootstrap intervals over instances.
- Ranks follow the evaluator's rule: weight × |coefficient| × better-direction improvement over the reference, divided by the metric's population standard deviation across a cohort.
  - **Core cohort**: reference, random control and the honest variants.
  - **Core+1**: the core cohort plus the one extra member being reported (shortcuts, the alternative zero prior).

**Evidence files:**
- `eval/test/hardening_summary.json`: final audit, hidden test and development.
- `eval/test/hardening_rms035_summary.json`: the same audit with the first 3D deformation (RMS 0.35).
- `eval/test/hardening_two_experiment_summary.json`: the previous iteration's final audit ("before").

## What changed

### 1. Two new scored experiments (the paper's main-text settings)

| | `unicycle_paper_constraints` | `quadcopter_hourglass` |
|---|---|---|
| Dynamics | 4D unicycle (px, py, θ, v), controls (ω, a) | 12D quadrotor (position, world velocity, ZYX Euler angles, body rates), controls (thrust, 3 torques), Sabatino's mass and inertia |
| Δt, T | 1, 30 (paper's default) | 1, 20 (paper) |
| Constraints | App. B.1 g1, g2, g3, cycled over instances | App. B.1 hourglass g5 |
| Workspace | [-15,15]² | [-6,6]³ |
| Hidden test / development | 72 / 24 | 96 / 24 |
| Uniform states per instance | 4,096 (paper: 2,500) | 8,192 (paper: 27,000 in total) |
| Boundary band / near-boundary | 0.5 / 0.175 | 0.35 / 0.125 |

- **Hidden instances.** Each is a private rotation, scale (0.88–1.08), translation and smooth displacement-field deformation of the paper's formula, with parameters only in `eval/test/system_tasks.py`. The test field is Gaussian bumps; development (`eval/train/system_tasks.py`, independent code and seeds) uses a sinusoidal field of matched size.
  - Measured RMS displacement on the boundary band: unicycle 0.27 (test) vs 0.32 (development); quadrotor 0.52 vs 0.52.
- **Expert (evaluator-owned).** It solves Prob. 1 exactly in position space, then lifts each path to the system's full state/control trajectory. The reduction is exact: with unbounded controls and free non-position endpoint states, the dynamics place no restriction on positions.
  - Warm start: a shortest grid path around the margin-inflated obstacle; then SLSQP on the discrete Prob. 1.
  - Job checks (`eval/test/check_system_oracle.py`) confirm feasibility, Prob. 1 stationarity residual ≤ 2.5e-6, and lifted dynamics exact to 4e-15 against independently written forward models.
- **Budget and initial demonstrations.** Same as before: four shared initial crossings, twelve demonstrations, forty requests. The 3D crossings are the cube's body diagonals.
- **Metrics and gate.** Same five metrics and the same success gate as the 2D experiments.

### 2. The reference now runs the paper's computational steps in these systems

- **Problems 7–8** (`method/kkt.py`) are solved as linear programs over the full KKT system of each system's dynamics:
  - Control and non-position-state stationarity rows are free of the unknown constraint, so they must hold with the dynamics and endpoint multipliers alone.
  - The L1 residual of a position row certifies tightness.
  - P8 gives the KKT-compatible gradient.
  - Jacobians are analytic, or complex-step for the quadrotor.
- **GP, posterior draws and Problems 3–6** (`method/gp_acl.py`) are dimension-general.
  - Problem 4 in 3D searches the plane H (Eq. 8) over 36 in-plane directions; in 2D H is a line.
  - Everything else is unchanged: tied prior mean, noise floor 1e-4, P = 4, α = 0.3, 1,000 RFF, β = 0.55/0.3/0.55, δ = 0.001, η = 1.
- **The 2D experiments** use the same general KKT code. Their results reproduce the previous iteration's (reference on test: g3 accuracy 0.9792 vs 0.9791 before).

### 3. Weights

Unicycle 0.35, quadrotor 0.25, double-integrator g3 0.2, complex boundaries 0.2 (before: g3 0.6, complex 0.4).

The paper's main text and its headline claim ("high dimensions") are the unicycle and quadrotor results; the double integrator is in its appendix. The complex-boundary task keeps weight as the setting where the method falls short. Metric coefficients are unchanged.

### 4. Evaluator robustness (no score changes)

The first final job stalled (see Jobs). Changes:
- the producer's stderr is drained concurrently, so a chatty method can never block on a full pipe;
- worker processes are spawned, not forked from a parent holding all records;
- frozen and run records are compact;
- checkpoints are throttled;
- a private per-instance cache holds each instance's evaluation states and masks.

### 5. Private 3D deformation enlarged after the shortcut audit

See Shortcuts.

## Paper direction: reference vs random, same demonstration budget

Hidden test, random − reference (negative means the reference is better):

| Experiment | Uniform accuracy | Boundary balanced | Near-boundary balanced |
|---|---|---|---|
| Unicycle g1–g3 (72) | **+0.0001 [−0.0058, +0.0058]** | −0.067 [−0.088, −0.045] | −0.085 [−0.107, −0.064] |
| — g1 (24) | **+0.0106 [+0.0002, +0.0197]** | | |
| — g2 (24) | −0.0097 [−0.0189, −0.0008] | | |
| — g3 (24) | −0.0007 [−0.0097, +0.0091] | | |
| Quadrotor g5 (96) | −0.0030 [−0.0053, −0.0006] | −0.033 [−0.044, −0.023] | −0.042 [−0.052, −0.033] |
| Double integrator g3 (72) | −0.0017 [−0.0061, +0.0031] | −0.023 [−0.036, −0.009] | −0.028 [−0.039, −0.018] |
| Complex boundaries (72) | −0.0028 [−0.0052, −0.0003] | −0.043 [−0.062, −0.025] | −0.050 [−0.067, −0.032] |

Development, random − reference uniform accuracy:
- unicycle −0.0134 [−0.0221, −0.0040], beyond noise;
- quadrotor −0.0010 [−0.0049, +0.0030];
- boundary metrics favour the reference beyond noise in every experiment.

**Stated plainly:**
- **Quadrotor.** The reference beats random beyond paired noise on every scored metric.
- **Unicycle.**
  - The reference beats random beyond noise on the boundary metrics, by about 8 points near the boundary.
  - On uniform accuracy (the paper's metric) it does not: the two tie.
  - Per constraint, the reference wins beyond noise on g2 and ties on g3. **Random is ahead on g1 by 0.011.**
  - Why: g1 is the largest obstacle (semi-axes ≈ 8.2 × 4.8). The reference's dense contacts fit a short lengthscale, so the posterior mean deep inside the obstacle relaxes to the tied prior mean, which is "safe" (hollow interior). Its g1 band recalls are far better than random's (safe 0.990 vs 0.953, unsafe 0.859 vs 0.660), but interior states are misclassified.
  - The fill variants, which close interiors, gain +0.006 uniform accuracy on the unicycle.
- **Double integrator g3.** Uniform accuracy remains in the paper's direction but within noise (unchanged from before).
- **Overall.** The combined score ranks random below the reference: −1.68 [−2.09, −1.24] in the core cohort.

## Do better results score higher?

Hidden test, core cohort rank relative to the reference:

| Member | Before (2 experiments, 72 each) | After (4 experiments) | After, by experiment (g3 / complex / unicycle / quadrotor) |
|---|---:|---:|---|
| Fill + coverage top-up (`better_fill_topup`) | +0.884 [0.644, 1.146] | **+0.357 [0.118, 0.601]** | +0.15 / +0.27 / +0.19 / −0.25 |
| Coverage top-up (`better_topup`) | +0.164 [0.017, 0.335] | **+0.180 [0.059, 0.308]** | −0.05 / +0.12 / +0.03 / +0.09 |
| Fill (`better_fill`) | +0.728 [0.531, 0.918] | +0.174 [−0.044, 0.378] | +0.20 / +0.15 / +0.16 / −0.33 |
| P = 2 posterior draws (`weaker_two_draws`) | −0.390 [−0.663, −0.123] | −0.192 [−0.426, +0.046] | −0.07 / −0.13 / +0.02 / −0.01 |
| Simple coverage sweep | −1.101 [−1.627, −0.605] | −0.485 [−0.914, −0.050] | |
| Learned prior mean | −1.646 [−2.125, −1.168] | −1.199 [−1.627, −0.775] | |
| Random-query control | −1.269 [−1.784, −0.753] | **−1.680 [−2.095, −1.242]** | −0.26 / −0.37 / −0.67 / −0.38 |
| No acquisition (four initial demonstrations) | −3.249 [−3.727, −2.787] | −4.162 [−4.576, −3.752] | |
| Raw (unnormalized) KKT gradients | −4.562 [−5.135, −3.999] | −6.336 [−6.878, −5.795] | |
| No gradient observations | fails gate | fails gate | |
| Paper's zero-mean prior (core+1) | excluded | −3.434 [−3.929, −2.975] | |

- **Better variants.** Two are above the reference beyond noise; fill is above it on average.
  - On the quadrotor only the top-up variant helps. The star-shaped interior fill hurts there: the hourglass waist is not star-shaped about the contact centroid.
- **Weaker members.** Every ablation, the simple rule and the random control are below the reference; P = 2 is below it but now within noise.
- **Development agrees.** Fill + top-up +0.23, top-up +0.20, fill +0.12; random −3.17, no acquisition −6.14.
- **Evaluator cohort** (every passing member, as `scripts/evaluate_test.sh` ranks them): fill + top-up +0.25, top-up +0.17, fill +0.10, P = 2 −0.22, sweep −0.46, paper-formula fit −0.62, family fit −1.02, learned prior −1.28, random −1.56, zero prior −3.35, no acquisition −3.54, raw gradients −5.17.

## Shortcuts

| Shortcut | Before | After |
|---|---|---|
| **Paper-formula fit** (`shortcut_paper_fit`): relay the reference's acquisition, fit the paper's published g1/g2/g3 (2D) or g5 (3D) with free rotation, scale and centre to every tight state and normal, classify with the best fit | Development pilot, 3D RMS 0.18: **+0.455, highest of all members** (evaluator cohort; quadrotor balanced +0.10). Hidden test, 3D RMS 0.35: **−0.066 [−0.484, +0.332]** (core+1), quadrotor term **+0.198** | Hidden test, 3D RMS 0.52: **−0.257 [−0.652, +0.132]**, quadrotor term +0.007, unicycle −0.264. Development −0.673 |
| Fit of the visible development double-integrator generators (`shortcut_family_fit`; reference elsewhere) | +0.313 [0.037, 0.588] (hybrid, core) | −0.842 [−1.023, −0.663] |
| Interface probes: over-budget requests, false `n_demos`, forged flags/fields, impossible labels, constant safe, contact disk | failed (earlier audits) | all fail the run (protocol/budget rejection or success gate) |

**Paper-formula fit: the fix.** The private 3D deformation was enlarged on both splits, matched in size. It was raised from 0.18 to 0.35 and then to 0.52 RMS (test bump amplitude 0.85, width 1.8; development waves amplitude 0.47, wavelength 11).
- At 0.52 the published formula still gains uniform accuracy (quadrotor +0.011, unicycle +0.014), because it closes the reference's hollow interiors. It loses beyond noise near the boundary (quadrotor −0.067, unicycle −0.134).
- Net, it ranks below the reference in both new experiments and overall.
- What larger deformation cost the honest methods:
  - the reference's quadrotor accuracy fell from 0.9771 to 0.9729;
  - the reference − random accuracy gap on the quadrotor shrank from +0.0073 to +0.0030, still beyond noise;
  - the better-variant ordering is unchanged.

### Left open, and why

- **The paper-formula fit is below the reference but not beyond noise** (upper end +0.13).
  - A hybrid was not run: published formula in the far field, the learned boundary near the observed contacts. It would keep the formula's uniform-accuracy gain without its boundary loss, and would likely outrank the reference.
  - Its gain is the interior-filling the honest fill variants also find, here supplied by a hand-written published formula.
  - Closing it entirely would need deformations large enough that the instances stop being the paper's shapes.
  - `scoring/CONSTRAINTS.md` forbids hand-coding published formulas, and the grader can detect them (the App. B.1 constants appear in the code).
- **Reading `eval/test/` at test time.** Submissions run as subprocesses with filesystem access. No small fix closes this; it is left to the grader, as before.
- **Membership probing through rejected requests.** A rejection reveals an endpoint inside the obstacle, at most 40 bits per instance, against 12,288–16,384 evaluation states.
- **Copying the reference** ties it; that is not a loophole.

## Reference before and after (hidden test)

| | Accuracy | Balanced | Near-boundary | Safe recall | Unsafe recall |
|---|---:|---:|---:|---:|---:|
| g3, before | 0.9791 | 0.7565 | 0.7354 | 0.9535 | 0.5594 |
| g3, after | 0.9792 | 0.7565 | 0.7356 | 0.9544 | 0.5586 |
| complex, before | 0.9521 | 0.7806 | 0.7476 | 0.9493 | 0.6119 |
| complex, after | 0.9521 | 0.7809 | 0.7480 | 0.9500 | 0.6118 |
| **unicycle (new)** | 0.9725 | 0.8913 | 0.8649 | 0.9750 | 0.8076 |
| **quadrotor (new)** | 0.9729 | 0.7621 | 0.6800 | 0.7480 | 0.7763 |

Random control on the new experiments (accuracy / balanced / near-boundary / safe recall / unsafe recall):
- unicycle: 0.9726 / 0.8247 / 0.7795 / 0.8629 / 0.7865;
- quadrotor: 0.9699 / 0.7288 / 0.6379 / 0.6715 / 0.7862.

`scoring/scores_test.json` and `scores_train.json` contain only the reference and the frozen random control. Both were rerun end to end from the final code (job `e87b2f87-237`, bootstrap of both splits).
- No method in that run has a wall-clock limit.
- The job watchdog never fired (no aborts in any job).
- The hidden test and `scripts/reproduce.sh` do not use it.

## Headroom

- No scored metric of the reference is at its bound.
  - Quadrotor near-boundary balanced accuracy is 0.68 and unicycle near-boundary 0.86.
  - Accuracy is 0.95–0.98 in all four experiments.
- Two honest variants beat the reference beyond noise.
- The quadrotor experiment is where the method falls short most: sparse trajectories touch only curves on a 2D surface, and the reference's 3D boundary is coarse.
- Brute force is bounded by the evaluator-owned oracle (12 demonstrations, 40 requests); test states are revealed only after acquisition.
- The simple coverage sweep, a short hand-written acquisition rule, ranks below the reference.

## Not fixed, stated plainly

- **Unicycle uniform accuracy:** reference and random tie (g1: random ahead), for the hollow-interior reason above. Boundary metrics favour the reference beyond noise.
- **Double-integrator g3 uniform accuracy:** the gap stays within noise (unchanged).
- **Paper accuracies not reached:**
  - unicycle g1 / g2 / g3: 0.955 / 0.989 / 0.974, against the paper's 0.9996 / 1.0 / 0.9956;
  - quadrotor: 0.973, against 0.9919;
  - our random control matches the paper's on the unicycle mean (0.9726 vs 0.9731) and is much stronger on the quadrotor (0.970 vs 0.9423); the shortfall is our reference (unicycle mean 0.9725 vs the paper's 0.9984).
- **Instances differ from the paper's exact formulas.** They are private rotations, scalings, translations and deformations; the paper's numbers are kept under `_paper_reported_*` in `scoring/reference.json`.
- **Quadrotor model.** It uses world-frame translational velocity rather than Sabatino's body-frame velocity: the same physics in different coordinates.
- **Expert solve.** The expert solves Prob. 1 in position space and lifts the path. That is exact here, but it is not an IPOPT solve over the full state/control trajectory.
- **Not reproduced:** the 7-DOF arm and hardware experiments, the 3D double integrator (g6), and g4.
- **Unspecified choices**, selected or confirmed on development only:
  - four initial crossings, twelve demonstrations and forty requests;
  - Alg. 1 iterated (P = 4 per batch) until the budget is spent, rather than a fixed N_iters = 3;
  - the tied prior mean with ρ = 0.1;
  - the 1e-4 noise floor;
  - unit-norm gradients;
  - the one-lengthscale Problem 3 region.

## Jobs

| Job | What it ran | Outcome |
|---|---|---|
| `a9a7ce06-642` | Smoke test of the new tasks, expert, KKT and 3D GP | Checks passed; timing about 9 s (unicycle) and 27 s (quadrotor) per reference instance. Its accuracy numbers were invalid (my smoke script reused a stateful evaluation stream; evaluation is now idempotent) |
| `d2baf85e-52c` | Development pilot, all members | Reference beats random on the unicycle; the paper-formula fit ranked highest (+0.455). 3D deformation enlarged |
| `92153af0-4dd` | Final job | Stalled: the random control's unicycle methods stopped completing after their queries, with no error output. Cancelled; evaluator made robust (stderr drain, spawned workers, compact records) |
| `b2b22cca-e77` | Final job, 3D RMS 0.35 | Exit 0, validation passed; the paper-formula fit still gained on the quadrotor (`hardening_rms035_summary.json`) |
| `e87b2f87-237` | Final job, 3D RMS 0.52 | Exit 0, validation passed; this document's "after" numbers |
