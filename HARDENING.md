# Hardening

Final iteration. `REVIEW.md` listed one weakness: the paper's headline claim (better uniform-space accuracy than random sampling) did not reproduce; random tied GP-ACL. Every number below comes from a completed job. Comparisons are paired: every method sees the same obstacles, initial trajectories and query points. Intervals are 95% paired bootstrap intervals.

Ranks follow the gym's rule: weight × |coefficient| × better-direction improvement over the reference, divided by the metric's population standard deviation across the ranked cohort.
- **Core cohort**: every passing audit member except `alt_zero_prior`, whose collapsed accuracy would otherwise set the accuracy scale for everyone (same convention as the previous iteration).
- **Evaluator cohort**: everything the evaluator ranked.

Evidence files:
- `eval/test/hardening_summary.json`: final audit (test and development).
- `eval/test/hardening_eighth_summary.json`: the previous iteration's audit ("before").
- `eval/train/hardening_pilot_summary.json`: development pilot (diagnosis).
- `eval/train/noise_floor_selection.json`: development selection.
- `eval/train/hardening_rejection_diagnostic.json`: rejected requests.

Jobs:

| Job | What it ran | Outcome |
|---|---|---|
| `ecec050e-3cf` | Development selection of a partial shared initial set | Cancelled; every candidate rejected (below) |
| `3216cb5c-ccf` | Development pilot: contact quality, error localization, classifier rescoring, three end-to-end settings | Diagnosis below |
| `1a6389bd-3dd` | Final job: noise-floor selection, controls on both splits, development and 72-instance hidden-test audits | Exit 0; `validate.py --post-design` passed |
| `808829f2-232` | Development diagnostic of rejected requests | No solver failures |
| `59bd0533-310` | Same as the final job with an 80-request allowance | Cancelled after selection and controls: no gain; reverted |

## Diagnosis: what blocked the headline

Uniform-space errors were dominated by artifacts of the reference's GP, not by acquisition. The development pilot found (previous setup and reference):

- **Tight states are clean.** All 6,799 flagged tight states lie within 0.05 of the true boundary, with estimated gradients within cos 0.9 of the true normal.
- **g3: false "unsafe" far from any observation.** 0.0178 of the workspace (reference) and 0.0233 (random) was predicted unsafe more than 4 units from every tight state. The cause is the fitted noise-variance floor of 1e-10. Dense, nearly coincident tight states make the training covariance numerically singular, and the posterior mean then carries spurious unsafe regions.
- **Complex-boundary task: hollow interiors.** False "safe" was 0.061 (reference) vs 0.050 (random). The reference's denser contacts fit a shorter lengthscale, so interiors relax to the safe prior sooner.

Re-fitting only the final classifier to the frozen development demonstrations, with GPyTorch's inferred-noise floor (1e-4):
- far false-unsafe fell from 0.0178 to 0.0024;
- reference − random uniform accuracy moved from +0.0031 [−0.0030, +0.0104] to **+0.0092 [+0.0007, +0.0175]** on g3, and from −0.0103 [−0.0143, −0.0064] to +0.0007 [−0.0033, +0.0045] on the complex-boundary task.

## What changed

**1. The reference's GP noise floor (restoring a well-conditioned Eq. 15).**
- The fitted noise variance is now bounded below by 1e-4. That is GPyTorch's inferred-noise floor; the cited training procedure fits its GP with GPyTorch.
- It was chosen on development by a rule fixed before the job (`scripts/select_noise_floor.py`): the reference with the best uniform accuracy (the paper's headline quantity), among candidates whose reference passes the gate and outranks its own random control. Ties within 0.0005 go to the larger floor.

| Development, reference accuracy | g3 | complex | mean | Reference − random, g3 | Reference − random, complex |
|---|---:|---:|---:|---|---|
| **1e-4 (chosen)** | 0.9811 | 0.9565 | **0.9688** | +0.0022 [−0.0087, +0.0125] | +0.0051 [−0.0007, +0.0106] |
| 1e-5 | 0.9809 | 0.9529 | 0.9669 | +0.0008 [−0.0074, +0.0082] | +0.0029 [−0.0025, +0.0083] |
| 1e-6 | 0.9741 | 0.9457 | 0.9599 | −0.0016 [−0.0074, +0.0042] | −0.0037 [−0.0081, +0.0009] |

- Everything else in the reference is unchanged: Alg. 1, Probs. 3–8, the tied prior mean, P=4, α=0.3. The random control uses the same GP. The code default is `NOISE_FLOOR = 0.0001`, and the manifests pass `--noise-floor 0.0001`.

**2. Hidden test enlarged from 36 to 72 obstacles per experiment.**
- Per-instance paired accuracy differences on g3 vary with an SD of about 0.03, so 36 instances could not resolve the effect sizes this task produces.
- The first 36 instances are unchanged. Development stays at 24.

**3. New audit members.**
- `simple_coverage_sweep`: a short hand-written acquisition rule (crossings aimed at the least-covered arc).
- `variant_previous_floor`: the previous reference, i.e. the 1e-10 floor.

**Tried and rejected:**

| Change | Result | Decision |
|---|---|---|
| Partial shared initial set (2 or 1 crossings; 1 crossing with ρ=0.01), closer to the paper's Fig. 8 | Alg. 1 stalled. On the complex task the reference acquired 0.4 demonstrations on average from one crossing. Random outranked the reference in every candidate (rank +1.50, +3.00, +3.49), and reference − random accuracy was negative in all six candidate/experiment cells (five beyond noise) | Kept four crossings |
| Request allowance 80 instead of 40 | Development g3 gap +0.0021 (was +0.0022). Hidden test: random − reference on g3 −0.0019 [−0.0065, +0.0028] (was −0.0017). The extra requests were mostly rejected too: g3 demonstrations rose from 8.32 to 8.46, and 26 of 72 g3 instances stopped at Alg. 1's 10-batch limit | Reverted to 40: a larger allowance only enlarges the membership leak of rejected requests |
| Learned constant prior (GPyTorch ConstantMeanGrad style), floor 1e-4 | Random ahead on both experiments (development) | Rejected |
| Paper's zero-mean posterior, likelihood only, floor 1e-4 | Uniform accuracy collapses (0.92 / 0.83) | Rejected |

**Unchanged:** the evaluator code, protocol, metrics, gates, coefficients and weights, the four shared crossings, the 12-demonstration budget and 40-request allowance, the instance families, and the paper-reported numbers in `scoring/reference.json`.

## Paper direction (reference vs random, hidden test, paired)

| Random − reference | Before: g3 (36) | After: g3 (72) | Before: complex (36) | After: complex (72) |
|---|---|---|---|---|
| Uniform accuracy | +0.0036 [−0.0007, +0.0077] | **−0.0017 [−0.0062, +0.0030]** | +0.0078 [+0.0046, +0.0110] | **−0.0028 [−0.0052, −0.0003]** |
| Boundary balanced accuracy | −0.048 [−0.074, −0.022] | −0.023 [−0.036, −0.008] | −0.032 [−0.055, −0.009] | −0.043 [−0.062, −0.025] |
| Near-boundary balanced accuracy | −0.045 [−0.064, −0.026] | −0.028 [−0.039, −0.018] | −0.045 [−0.068, −0.023] | −0.049 [−0.067, −0.032] |

- The reference now leads random on uniform accuracy in both experiments. The lead is beyond noise on the complex-boundary task but **not on g3**.
- Boundary metrics favour the reference beyond noise in both.
- Development agrees in direction: reference − random accuracy +0.0022 (g3) and +0.0051 (complex).
- The combined score ranks random well below the reference: −1.269 [−1.784, −0.753] in the core cohort.

## Do better results score higher?

Yes. Ranks relative to the reference on the hidden test:

| Member | Before (core, 36) | After (core, 72) | After (evaluator cohort) |
|---|---:|---:|---:|
| Fill + coverage top-up (`better_fill_topup`) | +1.331 [0.992, 1.673] | **+0.884 [0.644, 1.146]** | +0.607 [0.398, 0.830] |
| Fill (`better_fill`) | +0.887 [0.631, 1.142] | **+0.728 [0.531, 0.918]** | +0.495 [0.322, 0.666] |
| Coverage top-up (`better_topup`) | +0.314 [0.136, 0.535] | **+0.164 [0.017, 0.335]** | +0.150 [0.022, 0.296] |
| Previous reference, floor 1e-10 (`variant_previous_floor`) | (was the reference) | −0.064 [−0.566, +0.446] | +0.316 [−0.152, +0.790] |
| P=2 posterior draws (`weaker_two_draws`) | +0.194 [−0.312, 0.676] | **−0.390 [−0.663, −0.123]** | −0.357 [−0.599, −0.114] |
| Simple coverage sweep (`simple_coverage_sweep`) | n/a | −1.101 [−1.627, −0.605] | −1.055 [−1.521, −0.593] |
| Random-query control | −0.777 [−1.368, −0.192] | −1.269 [−1.784, −0.753] | −1.213 [−1.686, −0.737] |
| Learned prior mean | −0.780 [−1.507, −0.046] | −1.646 [−2.125, −1.168] | −1.360 [−1.808, −0.926] |
| No acquisition (four initial demonstrations) | −1.837 | −3.249 [−3.727, −2.787] | −2.999 |
| Raw (unnormalized) KKT gradients | −4.281 | −4.562 [−5.135, −3.999] | −3.987 |
| No gradient observations | fails gate | fails gate | fails gate |
| Paper's zero-mean prior (`alt_zero_prior`) | excluded | excluded | −3.274 [−3.746, −2.807] |

- Every better variant is above the reference beyond paired noise. Every ablation, the simple rule and the random control are below it.
- P=2 versus P=4 was an inseparable pair before; it is now separated beyond noise.
- Paired gains of `better_fill_topup` over the reference:
  - g3: accuracy +0.0020 [0.0003, 0.0037], balanced +0.014, near-boundary +0.006;
  - complex: accuracy +0.0145 [0.0115, 0.0177], balanced +0.018, near-boundary +0.014.
- It generalizes from development: there it ranks +1.165 [0.686, 1.705].
- **The previous reference ties the new one.** On the same 72 hidden instances it is worse on uniform accuracy (g3 −0.0150, complex −0.0142, both beyond noise) but better on g3 boundary metrics (balanced +0.061, near-boundary +0.047). The 1e-4 floor smooths the model and gets fewer accepted demonstrations on g3 (8.3 vs 11.1). This is a genuine trade-off the combined score weighs as even. It is an honest alternative the test cannot separate, not a loophole.

## Shortcuts

No shortcut was changed in this iteration; the probe was rerun on the new setup.

| Shortcut | Before | After |
|---|---|---|
| Write down the visible complex-boundary generator, fitted to all contacts, on both experiments (`shortcut_family_fit`) | −2.309 (core) | −2.725 [−3.175, −2.287] |
| Same fit on the complex task only, reference on g3 (`hybrid_stress_family_fit`) | +0.539 [0.219, 0.873] core; +0.337 three-member | +0.313 [0.037, 0.588] core; +0.255 three-member |

The interface shortcuts are not replayed because the evaluator code did not change: public answer oracle, grid ordering, grid resizing, false/negative/NaN/−inf counts, excess budget, replaced truth, moved baseline fields, impossible/empty/null labels, forged flags or scores, constant classifiers, fixed disks, static predictions. All failed the whole run in earlier audits.

### Left open, and why

- **Generic closed-shape fill on the complex-boundary task (hybrid, +0.31).**
  - On hidden instances the development generator is the wrong family. It loses near-boundary balanced accuracy (−0.040 [−0.051, −0.028]) and gains uniform accuracy (+0.026) only by closing the obstacle interior, which the reference leaves partly "safe".
  - It stays below both honest variants that fill interiors without a hand-written generator (+0.728, +0.884).
  - Removing it would mean changing the reference method or the complex-boundary family. The separate grader can judge whether a submission hard-codes the development generator.
- **Reading `eval/test/` at test time.** Submissions run as subprocesses with filesystem access. No small fix closes this; it is left to the grader, as before.
- **Membership probing through rejected requests.** A rejected request reveals that an endpoint is inside the obstacle, at most 40 bits per instance. That is not enough to label the 12,288 test points. Raising the allowance to 80 did not help the reference, so the allowance stays at 40.
- **Copying the reference** ties it; not a loophole.

## Reference before and after (hidden test)

| | Accuracy | Balanced | Near-boundary | Safe recall | Unsafe recall |
|---|---:|---:|---:|---:|---:|
| g3, before (36, floor 1e-10) | 0.9637 | 0.8084 | 0.7783 | 0.9453 | 0.6716 |
| g3, previous reference on the 72 | 0.9642 | 0.8173 | 0.7826 | 0.9172 | 0.7174 |
| **g3, after (72, floor 1e-4)** | **0.9791** | 0.7565 | 0.7354 | 0.9535 | 0.5594 |
| complex, before (36) | 0.9371 | 0.7654 | 0.7463 | 0.9724 | 0.5583 |
| complex, previous reference on the 72 | 0.9379 | 0.7572 | 0.7387 | 0.9732 | 0.5413 |
| **complex, after (72)** | **0.9521** | 0.7806 | 0.7476 | 0.9493 | 0.6119 |

Random control after: g3 0.9774 / 0.7340 / 0.7074 / 0.7997 / 0.6683; complex 0.9493 / 0.7375 / 0.6985 / 0.9227 / 0.5524.

`scoring/scores_test.json` and `scores_train.json` contain only the reference and the frozen random control. Both were rerun end to end from the final code in job `1a6389bd-3dd` (bootstrap of both splits). The task files and method are byte-identical to that job's.

## Headroom

The reference is at the bound of no scored metric. On test it scores g3 0.979 / 0.757 / 0.735 / 0.954 / 0.559 and complex 0.952 / 0.781 / 0.748 / 0.949 / 0.612.

Three variants beat it beyond noise; the largest gains are on the complex-boundary experiment, the setting where the method falls short. Brute force is bounded by the evaluator-owned oracle (12 demonstrations, 40 requests). The simple coverage sweep, a short hand-written acquisition rule, ranks below the reference.

## Weights

Unchanged. The paper reports uniform accuracy as its headline number (coefficient 1.0) and stresses boundary errors (0.5, 0.6, recalls 0.25 each). g3 is the paper's own constraint (weight 0.6); the complex-boundary experiment is where the method falls short (0.4).

## Not fixed, stated plainly

- **The headline effect on g3 is in the paper's direction but not beyond noise:** reference − random +0.0017 [−0.0030, +0.0062] on 72 hidden instances. It is beyond noise on the complex-boundary task (+0.0028 [+0.0003, +0.0052]).
  - Rejected-request diagnostic (development): every one of the reference's 962 rejected requests had an endpoint inside the true obstacle, and the expert's solver never failed.
  - The rejections come from the paper's endpoint rule (Eq. 10, β = 0.3 / 0.55), which places endpoints just outside the sampled boundary.
  - So the reference gets 8.3 of 12 demonstrations on g3 (random gets 12). A rejection gives the GP no information.
  - Neither of the two setup levers that address this helped: a partial initial set and a larger request allowance. Spending the unused budget is a measured better variant (`better_topup`), not the paper's method.
- **The paper's 0.9956 g3 accuracy is not reached** (reference 0.979). The paper's random baseline (0.9652) is below ours (0.977), because the restored noise floor also removes the random control's spurious regions.
- **The noise-floor fix is a trade-off on g3, not a strict improvement.** Accuracy improves beyond noise and boundary metrics drop. The previous reference ties the new one in rank.
- **Scope:** only the 2D double integrator is reproduced; the unicycle, quadcopter, 7-DOF arm and hardware experiments are not.
- **Unspecified choices:** the reference depends on choices the paper leaves open, selected or confirmed on development only:
  - four initial crossings, a twelve-demonstration budget and forty requests;
  - P=4, the one-lengthscale Problem 3 region;
  - the tied mean with ρ=0.1, η=1, unit-norm gradients;
  - the 1e-4 noise floor.
