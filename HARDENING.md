# Hardening

Final iteration. Every number comes from a completed job. All hidden-test comparisons are paired: every method sees the same 36 obstacles per experiment, the same four initial trajectories and the same query points. Intervals are 95% paired bootstrap intervals.

Ranks follow the gym's rule: weight × |coefficient| × better-direction improvement over the reference, divided by the metric's population standard deviation across the cohort. Two cohorts are reported:
- **Core cohort**: every passing attempt in the hidden-test audit, except the rejected zero-mean prior (item 1 below). That prior's collapsed uniform accuracy (0.72) would otherwise set the accuracy scale for everyone.
- **Evaluator audit cohort**: everything the evaluator ranked, including that outlier.

Jobs:
- `3b3ba72f-7f9`: the combined audit, exit 0; `validate.py --post-design` passed.
- `VERIFY_JOB`: end-to-end reproduction from the final code in a fresh-clone state, compared with the frozen records.
- `05339383-a8e` (previous iteration): the undeformed-g3 formula-fit measurement used as a "before" below.

Evidence:
- `eval/test/hardening_summary.json`: every number below.
- `eval/test/hardening_seventh_summary.json`: the previous audit.
- `tmp/prior_selection.json`: the development prior selection.

## What changed this iteration

**1. The paper's zero-mean GP posterior was tested as the reference and rejected (development split only).**
- The previous reference ties the GP prior mean to −0.1σ. Far-from-data states then default to safe, so poorly observed obstacle interiors are classified safe.
- This is why the reference does not beat random on the paper's headline quantity, uniform-space accuracy.
- The paper's own posterior (Eqs. 15–16) is zero-mean, and its Fig. 8 shows filled interiors, so restoring it was in scope.
- It was selected by a rule fixed before the job (`scripts/select_prior.py`). Keep zero mean only if, on development:
  - its reference and its random control both pass the gate;
  - its reference beats random on g3 uniform accuracy;
  - its reference outranks random overall.

It failed badly:

| Development, uniform accuracy | g3 | stress |
|---|---:|---:|
| zero-mean reference | **0.7335** | **0.7283** |
| zero-mean random | 0.8453 | 0.8431 |
| tied reference (kept) | 0.9689 | 0.9376 |
| tied random | 0.9657 | 0.9475 |

Why it fails:
- With zero mean, the training hinge on demonstrated states drives the fitted lengthscale to its upper bound (12.5).
- Gradient tails then extrapolate as half-plane "unsafe" regions across the workspace.
- Its boundary metrics are fine (dev g3 balanced accuracy 0.809, near-boundary 0.775).

The tied prior therefore stays. The reference is the same method as in the previous iteration, and on the unchanged g3 instances its test metrics are identical. The code default is now `tied`. The manifests pass `--prior tied` explicitly.

**2. The hidden complex-boundary (stress) family is now private.**
- The stress generator in `eval/train/tasks.py` (visible) was the same family as the hidden test's, with fixed amplitudes for its two dominant harmonics.
- Fitting that written-down generator to the contacts beat the reference (shortcut table below).
- The hidden test now draws its own family: a different set of dominant harmonics with random amplitudes. Its form stays in `eval/test/tasks.py`.
- The new parameters come from a separate seed stream, so g3 test instances and all evaluation coordinates are unchanged.
- Difficulty is matched (`scripts/check_tasks.py`). Radius variation is RMS 0.1398 (development) vs 0.1410 (test) of the mean radius, slope RMS 0.655 vs 0.692, and narrowest radius 0.7465 vs 0.7466.
- The reference scores alike on the two splits:

| Stress experiment, reference | Accuracy | Balanced accuracy | Near-boundary BA |
|---|---:|---:|---:|
| Development | 0.9376 | 0.7665 | 0.7454 |
| Test, previous family | 0.9368 | 0.7667 | 0.7493 |
| Test, new family | 0.9371 | 0.7654 | 0.7463 |

**3. New audit members.**
- `topup`: an acquisition variant. When Alg. 1 stops early, it spends the unused demonstration budget on crossings aimed at the least-covered arc. The reference leaves budget unused: 10.25 of 12 demonstrations on average on stress, as few as 4.
- The `family_fit` probe now also fits the visible stress generator: a MAP fit of centre, scale, two phases and 30 high-harmonic coefficients under the generator's own prior.

**Unchanged:** the evaluator, protocol, metrics, gates, coefficients, weights (g3 0.6, stress 0.4), the paper-reported numbers in `scoring/reference.json`, g3 instances and the paper's training method.

## Do better results score higher?

Yes, in both cohorts.

| Variant | Before (previous audit) | After, core cohort | After, evaluator audit cohort |
|---|---:|---:|---:|
| Reference GP-ACL | 0 | 0 | 0 |
| Better: interior fill + coverage top-up (`better_fill_topup`) | n/a | **+1.331 [1.005, 1.680]** | +0.687 [0.448, 0.945] |
| Better: interior fill (`better_fill`) | +1.088 [0.789, 1.378] | +0.887 [0.623, 1.149] | +0.408 [0.239, 0.580] |
| Better: coverage top-up (`better_topup`) | n/a | +0.314 [0.131, 0.528] | +0.256 [0.101, 0.437] |
| Random-query control | −0.906 [−1.586, −0.230] | **−0.777 [−1.347, −0.195]** | −1.110 [−1.561, −0.661] |
| P=2 posterior draws (`weaker_two_draws`) | +0.151 [−0.409, +0.681] | +0.194 [−0.290, +0.653] | +0.262 [−0.091, +0.611] |
| Learned prior mean | −0.759 [−1.579, +0.069] | −0.780 [−1.518, −0.051] | −1.440 [−1.972, −0.902] |
| No acquisition (initial 4 demonstrations) | −2.156 | −1.837 [−2.456, −1.226] | −2.084 [−2.520, −1.676] |
| Raw (unnormalized) KKT gradients | −4.328 | −4.281 [−5.092, −3.526] | −3.625 [−4.130, −3.133] |
| No gradient observations | fails gate | fails gate | fails gate |
| Paper's zero-mean prior (`alt_zero_prior`) | n/a | excluded | −3.275 [−3.699, −2.848] |

- Every better variant is above the reference beyond paired noise.
- Every ablation and the random control are below it. The learned prior is now clearly below in both cohorts.
- P=2 versus P=4 remains an honest alternative the test cannot separate. It is a headroom question, not a loophole.

Paired gains of the best variant (`better_fill_topup` − reference, test):

| | Accuracy | Balanced accuracy | Near-boundary BA |
|---|---|---|---|
| g3 | +0.0027 [0.0016, 0.0038] | +0.026 [0.014, 0.039] | +0.008 [0.001, 0.015] |
| stress | +0.0229 [0.0175, 0.0283] | +0.032 [0.016, 0.050] | +0.023 [0.009, 0.040] |

It also generalizes from development: there it ranks +1.018 [0.623, 1.414] against the reference.

## Shortcuts

| Shortcut | Before | Fix | After |
|---|---|---|---|
| **Write down the visible stress generator** (fit it to all contacts; reference on g3, fit on stress) | **+1.572 [1.121, 1.942]**: near-boundary BA +0.054, accuracy +0.054 on stress | Private hidden stress family | **+0.337 [−0.248, +0.950]** in the same three-member cohort (+0.539 [0.223, 0.864] in the core cohort); near-boundary BA now **−0.044** |
| Same fit applied to both experiments | −1.377 | (same) | −2.464; on g3 the stress model always fits contacts best (RMS ≤ 0.016) and is wrong |
| Fit the paper's g3 polynomial (previous iteration) | +2.584 on undeformed g3 (job `05339383-a8e`) | Private per-instance g3 deformation (previous iteration) | −0.184 [−1.699, +1.074]; g3 instances and reference unchanged since |

How the stress "before" was measured:
- It comes from the development split. There the stress family is the visible one, as on the old hidden test.
- The cohort is the reference, the frozen random control and the probe.
- "After" uses the same cohort on the hidden test.

The earlier interface shortcuts are not replayed, because the evaluator code did not change:
- public answer oracle;
- grid ordering;
- grid resizing;
- false, negative, NaN and −inf counts;
- excess budget;
- replacing the public truth module;
- moving baseline fields;
- impossible, empty or null labels;
- forged flags or scores;
- constant classifiers;
- fixed disks;
- static legacy predictions.

All failed the whole run in the previous audit.

### Left open, and why

- **Generic closed-shape fitting on stress (+0.34 to +0.54 vs the reference, below every honest better variant).**
  - The hand-written development generator no longer answers hidden instances: it now loses 0.044 near-boundary balanced accuracy.
  - It still gains uniform accuracy (+0.032) because any closed contour fills the obstacle interior, and the reference leaves interiors partly "safe" (see Not fixed).
  - Honest shape-aware variants gain more from the same effect: fill +0.887, fill+top-up +1.331 in the core cohort.
  - Removing what remains would mean changing the reference method, which is out of scope. A stress family with no rough-round structure would change the experiment's difficulty.
  - The separate grader can judge whether a submission hard-codes the development generator.
- **Reading `eval/test/` at test time.** Submissions run as subprocesses with filesystem access, so a method could import the private generator and match instances. No small fix closes this. It is left to the separate grader, as in previous iterations.
- **Copying the reference** ties it. That is not a loophole.

## Reference before and after

- **g3**: identical to the previous iteration: accuracy 0.9637, balanced 0.8084, near-boundary 0.7783, safe recall 0.9453, unsafe recall 0.6716. The method and instances are unchanged.
- **stress**: before (old family) 0.9368 / 0.7667 / 0.7493 / 0.9751 / 0.5582; after (private family) 0.9371 / 0.7654 / 0.7463 / 0.9724 / 0.5583.

`scoring/scores_test.json` and `scores_train.json` contain only the reference and the frozen random control, both regenerated from the final code. The paper's reported numbers are unchanged (`_paper_reported_*`).

## Paper direction (reference vs random, paired)

| Test | Random − reference: g3 | Random − reference: stress |
|---|---|---|
| Uniform accuracy | +0.0036 [−0.0008, +0.0078] | +0.0078 [+0.0046, +0.0110] |
| Boundary balanced accuracy | −0.048 [−0.075, −0.023] | −0.032 [−0.055, −0.010] |
| Near-boundary balanced accuracy | −0.045 [−0.064, −0.026] | −0.045 [−0.068, −0.023] |

Development agrees:
- g3 uniform accuracy: random −0.0031 [−0.0104, +0.0032].
- Boundary balanced accuracy: −0.043 (g3) and −0.030 (stress).
- Near-boundary BA: −0.036 (g3) and −0.050 (stress).

GP-ACL's added demonstrations touch the obstacle 99.6% (g3) and 95.1% (stress) of the time, versus 18% and 16% for random. They add 15.3 and 7.3 new unit boundary cells per instance, versus 4.0 and 2.1.

The combined score ranks the reference above random beyond noise. The paper's emphasis on fewer errors near boundaries is reproduced beyond noise in both experiments and both splits.

## Headroom

The reference is at the bound of no scored metric. On test:
- g3: 0.964 / 0.808 / 0.778 / 0.945 / 0.672.
- stress: 0.937 / 0.765 / 0.746 / 0.972 / 0.558.

Three variants beat it beyond noise (table above). The largest gains are on the stress experiment, the setting added where the paper's method falls short. No variant approaches the bounds. Brute force is bounded by the evaluator-owned oracle with its twelve-demonstration and forty-request caps.

## Weights

Unchanged. The paper reports uniform accuracy as its headline number (coefficient 1.0) and stresses boundary errors (0.5, 0.6, recalls 0.25 each). g3 is the paper's own constraint (0.6); the stress experiment is where the method falls short (0.4).

## Not fixed, stated plainly

- **The paper's headline uniform-accuracy advantage is not reproduced.**
  - On g3, random and GP-ACL tie (test +0.0036 [−0.0008, +0.0078] for random; development −0.0031 for random). On stress, random is slightly better beyond noise.
  - The cause is the GP's posterior far from data. With the tied prior, poorly observed interiors default to safe.
  - The paper's zero-mean posterior, the faithful alternative, collapses uniform accuracy to 0.73.
  - The paper's 0.9956 is not reached. Our random control (0.967) is close to the paper's (0.9652).
- **Scope.** Only the 2D double integrator is reproduced. The unicycle, quadcopter, 7-DOF arm and hardware experiments are not.
- **Unspecified choices.** The reference depends on choices the paper leaves open (four initial crossings, a twelve-demonstration budget, P=4, the one-lengthscale P3 region, the tied mean with ρ=0.1, η=1, unit-norm gradients), selected or confirmed on development only.
- **The P=2 variant** is indistinguishable from P=4.
