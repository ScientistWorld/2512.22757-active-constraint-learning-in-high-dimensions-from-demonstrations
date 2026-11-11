# Hardening

Final iteration. All numbers below come from completed jobs. Development selection used only the development split. Every hidden-test comparison is paired: each method sees the same 36 obstacles per experiment, the same initial trajectories and the same query points. Intervals are 95% paired bootstrap intervals. Ranks follow the gym's rule (weight × |coefficient| × better-direction improvement ÷ cohort standard deviation). They use the scales of the audit cohort: the reference, the frozen random control, every variant and every probe that passed the success gate. Complete evidence is in `eval/test/hardening_summary.json`, and every scored test number is in `eval/test/hardening_scores.json`. Job IDs:
- `fcaa3a0a-3f9`: development exploration.
- `c7324ec1-f1e`: final combined audit, exit 0. `validate.py --post-design` passed.
- UNDEFORMED_JOB: the undeformed-shortcut measurement; see the shortcut table.

## Do better results now score higher?

Yes. Before this iteration they did not.

| Variant | Before (previous iteration's final audit) | After (this audit) | Paired 95% interval |
|---|---:|---:|---|
| Reference GP-ACL | 0 | 0 | anchor |
| Random-query control | **+0.566** (above the reference) | **−0.906** | [−1.586, −0.230] |
| Better: interior fill classifier | n/a | **+1.088** | [0.789, 1.378] |
| Contact/normal radial fit (previous "better" candidate) | −4.045 (order 6), −2.891 (order 12) | −1.418 (`alt_periodic`) | [−2.407, −0.509] |
| Previous packaged reference (`weaker_legacy`) | 0 (it was the anchor) | −0.683 | [−1.332, −0.038] |
| Learned constant prior mean | n/a | −0.759 | [−1.579, +0.069] |
| No acquisition (initial demonstrations only) | −2.085 | −2.156 | [−2.890, −1.445] |
| Raw (unnormalized) KKT gradients | −4.439 | −4.328 | [−5.240, −3.433] |
| No gradient observations | failed gate | failed gate (raw rank −8.48) | [−9.02, −7.93] |
| P=2 posterior draws instead of 4 (`weaker_two_draws`) | n/a | +0.151 | [−0.409, +0.681] |

Before, the "better" variants scored below the reference and random sat above it, so the gym had neither a working reference nor any demonstrated headroom. Now:
- Random falls below the reference beyond paired noise.
- A genuine improvement (the fill classifier) lies above it beyond noise.
- The previous method and the ablations fall below it.

P=2 versus P=4 is an honest alternative that the test cannot tell apart. Its name `weaker_two_draws` is historical: P=4 was chosen because it ranked +0.69 over P=2 on development, but on test the two tie.

## What was wrong and what changed

**1. Problem 3 never optimized (bug).** The previous code started P3's local solver at observed tight states. The posterior variance is minimal there, with zero gradient, so the solver returned its start.
- Consequence: κ_MI sat on known contacts, and every batch re-requested almost the same start/goal pair (one test instance repeated one pair six times).
- Diagnosis from the previous ledgers: ~180 added contacts per g3 instance but only 0.6 contacts more than one unit from earlier ones.
- Fix: P3 is solved over the posterior sample's zero set. All zero crossings on a grid finer than the fitted lengthscale are enumerated, the best distinct ones are polished with the continuous constrained NLP, and the highest-variance point is kept.
- Search region: within one fitted lengthscale of the observed tight states. Spurious zero contours of a draw far from all data carry no information about where demonstrations can touch the constraint. This radius was chosen on development: 0.5 ranked −1.17 and 1.5 ranked +0.19 relative to 1.0.

**2. Unexplored space counted as safe.** The constant prior mean was learned freely and settled near −0.9 × signal. Unexplored states therefore had prior safety probability Φ(0.9) ≈ 0.82. That clears Eq. 10's β = 0.55 with no evidence.
- Consequence: endpoints landed inside the true obstacle and the expert rejected them (61–80% of requests). The inferred obstacle was also a hollow crescent.
- Context: the previous iteration chose ρ = 0.1 so that unexplored states would sit at Φ(0.1) = 0.54 < β. The learned constant overrode that intent. The paper's own posterior (Eqs. 15–16) is zero-mean, which is the regime where β > 0.5 requires evidence.
- Fix: the constant is now tied to −ρσ. The training loss (Chou et al. Sec. IV-B marginal likelihood plus feasibility hinge) still optimizes length, signal and noise. A finite-difference check of the tied gradient runs in every job.
- Development result: the learned constant ranked −1.98 and accepted only 3.3 of 8 demonstrations on g3.

**3. Batching.** Alg. 1 is now iterated with a refit after every batch until the twelve-demonstration budget is spent, instead of redrawing samples from one stale fit. The paper's Fig. 5 runs five iterations; N_iters is set by the budget. A request that nearly duplicates an earlier one is skipped, since a deterministic expert would return the same demonstration. P = 4 posterior draws per batch; the paper does not specify P.

**4. Setup.** Four shared initial crossings now replace two, so every method starts from coarse coverage from several sides. Alg. 1 takes S_D^(1) as input and the paper does not specify it; its Fig. 5 starts both methods from a usable first model. Every method still acquires at most eight further demonstrations within forty requests.

**5. Public-formula shortcut.** The exact g3 polynomial is public in the paper and in the development generator. A method could fit its four parameters (rotation, scale, centre) to contacts. Hidden g3 instances now carry a private smooth radial deformation of the polynomial: localized angular bumps and dents, RMS 3.99% of the radius. Development uses a different family (low harmonics, RMS 4.06%), so difficulty matches but the exact form does not transfer. Gradient and boundary consistency of both generators are checked in every job (relative gradient error < 1e-8).

**6. Unchanged.** The paper's training method is unchanged: the Problems 7–8 KKT programs, unit-norm gradient estimates (identified only up to scale), 1,000 random Fourier features, α = 0.3, δ = 0.001, β = 0.55/0.3/0.55, the continuous P4–P6 ray programs, Eq. 10 read as a safe probability, and posterior-mean classification. Also unchanged: metrics, coefficients, weights, gates, the paper-reported numbers, the 36 hidden instances per experiment and the evaluator-owned controls. The development split was cut to 24 instances per experiment to save compute. Evaluators now run instances in parallel on available CPUs; records are still saved in index order.

## Paper direction (reference vs random, hidden test)

| Experiment | Metric | Reference | Random | Random − reference [95%] |
|---|---|---:|---:|---|
| g3 | uniform accuracy | 0.9637 | 0.9673 | +0.0036 [−0.0008, +0.0078] |
| g3 | boundary balanced accuracy | 0.8084 | 0.7604 | −0.0480 [−0.0751, −0.0220] |
| g3 | near-boundary balanced accuracy | 0.7783 | 0.7337 | −0.0446 [−0.0647, −0.0253] |
| g3 | safe / unsafe recall | 0.9453 / 0.6716 | 0.7782 / 0.7427 | −0.167 / +0.071 |
| stress | uniform accuracy | 0.9368 | 0.9447 | +0.0079 [+0.0047, +0.0109] |
| stress | boundary balanced accuracy | 0.7667 | 0.7378 | −0.0289 [−0.0524, −0.0058] |
| stress | near-boundary balanced accuracy | 0.7493 | 0.7029 | −0.0464 [−0.0679, −0.0243] |
| stress | safe / unsafe recall | 0.9751 / 0.5582 | 0.9224 / 0.5532 | −0.053 / −0.005 |

Development tells the same story: random ranks −2.60 in the two-method cohort, with g3 boundary balanced accuracy −0.043 [−0.071, −0.016] and stress −0.030 [−0.051, −0.008].

GP-ACL acquisition works as the paper describes:
- 99.6% (g3) and 96.9% (stress) of its added demonstrations touch the obstacle, versus 18% and 17% for random.
- They add 15.3 and 7.6 new unit boundary cells per instance, versus 4.0 and 2.3.

The paper's emphasis on lower error near boundaries is reproduced beyond noise in both experiments. **The paper's headline uniform-space accuracy is not reproduced:** random is level on g3 and slightly better on the stress task. The near-zero prior mean that Eq. 10 needs leaves a weaker safe default far from data, and dense ACL contacts favour short lengthscales that leave large interiors partly unclassified. The "fill" variant fixes exactly that, which is why it is the headroom result. The legacy and learned-prior variants show the trade-off: higher uniform g3 accuracy (0.982–0.984) but clearly worse boundary recovery. The combined score, with unchanged coefficients, ranks the reference above random.

## Shortcuts

The redteam shortcuts were closed by earlier iterations. This iteration replayed them all through the unchanged no-argument test path on the new setup. "Failed" means the attempt fails the whole run and gets null rank (zero scores, ordered below every success).

| Shortcut | Before (REDTEAM) | Fix | After |
|---|---|---|---|
| Public answer oracle | +0.636, perfect | private generator, unseen instances | failed |
| Signed-offset grid ordering | +0.544 | random coordinates, shuffled order | failed |
| Shrink grid | +0.636 on 32 points | evaluator-owned test size | failed |
| False / negative / NaN / −inf demo count | tie / +0.197 / accepted | ledger authentication, strict parsing | failed |
| Excess budget | accepted | parent-owned caps | failed |
| Replace public truth module | +0.636 | evaluator does not import `data/` | failed |
| Move baseline fields | paying tie | frozen raw controls, strict output schema | failed |
| Impossible, empty or null labels; forged flags/scores | accepted or rejected | strict binary output of evaluator size | failed |
| Constant safe / constant unsafe | lost | whole-run recall gate | failed |
| Fixed disk / contact disk | n/a | per-instance pose and shape | failed gate |
| Seven saved legacy artifacts | n/a | static predictions cannot answer unseen instances | failed |
| **New: public-formula fit (reference acquisition, then fit the paper's exact g3 polynomial to all tight states and normals; polar interpolation if the fit is poor)** | FAMILY_FIT_BEFORE | private per-instance deformation of the hidden g3 shapes | **−0.184 [−1.699, +1.074]** |

After the fix, the formula fit still wins uniform g3 accuracy (+0.013) but loses near-boundary balanced accuracy (−0.069). On stress it loses boundary balanced accuracy (−0.058). It no longer pays.

**Left open:**
- **File and evaluator tampering, resource abuse.** Reading `eval/test/`, editing evaluator caches or source, and resource abuse are left to the separate grader.
- **Copying the reference.** A method that reproduces the reference's predictions ties it. That is not a loophole.
- **Stress-family fit.** The stress generator's functional form is the same on development and test. A fit of its ~40 parameters from sparse contacts would amount to a generic radial Fourier fit, and the tested radial fit (`alt_periodic`) ranks −1.418, so this was not addressed further.
- **Star-shape exploitation.** Both families are star-shaped. The fill variant exploits this. It is a legitimate shape prior, and it is the demonstrated headroom.

## Headroom

The reference sits at the bound of no scored metric:

| Experiment | Accuracy | Boundary BA | Near-boundary BA | Safe recall | Unsafe recall |
|---|---:|---:|---:|---:|---:|
| g3 | 0.964 | 0.808 | 0.778 | 0.945 | 0.672 |
| stress | 0.937 | 0.767 | 0.749 | 0.975 | 0.558 |

The fill variant beats it by +1.088 [0.789, 1.378]. Its uniform accuracy gains are +0.0025 [0.0014, 0.0035] on g3 and +0.0192 [0.0132, 0.0254] on stress. Its boundary balanced accuracy gains are +0.026 and +0.013. No tested variant approaches the bounds. The stress experiment (weight 0.4), added by an earlier iteration where the paper's method falls short, remains the place where the gap is largest.

Brute force is bounded by the twelve-demonstration and forty-request caps, and the evaluator owns the oracle. The only route to the hidden obstacle is the authenticated oracle; the public-formula private search was measured above.

## Reference score before and after

These raw numbers are not on the same setup, so they are not directly comparable:
- **Previous reference, test g3:** accuracy 0.9766, balanced accuracy 0.7265, near-boundary BA 0.7051.
- **Previous reference, test stress:** 0.9261 / 0.6379 / 0.6201.
- **Final reference, test g3:** 0.9637 / 0.8084 / 0.7783.
- **Final reference, test stress:** 0.9368 / 0.7667 / 0.7493.

The new setup adds deformations and four initial crossings, and its demonstration budget is twelve. The reference rank is zero by definition. `scoring/scores_test.json` and `scoring/scores_train.json` contain only the reference and the frozen random control. They were regenerated from the final code. The paper's reported numbers are unchanged in `scoring/reference.json` under `_paper_reported_*`.

Weights and coefficients are unchanged (g3 0.6, stress 0.4; accuracy 1.0, balanced accuracy 0.5, near-boundary 0.6, recalls 0.25 each). The paper reports uniform accuracy but stresses boundary errors, and both kinds of metric keep their weight.

## Not fixed, stated plainly

- **Uniform accuracy:** GP-ACL does not beat random on the paper's headline uniform-space accuracy here; see above. The combined score favours GP-ACL through the boundary metrics the paper emphasizes.
- **Scope:** only the 2D double integrator is reproduced. The paper's unicycle, quadcopter, 7-DOF arm and hardware results are not, and the absolute accuracies (0.9956 versus 0.9652) are not reached at this scale.
- **Unspecified choices:** the reference depends on choices the paper does not fix:
  - four initial crossings;
  - a twelve-demonstration budget;
  - P = 4;
  - the one-lengthscale P3 region;
  - the tied prior mean with ρ = 0.1;
  - η = 1;
  - unit-norm gradient estimates;
  - Eq. 10 read as a safe probability (the printed sign contradicts the paper's avoid-set convention).

  All except η and the gradient normalization were selected or confirmed on development data only.
- **Two weaker variants are not clearly weaker on test:** the learned-prior interval touches zero (upper bound +0.069), and P = 2 is indistinguishable from P = 4.
- **Cosmetic:** under parallel evaluation, a probe that fails on every instance still runs its already-queued instances before the run is marked failed. Scoring is unaffected.
