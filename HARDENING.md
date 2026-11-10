# Hardening

Status: four combined verifications ran all variants and probes. The fourth passes structural validation and reference quality, closes every replayed shortcut, and establishes paired headroom. Random acquisition still outranks the reference on both splits, so the scientific reproduction requirement remains unmet. The fifth sparse-initialization run and sixth feasible-initialization run failed development quality gates and were cancelled. The next combined job restores the literal CDF sign in the paper's Equation 10; its outcome is pending. Historical before-results are from `REDTEAM.md`; they are on a different task and ranking scale.

## Findings and changes

The previous test exposed its answer generator, let submissions choose the test grid and comparison metadata, and rewarded unverified demonstration counts. Its active-versus-random effect also reversed across seeds. The previous simplified tight-point and query searches did not implement the paper's optimization programs faithfully enough to justify its claimed milestone.

The implementation now solves the reduced KKT Programs 7/8, derivative-observation GP marginal likelihood, conditioned posterior samples (1,000 RFF, alpha 0.3), and continuous constrained Programs 3–6. It uses T=30 and the paper's double-integrator setting, with positions in the paper's physical units (bounds [-10,10]). Free velocities and controls are eliminated exactly and reconstructed; no velocity/control limits or endpoint velocities are assumed. The simulator uses the paper's squared successive-position cost and locally optimal constrained trajectories. Equation 10 is implemented literally as Phi(+mean/sd). This conflicts with the prose describing a safety probability for g<=0; earlier jobs used a negative mean to match that prose, an interpretation now withdrawn in favor of the printed computational step. Eta=1 (previously 0.1), four initial demonstrations (the cancelled fifth run used one), twenty total demonstrations, and eighty proposal attempts are explicit benchmark settings. Three fits are used. These choices and the added complex-boundary experiment are not presented as the full high-dimensional or hardware reproduction.

Private instance generation, seeds, truth, initial expert trajectories and all frozen comparison raw records are confined to `eval/test/`. Training has an independent implementation and different instances from the same distribution. An online command interface evaluates unseen instances; prediction points arrive only after acquisition ends. Signed-distance order and submission-selected grids are gone. There is no finite public candidate enumeration to solve: acquisition ranges continuously over endpoints on new hidden obstacles.

Every metric is recomputed from binary predictions against evaluator truth. Budgets use the evaluator's query ledger; no count earns a bonus. Invalid values or task failures fail the whole run. Baselines, anchors, test size, operating points and scoring configuration belong to each evaluator. Submissions cannot replace them through output fields or files. All manifests on disk are discovered without a train-score whitelist. Raw response caches are versioned by evaluator, task, command and local Python dependencies.

Rank sums experiment weight times coefficient magnitude times better-direction improvement divided by cohort standard deviation. Failed attempts sort below successful attempts. Zero-variance terms contribute zero. Arithmetic checks cover sign, direction, units, failures and zero variance. The count coefficient was removed: the scientifically relevant constraint is a fixed verified cap, and a constant cap cannot support a useful standardized reward.

## Before-results and planned paired check

The old reference had test accuracy 0.954645, balanced accuracy 0.954823 and near-boundary balanced accuracy 0.899525; its rank was zero by definition. The old random baseline had accuracy 0.931148 and rank -0.244327. Across seeds the active mean accuracy was 0.929872 versus random 0.946084; reference seed rank spread was 0.579845. The old better contact/normal fit ranked +0.473367 (a rounded low-order version +0.636216 and perfect accuracy); dropping gradients ranked -4.462129 and dropping acquisition -0.425626.

The combined job runs the final reference and random control, two contact/normal fits, a normalized-gradient alternative, and weaker gradient/acquisition ablations on identical instances in both splits. Thirty-six paired obstacles per experiment, fixed learner seeds and 12,288 shuffled points per obstacle replace a single lucky seed. Paired bootstrap intervals will quantify candidate-minus-reference effects. The stress experiment has additional fine boundary variation that a simple low-order fit cannot exactly represent. Its weight is 0.4; the actual paper constraint retains weight 0.6. Accuracy remains primary, with boundary and class recall metrics preventing easy broad-region accuracy from hiding boundary failure. The new experiment and weights target the paper's sparse-demonstration and complex-geometry limitation, not agreement with reported numbers.

## First verification and numerical correction

Job 5f293a40-e15 ran every comparison and shortcut. GP/KKT and rank arithmetic checks passed. The final validation failed because the reference and random baseline both failed the 0.60 recall constraint, making their published scores zero. This was a scientific failure, not a train/test wiring failure; the underlying train and test predictions and metrics differed.

The reference's raw test accuracy/boundary-balanced accuracy were 0.646769/0.714698 on g3 and 0.807251/0.658861 on the stress task. Stress safe recall was only 0.488747. Random sampling reached 0.655802/0.760413 on g3 and 0.859762/0.615766 on stress, with stress safe recall 0.296896. Several active instances stayed at their four initial demonstrations, despite an available budget of twelve. Thus this run did not reproduce the paper's active-over-random claim. Two contact/normal alternatives ranked +2.634643 and +3.976682, and a normalized-gradient alternative ranked +4.975243, but improvements over a failed reference do not establish a satisfactory gym. The complete first-run audit is retained in `eval/test/hardening_first_summary.json`.

The earlier position/5 conversion also divided the demonstration cost and raw KKT field scale while leaving the paper's fixed delta=0.001 unchanged. The correction restores physical coordinates, corresponding kernel length scales and narrow-band widths. It retains the actual squared-position cost and unnormalized P8 observations. GP hyperparameters still optimize marginal likelihood, now from three deterministic initializations with broad bounds. P3 uses analytic posterior/sample derivatives and dimensionless objective/residual scaling. In 2D, P4--6 are exactly scalar constrained ray problems: bracketing plus continuous root refinement finds feasible intervals and their nearest/furthest endpoints, avoiding ill-conditioned raw-tau SLSQP solves. Unit ray coordinates only reparameterize tau; Equation 7 still offsets along the raw gradient. No infeasible fallback or new training algorithm is introduced. Up to sixteen independent posterior draws per acquisition stage replace twelve, within the same fixed request/demonstration caps and three paper iterations.

Both evaluators and every reference/control/variant will rerun from the corrected code and physical units. Unchanged evaluator/interface probes use saved raw records where possible. The job preserves source-versioned per-instance checkpoints across preemptions; the first run was preempted twice and previously restarted completed computations.

## Second verification and acquisition setup correction

Job c7495dc7-d8c completed every run and audit. The physical-unit reference passed both recall gates, with test g3 accuracy/balanced accuracy 0.613993/0.757002 and stress 0.830058/0.753100. Random acquisition reached 0.615458/0.770495 and 0.864685/0.743855 respectively; stress safe recall 0.589188 failed its task. That gate failure does not establish the paper's accuracy effect. Training g3 accuracy favored active acquisition (0.677490 versus 0.624125), while stress did not (0.814372 versus 0.820068). The effect remains unproven at twelve paired tasks.

The order-12 contact/normal alternative ranked +2.925454, paired 95% interval [1.799563, 4.429159]. The normalized-gradient alternative ranked +4.358173 [2.799246, 6.052645]. The order-6 fit ranked +1.051076, but its interval [-0.015697, 2.481903] includes zero. None reaches every metric's bound. The weaker variants and all adapted prediction/count/budget shortcuts failed; forged baseline fields attached to authentic reference outputs tied. Complete measurements remain in eval/test/hardening_second_summary.json. These are interim results on the earlier task size, not the final verification.

P3 previously initialized from low-magnitude sampled zeros across the whole box and preferred the largest variance among distant candidates. A zero-mean prior draw has spurious contours away from the inferred obstacle, so this initialization can solicit uninformative demonstrations. Its continuous constrained optimization now starts at observed contacts with small local perturbations, matching the local-NLP solution regime; its objective and constraints are unchanged. Four posterior draws per round support up to eight accepted trajectories per acquisition stage. The same total cap for both active and random methods increases to twenty demonstrations and eighty proposals. This changes data scale rather than the paper's training algorithm; the paper does not specify its demonstration count or posterior draw count. Three fits, raw KKT gradients, marginal-likelihood training, delta and chance levels are retained.

Both splits now use thirty-six paired instances per experiment to resolve task variation. All reference/control/variant runs must rerun on these changed inputs. The final-response schema now rejects forged comparison fields, flags, settings and scores, rather than ignoring them. This closes the adapted baseline-forgery probe that previously tied while returning reference predictions. A legitimate predictor returning the same predictions without falsifying comparison metadata can still tie. The publisher stores scaled, split-specific baseline measurements and failures under _measured_reference_runs; a failed score of zero is not a paper-reported baseline target. Paper numbers remain unchanged. Rank standard deviations use the actual successful cohort, without duplicating the anchor; metric-only edits rescore raw outputs rather than rerun acquisition.

## Third verification and remaining scientific correction

Job b3807db0-4f1 completed with exit 0; `validate.py --post-design` passed. All twelve adapted shortcuts and seven legacy artifacts failed. A forged comparison attached to valid reference predictions now fails the strict schema instead of tying. The third complete audit is preserved in `eval/test/hardening_third_summary.json`.

The following ranks use the same successful audit cohort. Their magnitude depends on the cohort standard deviations; the reference-only score files use a smaller cohort and therefore different scales. Intervals are 95% paired bootstrap intervals with each comparison using identical obstacles and initial observations.

| Variant | Old rank | Third verification test rank | Paired interval |
|---|---:|---:|---|
| Reference | 0 | 0 | Anchor |
| Random acquisition | -0.244327 | +1.029839 | [0.441270, 1.664275] |
| Order-6 contact/normal fit | +0.473367 | +0.736948 | [0.248407, 1.243594] |
| Order-12 contact/normal fit | Not previously measured | +2.319953 | [1.839181, 2.821363] |
| Normalized-gradient fit | Not previously measured | +3.444084 | [2.944394, 3.983856] |
| No acquisition | -0.425626 | -1.328212 | [-1.910500, -0.758164] |
| No gradients | -4.462129 | Failed | Both recall requirements fail |

Better and weaker methods separate beyond paired uncertainty, and no scored reference metric is at its bound of one. No simple tested fit reaches all bounds: the stress experiment retains recovery error even for the stronger variants. This establishes headroom but does not excuse the random baseline outperforming the reference.

| Experiment | Reference accuracy / boundary BA | Random accuracy / boundary BA |
|---|---|---|
| Test g3 | 0.665738 / 0.807662 | 0.680840 / 0.838752 |
| Test stress | 0.806396 / 0.738615 | 0.841621 / 0.773858 |
| Train g3 | 0.634135 / 0.783874 | 0.669752 / 0.833645 |
| Train stress | 0.821038 / 0.765041 | 0.838725 / 0.782521 |

Random-minus-active boundary BA is +0.031091 [0.0093, 0.0528] on test g3 and +0.035242 [0.0086, 0.0636] on stress. The corresponding train differences also favor random. Both controls pass the fixed recall thresholds; changing the thresholds or weights to conceal this reversal would not reproduce the paper. The reference rank remains zero by definition; its raw g3 accuracy changed from the old 0.954645 to 0.665738 on the harder, unseen setup. These raw values are not measurements of the same test.

Saved ledgers show incomplete active allowances: test g3 averages 18.06 demonstrations, with 13/36 below twenty; stress averages 18.00, with 18/36 below twenty and one instance still at four. Random uses twenty everywhere. Of accepted added trajectories, 66.0% (g3) and 65.7% (stress) have identified contacts, versus 37.2% and 35.9% for random. Active acquisition is more contact-efficient, but this does not yet translate into better recovery. The reduced KKT observations must retain the paper's raw magnitudes; promoting the normalized-gradient variant to the reference would change the protected training method.

The next combined correction keeps the likelihood, observation noise model, kernel family, KKT programs, training data, budgets, seeds, test size and all scoring coefficients fixed. It supplies analytic length-scale/signal/noise derivatives to the same marginal-likelihood optimizer, averages the objective over observations for numerical conditioning, and permits up to 200 optimizer iterations. A finite-difference hypergradient check is included in the job. It increases the unspecified Equation 7 step from eta=0.1 to eta=1 in physical units and permits up to twenty-four four-sample acquisition rounds per stage instead of eight, still bounded by eighty actual oracle proposals and twenty accepted demonstrations. Unused first-stage demonstration allowance carries into the second stage; the paper's two acquisition stages and three GP fits remain unchanged, as do alpha, RFF count, delta and chance levels. This addresses premature numerical termination and shallow offsets without adding a surrogate, normalizing the KKT gradients, or discarding straight expert trajectories.

The cited earlier GP paper also contains a demonstration-feasibility hinge loss, but the target paper explicitly specifies marginal likelihood in Section 2.3 and its Appendix A.2 posterior. No hinge loss or additional training objective is introduced on the strength of that citation. All changed methods and controls will rerun from source, and the shortcut replays and paired estimates will be checked in the same job. Until its results arrive the milestone remains `method_runs`.

## Fourth verification and sparse-initialization correction

Job f6c52544-4c0 completed with exit 0. The likelihood hypergradient check and structural validation passed. All twelve adapted shortcuts and seven legacy artifacts failed again. The fourth audit is preserved in `eval/test/hardening_fourth_summary.json`.

| Variant | Old rank | Fourth verification test rank | Paired 95% interval |
|---|---:|---:|---|
| Reference | 0 | 0 | Anchor |
| Random acquisition | -0.244327 | +0.603014 | [-0.059525, 1.300636] |
| Order-6 contact/normal fit | +0.473367 | +0.578652 | [0.168987, 0.997900] |
| Order-12 contact/normal fit | Not previously measured | +1.988915 | [1.597440, 2.406720] |
| Normalized-gradient fit | Not previously measured | +3.457279 | [3.059205, 3.893139] |
| No acquisition | -0.425626 | Failed | Recall requirement fails |
| No gradients | -4.462129 | Failed | Both recall requirements fail |

The reference remains zero by definition. Its raw test g3 accuracy is now 0.633213, boundary balanced accuracy 0.831427, versus random 0.680840 and 0.838752. On stress these are 0.829508/0.748706 versus random 0.836283/0.773515. Train g3 is 0.646023/0.822007 versus random 0.669752/0.833645; train stress is 0.822795/0.760952 versus random 0.838725/0.782521. Random ranks +0.519918 on train. Improved boundary recovery after the numerical correction has not produced the paper's direction. No scored reference metric is at one; the stronger alternatives remain below the bounds, including on stress. Headroom and shortcut closure are supported, while the core claim is still unproven.

The next change is to the common initial dataset, whose size the paper does not specify. Four initial long crossings cover an average 31.9% of twenty-four angular sectors on test g3 and 45.1% on stress, leaving less scope for an acquisition benefit. The first trajectory alone covers 9.8%/15.7%, and has at least two identified contacts on every train and test instance. Both independent simulators now supply this one trajectory, preserving the original world-coordinate endpoints. Every method retains the same total twenty-demonstration and eighty-request allowances. Geometry, seeds, test size, training algorithm and coefficients are unchanged. This is a test of the sparse-demonstration regime, not an adjustment toward a reported score. All reference/control/variant and affected shortcut runs must rerun on the new inputs; old measured scores are interim until that job completes. The combined report now includes paired rank intervals on development as well as test, and budget/contact diagnostics derived from raw ledgers.

## Fifth verification and expert-oracle initialization correction

Job f9466b7c-33b completed the full development audit. Every active variant, including the reference, failed one or both fixed recall requirements; random alone succeeded. The job was cancelled during the first test variant (36 g3 and 33 stress instances complete). There is no complete fifth-job hidden-test comparison. All development scores and partial-test counts are preserved in `eval/test/hardening_fifth_summary.json`; published reference-only scores remain from the fourth completed setup.

| Development method | g3 boundary BA | Stress boundary BA | Whole-run result |
|---|---:|---:|---|
| Reference | 0.608238 | 0.605174 | Failed |
| Random acquisition | 0.806611 | 0.744179 | Passed |
| Order-6 contact/normal fit | 0.614304 | 0.619474 | Failed |
| Order-12 contact/normal fit | 0.649447 | 0.625073 | Failed |
| Normalized-gradient fit | 0.693314 | 0.662915 | Failed |
| No acquisition | 0.604845 | 0.606684 | Failed |
| No gradients | 0.500000 | 0.500000 | Failed |

Failed methods have null rank, not zero-reference improvement. The sparse setup cannot become the ranking anchor. Reducing common initialization left many active instances without useful acquisition: the reference used a mean 7.28 g3 and 16.53 stress demonstrations out of twenty. This regime is withdrawn, rather than relaxing the success constraints.

Raw trajectories show a likely upstream problem: 26/36 initial g3 demonstrations have raw KKT gradient norm above one and jump across the narrow waist between adjacent knots. The old oracle solved five infeasible straight/bowed initial guesses and selected the cheapest feasible result. These jumps are allowed by the discrete P1 constraints; they are not claimed to be invalid trajectories. However, their large raw gradients differ sharply in magnitude from those of short local queries, and a joint derivative-observation GP must fit both. Changing the 1e-4 contact tolerance to 1e-5 barely changes the identified contacts and would not address that discrepancy.

The next combined verification restores the four fixed common initial endpoint pairs and initializes the expert from feasible polar routes. It returns the first successful feasible local optimum, trying the other angular direction only if the first solve fails. A feasible straight line remains the exact unconstrained optimum. This retains the paper's cost, discrete constraints, dynamics, T=30 and local-optimality regime; it changes solver initialization and local-solution selection, which the paper leaves unspecified. No continuous-collision, speed or smoothness constraint is added, and no reference-gradient normalization or new training loss is introduced. Both independent simulators use the same procedure with their own hidden instances. The GP-ACL algorithm, budgets, seeds, test size and coefficients remain unchanged.

The job first verifies expert feasibility, reconstructed dynamics, stationarity and complementary slackness on representative initial paths, then reruns all changed controls and variants and the affected shortcut probes in one job. It reports gradient and longest-edge distributions from the raw ledgers. Source digests invalidate every affected method/oracle cache. There is no verified improvement from this correction yet. The reference's paper-direction requirement remains unresolved and the milestone stays `method_runs`.

## Sixth verification and restoration of printed Equation 10

Job 3df03bad-304 completed the development reference and random controls, then was cancelled after six g3 test reference instances. The independent expert check passed feasibility, reconstructed dynamics and KKT stationarity: representative maximum residuals were 3.29e-7 (g3) and 3.27e-7 (stress). Feasible warm starts addressed large initial trajectory jumps, but did not restore the scientific result. Evidence is preserved in `eval/test/hardening_sixth_summary.json`.

| Development method / experiment | Accuracy | Boundary balanced accuracy | Near-boundary balanced accuracy | Safe recall | Unsafe recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference / g3 | 0.766093 | 0.704023 | 0.690386 | 0.831022 | 0.577024 |
| Random / g3 | 0.873678 | 0.700312 | 0.665122 | 0.641982 | 0.758642 |
| Reference / stress | 0.821906 | 0.737435 | 0.660748 | 0.680756 | 0.794115 |
| Random / stress | 0.833842 | 0.768865 | 0.681805 | 0.701984 | 0.835745 |

The reference fails the fixed 0.60 unsafe-recall gate on g3, so its rank is null. Random alone succeeds and has rank zero in this two-control cohort. No complete test comparison or new headroom result is claimed. Fourth-job published measurements remain interim; failed controls do not replace a verified reference anchor.

Rechecking [the paper's Equation 10](https://arxiv.org/html/2512.22757v1) exposed a remaining procedural departure: the implementation negated the posterior mean inside the CDF to match the prose's safe-set convention. The next job restores the printed positive sign in both endpoint programs. The independent sampled-function constraint and the expert's rejection of truly infeasible endpoints remain in place. A numerical regression check pins both endpoint inequalities. This is a restoration of a paper computational step, not a new training method. The prose/equation inconsistency is unresolved; we do not describe the positive-sign expression as the probability of g<=0.

No other scientific setting changes: feasible expert initialization, four initial trajectories, raw P8 gradient magnitudes, joint marginal likelihood, three fits, shared seeds, thirty-six paired instances, budgets, recall gates and weights are fixed. Per-stage GP fit diagnostics go to stderr and are retained as opaque raw text, never read by scoring. The combined job reruns changed reference/control/variant commands on both splits, replays the fixed shortcuts, measures paired uncertainty, and publishes only successful reference runs. Results from the restored equation are pending.

## Printed-equation audit interrupted by infrastructure

Job c5165ee2-143 was preempted for a SLURM worker drain; its stderr reports that the worker node died. It passed the GP/KKT/ray, printed Equation 10, rank arithmetic and expert checks. Twelve development g3 reference instances were saved with no protocol failures. No complete control comparison, test score or new headroom result exists. This interruption does not justify a scientific patch. The unchanged combined audit resumes from source-versioned, atomically saved records; the entrypoint prints a checkpoint inventory for progress reporting.

## Shortcut replay

After-values below are the completed fourth verification, before the pending printed-equation rerun. All failures sort below every successful attempt.

| Shortcut | Before rank / result | Fix | After |
|---|---|---|---|
| Public answer oracle | +0.636216; perfect, zero demos | Private generator and unseen obstacle instances | Failed; rank null, score 0 |
| Signed-offset grid ordering | +0.544332; accuracy 0.993989 | Independent random coordinates, shuffled labels/order | Failed; rank null, score 0 |
| Shrink grid to eight angles | +0.636216; perfect on 32 points | Evaluator-owned full test size | Failed; rank null, score 0 |
| False zero demo count | Approximately zero, slightly positive tie | Authenticated ledger; inconsistent declaration fails | Failed; rank null, score 0 |
| Negative demo count | +0.197320 | Nonfinite/impossible/inconsistent declarations fail; no count reward | Failed; rank null, score 0 |
| Replace public truth | +0.636216; perfect | No evaluator dependency on public `data/` | Failed; rank null, score 0 |
| Replace baseline predictions/count | Zero, paying tie while falsifying comparison | Immutable evaluator-owned baseline raw predictions; forged comparison fields rejected | Failed; rank null, score 0 |
| Impossible labels equal to two | Accepted with rank -9.730710 | Strict finite binary labels | Failed; rank null, score 0 |
| Forged scores/flags | Already lost at -4.462129 | Recomputed metrics and success gate | Failed; rank null, score 0 |
| Nonfinite counts | Previously accepted, finite rank unconfirmed | Strict JSON parser and count validation | Failed; rank null, score 0 |
| Excess demonstration budget | Accepted, rank -0.225793 | Parent-owned accepted-demo and proposal caps | Failed; rank null, score 0 |

The old saved shortcut artifacts are also passed directly to the new test; a static array cannot answer unseen instances. Adapted probes exercise the same attacks through the online interface. A persistent change to the former public truth module and an overwritten baseline manifest are included in the same job. Only the required valid reference runs populate `scoring/scores*.json`; probes and variants are retained in the audit summary.

## Remaining limits

No claim of full reproduction, high-dimensional performance, hardware performance, or worst-case safety is made. The 0.60 safe/unsafe recall gate is an explicit benchmark success threshold, not a paper theorem. Per-instance active learning is necessary to solve this task; test scoring never trains an offline model or retrains the fixed baseline. The evaluator reuses its own raw observations when rescoring unchanged commands.

File access to private test files, tampering with evaluator source or caches, and resource abuse are left to the separate attempt grader. No defenses against leftover processes or Python environments were added. Copying a legitimate reference predictor can tie its predictions. Forged comparison fields violate the strict output schema and fail, even if predictions otherwise match the reference. An honest alternative with indistinguishable predictions is a headroom question, not a metric-forgery loophole. The fourth run demonstrates paired headroom and matched train/test difficulty, but fails to demonstrate the paper's active-versus-random claim. This remains an explicit unresolved requirement pending the printed-equation rerun; no `core_claim` or completed hardening status is asserted.
