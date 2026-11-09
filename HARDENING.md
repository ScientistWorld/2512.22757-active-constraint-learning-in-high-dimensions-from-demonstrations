# Hardening

Status: all fixes below are applied together; the combined verification job is pending. Numerical after-results must be filled from `eval/test/hardening_summary.json`, not inferred from the patches. Historical before-results are from `REDTEAM.md`; they are on a different task and ranking scale.

## Findings and changes

The previous test exposed its answer generator, let submissions choose the test grid and comparison metadata, and rewarded unverified demonstration counts. Its active-versus-random effect also reversed across seeds. The previous simplified tight-point and query searches did not implement the paper's optimization programs faithfully enough to justify its claimed milestone.

The implementation now solves the reduced KKT Programs 7/8, derivative-observation GP marginal likelihood, conditioned posterior samples (1,000 RFF, alpha 0.3), and continuous constrained Programs 3–6. It uses T=30 and the paper's double-integrator setting, with positions in units of five paper units. Free velocities and controls are eliminated exactly and reconstructed; no velocity/control limits or endpoint velocities are assumed. The simulator uses the paper's squared successive-position cost and locally optimal constrained trajectories. The paper's safe convention requires Phi(-mean/sd); the positive sign printed in its chance expression conflicts with that convention. Eta=0.1, four initial demonstrations, twelve total demonstrations, and forty proposal attempts are explicit benchmark settings. Three fits are used. These choices and the added complex-boundary experiment are not presented as the full high-dimensional or hardware reproduction.

Private instance generation, seeds, truth, initial expert trajectories and all frozen comparison raw records are confined to `eval/test/`. Training has an independent implementation and different instances from the same distribution. An online command interface evaluates unseen instances; prediction points arrive only after acquisition ends. Signed-distance order and submission-selected grids are gone. There is no finite public candidate enumeration to solve: acquisition ranges continuously over endpoints on new hidden obstacles.

Every metric is recomputed from binary predictions against evaluator truth. Budgets use the evaluator's query ledger; no count earns a bonus. Invalid values or task failures fail the whole run. Baselines, anchors, test size, operating points and scoring configuration belong to each evaluator. Submissions cannot replace them through output fields or files. All manifests on disk are discovered without a train-score whitelist. Raw response caches are versioned by evaluator, task, command and local Python dependencies.

Rank sums experiment weight times coefficient magnitude times better-direction improvement divided by cohort standard deviation. Failed attempts sort below successful attempts. Zero-variance terms contribute zero. Arithmetic checks cover sign, direction, units, failures and zero variance. The count coefficient was removed: the scientifically relevant constraint is a fixed verified cap, and a constant cap cannot support a useful standardized reward.

## Before-results and planned paired check

The old reference had test accuracy 0.954645, balanced accuracy 0.954823 and near-boundary balanced accuracy 0.899525; its rank was zero by definition. The old random baseline had accuracy 0.931148 and rank -0.244327. Across seeds the active mean accuracy was 0.929872 versus random 0.946084; reference seed rank spread was 0.579845. The old better contact/normal fit ranked +0.473367 (a rounded low-order version +0.636216 and perfect accuracy); dropping gradients ranked -4.462129 and dropping acquisition -0.425626.

The combined job runs the final reference and random control, two contact/normal fits, a normalized-gradient alternative, and weaker gradient/acquisition ablations on identical instances in both splits. Twelve paired obstacles per experiment, fixed learner seeds and 12,288 shuffled points per obstacle replace a single lucky seed. Paired bootstrap intervals will quantify candidate-minus-reference effects. The stress experiment has additional fine boundary variation that a simple low-order fit cannot exactly represent. Its weight is 0.4; the actual paper constraint retains weight 0.6. Accuracy remains primary, with boundary and class recall metrics preventing easy broad-region accuracy from hiding boundary failure. The new experiment and weights target the paper's sparse-demonstration and complex-geometry limitation, not agreement with reported numbers.

## Shortcut replay

| Shortcut | Before rank / result | Fix | After |
|---|---|---|---|
| Public answer oracle | +0.636216; perfect, zero demos | Private generator and unseen obstacle instances | Pending |
| Signed-offset grid ordering | +0.544332; accuracy 0.993989 | Independent random coordinates, shuffled labels/order | Pending |
| Shrink grid to eight angles | +0.636216; perfect on 32 points | Evaluator-owned full test size | Pending |
| False zero demo count | Approximately zero, slightly positive tie | Authenticated ledger; inconsistent declaration fails | Pending |
| Negative demo count | +0.197320 | Nonfinite/impossible/inconsistent declarations fail; no count reward | Pending |
| Replace public truth | +0.636216; perfect | No evaluator dependency on public `data/` | Pending |
| Replace baseline predictions/count | Zero, paying tie while falsifying comparison | Immutable evaluator-owned baseline raw predictions; candidate fields ignored | Pending |
| Impossible labels equal to two | Accepted with rank -9.730710 | Strict finite binary labels | Pending |
| Forged scores/flags | Already lost at -4.462129 | Recomputed metrics and success gate | Pending |
| Nonfinite counts | Previously accepted, finite rank unconfirmed | Strict JSON parser and count validation | Pending |
| Excess demonstration budget | Accepted, rank -0.225793 | Parent-owned accepted-demo and proposal caps | Pending |

The old saved shortcut artifacts are also passed directly to the new test; a static array cannot answer unseen instances. Adapted probes exercise the same attacks through the online interface. A persistent change to the former public truth module and an overwritten baseline manifest are included in the same job. Only the required valid reference runs populate `scoring/scores*.json`; probes and variants are retained in the audit summary.

## Remaining limits

No claim of full reproduction, high-dimensional performance, hardware performance, or worst-case safety is made. The 0.60 safe/unsafe recall gate is an explicit benchmark success threshold, not a paper theorem. Per-instance active learning is necessary to solve this task; test scoring never trains an offline model or retrains the fixed baseline. The evaluator reuses its own raw observations when rescoring unchanged commands.

File access to private test files, tampering with evaluator source or caches, and resource abuse are left to the separate attempt grader. No defenses against leftover processes or Python environments were added. Copying a legitimate reference predictor can tie its predictions; ignored forged comparison fields do not justify failing otherwise valid outputs. An honest alternative with indistinguishable predictions is a headroom question, not a metric-forgery loophole. Final headroom, reference effect, train/test comparability and any unresolved failures must be reported from the job before completion.
