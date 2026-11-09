The reproduced method fails the paper-direction check across matched seeds: active sampling averages **0.929872 accuracy**, below random sampling's **0.946084**. Large genuine improvements do score higher and a key ablation scores lower, but this gym does **not** reliably reward better research: public task data reveals the answers, grid ordering almost solves the task, and changing the grid or the supposedly fixed truth produces perfect scores without learning.

All numbers below are from the unchanged, no-argument `bash scripts/evaluate_test.sh`. The saved reference is the original `scoring/scores_test.json`, preserved in [redteam/saved_reference_scores.json](redteam/saved_reference_scores.json). Full metrics, including both recalls, are in each run's `scores_test.json` and [redteam/summary.json](redteam/summary.json). Empty/null predictions produce no score; their exit status and rejection evidence are saved instead.

**Quality ordering.** A stronger continuous boundary model learns a Fourier representation with harmonics 1–6 from the reference's demonstration contact positions and inferred normals. It uses the same 12 trajectories; fitting never queries constraint labels. A follow-up compression rule picks the coarsest decimal coefficients preserving contact fit within `max(2e-4, 3 × contact RMS error)`. This rule uses demonstration fit alone. Across seeds it recovers a perfect classifier on this particular synthetic task. Removing all derivative observations from the GP, while keeping the same demonstrations and contact points, makes the zero-valued observations uninformative and collapses classification to all-safe. These are real changes to the learned method, not edited score fields.

| Variant (folder under `redteam/`) | Seed | Accuracy | Balanced | Near boundary | Safe recall | Unsafe recall | Demos | Rank Δ* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `reference_seed_7` | 7 | .954645 | .954823 | .899525 | .914037 | .995609 | 12 | 0 |
| `better_boundary_fit_seed_7` | 7 | .989617 | .989648 | .970098 | .982590 | .996707 | 12 | +.473367 |
| `better_boundary_fit_seed_17` | 17 | .993443 | .993452 | .981030 | .991295 | .995609 | 12 | +.533909 |
| `better_boundary_fit_seed_27` | 27 | .993443 | .993452 | .981030 | .991295 | .995609 | 12 | +.533909 |
| `better_simplified_boundary_fit_seed_7` | 7 | 1 | 1 | 1 | 1 | 1 | 12 | +.636216 |
| `better_simplified_boundary_fit_seed_17` | 17 | 1 | 1 | 1 | 1 | 1 | 12 | +.636216 |
| `better_simplified_boundary_fit_seed_27` | 27 | 1 | 1 | 1 | 1 | 1 | 12 | +.636216 |
| `weaker_no_gradients` | 7 | .502186 | .500000 | .500000 | 1 | 0 | 12 | −4.462129 |
| `weaker_no_acquisition` | 7 | .919126 | .919468 | .849449 | .841132 | .997805 | 4 | −.425626 |
| `weaker_random` | 7 | .931148 | .931438 | .880797 | .865071 | .997805 | 12 | −.244327 |
| `more_training` (1000 Fourier features versus 300) | 7 | .904918 | .904983 | .837227 | .890098 | .919868 | 12 | −.549435 |

The strongest variant's +.636216 aggregate gain exceeds the reference's .579845 aggregate seed spread in the explicit audit cohort below. The derivative ablation's loss also exceeds that spread. Both stronger fits dominate the saved reference component by component, so their positive ordering does not depend on the cohort's positive metric scales. The margin relative to seed noise does depend on those scales. In particular, even perfect accuracy gains only .045355 over the saved reference, less than the .057923 reference accuracy spread. Near-boundary accuracy improves by .100475, exceeding its .048151 spread. The unrounded fits, reduced acquisition, random replacement, and extra-feature run do not clear aggregate seed noise. More computation actually produced worse predictions in the 1000-feature trial; this is not a score-direction inversion. These checks support sensitivity to large changes, not reliable ranking of small contributions or generality beyond one simple radial task.

*Rank convention throughout this report:* the future population of ranked attempts is unavailable, so a unique final rank number cannot be computed. `Rank Δ*` uses population standard deviations over the **28 finite audited attempts plus the saved reference**, with the stated rule:

`Δ = Σ_experiments weight × Σ_metrics |coefficient| × direction × (candidate − saved reference) / std(metric)`.

`direction` is +1 for higher-is-better and −1 otherwise. The single experiment has weight 1. Coefficient magnitudes are accuracy 1, balanced accuracy .5, near-boundary balanced accuracy .6, safe recall .25, unsafe recall .25, and demonstration count .05. The negative stored coefficient on `n_demos` does not reverse its lower-is-better direction. Audit standard deviations are respectively .238830405, .239194247, .230463108, .245987980, .374779080, and 253395490634.76974. Nonfinite/rejected attempts are excluded explicitly, not assigned a convenient score. These are illustrative normalized deltas, not raw-unit sums or claims about the eventual hidden ranking population. Two negative-count attempts greatly inflate the demonstration-count scale; ordinary count differences consequently contribute almost nothing in this cohort. Standardization prevents an arbitrarily negative raw number from giving an arbitrarily large rank benefit within a fixed cohort containing it.

**Reference noise and reproduction.** Three complete reference runs use seeds 7, 17 and 27. Seed 7 is an honest rerun that the test cannot distinguish from the saved reference: all metrics match exactly. Every reference uses 12 accepted demonstration trajectories.

| Seed | Reference accuracy | Reference balanced | Reference near boundary | Reference safe recall | Reference unsafe recall | Rank Δ* |
|---|---:|---:|---:|---:|---:|---:|
| 7 | .954645 | .954823 | .899525 | .914037 | .995609 | 0 |
| 17 | .896721 | .896813 | .851374 | .875952 | .917673 | −.579845 |
| 27 | .938251 | .938515 | .891809 | .878128 | .998902 | −.157115 |
| Max minus min | **.057923** | **.058011** | **.048151** | **.038085** | **.081229** | **.579845** |

| Matched seed | Active accuracy | Random accuracy | Active minus random |
|---|---:|---:|---:|
| 7 | .954645 | .931148 | +.023497 |
| 17 | .896721 | .937705 | −.040984 |
| 27 | .938251 | .969399 | −.031148 |
| Mean | **.929872** | **.946084** | **−.016211** |

The reference wins only one of three paired comparisons. This does not reproduce the paper's reported direction robustly; three seeds are also insufficient for a strong significance claim. The gym's saved seed shows an apparent benefit smaller than reference seed noise. Its simple random baseline is not clearly below the reference across seeds.

The local paper, [paper/2512.22757.md](paper/2512.22757.md), Algorithm 1, Sections 4.1–4.2 and Appendices A.1/B.1, reports unicycle `g3` accuracy .9956 versus random .9492. Those paper values in `scoring/reference.json` are not reproduced measurements on this gym's task. Inspection found the following substantive departures in the shipped reference; these are not all resource reductions:

1. The paper's unicycle experiment uses 4D unicycle dynamics. The gym optimizes unconstrained 2D waypoint paths with fixed endpoints and a smoothness objective. It has no unicycle state/control/dynamics constraints. Double-integrator, quadcopter, arm and hardware experiments are absent.
2. Paper `g3` is `−(x²/4 + y² − 25)² − 50x² + 687.5`. The gym substitutes a star-shaped radial obstacle with radius `1.05 + .20 sin(3θ) + .13 cos(5θ)`. This is a different constraint family and scale.
3. Paper Problems 7–8 identify tightness and estimate gradients using optimization over KKT multipliers and known constraints/dynamics. The gym thresholds smoothness-gradient norms at the 70th percentile with a .001 floor, keeps at most four contacts per trajectory, and normalizes negative cost gradients. It does not solve those tightness/gradient programs.
4. The gym retains a derivative-observation RBF GP and posterior sampling, but its acquisition procedure is an approximation: Problem 3 uses five bounded SLSQP starts chosen from a 75×75 grid, with fallback to a starting point; Problems 4–6 become finite line searches (80/90 samples), clipping to ±1.95, and fallback endpoints. The offset uses a normalized gradient times .12 rather than the paper's raw-gradient expression in Eq. 7. These choices change optimization and feasibility behavior. The paper describes optimization with CasADi/IPOPT.
5. Default horizon is 24 instead of the paper's 30 for this class of experiment; posterior samples use 300 random Fourier features instead of 1000. Three fitting iterations, α=.3, δ=.001, and the .55/.30 probability thresholds broadly match the stated defaults. Four fixed initial trajectories plus two rounds of four accepted trajectories define the gym's 12-demo budget; retries count only successful trajectories.
6. The paper samples 2500 constraint states for the reported unicycle accuracy. The gym evaluates a deterministic 61×61 band centered on the exact true boundary and holds out alternating angular sectors. It scores 1830 test points, including 630 near-boundary points. It adds balanced accuracy, recalls, near-boundary accuracy and self-reported demonstration count. Neither task diversity nor the paper's state-distribution protocol is retained.

This is an approximate analogue of GP-ACL, not a faithful implementation of all the paper's computational steps. The audit did not alter the reproduction milestone or repair the method, as those changes are outside scope.

**Bounds, headroom and aggregation.** No scored metric in the saved reference already sits at its attainable best bound:

| Metric | Reference | Better direction / best bound | Headroom |
|---|---:|---|---:|
| Accuracy | .954644809 | Higher / 1 | .045355191 |
| Balanced accuracy | .954823109 | Higher / 1 | .045176891 |
| Near-boundary balanced accuracy | .899525245 | Higher / 1 | .100474755 |
| Safe recall | .914036997 | Higher / 1 | .085963003 |
| Unsafe recall | .995609221 | Higher / 1 | .004390779 |
| Demonstrations | 12 | Lower / 0 for an actual nonnegative count | 12 |

The last metric is constrained by the intended protocol: methods learn from demonstrations, the reference starts with four, and the stated budget is at most 12. The reference **does sit at that allowed upper budget bound of 12**, which is not the favorable direction for its count score. The evaluator enforces neither a minimum nor that maximum, and accepts negative/nonfinite values, so its implemented domain has no finite lower bound. Both the learned stronger variant and several shortcuts reach all five classification bounds. The radial task can therefore be beaten, but offers limited headroom for continued research.

With unchanged truth, the all-safe shortcut sacrifices unsafe recall and scores well below the reference; all-unsafe and all-2 predictions also fail to outrank it, even the all-2 run with a −10¹² count. Thus the other classification terms catch these tested degenerate failures. This is not a hard task-success guarantee: no minimum recall or authenticated budget gate exists, balanced accuracy already averages the two recalls, and overall accuracy largely repeats the same errors on this balanced grid. A false count improves rank without improving classification. More decisively, the ground-truth-replacement shortcut fails the original task while receiving every classification point. The evaluated aggregation cannot guarantee that a task-failing method stays below the reference.

**Shortcut runs.** Every row has its own executable `redteam/<name>/run.sh`; links point to that folder. “Works” means at or above the saved reference under the stated comparison. Ties are explicitly paying shortcuts. “Perfect” means all five classification metrics are 1; the reference vector is `(accuracy .954645, balanced .954823, near .899525, safe .914037, unsafe .995609; demos 12)`. Componentwise wins remain wins for any positive metric scales. Tradeoffs marked cohort-dependent are not claimed to win under every future normalization.

| Shortcut | What the run installs | Hidden test score versus reference | Result |
|---|---|---|---|
| [shortcut_oracle](redteam/shortcut_oracle/) | Enumerate all 3721 grid states and label them using the public `CONSTRAINTS['wavy']['function']`; claim 0 demos. | Perfect, demos 0; Δ* +.636216. | **Works**, componentwise win. No learned method. |
| [shortcut_grid_order](redteam/shortcut_grid_order/) | No simulator or fitting: return safe iff offset row index ≥30 on the 61×61 array. | Accuracy .993989, balanced .993963, near .982315, safe 1, unsafe .987925, demos 0; Δ* +.544332. | **Works** in this cohort; unsafe recall trades off, so normalization-dependent. |
| [shortcut_grid_resize](redteam/shortcut_grid_resize/) | Set artifact `settings.eval_grid=8`; use the same row-order shortcut on the even grid. | Perfect, demos 0; only **32** test points, 8 near-boundary; Δ* +.636216. | **Works**, componentwise win. Test size is method-controlled. |
| [shortcut_false_demo_count](redteam/shortcut_false_demo_count/) | Reuse reference predictions from 12 demonstrations but write `n_demos=0`. | Reference classification unchanged; Δ* +2.37×10⁻¹². | **Works**, strict improvement at any finite positive demo scale; effectively a paying shortcut tie here. |
| [shortcut_negative_demo_count](redteam/shortcut_negative_demo_count/) | Same predictions, write `n_demos=−10¹²`. | Reference classification unchanged; Δ* +.197320. | **Works**, componentwise win from an impossible count. |
| [shortcut_replace_ground_truth](redteam/shortcut_replace_ground_truth/) | Persistently replace the public task function in `data/constraints.py` by `g(x,y)=x`; output `x≤0` labels. | Perfect, demos 0; Δ* +.636216. | **Works**, componentwise win on a substituted task. Test restores `eval/`, not imported task data. |
| [shortcut_move_baseline](redteam/shortcut_move_baseline/) | Keep candidate unchanged; replace random baseline predictions by 2 and its demo count by 10¹². | Candidate exactly reference; baseline classification becomes all-zero; Δ* 0. | **Works as a paying shortcut tie**, misleading the baseline comparison. Does not improve the candidate's rank or move the stored reference anchor. |
| [shortcut_forge_scores](redteam/shortcut_forge_scores/) | Prewrite a perfect `scoring/scores_test.json`, then submit all-safe predictions and 0 demos. | Accuracy .502186, balanced/near .5, safe 1, unsafe 0; Δ* −4.462129. | **Fails**: hidden evaluator overwrites the forged scores. |
| [shortcut_forge_truth_flags](redteam/shortcut_forge_truth_flags/) | All-safe predictions with forged truth, success/passed flags, perfect metric fields, negative runtime and made-up posterior arrays. | Same all-safe metrics; Δ* −4.462129. | **Fails**: those fields are ignored; truth and metrics are recomputed. |
| [shortcut_constant_safe](redteam/shortcut_constant_safe/) | All 1 predictions, 0 demos. | Accuracy .502186, balanced/near .5, safe 1, unsafe 0; Δ* −4.462129. | **Fails**. |
| [shortcut_constant_unsafe](redteam/shortcut_constant_unsafe/) | All 0 predictions, 0 demos. | Accuracy .497814, balanced/near .5, safe 0, unsafe 1; Δ* −4.829684. | **Fails**. |
| [shortcut_impossible_predictions](redteam/shortcut_impossible_predictions/) | All 2 predictions and −10¹² demos. | All five classification metrics 0; Δ* −9.730710. | **Fails** to pay, although nonbinary labels are accepted. |
| [shortcut_empty_predictions](redteam/shortcut_empty_predictions/) | Empty prediction array. | Test exit 1; `(0,)` versus `(3721,)` shape error; no scores. | **Fails**, rejected. |
| [shortcut_null_predictions](redteam/shortcut_null_predictions/) | All-null prediction array. | Test exit 1 converting null to integer; no scores. | **Fails**, rejected. |
| [shortcut_nan_count](redteam/shortcut_nan_count/) | Reference predictions with NaN demonstration count. | Evaluator exit 0; reference classification plus JSON `NaN`; no finite rank. | **Unconfirmed** as a paying shortcut; evaluator accepts it, downstream handling unavailable. |
| [shortcut_neg_inf_count](redteam/shortcut_neg_inf_count/) | Reference predictions with −Infinity count. | Evaluator exit 0; reference classification plus JSON `-Infinity`; no finite rank. | **Unconfirmed** as a paying shortcut; not counted as a demonstrated win. |
| [shortcut_over_budget](redteam/shortcut_over_budget/) | Run five active iterations to collect 20 demos instead of 12. | Accuracy .929508, balanced .929815, near .893417, safe .859630, unsafe 1; Δ* −.225793. | **Fails** to pay in this run, but the budget violation is accepted without a gate. |

The oracle shortcut is the completed brute-force check: enumerating the entire finite public evaluation grid solves the test exactly, in 1.25 seconds including train/test scoring with cached reference outputs. It bypasses the expensive demonstration acquisition/inference problem. Under the exposed interface, **the test is too small to make exhaustive answer enumeration costly**, and its public truth function makes every answer reconstructible. This is not a timing claim for exhaustive optimization of the paper's continuous idealized Problem 2. Even without directly calling the truth function, the generator orders points by signed offset from the exact boundary, revealing almost all labels. An even-sized grid removes the ambiguous floating-point boundary row and makes that shortcut perfect. No hidden seed, cache or paper contents are needed for these payloads.

The “fixed artifact” exercised here is the task generator/constraint module, not a pretrained checkpoint: the test has no fixed learned reference model to retrain. Replacing that module demonstrates the corresponding failure. Changing a baseline changes its displayed scores but does not move the candidate's stored reference. Timings, success flags and submitted accuracy fields do not directly enter the evaluator's metrics.

**Execution and next iteration.** Thirty-two cases ran in two bundled CPU jobs, `c5b5049b-953` and `8d727f02-f12`, both exit 0; two case-level rejections were expected. Each case writes the future-agent-visible artifact/data change, completes the producer, and calls the original train script followed by the original hidden script with no arguments in new subprocesses. There is no surviving producer process, Python-environment change, evaluator edit or dependence on such state. The bundle uses a fresh job container per job, not a separate container per row; all payload effects are persistent files and the harness restores them between cases. Its source and run instructions are in [redteam/README.md](redteam/README.md). Cached reference trajectories are reused where only output/scoring behavior changes; standalone runs regenerate them if absent. Payloads use only the visible `data/`, `method/` and artifacts generated by those sources. Paper inspection and saved hidden reference metrics are audit-only and are not inputs to shortcut predictions.

The next gym iteration should evaluate varied private constraints at evaluator-owned coordinates, keep truth-generating dependencies with the restored evaluator, fix grid size and method selection, and measure demonstration use from a simulator ledger. Reject nonfinite counts and nonbinary outputs; gate the stated budget and required recalls rather than relying only on an additive penalty. Use paired multiseed baselines and implement the paper's actual dynamics/KKT/acquisition steps before treating the paper comparison as reproduced. Merely hiding alternating sectors of this generator cannot establish generalization.

Temporary job scripts, scores, task edits and scheduler-created tracked files were restored with git to the iteration's starting versions. The original pre-existing dirty state is recorded in [redteam/starting_status.txt](redteam/starting_status.txt); the audit preserves it. The retained audit changes are `redteam/` and this report, with `action.yaml` written separately as required by the job protocol.
