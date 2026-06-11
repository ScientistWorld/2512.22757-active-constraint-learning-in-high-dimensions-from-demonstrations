# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-11 16:41] - core_claim
- Verified existing GP-ACL reproduction artifacts and scoring contract.
- `python validate.py --compare` passes.
- Saved scores support the core claim on the scaled nonlinear 2D obstacle-recovery task: GP-ACL accuracy 0.9492 versus random-query baseline accuracy 0.9371 with the same 12 demonstrations.

### [2026-06-11 16:45] - method_runs
- Strengthened the method implementation to use the paper's conditioned random Fourier feature posterior-sample formula and marginal-likelihood GP hyperparameter fitting.
- Replaced the active query boundary pick with a constrained solve for the maximally-informative tight state, and added the 2D version of the hyperplane-direction search.
- The previous core-claim scores are now stale until the revised method is rerun in the compute container.

### [2026-06-11 17:00] - method_runs
- Audited method and evaluation separation after milestone reset.
- Fixed `scripts/reproduce.sh` to clear managed output artifacts before an end-to-end rerun, preventing stale scores if a later stage fails.

### [2026-06-11 17:18] - method_runs
- Job `a86544e3-6dd` reran successfully with fresh artifacts, but the strengthened method only scored 0.9315 versus 0.9258 for random sampling.
- Diagnosed the weak result as active query under-acceptance: GP-ACL ended with 10 demonstrations while the baseline used 12.
- Updated the method driver to keep issuing active acquisition rounds until the requested accepted-demonstration budget is filled, bounded by `--max-query-rounds`.

### [2026-06-11 17:35] - core_claim
- Job `13644732-d99` reran successfully after the equal-budget active-query fix.
- GP-ACL scored 0.9497 accuracy versus 0.9258 for random sampling on the scaled nonlinear 2D boundary-band obstacle recovery task.
- Both methods used 12 demonstrations, so the result supports the paper's central claim that active constraint-learning queries improve recovery without increasing the demonstration budget.

### [2026-06-11 17:30] - core_claim
- Expanded evaluator reporting to include the constraint-balance metrics declared in `scoring/reference.json`.
- Submitting a CPU validation job to regenerate `scoring/scores.json` and rerun `python validate.py --compare` through the standard reproduction path.

### [2026-06-11 17:25] - core_claim
- Added a second scaled nonlinear 2D obstacle setting (`scaled_2d_pinched_recovery`) to test transfer beyond the original wavy boundary benchmark.
- Updated the standard run scripts and evaluator to score both settings from method-written prediction artifacts.
- Submitting a CPU validation job; the current verified milestone remains `core_claim` until those results are available.

## Stop Justification
- Completed at milestone `core_claim`.
