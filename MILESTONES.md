# Reproduction Milestones

**Current: method_runs**

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

## Stop Justification
- Not stopped. A CPU validation job is being submitted to rerun the revised reproduction.
