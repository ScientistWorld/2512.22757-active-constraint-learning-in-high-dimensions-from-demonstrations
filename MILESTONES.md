# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-11 16:30] - core_claim
- Implemented and ran a scaled GP-ACL reproduction of the paper's active constraint learning algorithm.
- Verified the core claim on a nonlinear 2D boundary-band obstacle recovery benchmark: GP-ACL outperformed the random-query baseline under the same 12-demonstration budget.
- `python validate.py --compare` passes, including scoring structure and eval/method separation checks.

### [2026-06-11 16:30] - method_runs
- Continuation audit reset the active milestone to `method_runs` pending revalidation.
- Fixed evaluator honesty issue by recomputing held-out labels in `eval/` from `data/constraints.py` instead of trusting output artifact labels.
- Local structural validation passes; full end-to-end scoring needs a container job because login-node Python lacks `numpy`.

### [2026-06-11 16:38] - core_claim
- CPU container job `fc134803-4b7` completed successfully after the evaluator fix.
- Strengthened evaluator reproduced the same core result: GP-ACL accuracy 0.9492 versus random-query baseline 0.9371 with 12 demonstrations each.
- `python validate.py --compare` passes with the updated scores and method-independent evaluator.

## Stop Justification
- Completed at milestone `core_claim`.
