# Constraints

## Interface

- Use the newline-delimited JSON interface documented in `briefing/evaluation.md`.
- Learn only from the initial demonstrations and those returned by the evaluator.
- Forbidden, because they bypass the task:
  - direct access to private task parameters, labels, evaluator files or raw test caches;
  - external observations or dense constraint queries;
  - hardcoded instance answers.

## Budget

- The total demonstration cap is twelve, including four fixed initial trajectories.
- At most forty additional requests are allowed, including rejected requests.
- The evaluator records acquisition and generates trajectories (positions plus each system's states and controls). A candidate's counts or trajectories cannot replace that record.
- Test coordinates are revealed only after acquisition ends.

## Fixed by the evaluator

- Keep these fixed: the systems' dynamics, the demonstration objective, the task distributions, measurement sizes, thresholds and the evaluator-owned controls.
- Do not edit `eval/`, its frozen comparisons, or `scoring/reference.json`.
- `random_baseline` is a reserved result identifier; candidate commands cannot replace that comparison.
- No metric depends on runtime.

## Outputs and failure

- Outputs must be finite binary predictions, exactly one per evaluator-provided point.
- Any of the following fails the whole run:
  - a budget or protocol violation;
  - on any experiment, a mean boundary balanced accuracy below 0.55;
  - on any experiment, a class recall below 0.10.

## What counts as a solution

- Future methods may use a different representation and acquisition procedure, provided they solve the same information-limited task.
- The public `data/constraints.py` examples do not provide private-instance truth.
- The development generators in `eval/train/` are not the hidden families.
- Hidden instances are privately deformed versions of the published obstacle shapes. Hand-coding a development generator or a published formula is not a solution.
