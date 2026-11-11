# Constraints

Use the newline-delimited JSON interface documented in `briefing/evaluation.md`. Learn only from the initial demonstrations and those returned by the evaluator. Direct access to private task parameters, labels, evaluator files or raw test caches is forbidden. External observations, dense constraint queries and hardcoded instance answers bypass the task.

The total demonstration cap is twelve, including four fixed initial trajectories. At most forty additional requests are allowed, including rejected requests. The evaluator records acquisition and generates trajectories; a candidate's counts or trajectories cannot replace that record. Test coordinates are revealed only after acquisition ends.

Keep the simulator dynamics, demonstration objective, task distributions, measurement size, thresholds and evaluator-owned controls fixed. Do not edit `eval/`, its frozen comparisons or `scoring/reference.json`. `random_baseline` is a reserved result identifier; candidate commands cannot replace that comparison. No metric depends on runtime.

Outputs must be finite binary predictions, exactly one per evaluator-provided point. A budget/protocol violation or, on either experiment, mean boundary balanced accuracy below 0.55 or a class recall below 0.10 fails the whole run. Future methods may use a different representation and acquisition procedure, provided they solve the same information-limited task. The public `data/constraints.py` examples do not provide private-instance truth.
