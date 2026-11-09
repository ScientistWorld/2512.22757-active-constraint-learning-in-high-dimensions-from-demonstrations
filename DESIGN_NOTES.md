# Train and test design

The current design supersedes the historical angular-sector split. It uses independent implementations in eval/train/ and eval/test/ with different obstacle instances and coordinates from matching distributions. Only the private side contains its parameters, answer generator and raw fixed comparisons. Neither side imports the method or the other split.

Methods are commands implementing the same online protocol in both splits. They acquire expert trajectories from an evaluator-owned oracle, then classify a shuffled held-out batch. All methods see identical instances and initial observations within a split. The evaluator owns the test size, truth, budgets, recall gate, weights and baseline/reference records. It recomputes metrics from raw predictions and reconstructs the budget from its own ledger.

The paper constraint retains 60% weight. An additional complex-boundary stress task receives 40% to measure the sparse-data and fine-geometry limitation. Fixed seeds remove acquisition seed variability; twelve independent obstacles per task and dense boundary samples support paired estimates. After the combined job, HARDENING.md records whether actual improvements exceed paired uncertainty, whether the reference beats random queries, and any limits that remain.
