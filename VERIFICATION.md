# Verification Report

## Initial Assessment

The workspace claimed `core_claim` and contained valid saved results for one scaled 2D obstacle-recovery experiment. That claim was broadly justified: GP-ACL outperformed the random-query baseline under the same 12-demonstration budget, and the baseline stayed below 95% accuracy.

However, the handoff state had started expanding the gym to a second obstacle setting. That setting was present in `reference.json`, run scripts, and documentation, but it had not been run and was absent from `scoring/scores.json`.

## Issues Found

- **Critical:** `reference.json`, `TARGETS.md`, `EXPERIMENTS.md`, and the standard run scripts listed `scaled_2d_pinched_recovery`, but `scores.json`, `scores_train.json`, and `scores_test.json` only contained `scaled_2d_obstacle_recovery`. This would create a phantom target and an inconsistent gym contract.
- **Minor:** `briefing/evaluation.md` described the proposed method's active query search in a method-agnostic file.
- **Minor:** `PROGRESS.md` and `MILESTONES.md` still described the unvalidated second setting as pending submission.

## Fixes Applied

- Removed the unvalidated second setting from `scoring/reference.json`, `scoring/TARGETS.md`, `scoring/EXPERIMENTS.md`, `scoring/CONSTRAINTS.md`, `data/constraints.py`, `eval/evaluate_outputs.py`, and the standard run scripts.
- Reworded `briefing/problem.md` and `briefing/evaluation.md` so from-scratch agents see the task and scoring interface without method details.
- Updated `PROGRESS.md` and `MILESTONES.md` to reflect the verified one-experiment `core_claim` state.
- Revalidated the workspace with `python validate.py --compare`, method/eval py-compilation, and a `shared/` reference scan.
- Checked `scripts/evaluate.sh` for standalone use. Running it directly on the login-node Python is blocked by the expected missing `numpy` dependency; `environment/setup.sh` installs `numpy` and `scipy` for the compute container where the reproduced scores were generated.

## Final Judgment

The workspace is ready to serve as a research gym for the `core_claim` milestone. Improve-mode agents can run the packaged method and baseline, while from-scratch agents can use the problem definition, data module, and evaluator without importing the paper method. The scoring contract now matches the experiments actually reproduced.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: high
- Key limitation: scaled 2D procedural benchmark only
