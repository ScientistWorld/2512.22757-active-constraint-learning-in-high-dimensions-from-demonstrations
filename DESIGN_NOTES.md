# Train/Test Evaluation Design Notes

## Checks Performed

- Read `eval/train/evaluate.py` and `eval/test/evaluate.py` side by side for copy-paste errors, hardcoded method defaults, cross-imports, and score output paths.
- Confirmed both slice scripts call their matching evaluator and write through the evaluator defaults rather than `scoring/scores.json`.
- Confirmed the split evaluators score method-produced JSON artifacts from `outputs/` or explicit artifact paths instead of importing or rerunning `method/`.
- Checked `scoring/reference.json` for paper-agreement, runtime, throughput, or speed metrics and found none.
- Verified `scoring/scores_train.json` and `scoring/scores_test.json` are absent.

## Changes Made

- Replaced the original even/odd flattened-index split with an angular-sector split:
  - `eval/train/` scores visible alternating angular sectors.
  - `eval/test/` scores the complementary held-out angular sectors.
  - This makes a train-tuned shortcut less likely to transfer trivially than neighboring-point interpolation on the same boundary band.
- Added `near_boundary_balanced_accuracy` in both train and test evaluators.
  - This metric focuses on the hardest part of the task, close to the obstacle boundary, and makes coarse safe/unsafe heuristics easier to detect.
- Updated `scoring/reference.json` so every active metric has `higher_is_better`, `coefficient`, and `description`.
- Added the new near-boundary metric to the experiment scoring with a positive coefficient.

## Static Validation

- `bash -n scripts/evaluate_train.sh`
- `bash -n scripts/evaluate_test.sh`
- `python -m json.tool scoring/reference.json`
- `python -m py_compile eval/train/evaluate.py eval/test/evaluate.py`
- Grepped for train/test cross-imports and forbidden paper-agreement or runtime metric names.

## Runtime Concerns For Next Iteration

- I did not run the evaluators or generate scores in this design-only pass.
- The train/test split still relies on prediction arrays over the same fixed procedural evaluation grid. This is lightweight and artifact-compatible, but a future agent that uses the visible `data.constraints` oracle could potentially predict both slices directly. A stronger future design would hide the held-out label generator or require methods to expose a callable predictor that can be queried on private test coordinates.
- The near-boundary metric may expose sharper regressions than overall accuracy; the runner iteration should check that it is not saturated by the existing reproduction or by the random baseline.
