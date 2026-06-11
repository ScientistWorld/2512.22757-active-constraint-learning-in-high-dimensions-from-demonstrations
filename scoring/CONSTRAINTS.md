# Scientist Constraints

## Data and Demonstration Budget
Use the procedural 2D constraint tasks in `data/constraints.py` and the same fixed evaluation grids. Do not add external demonstrations, change the hidden constraints during evaluation, or increase the number of demonstrations beyond the budget reported in `scores.json`.

## Problem Formulation
Predictions must be safe/unsafe labels over the same continuous constraint-state space. Methods must learn from demonstration trajectories or their derived optimality information, not from direct queries of the hidden constraint function except through the simulator that generates demonstrations.

## Evaluation
Use `scripts/evaluate.sh` without editing `eval/` or `scoring/reference.json`. Improvements must be reflected in the JSON artifacts under `outputs/` and scored by the shared evaluator.

## Compliance Checklist
- [ ] Uses the provided `wavy` and `pinched` tasks and boundary-band evaluation grids.
- [ ] Does not exceed the recorded demonstration budget.
- [ ] Does not train on evaluation labels or directly sample the hidden constraint for fitting.
- [ ] Scores are produced by `scripts/evaluate.sh`.
