# Targets

## Experiments and metrics

Four experiments are scored:

| Experiment | Weight |
|---|---:|
| `unicycle_paper_constraints` | 0.35 |
| `quadcopter_hourglass` | 0.25 |
| `double_integrator_g3` | 0.20 |
| `complex_boundary_stress` | 0.20 |

Each experiment scores the same scientific quantities, all higher-is-better:

| Metric | Coefficient | What it measures |
|---|---:|---|
| `accuracy` | 1.0 | Uniform-space accuracy (the paper's recovery accuracy) |
| `balanced_accuracy` | 0.5 | Boundary band, both classes |
| `near_boundary_balanced_accuracy` | 0.6 | Local recovery errors close to the boundary |
| `safe_recall` | 0.25 | Feasible-space recovery |
| `unsafe_recall` | 0.25 | Obstacle detection |

## Constraints

- Demonstration use is authenticated and capped, not rewarded through a submitted count.
- Each experiment must reach a mean boundary balanced accuracy of at least 0.55 and recall each class at least 0.10. Violations or impossible outputs fail the entire attempt.
- Paired comparisons use the same obstacles, initial demonstrations and evaluation states.

## Paper numbers

The paper's original numbers are kept in `reference.json` under `_paper_reported_*`, outside the results to be matched at this scale:
- unicycle g1, g2 and g3 over 2,500 states;
- quadrotor g5 over 27,000 states;
- double integrator g3.

The scored reference and the random control are end-to-end measurements on the final setup. Their raw comparison records are fixed by the evaluator.
