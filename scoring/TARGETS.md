# Targets

Both `double_integrator_g3` (weight 0.60) and `complex_boundary_stress` (weight 0.40) score the following scientific quantities:

- Uniform-space `accuracy` (coefficient 1.0).
- Boundary-band `balanced_accuracy` (0.5).
- `near_boundary_balanced_accuracy` (0.6), emphasizing local recovery errors.
- `safe_recall` (0.25) and `unsafe_recall` (0.25), preserving both feasible-space recovery and obstacle detection.

All are higher-is-better. Demonstration use is authenticated and capped, rather than rewarded through a submitted count. Each experiment must achieve mean safe and unsafe boundary recall of at least 0.50 (chance); violations or impossible outputs fail the entire attempt. Paired comparisons use the same obstacles and initial demonstrations.

The paper's original numbers are preserved under `_paper_reported_*` in `reference.json`, outside the results to be matched at this scale. The scored reference and control are end-to-end measurements on the final setup, with raw comparison records fixed by the evaluator.
