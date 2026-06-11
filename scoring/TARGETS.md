# Evaluation Targets

## Primary: scaled_2d_obstacle_recovery
Tests whether a learner can recover the safe/unsafe split of an unknown 2D obstacle from sparse demonstrations on held-out boundary-band states.
- **Metrics**: accuracy, balanced_accuracy, and near_boundary_balanced_accuracy (higher is better)

## Constraint-Recovery Balance: scaled_2d_obstacle_recovery
Tests whether the learner recovers both sides of the boundary instead of overpredicting only safe or unsafe states.
- **Metrics**: safe_recall and unsafe_recall (higher is better)

## Constraint: scaled_2d_obstacle_recovery
Tests whether the learner preserves the sparse-demonstration premise instead of improving only by collecting more trajectories.
- **Metric**: n_demos (lower is better for comparable accuracy)
