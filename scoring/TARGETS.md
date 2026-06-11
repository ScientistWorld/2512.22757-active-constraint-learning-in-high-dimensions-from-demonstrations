# Evaluation Targets

## Primary: scaled_2d_obstacle_recovery
Tests whether a learner can recover the safe/unsafe split of an unknown 2D obstacle from sparse demonstrations.
- **Metric**: accuracy (higher is better)

## Constraint: scaled_2d_obstacle_recovery
Tests whether the learner preserves the sparse-demonstration premise instead of improving only by collecting more trajectories.
- **Metric**: n_demos (lower is better for comparable accuracy)
