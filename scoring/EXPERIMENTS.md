# Experiments

All four experiments share these features:
- **Budget:** a fixed allowance of twelve demonstrations, four of them common initial trajectories that give coarse coverage, and at most forty start/goal requests.
- **Uniform-space accuracy:** the fraction of uniformly sampled workspace states classified correctly, the quantity the paper reports. It measures overall recovery.
- **Boundary-band metrics:** recalls and balanced accuracy measure both sides of the safety boundary, and a narrower band resolves small boundary errors.
- **Paired evaluation:** every method sees the same instances, initial trajectories and evaluation states.
- **Random-query control:** the evaluator's fixed control, which chooses demonstrations without regard to informativeness. It establishes the value of choosing informative demonstrations.
- **Hidden shapes:** in every experiment, the hidden shapes come from a different deformation family than the development shapes, matched in size. What transfers is recovering each instance from its own demonstrations, not the development generator or a published formula.
- **Failure:** any protocol or budget violation, or failure of the minimum boundary balanced-accuracy/recall requirement on any experiment, fails the entire attempt. The evaluator's comparison methods and success thresholds are fixed; neither is provided by a submission.

## unicycle_paper_constraints (weight 0.35)

Measures recovery of the paper's three published planar obstacle shapes from demonstrations of a nonlinear 4D unicycle.
- Instances cycle through the three shapes.
- Each instance is independently rotated, rescaled, translated and smoothly deformed, so no single published formula answers it.
- 72 hidden test instances; 24 development instances.

## quadcopter_hourglass (weight 0.25)

Measures recovery of a three-dimensional hourglass-shaped obstacle from demonstrations of a 12D quadrotor.
- The paper's high-dimensional dynamics setting.
- Sparse trajectories cover only curves on a 2D surface, so much of the boundary stays unobserved.
- Orientation, size, position and a smooth deformation vary per instance.
- 96 hidden test instances; 24 development instances.

## double_integrator_g3 (weight 0.2)

Measures recovery of a complex nonlinear planar obstacle from double-integrator demonstrations.
- Orientation, scale, position and smooth shape deformation vary per instance, so recovery must go beyond memorizing one scene or one formula.
- 72 hidden test instances; 24 development instances.

## complex_boundary_stress (weight 0.2)

Measures recovery when sparse demonstrations leave complex local geometry unresolved.
- Shape, orientation, position and small-scale curvature vary across instances.
- Provides room for improving recovery, rather than matching a nearly solved shape.
- 72 hidden test instances; 24 development instances.
