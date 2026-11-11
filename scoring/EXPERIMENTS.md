# Experiments

## double_integrator_g3

Measure recovery of a complex nonlinear safety boundary from a fixed, small demonstration allowance. Independent obstacle orientations, scales, positions and smooth shape deformations test recovery beyond memorization of one scene or one formula. Uniform-space accuracy measures overall recovery; boundary-band recalls and balanced accuracy measure both sides of the safety boundary. A narrower band resolves small boundary errors. The fixed random-query control establishes the value of choosing informative demonstrations.

## complex_boundary_stress

Measure recovery when sparse demonstrations leave more complex local geometry unresolved. Shape, orientation, position and small-scale curvature vary across instances. This experiment provides room for improving recovery rather than matching a nearly solved single shape. It uses the same observation budget, classification metrics and paired evaluation protocol as the first experiment.

Each experiment has thirty-six hidden test obstacles and a separate development sample of twenty-four. Four common initial trajectories give coarse coverage and leave eight of the twelve demonstrations for exploration. Both uniform-space and boundary-focused quantities matter. Any protocol/budget violation or failure of the minimum boundary balanced-accuracy/recall requirement fails the entire attempt. The evaluator's comparison methods and success thresholds are fixed; neither is provided by a submission.
