# Experiments

## double_integrator_g3

Measure recovery of a complex nonlinear safety boundary from a fixed, small demonstration allowance. Independent obstacle orientations/scales test recovery beyond memorization of one scene. Uniform-space accuracy measures overall recovery; boundary-band recalls and balanced accuracy measure both sides of the safety boundary. A narrower band resolves small boundary errors. The fixed random-query control establishes the value of choosing informative demonstrations.

## complex_boundary_stress

Measure recovery when sparse demonstrations leave more complex local geometry unresolved. Shape, orientation and small-scale curvature vary across instances. This experiment provides room for improving recovery rather than matching a nearly solved single shape. It uses the same observation budget, classification metrics and paired evaluation protocol as the first experiment.

Each experiment has thirty-six obstacles and separate development/test samples. One common initial trajectory leaves the rest of the twenty-demonstration allowance for exploration. Both uniform-space and boundary-focused quantities matter. Any protocol/budget violation or failure of either minimum recall requirement fails the entire attempt. The evaluator's comparison methods and success thresholds are fixed; neither is provided by a submission.
