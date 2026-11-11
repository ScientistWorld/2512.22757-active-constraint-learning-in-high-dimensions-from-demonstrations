# Train and test design

`eval/train/` and `eval/test/` are independent implementations with different seeds and instances. g3 shapes carry a smooth per-instance deformation. The two splits use different private families of matching size: RMS 4.06% of the radius on development (24 instances per experiment) and 3.99% on the hidden test (36 per experiment). An exact public formula therefore does not answer hidden instances.

Methods are commands that speak the same online protocol on both splits. They acquire demonstrations from the evaluator's oracle within twelve demonstrations (four shared initial crossings) and forty requests. They then classify a shuffled batch of points revealed only after acquisition ends.

The evaluator owns truth, budgets, gates, weights and frozen raw reference/control records. It recomputes every metric from raw predictions. Independent instances may run in parallel; records are kept in index order.

The paper's g3 constraint keeps 60% weight. A complex-boundary stress task (40%) measures where the method falls short.
