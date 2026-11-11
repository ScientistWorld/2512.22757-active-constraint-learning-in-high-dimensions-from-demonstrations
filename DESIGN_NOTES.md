# Train and test design

`eval/train/` and `eval/test/` are independent implementations with different seeds and instances.

In both experiments the hidden shape family differs from the development one at matched difficulty:
- g3 shapes carry a smooth per-instance deformation from different private families (RMS 4.06% vs 3.99% of the radius).
- Complex-boundary shapes use a different private family of matched radius variation, slope and narrowest radius.

Hand-coding a development generator therefore does not answer hidden instances. Development has 24 instances per experiment; the hidden test has 36.

Methods are commands that speak the same online protocol on both splits. They acquire demonstrations from the evaluator's oracle within twelve demonstrations (four shared initial crossings) and forty requests. They then classify a shuffled batch of points revealed only after acquisition ends.

The evaluator owns truth, budgets, gates, weights and frozen raw reference/control records. It recomputes every metric from raw predictions.

The paper's g3 constraint keeps 60% weight. A complex-boundary stress task (40%) measures where the method falls short.
