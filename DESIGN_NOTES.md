# Train and test design

`eval/train/` and `eval/test/` are independent implementations with different seeds and instances.

## Shape families

In every experiment the hidden shape family differs from the development one, at matched difficulty:

| Experiment | Hidden test | Development | Matched by |
|---|---|---|---|
| Double integrator g3 | Private radial deformation | Different private radial deformation | RMS 4.06% vs 3.99% of the radius |
| Complex boundaries | Private family | Different private family | Radius variation, slope and narrowest radius |
| Unicycle (g1–g3), quadrotor (g5) | Gaussian-bump displacement field | Sinusoidal displacement field | Boundary RMS: unicycle 0.27 vs 0.32, quadrotor 0.52 vs 0.52 |

Neither a development generator nor a published App. B.1 formula answers hidden instances.

## Instance counts

| Experiment | Hidden test | Development |
|---|---:|---:|
| Double integrator g3 | 72 | 24 |
| Complex boundaries | 72 | 24 |
| Unicycle | 72 | 24 |
| Quadrotor | 96 | 24 |

## Protocol

Methods are commands that speak the same online protocol on both splits.
- The header names the system: double integrator, unicycle or quadrotor.
- Methods acquire demonstrations from the evaluator's expert within twelve demonstrations (four shared initial crossings) and forty requests.
- They then classify a shuffled batch of states, revealed only after acquisition ends.

The evaluator owns truth, the expert, budgets, gates, weights and the frozen raw reference/control records. It recomputes every metric from raw predictions.

## Weights

| Experiment | Weight |
|---|---:|
| Unicycle | 0.35 |
| Quadrotor | 0.25 |
| Double integrator g3 | 0.2 |
| Complex-boundary stress | 0.2 |

The unicycle and quadrotor are the paper's main-text, high-dimensional settings.
