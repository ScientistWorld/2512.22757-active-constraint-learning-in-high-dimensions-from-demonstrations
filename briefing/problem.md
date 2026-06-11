# Problem Definition

## Research Question

Learn an unknown safety constraint in a continuous robot state or configuration space from a small sequence of expert demonstrations. Demonstrations are feasible, locally optimal trajectories between requested start and goal states, but the learner does not directly observe the hidden obstacle or constraint function that shaped them.

## Why It Matters

Robots deployed around unknown obstacles or task restrictions need to infer safety boundaries from limited expert behavior. A good learner should recover the safe/unsafe split accurately while requiring few demonstrations, because demonstrations may be expensive or risky to collect.

## Success Criteria

A successful system uses a small budget of demonstrations, fits a constraint representation from the resulting trajectories, and correctly classifies held-out constraint states as safe or unsafe. It should improve over a random start/goal query strategy under the same demonstration budget.
