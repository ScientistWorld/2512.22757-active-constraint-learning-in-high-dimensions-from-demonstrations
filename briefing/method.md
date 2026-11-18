# Paper's Method

## Key Contribution

The paper introduces Gaussian Process Active Constraint Learning (GP-ACL). It is an iterative demonstration-query algorithm for learning unknown continuous safety constraints from locally optimal demonstrations.

## Approach

Each iteration has five steps:
1. Extract tight states and constraint-gradient estimates from the demonstrations, using KKT optimality conditions.
2. Train a Gaussian-process representation of the unknown constraint on these zero-valued boundary observations and gradient observations.
3. Sample plausible constraint functions from the posterior.
4. Find high-uncertainty points on the sampled boundaries.
5. Generate start/goal pairs in directions parallel and orthogonal to the sampled boundary gradient.

The demonstrations from those start/goal pairs are added to the dataset.

## Main Claims

**Primary claim.** Actively selected demonstrations recover unknown nonlinear constraints more accurately than random start/goal selection with the same sparse demonstration budget. The paper shows this on 4D unicycle and 12D quadrotor dynamics, and on double integrators in its appendix.

**Secondary claims.** The method applies to high-dimensional nonlinear dynamics, and handles both simulation and robot-hardware settings.

## Packaged specialization (`method/gp_acl.py`, `method/kkt.py`)

**Problems 7–8 (KKT extraction).**
- Solved as linear programs over the full KKT system of each system's known dynamics: double integrator, 4D unicycle and 12D quadrotor.
- The stationarity rows of controls and of non-position states are free of the unknown constraint, and must hold with the dynamics multipliers alone.
- An L1 residual on a position row certifies tightness. The P8 gradient is the KKT-compatible residual.
- Each gradient estimate is kept as a unit direction, because the estimates are identified only up to a positive scale.

**GP training.**
- A value/derivative GP is trained in 2D or 3D constraint space.
- The loss is that of the GP constraint-learning work the paper cites for its training: negative marginal likelihood plus a hinge keeping demonstrated states predicted safe.
- The fitted observation-noise variance is bounded below by 1e-4, the inferred-noise floor of GPyTorch.
- The constant prior mean is tied to −0.1 × signal, so unexplored states keep prior safety probability Φ(0.1) = 0.54, just below the paper's endpoint level β = 0.55.
- The paper's own posterior is zero-mean. That setting was tested as the reference and rejected on the development split.

**Posterior sampling.** The conditioned posterior is sampled with 1,000 random Fourier features and α = 0.3.

**Problem 3.** Solved over the sampled boundary, within one fitted lengthscale of the observed tight states: zero-crossing enumeration on a grid, then continuous polishing.

**Problems 4–6.**
- Problem 4 searches the hyperplane orthogonal to the sampled gradient. In 2D that hyperplane is a line (two directions); in 3D it is a plane, searched over 36 in-plane directions.
- Problems 5–6 are continuous scalar root solves, with Eq. 10 read as a safe probability.

**Algorithm 1.** Iterated, refitting after every batch of P = 4 posterior draws, until the demonstration budget is spent. Repeated start/goal requests are skipped.

See `HARDENING.md` for unspecified settings and deviations.
