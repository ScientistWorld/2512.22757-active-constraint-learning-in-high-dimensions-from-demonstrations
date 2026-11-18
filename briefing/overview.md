# Overview

This workspace studies how to recover unknown robot safety constraints from a limited set of expert demonstrations. The source is *Active Constraint Learning in High Dimensions from Demonstrations* (paper 2512.22757).

The gym has four settings, each with independent development and test instances:
- **Paper's main-text unicycle setting.** Three published obstacle shapes, with 4D unicycle trajectories.
- **Paper's quadrotor setting.** A 3D hourglass obstacle, with 12D quadrotor trajectories.
- **Double integrator, nonlinear obstacles.** Thirty-step trajectories.
- **Double integrator, more complex boundaries.**

Every instance is privately rotated, rescaled, moved and deformed. Under an authenticated demonstration budget, the gym measures:
- uniform-space classification accuracy, the paper's quantity;
- both sides of the boundary.

See `problem.md` and `evaluation.md` for the task and the command protocol.

Paper-reported results remain in `scoring/reference.json` under underscore-prefixed keys. Measured reference results are published from end-to-end reference runs on the final setup. `HARDENING.md` records the audit, ranking checks and remaining limitations.
