# Overview

This workspace studies recovery of unknown robot safety constraints from a limited set of expert demonstrations. The source is *Active Constraint Learning in High Dimensions from Demonstrations*, paper 2512.22757.

The final gym uses thirty-step double-integrator trajectories on varied nonlinear obstacles, with independent development and test instances. It measures overall classification accuracy and both sides of the boundary under an authenticated demonstration budget. A second setting adds more complex curvature to expose remaining recovery errors. See `problem.md` and `evaluation.md` for the task and command protocol.

Paper-reported results remain in `scoring/reference.json` under underscore-prefixed keys. Measured reference results are published from end-to-end reference runs on the final setup. `HARDENING.md` records the audit, ranking checks and remaining limitations.
