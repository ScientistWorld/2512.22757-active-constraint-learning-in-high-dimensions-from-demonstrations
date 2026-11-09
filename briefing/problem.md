# Problem

Recover an unknown continuous safety boundary from sparse expert demonstrations. The demonstrations are locally optimal paths that avoid a hidden obstacle. You may request start and goal positions, observe the resulting state/control trajectories, and then classify new positions as safe or unsafe. Direct constraint values and labels are unavailable.

The simulator uses two-dimensional positions with unconstrained double-integrator dynamics, a one-unit time step, thirty states and the sum of squared successive position changes as its trajectory cost. Only endpoint positions are prescribed. The observation space includes positions, velocities and controls. Position coordinates are scaled uniformly by five relative to the source experiments.

There are two measurement settings: a complex nonlinear obstacle family and more varied boundaries with small-scale curvature. Each instance has a different hidden shape or orientation. Methods must infer each instance from its own demonstrations rather than memorize one public obstacle. Four initial trajectories are shared; the total budget is twelve demonstrations with at most forty additional start/goal requests. Rejected requests also consume this request allowance.

Success means accurate recovery of both the feasible region and the obstacle, especially close to their shared boundary. A method that predicts everything safe sacrifices obstacle detection; predicting everything unsafe sacrifices feasible motion. Compare methods under the fixed observation budget and on the same unseen inputs. The evaluation contract is in `briefing/evaluation.md`.
