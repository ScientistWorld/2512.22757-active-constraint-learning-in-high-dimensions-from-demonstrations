# Problem

Recover an unknown continuous safety constraint from sparse expert demonstrations. Each demonstration is a locally optimal trajectory that avoids a hidden obstacle. You may request start and goal positions, observe the resulting state/control trajectories, and then classify new positions as safe or unsafe. Direct constraint values and labels are unavailable.

## Systems

The expert's trajectories come from known, discrete-time dynamics with time step 1. Its cost is the sum of squared successive position changes. Only the start and goal positions are prescribed, and the hidden constraint applies to the position at every time step. Controls are unbounded.

| System | Constraint space | State | Control | Dynamics | Horizon |
|---|---|---|---|---|---|
| Double integrator | 2D positions in [-15,15]^2 | position p, velocity v | acceleration a | p+ = p+v, v+ = v+a | 30 |
| Unicycle (4D) | 2D positions in [-15,15]^2 | (px, py, heading θ, speed v) | (turn rate ω, acceleration a) | px+ = px+v cosθ, py+ = py+v sinθ, θ+ = θ+ω, v+ = v+a | 30 |
| Quadrotor (12D) | 3D positions in [-6,6]^3 | (position p, world velocity v, ZYX Euler angles η = (roll, pitch, yaw), body rates ω) | (thrust F, body torques τ) | p+ = p+v, v+ = v+R(η)e3 F/m − g e3, η+ = η+W(η)ω, ω+ = ω+J⁻¹(τ − ω×Jω) | 20 |

Quadrotor constants:
- m = 0.468 and g = 9.81.
- J = diag(4.856e-3, 4.856e-3, 8.801e-3).
- R(η) = Rz(yaw)·Ry(pitch)·Rx(roll).
- W(η) maps body rates to Euler-angle rates.

## Settings

There are four measurement settings:
- **Double integrator, polynomial obstacles.** A complex nonlinear obstacle family.
- **Double integrator, complex boundaries.** More varied boundaries with small-scale curvature.
- **Unicycle, three published obstacle shapes.** Instances cycle through the three shapes.
- **Quadrotor, an hourglass-shaped 3D obstacle.**

Each instance has a different hidden shape, orientation, size and position, including a smooth deformation. Methods must infer each instance from its own demonstrations rather than memorize one obstacle or a published formula.

## Budget

- Four initial trajectories are shared by every method.
- The total budget is twelve demonstrations, with at most forty additional start/goal requests.
- Rejected requests also consume the request allowance. A request is rejected when an endpoint is inside the obstacle, or when the endpoints are too close or infeasible.

## What success means

Success means accurate recovery of both the feasible region and the obstacle, especially close to their shared boundary:
- A method that predicts everything safe sacrifices obstacle detection.
- A method that predicts everything unsafe sacrifices feasible motion.

Compare methods under the fixed observation budget and on the same unseen inputs. The evaluation contract is in `briefing/evaluation.md`.
