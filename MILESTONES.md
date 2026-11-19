# Milestone

Current: `core_claim`.

GP-ACL (Alg. 1) is implemented step by step for the paper's double integrator, 4D unicycle and 12D quadrotor:
- KKT Problems 7–8 as linear programs over the full KKT system of each system's dynamics;
- a derivative-observation GP in 2D or 3D, trained with marginal likelihood plus the feasibility hinge of the cited training procedure (noise floor 1e-4);
- conditioned posterior draws (1,000 random Fourier features, α = 0.3);
- Problem 3 on the sampled boundary;
- Problems 4–6 with Eq. 10 (the hyperplane of Eq. 8 is searched in 3D);
- iterative refits until the twelve-demonstration budget is spent.

Hidden test, paired against the random-query baseline under the same budget:
- **Quadrotor (hourglass g5, 96 instances):** the reference is better beyond noise on every metric, uniform accuracy +0.0030 [+0.0006, +0.0053].
- **Unicycle (g1–g3, 72 instances):**
  - boundary balanced accuracy +0.067 and near-boundary +0.085, both beyond noise;
  - uniform accuracy ties: +0.010 on g2, tie on g3, −0.011 on g1 (hollow interior of the largest obstacle).
- **Double integrator:** boundary metrics beyond noise in both experiments; uniform accuracy beyond noise on the complex task, within noise on g3.
- **Combined rank:** random −1.68 [−2.09, −1.24].

**Not reproduced:**
- the paper's accuracies (unicycle 0.9996 / 1.0 / 0.9956, quadrotor 0.9919);
- the unicycle and double-integrator g3 uniform-accuracy gaps beyond noise;
- the 7-DOF arm and hardware experiments.

See HARDENING.md.
