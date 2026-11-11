# Milestone

Current: `core_claim`.

GP-ACL (Alg. 1) is implemented step by step for the paper's 2D double integrator:
- the reduced KKT Problems 7–8;
- a derivative-observation GP trained with marginal likelihood plus the feasibility hinge from the training procedure the paper cites;
- conditioned posterior samples (1,000 random Fourier features, α = 0.3);
- Problem 3 solved over the sampled boundary, and the continuous Problems 4–6 with Eq. 10;
- iterative refits until the demonstration budget is spent.

On 36 paired hidden instances per experiment, it beats the random-query baseline in combined score beyond paired noise (random −0.906 [−1.586, −0.230]). The gains are in boundary balanced accuracy (+0.048 on g3, +0.029 on stress) and near-boundary balanced accuracy (+0.045, +0.046), the boundary errors the paper emphasizes. The fixed twelve-demonstration budget is the constraint.

Not reproduced: the paper's headline uniform-space accuracy. Random is level on g3 (+0.004 [−0.001, +0.008]) and slightly better on the stress task (+0.008). Higher-dimensional dynamics and hardware are not reproduced. See HARDENING.md.
