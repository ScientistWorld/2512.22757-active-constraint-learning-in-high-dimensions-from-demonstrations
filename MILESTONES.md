# Milestone

Current: `core_claim`.

GP-ACL (Alg. 1) is implemented step by step for the paper's 2D double integrator:
- the reduced KKT Problems 7–8;
- a derivative-observation GP trained with marginal likelihood plus the feasibility hinge from the training procedure the paper cites;
- conditioned posterior samples (1,000 random Fourier features, α = 0.3);
- Problem 3 solved over the sampled boundary, and the continuous Problems 4–6 with Eq. 10;
- iterative refits until the demonstration budget is spent.

On 36 paired hidden instances per experiment, it beats the random-query baseline in combined score beyond paired noise (random −0.777 [−1.368, −0.192] in the core cohort). The gains are in boundary balanced accuracy (+0.048 on g3, +0.032 on stress) and near-boundary balanced accuracy (+0.045, +0.045), the boundary errors the paper emphasizes. The fixed twelve-demonstration budget is the constraint.

Not reproduced: the paper's headline uniform-space accuracy advantage. Random ties on g3 (+0.004 [−0.001, +0.008]) and is slightly better on the stress task (+0.008). The paper's zero-mean posterior was tested as the reference and collapsed uniform accuracy (0.73 on development). Higher-dimensional dynamics and hardware are not reproduced. See HARDENING.md.
