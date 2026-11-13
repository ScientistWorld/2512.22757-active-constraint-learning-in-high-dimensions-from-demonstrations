# Milestone

Current: `core_claim`.

GP-ACL (Alg. 1) is implemented step by step for the paper's 2D double integrator:
- the reduced KKT Problems 7–8;
- a derivative-observation GP trained with marginal likelihood plus the feasibility hinge from the training procedure the paper cites, with the observation-noise variance bounded below by GPyTorch's 1e-4 floor;
- conditioned posterior samples (1,000 random Fourier features, α = 0.3);
- Problem 3 solved over the sampled boundary, and the continuous Problems 4–6 with Eq. 10;
- iterative refits until the demonstration budget is spent.

On 72 paired hidden instances per experiment it beats the random-query baseline under the same twelve-demonstration budget:
- combined score: random −1.269 [−1.784, −0.753] (core cohort);
- uniform-space accuracy, the paper's headline quantity, in the paper's direction in both experiments: +0.0017 [−0.0030, +0.0062] on g3 (not beyond noise) and +0.0028 [+0.0003, +0.0052] on the complex-boundary task (beyond noise);
- boundary and near-boundary balanced accuracy beyond noise in both: +0.023 / +0.028 on g3, +0.043 / +0.049 on complex.

Not reproduced: the paper's g3 accuracy of 0.9956 (ours 0.979), the g3 accuracy gap beyond noise, higher-dimensional dynamics and hardware. See HARDENING.md.
