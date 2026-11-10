# Progress

Final hardening iteration (11). Job 34886e68-1b3 (printed-Equation-10 audit) was cancelled at the iteration boundary after 20 development reference instances. Under the printed sign, acquisition largely failed: one instance spent all 80 requests and accepted no demonstration. The partial run gives no complete comparison.

Restorations applied together, from the paper and the work it cites for GP training:
- Eq. 10 reads as the prose's safe probability, Phi(-mean/sd) >= beta, consistent with the paper's avoid set {g>0}.
- GP training uses the cited Chou et al. (2022) Sec. IV-B loss: negative marginal log likelihood plus the demonstration-feasibility hinge on mean + rho*sd (rho=1), with a learned constant prior mean. The previous zero-mean fit predicted large unsafe regions far from any data (reference uniform accuracy 0.63, versus the paper's clean far field).
- The paper's [-15,15]^2 workspace for g3 and a sparse budget: two shared initial demonstrations, ten in total, forty requests; Alg. 1 batches of P=2 posterior draws.

Gym fixes: per-instance obstacle translation (a fixed origin-centred shape no longer answers the task), a new centred-disk shortcut probe, and a chance-level (0.50) recall gate. The combined job checks numerics and the oracle, runs both controls on both splits and publishes them, then runs test variants and fixed shortcuts.
