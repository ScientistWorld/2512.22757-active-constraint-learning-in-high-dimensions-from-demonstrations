# Progress

## What Works

- Paper analysis completed for the core GP-ACL loop and reported accuracy targets.
- Source code is in place for a scaled 2D obstacle-recovery reproduction.
- `python validate.py --reference-only` passes after staging source files.

## Results

- No reproduced scores yet. The local Python environment lacks `numpy`, so runtime validation is deferred to the container job where `environment/setup.sh` installs dependencies.
- Job `aa971aaa-2d4` built the container overlay and installed `numpy`/`scipy`, but failed before running the experiment because `baseline/random_sampling.py` could not import `method.gp_acl`.

## Remaining

- Run the full `scripts/reproduce.sh` pipeline in a CPU job.
- Inspect `scoring/scores.json`; if GP-ACL does not beat the random baseline, tune the reduced-scale benchmark while preserving the algorithm.
- After successful run, update milestone to `method_runs` or `core_claim` as justified by the results.

## Issues

- The login-node Python environment does not include `numpy`; no local package installation was performed.
- Fixed the first container runtime failure by exporting `PYTHONPATH=/home/user` in script entrypoints and adding package marker files.

## Deviations from Paper

- The paper evaluates double integrator, unicycle, quadcopter, and robot-arm systems with horizons up to 30. This reproduction starts with a scaled 2D point-trajectory setting and horizon 24 to keep the first job CPU-feasible.
- The paper solves several query subproblems with IPOPT/CasADi. This reproduction uses grid and line searches for the same subproblem objectives and constraints, preserving the GP posterior sampling, maximally informative boundary selection, and parallel/perpendicular start-goal construction.
- The paper optimizes GP hyperparameters by marginal likelihood. This initial reproduction uses fixed RBF hyperparameters to reduce setup risk; the GP still conditions jointly on boundary values and derivative observations.
