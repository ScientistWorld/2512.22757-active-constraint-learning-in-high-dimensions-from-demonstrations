#!/bin/bash
set -eu
cd /home/user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for name in random_baseline_seed_17 random_baseline_seed_27 weaker_no_gradients better_simplified_boundary_fit_seed_7 better_simplified_boundary_fit_seed_17 better_simplified_boundary_fit_seed_27; do
    bash "redteam/$name/run.sh"
done
python3 -m redteam.summarize
