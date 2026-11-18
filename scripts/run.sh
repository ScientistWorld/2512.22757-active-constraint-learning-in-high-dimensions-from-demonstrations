#!/bin/bash
# Smoke test of the new unicycle/quadrotor experiments (development split only).
set -uo pipefail
cd /home/user
export PYTHONPATH=/home/user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c "import os; print('CPUs available:', len(os.sched_getaffinity(0)), flush=True)"
(cd eval/test && timeout 1200 python check_system_oracle.py) || echo 'CHECK FAILED: test system oracle'
timeout 900 python scripts/check_numerics.py || echo 'CHECK FAILED: numerics'
timeout 2400 python scripts/smoke_systems.py 6 4 reference,random,topup || echo 'SMOKE FAILED'
