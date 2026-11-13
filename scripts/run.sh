#!/bin/bash
# Development-only diagnostic of rejected demonstration requests (nothing
# frozen, no hidden-test data). run.sh is then restored to end-to-end
# reproduction (scripts/reproduce.sh).
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/dev_rejection_diag.py gp_acl f6_reference variant_previous_floor
