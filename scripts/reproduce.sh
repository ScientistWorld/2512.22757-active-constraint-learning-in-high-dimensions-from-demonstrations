#!/bin/bash
# Rebuild raw reference/baseline runs from the final method, then rescore them.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p outputs/previous
find outputs -maxdepth 1 -type f -name '*.json' -exec mv -t outputs/previous {} +
bash scripts/baseline.sh
bash scripts/method.sh
# Source-versioned records allow resumption after preemption. A fresh clone
# has no per-attempt runs, so both controls are reproduced end to end.
bash scripts/evaluate_train.sh --bootstrap
bash scripts/evaluate_test.sh --bootstrap
python scripts/publish_reference.py
python validate.py
