#!/bin/bash
# Verification job: summarize the hardening audit, then reproduce the
# reference and random control end to end from the final code in a
# fresh-clone state (no oracle, run or fit caches) and check that the frozen
# comparison records are reproduced exactly. Afterwards run.sh is just
# scripts/reproduce.sh.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/summarize_hardening.py > tmp/hardening_summary_stdout.txt
head -c 6000 tmp/hardening_summary_stdout.txt
python scripts/check_numerics.py
cp eval/train/fixed_raw.json tmp/fixed_raw_train_audit.json
cp eval/test/fixed_raw.json tmp/fixed_raw_test_audit.json
rm -rf eval/train/oracle_cache eval/test/oracle_cache eval/train/runs eval/test/runs tmp/gp_fit_cache
bash scripts/reproduce.sh
python scripts/compare_frozen.py tmp/fixed_raw_train_audit.json eval/train/fixed_raw.json
python scripts/compare_frozen.py tmp/fixed_raw_test_audit.json eval/test/fixed_raw.json
python validate.py --post-design
