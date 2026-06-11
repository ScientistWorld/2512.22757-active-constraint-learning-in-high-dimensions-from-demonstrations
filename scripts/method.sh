#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

for constraint in wavy pinched; do
  python method/gp_acl.py \
    --mode active \
    --constraint "${constraint}" \
    --iterations 3 \
    --pairs-per-iter 4 \
    --horizon 24 \
    --eval-grid 61 \
    --n-features 300 \
    --alpha 0.3 \
    --output "outputs/gp_acl_${constraint}.json"
done

bash scripts/evaluate.sh
