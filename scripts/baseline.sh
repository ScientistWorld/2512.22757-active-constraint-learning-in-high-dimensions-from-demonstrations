#!/bin/bash
# Run a baseline method from the paper.
#
# Baselines are simple reference methods the paper compares against. 
# Implement this only if applicable, i.e. the paper has baselines.
# You do not need to implement all baselines to begin — 
# just enough to validate that the evaluation pipeline works 
# and provide a reference point. You can decide to expand more baselines later.
# Each baseline must be actually run and scored through evaluate.sh
# so the numbers are verified, not self-reported.
# You can decide what this baseline script should do yourself.
#
# Usage:
#   bash scripts/baseline.sh <name>

set -e

cd /home/user

python baseline/random_sampling.py \
  --constraint circle \
  --iterations 3 \
  --pairs-per-iter 4 \
  --horizon 24 \
  --eval-grid 61 \
  --n-features 300 \
  --alpha 0.3 \
  --output outputs/random_baseline_circle.json

bash scripts/evaluate.sh
