#!/bin/bash
# Evaluate the visible train slice of saved method outputs.
#
# Contract mirrors scripts/evaluate.sh: score method-agnostic JSON artifacts
# and write a scoring JSON file, without importing from method/.

set -e

cd /home/user
export PYTHONPATH=/home/user:${PYTHONPATH:-}

mkdir -p scoring

python eval/train/evaluate.py "$@"
