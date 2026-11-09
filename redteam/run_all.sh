#!/bin/bash
set -eu
cd /home/user
# Thread limits control CPU resource use, not scoring behavior.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while read -r name; do
    [[ -z "$name" ]] && continue
    bash "redteam/$name/run.sh"
done < redteam/cases.txt
python3 -m redteam.summarize
