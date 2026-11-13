#!/bin/bash
# Development-only diagnostic pilot (packager). Previous gym setup (four
# shared crossings); no hidden-test data are read and nothing is frozen.
#   1. Numerical checks of the method primitives.
#   2. Contact quality, error localization and final-classifier rescoring of
#      the frozen development records under alternative GP settings.
#   3. End-to-end development runs (reference and random control) of GP
#      settings that restore defaults of the cited training toolkit.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
START=$(date +%s)
elapsed() { echo $(( $(date +%s) - START )); }
python -c "import os; print('CPUs available:', len(os.sched_getaffinity(0)), flush=True)"
python scripts/check_numerics.py || echo 'CHECK FAILED: numerics'
python scripts/dev_classifier_pilot.py
echo "classifier pilot finished after $(elapsed) s"

mf() { printf '{"protocol":"acl-v2","command":%s}\n' "$3" > "$1/$2.json"; }
cmd() {
  local variant=$1 prior=$2; shift 2; local extra=""
  for word in "$@"; do extra="$extra,\"$word\""; done
  printf '["python","method/gp_acl.py","--variant","%s","--prior","%s","--rho","0.1"%s]' "$variant" "$prior" "$extra"
}
dir=tmp/pilot_e2e; mkdir -p "$dir"; rm -f "$dir"/*.json
for entry in "tf4|tied|--noise-floor|0.0001" "lf4|learned|--noise-floor|0.0001" "zml4|zero|--noise-floor|0.0001|--no-hinge"; do
  IFS='|' read -r -a parts <<< "$entry"
  id=${parts[0]}; prior=${parts[1]}; extra=("${parts[@]:2}")
  mf "$dir" "e2e_${id}_reference" "$(cmd reference "$prior" "${extra[@]}")"
  mf "$dir" "e2e_${id}_random" "$(cmd random "$prior" "${extra[@]}")"
  bash scripts/evaluate_train.sh --outputs-dir "$dir" --output tmp/pilot_e2e_scores.json
  python scripts/paired_report.py tmp/pilot_e2e_scores.json "e2e_${id}_reference" "e2e_${id}_random"
  echo "end-to-end $id finished after $(elapsed) s"
done
echo "pilot finished after $(elapsed) s"
