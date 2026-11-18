#!/bin/bash
# Development pilot of the extended gym (all four experiments, development split).
set -uo pipefail
cd /home/user
export PYTHONPATH=/home/user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c "import os; print('CPUs available:', len(os.sched_getaffinity(0)), flush=True)"
mf() { printf '{"protocol":"acl-v2","command":%s}\n' "$2" > "tmp/pilot/$1.json"; }
cmd() { local v=$1; shift; local extra=""; for w in "$@"; do extra="$extra,\"$w\""; done
  printf '["python","method/gp_acl.py","--variant","%s","--prior","tied","--noise-floor","0.0001"%s]' "$v" "$extra"; }
rm -rf tmp/pilot; mkdir -p tmp/pilot
mf gp_acl "$(cmd reference)"
mf random_baseline "$(cmd random)"
mf better_fill_topup "$(cmd fill_topup)"
mf better_topup "$(cmd topup)"
mf better_fill "$(cmd fill)"
mf weaker_no_acquisition "$(cmd no_acquisition)"
mf weaker_two_draws "$(cmd reference --draws 2)"
mf simple_coverage_sweep "$(cmd sweep)"
mf shortcut_paper_fit '["python","scripts/shortcut_probe.py","--attack","paper_fit","--prior","tied","--noise-floor","0.0001"]'
timeout 3000 bash scripts/evaluate_train.sh --bootstrap --outputs-dir tmp/pilot --output tmp/pilot_scores.json | grep -v "raw output saved" 
python scripts/pilot_report.py tmp/pilot_scores.json
