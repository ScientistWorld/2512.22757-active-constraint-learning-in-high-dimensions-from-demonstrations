#!/bin/bash
# Final hardening job (resumable: source-digest-versioned raw records, memoized
# GP fits, a cached development selection; a resubmission continues).
#   0. Development error map of the previous setup's frozen controls.
#   1. Development-only choice of the shared initial demonstration set (last
#      candidate: rho moved toward the paper's zero mean), by the rule fixed
#      in scripts/select_setup.py before this job ran.
#   2. The choice is written into eval/{train,test}/tasks.py, method/gp_acl.py
#      and the control manifests (scripts/apply_setup.py); job-only checks.
#   3. Reference and random control on both splits from the final code; raw
#      records frozen; scoring files published.
#   4. Development audit (generalization of the headroom and simple members).
#   5. Hidden-test audit: better, weaker, simple, shortcut and alternative
#      members in priority order under a time guard.
#   6. Reference-only scoring files again, validation, final summary.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
START=$(date +%s)
AUDIT_DEADLINE=${AUDIT_DEADLINE:-18000}
elapsed() { echo $(( $(date +%s) - START )); }
mkdir -p tmp outputs/previous tmp/setup_select
python -c "import os; print('CPUs available:', len(os.sched_getaffinity(0)), flush=True)"
check() { python "$@" || echo "CHECK FAILED: $*"; }

mf() { printf '{"protocol":"acl-v2","command":%s}\n' "$3" > "$1/$2.json"; }
cmd() {
  local variant=$1 prior=$2 rho=$3 extra=""; shift 3
  for word in "$@"; do extra="$extra,\"$word\""; done
  printf '["python","method/gp_acl.py","--variant","%s","--prior","%s","--rho","%s"%s]' "$variant" "$prior" "$rho" "$extra"
}
probe() { printf '["python","scripts/shortcut_probe.py","--attack","%s","--prior","%s","--rho","%s"]' "$1" "$2" "$3"; }

# 0. Where the previous setup's frozen development controls err (diagnostic).
if [ ! -s tmp/dev_error_map_before.json ]; then
  python scripts/dev_error_map.py gp_acl random_baseline || echo 'error map failed'
  if [ -s tmp/dev_error_map.json ]; then mv tmp/dev_error_map.json tmp/dev_error_map_before.json; fi
fi

# 1. Development-only setup selection.
CANDIDATES="init2_rho010:2:0.1 init1_rho010:1:0.1 init1_rho001:1:0.01"
if [ ! -s tmp/selected_setup ]; then
  for entry in $CANDIDATES; do
    IFS=: read -r id n rho <<< "$entry"
    python scripts/apply_setup.py --initial "$n" --sides train
    dir=tmp/setup_select/$id; mkdir -p "$dir"; rm -f "$dir"/*.json
    mf "$dir" "${id}_reference" "$(cmd reference tied "$rho")"
    mf "$dir" "${id}_random" "$(cmd random tied "$rho")"
    mf "$dir" "${id}_no_acquisition" "$(cmd no_acquisition tied "$rho")"
    bash scripts/evaluate_train.sh --outputs-dir "$dir" --output "tmp/setup_select/${id}_scores.json"
    echo "setup candidate $id finished after $(elapsed) s"
  done
  python scripts/select_setup.py | tee tmp/select_setup_stdout.txt
  tail -n 1 tmp/select_setup_stdout.txt > tmp/selected_setup
fi
read -r SETUP N_INITIAL RHO < tmp/selected_setup
echo "selected setup: $SETUP (initial crossings $N_INITIAL, rho $RHO) after $(elapsed) s"

# 2. Apply the choice everywhere it is defined; job-only checks.
python scripts/apply_setup.py --initial "$N_INITIAL" --rho "$RHO"
check scripts/check_tasks.py
check scripts/check_numerics.py
check eval/test/check_contract.py
check eval/test/check_oracle.py

# 3. Controls from the final code, frozen as evaluator-owned raw records.
find outputs -maxdepth 1 -type f -name '*.json' -exec mv -t outputs/previous {} +
bash scripts/method.sh
bash scripts/baseline.sh
# Identical manifests and sources share a digest: the evaluator reuses the
# selection's development records only if nothing changed since (else reruns).
for pair in "gp_acl:${SETUP}_reference" "random_baseline:${SETUP}_random"; do
  name=${pair%%:*}; source=${pair#*:}
  if [ -e "eval/train/runs/$source.json" ]; then cp "eval/train/runs/$source.json" "eval/train/runs/$name.json"; fi
done
bash scripts/evaluate_train.sh --bootstrap
bash scripts/evaluate_test.sh --bootstrap
python scripts/publish_reference.py || echo 'Reference publication refused: see failures above.'
echo "controls finished after $(elapsed) s"

# 4. Development audit cohort.
rm -rf tmp/dev_audit; mkdir -p tmp/dev_audit
mf tmp/dev_audit gp_acl "$(cmd reference tied "$RHO")"
mf tmp/dev_audit better_fill_topup "$(cmd fill_topup tied "$RHO")"
mf tmp/dev_audit simple_coverage_sweep "$(cmd sweep tied "$RHO")"
bash scripts/evaluate_train.sh --outputs-dir tmp/dev_audit --output eval/train/hardening_scores.json
python scripts/dev_error_map.py gp_acl random_baseline better_fill_topup simple_coverage_sweep || echo 'error map failed'
echo "development audit finished after $(elapsed) s"

# 5. Hidden-test audit cohort.
rm -rf tmp/audit_outputs; mkdir -p tmp/audit_outputs
mf tmp/audit_outputs gp_acl "$(cmd reference tied "$RHO")"
if [ "$RHO" = "0.1" ]; then OTHER_RHO=0.01; OTHER_NAME=variant_rho001; else OTHER_RHO=0.1; OTHER_NAME=variant_rho010; fi
audit() {
  bash scripts/evaluate_test.sh --outputs-dir tmp/audit_outputs --output eval/test/hardening_scores.json
  python scripts/summarize_hardening.py > tmp/hardening_summary_stdout.txt || echo 'summary failed'
  tail -n 90 tmp/hardening_summary_stdout.txt
}
for entry in "better_fill_topup|$(cmd fill_topup tied "$RHO")" \
             "better_topup|$(cmd topup tied "$RHO")" \
             "better_fill|$(cmd fill tied "$RHO")" \
             "weaker_no_acquisition|$(cmd no_acquisition tied "$RHO")" \
             "weaker_two_draws|$(cmd reference tied "$RHO" --draws 2)" \
             "shortcut_family_fit|$(probe family_fit tied "$RHO")" \
             "simple_coverage_sweep|$(cmd sweep tied "$RHO")" \
             "weaker_learned_prior|$(cmd reference learned "$RHO")" \
             "weaker_raw_gradients|$(cmd raw_gradients tied "$RHO")" \
             "weaker_no_gradients|$(cmd no_gradients tied "$RHO")" \
             "$OTHER_NAME|$(cmd reference tied "$OTHER_RHO")" \
             "alt_zero_prior|$(cmd reference zero "$RHO")"; do
  name=${entry%%|*}; command=${entry#*|}
  if [ "$(elapsed)" -gt "$AUDIT_DEADLINE" ]; then echo "time guard: skipping $name"; continue; fi
  mf tmp/audit_outputs "$name" "$command"
  echo "audit member $name started after $(elapsed) s"
  audit
done

# 6. Reference-only scoring documents, validation and final summary.
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python scripts/publish_reference.py
python validate.py --post-design || echo 'VALIDATION FAILED'
audit
echo "hardening job finished after $(elapsed) s"
