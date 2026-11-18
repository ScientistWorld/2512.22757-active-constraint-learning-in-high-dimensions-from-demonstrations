#!/bin/bash
# Final hardening job for the extended gym (resumable: source-digest-versioned
# raw records per instance, memoized GP fits and private oracle/evaluation
# caches; a resubmission continues where a killed job stopped).
#   1. Job-only checks (numerics, KKT programs, experts, deformation sizes).
#   2. Reference and random control on both splits from the final code, on
#      all four experiments; raw records frozen; scoring files published.
#   3. Development audit cohort.
#   4. Hidden-test audit: better, weaker, simple, shortcut and interface
#      members, one at a time under a time guard, then one cohort rescore.
#   5. Reference-only scoring files again, validation, final summary.
set -uo pipefail
cd /home/user
export PYTHONPATH=/home/user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
START=$(date +%s)
AUDIT_DEADLINE=${AUDIT_DEADLINE:-16500}
elapsed() { echo $(( $(date +%s) - START )); }
mkdir -p tmp outputs/previous
python -c "import os; print('CPUs available:', len(os.sched_getaffinity(0)), flush=True)"
check() { python "$@" || echo "CHECK FAILED: $*"; }
mf() { printf '{"protocol":"acl-v2","command":%s}\n' "$3" > "$1/$2.json"; }
cmd() {
  local variant=$1 prior=$2; shift 2; local extra=""
  for word in "$@"; do extra="$extra,\"$word\""; done
  printf '["python","method/gp_acl.py","--variant","%s","--prior","%s","--noise-floor","0.0001"%s]' "$variant" "$prior" "$extra"
}
probe() { printf '["python","scripts/shortcut_probe.py","--attack","%s","--prior","tied","--noise-floor","0.0001"]' "$1"; }

# 1. Checks.
check scripts/check_tasks.py
check scripts/check_numerics.py
check eval/test/check_contract.py
check eval/test/check_oracle.py
check eval/test/check_system_oracle.py
check scripts/check_deformation.py
echo "checks finished after $(elapsed) s"

# 2. Controls from the final code, frozen as evaluator-owned raw records.
find outputs -maxdepth 1 -type f -name '*.json' -exec mv -t outputs/previous {} +
bash scripts/method.sh
bash scripts/baseline.sh
bash scripts/evaluate_train.sh --bootstrap | grep -v 'raw output saved'
bash scripts/evaluate_test.sh --bootstrap | grep -v 'raw output saved'
python scripts/publish_reference.py || echo 'Reference publication refused: see failures above.'
echo "controls finished after $(elapsed) s"

# 3. Development audit cohort.
rm -rf tmp/dev_audit; mkdir -p tmp/dev_audit
mf tmp/dev_audit gp_acl "$(cmd reference tied)"
for entry in "better_fill_topup|$(cmd fill_topup tied)" "better_topup|$(cmd topup tied)" \
             "better_fill|$(cmd fill tied)" "weaker_no_acquisition|$(cmd no_acquisition tied)" \
             "weaker_two_draws|$(cmd reference tied --draws 2)" "simple_coverage_sweep|$(cmd sweep tied)" \
             "shortcut_paper_fit|$(probe paper_fit)"; do
  mf tmp/dev_audit "${entry%%|*}" "${entry#*|}"
done
bash scripts/evaluate_train.sh --outputs-dir tmp/dev_audit --output eval/train/hardening_scores.json | grep -v 'raw output saved'
echo "development audit finished after $(elapsed) s"

# 4. Hidden-test audit, one member at a time (records persist per instance).
rm -rf tmp/audit_outputs tmp/audit_single; mkdir -p tmp/audit_outputs tmp/audit_single
mf tmp/audit_outputs gp_acl "$(cmd reference tied)"
for entry in "better_fill_topup|$(cmd fill_topup tied)" \
             "better_fill|$(cmd fill tied)" \
             "better_topup|$(cmd topup tied)" \
             "weaker_no_acquisition|$(cmd no_acquisition tied)" \
             "weaker_two_draws|$(cmd reference tied --draws 2)" \
             "shortcut_paper_fit|$(probe paper_fit)" \
             "simple_coverage_sweep|$(cmd sweep tied)" \
             "weaker_raw_gradients|$(cmd raw_gradients tied)" \
             "weaker_learned_prior|$(cmd reference learned)" \
             "weaker_no_gradients|$(cmd no_gradients tied)" \
             "shortcut_family_fit|$(probe family_fit)" \
             "probe_over_budget|$(probe over_budget)" \
             "probe_false_count|$(probe false_count)" \
             "probe_constant_safe|$(probe constant_safe)" \
             "probe_impossible_predictions|$(probe impossible_predictions)" \
             "probe_forge_flags|$(probe forge_flags)" \
             "probe_contact_disk|$(probe contact_disk)" \
             "alt_zero_prior|$(cmd reference zero)"; do
  name=${entry%%|*}; command=${entry#*|}
  if [ "$(elapsed)" -gt "$AUDIT_DEADLINE" ]; then echo "time guard: skipping $name"; continue; fi
  echo "audit member $name started after $(elapsed) s"
  mf tmp/audit_outputs "$name" "$command"
  rm -rf tmp/audit_single/run; mkdir -p tmp/audit_single/run
  cp tmp/audit_outputs/gp_acl.json "tmp/audit_outputs/$name.json" tmp/audit_single/run/
  bash scripts/evaluate_test.sh --outputs-dir tmp/audit_single/run --output "tmp/audit_single/$name.json" \
    | grep -v 'raw output saved' | tail -n 2
done
bash scripts/evaluate_test.sh --outputs-dir tmp/audit_outputs --output eval/test/hardening_scores.json | grep -v 'raw output saved' | tail -n 3
echo "test audit finished after $(elapsed) s"

# 5. Reference-only scoring documents, validation and final summary.
bash scripts/evaluate_train.sh | grep -v 'raw output saved'
bash scripts/evaluate_test.sh | grep -v 'raw output saved'
python scripts/publish_reference.py
python validate.py --post-design || echo 'VALIDATION FAILED'
python scripts/summarize_hardening.py | tee tmp/hardening_summary_stdout.txt || echo 'summary failed'
echo "hardening job finished after $(elapsed) s"
