#!/bin/bash
# Final hardening job (resumable: source-digest-versioned raw records, memoized
# GP fits and a cached development selection; a resubmission continues).
#   1. Development-only choice of the reference's GP noise floor (the previous
#      1e-10 floor is numerically singular) by the rule fixed in
#      scripts/select_noise_floor.py before this job ran.
#   2. The choice is written into method/gp_acl.py and the control manifests
#      (scripts/apply_noise_floor.py); job-only checks.
#   3. Reference and random control on both splits from the final code (the
#      hidden test has 72 paired instances per experiment); raw records
#      frozen; scoring files published.
#   4. Development audit and error map.
#   5. Hidden-test audit: better, weaker, previous-reference, simple, shortcut
#      and alternative members in priority order under a time guard.
#   6. Reference-only scoring files again, validation, final summary.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
START=$(date +%s)
AUDIT_DEADLINE=${AUDIT_DEADLINE:-18000}
elapsed() { echo $(( $(date +%s) - START )); }
mkdir -p tmp outputs/previous
python -c "import os; print('CPUs available:', len(os.sched_getaffinity(0)), flush=True)"
check() { python "$@" || echo "CHECK FAILED: $*"; }

mf() { printf '{"protocol":"acl-v2","command":%s}\n' "$3" > "$1/$2.json"; }
cmd() {
  local variant=$1 prior=$2 floor=$3 extra=""; shift 3
  for word in "$@"; do extra="$extra,\"$word\""; done
  printf '["python","method/gp_acl.py","--variant","%s","--prior","%s","--noise-floor","%s"%s]' "$variant" "$prior" "$floor" "$extra"
}
probe() { printf '["python","scripts/shortcut_probe.py","--attack","%s","--prior","%s","--noise-floor","%s"]' "$1" "$2" "$3"; }

# 1. Development-only noise-floor selection.
if [ ! -s tmp/selected_noise_floor ]; then
  dir=tmp/floor_select; mkdir -p "$dir"; rm -f "$dir"/*.json
  for entry in f4:0.0001 f5:1e-05 f6:1e-06; do
    id=${entry%%:*}; floor=${entry#*:}
    mf "$dir" "${id}_reference" "$(cmd reference tied "$floor")"
    mf "$dir" "${id}_random" "$(cmd random tied "$floor")"
  done
  bash scripts/evaluate_train.sh --outputs-dir "$dir" --output tmp/noise_floor_scores.json
  python scripts/select_noise_floor.py tmp/noise_floor_scores.json | tee tmp/select_noise_floor_stdout.txt
  tail -n 1 tmp/select_noise_floor_stdout.txt > tmp/selected_noise_floor
fi
FLOOR=$(cat tmp/selected_noise_floor)
case "$FLOOR" in 0.0001) SEL=f4;; 1e-05) SEL=f5;; 1e-06) SEL=f6;; *) echo "unexpected floor $FLOOR"; exit 1;; esac
echo "selected noise floor: $FLOOR after $(elapsed) s"

# 2. Apply the choice; job-only checks.
python scripts/apply_noise_floor.py "$FLOOR"
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
for pair in "gp_acl:${SEL}_reference" "random_baseline:${SEL}_random"; do
  name=${pair%%:*}; source=${pair#*:}
  if [ -e "eval/train/runs/$source.json" ]; then cp "eval/train/runs/$source.json" "eval/train/runs/$name.json"; fi
done
bash scripts/evaluate_train.sh --bootstrap
bash scripts/evaluate_test.sh --bootstrap
python scripts/publish_reference.py || echo 'Reference publication refused: see failures above.'
echo "controls finished after $(elapsed) s"

# 4. Development audit cohort and error map.
rm -rf tmp/dev_audit; mkdir -p tmp/dev_audit
mf tmp/dev_audit gp_acl "$(cmd reference tied "$FLOOR")"
mf tmp/dev_audit better_fill_topup "$(cmd fill_topup tied "$FLOOR")"
mf tmp/dev_audit simple_coverage_sweep "$(cmd sweep tied "$FLOOR")"
mf tmp/dev_audit variant_previous_floor "$(cmd reference tied 1e-10)"
bash scripts/evaluate_train.sh --outputs-dir tmp/dev_audit --output eval/train/hardening_scores.json
python scripts/dev_error_map.py gp_acl random_baseline better_fill_topup simple_coverage_sweep variant_previous_floor || echo 'error map failed'
echo "development audit finished after $(elapsed) s"

# 5. Hidden-test audit cohort.
rm -rf tmp/audit_outputs; mkdir -p tmp/audit_outputs
mf tmp/audit_outputs gp_acl "$(cmd reference tied "$FLOOR")"
audit() {
  bash scripts/evaluate_test.sh --outputs-dir tmp/audit_outputs --output eval/test/hardening_scores.json
  python scripts/summarize_hardening.py > tmp/hardening_summary_stdout.txt || echo 'summary failed'
  tail -n 90 tmp/hardening_summary_stdout.txt
}
for entry in "better_fill_topup|$(cmd fill_topup tied "$FLOOR")" \
             "better_topup|$(cmd topup tied "$FLOOR")" \
             "better_fill|$(cmd fill tied "$FLOOR")" \
             "weaker_no_acquisition|$(cmd no_acquisition tied "$FLOOR")" \
             "variant_previous_floor|$(cmd reference tied 1e-10)" \
             "weaker_two_draws|$(cmd reference tied "$FLOOR" --draws 2)" \
             "shortcut_family_fit|$(probe family_fit tied "$FLOOR")" \
             "simple_coverage_sweep|$(cmd sweep tied "$FLOOR")" \
             "weaker_learned_prior|$(cmd reference learned "$FLOOR")" \
             "weaker_raw_gradients|$(cmd raw_gradients tied "$FLOOR")" \
             "weaker_no_gradients|$(cmd no_gradients tied "$FLOOR")" \
             "alt_zero_prior|$(cmd reference zero "$FLOOR")"; do
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
