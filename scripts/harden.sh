#!/bin/bash
# Final hardening job, resumable (source-digest-versioned raw records and
# memoized GP fits; a resubmission continues where it stopped).
#   0. Job-only numerical, task-generator, contract and oracle checks.
#   1. Development-only choice of the reference prior mean: the paper's zero
#      mean (Eqs. 15-16) versus the previous package's tied mean, by the rule
#      in scripts/select_prior.py (fixed before this job ran).
#   2. Reference and random control on both splits from the final code; their
#      raw records are frozen and the scoring files published.
#   3. Development audit: the family-fit shortcut on the visible generator
#      families (the "before" state of the hidden stress family) and the
#      combined better variant.
#   4. Hidden-test audit: shortcut, better and weaker variants, previous
#      reference, in priority order under a time guard.
#   5. Reference-only scoring files again, validation, final summary.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
START=$(date +%s)
AUDIT_DEADLINE=${AUDIT_DEADLINE:-17500}
elapsed() { echo $(( $(date +%s) - START )); }
mkdir -p tmp outputs/previous
python -c "import os; print('CPUs available:', len(os.sched_getaffinity(0)), flush=True)"
check() { python "$@" || echo "CHECK FAILED: $*"; }
check scripts/check_tasks.py
check scripts/check_numerics.py
check eval/test/check_contract.py
check eval/test/check_oracle.py

mf() { printf '{"protocol":"acl-v2","command":%s}\n' "$3" > "$1/$2.json"; }
cmd() {
  local variant=$1 prior=$2 extra=""; shift 2
  for word in "$@"; do extra="$extra,\"$word\""; done
  printf '["python","method/gp_acl.py","--variant","%s","--prior","%s"%s]' "$variant" "$prior" "$extra"
}
probe() { printf '["python","scripts/shortcut_probe.py","--attack","%s","--prior","%s"]' "$1" "$2"; }

# 1. Development-only prior choice.
if [ ! -s tmp/selected_prior ]; then
  rm -rf tmp/dev_select; mkdir -p tmp/dev_select
  for prior in zero tied; do
    mf tmp/dev_select "${prior}_reference" "$(cmd reference $prior)"
    mf tmp/dev_select "${prior}_random" "$(cmd random $prior)"
  done
  bash scripts/evaluate_train.sh --outputs-dir tmp/dev_select --output tmp/dev_select_scores.json
  python scripts/select_prior.py tmp/dev_select_scores.json | tee tmp/select_prior_stdout.txt
  tail -n 1 tmp/select_prior_stdout.txt > tmp/selected_prior
fi
PRIOR=$(cat tmp/selected_prior)
if [ "$PRIOR" = zero ]; then OTHER=tied; else OTHER=zero; fi
echo "selected reference prior mean: $PRIOR (alternative $OTHER) after $(elapsed) s"

# 2. Controls from the final code, frozen as evaluator-owned raw records.
find outputs -maxdepth 1 -type f -name '*.json' -exec mv -t outputs/previous {} +
mf outputs gp_acl "$(cmd reference $PRIOR)"
mf outputs random_baseline "$(cmd random $PRIOR)"
# Identical manifests share a digest: reuse the selection run's records.
for pair in "gp_acl:${PRIOR}_reference" "random_baseline:${PRIOR}_random"; do
  name=${pair%%:*}; source=${pair#*:}
  if [ -e "eval/train/runs/$source.json" ] && [ ! -e "eval/train/runs/$name.json" ]; then
    cp "eval/train/runs/$source.json" "eval/train/runs/$name.json"
  fi
done
bash scripts/evaluate_train.sh --bootstrap
bash scripts/evaluate_test.sh --bootstrap
python scripts/publish_reference.py || echo 'Reference publication refused: see failures above.'
echo "controls finished after $(elapsed) s"

# 3. Development audit cohort.
rm -rf tmp/dev_audit; mkdir -p tmp/dev_audit
mf tmp/dev_audit gp_acl "$(cmd reference $PRIOR)"
mf tmp/dev_audit shortcut_family_fit "$(probe family_fit $PRIOR)"
mf tmp/dev_audit better_fill_topup "$(cmd fill_topup $PRIOR)"
bash scripts/evaluate_train.sh --outputs-dir tmp/dev_audit --output eval/train/hardening_scores.json
echo "development audit finished after $(elapsed) s"

# 4. Hidden-test audit cohort.
rm -rf tmp/audit_outputs; mkdir -p tmp/audit_outputs
mf tmp/audit_outputs gp_acl "$(cmd reference $PRIOR)"
audit() {
  bash scripts/evaluate_test.sh --outputs-dir tmp/audit_outputs --output eval/test/hardening_scores.json
  python scripts/summarize_hardening.py > tmp/hardening_summary_stdout.txt || echo 'summary failed'
  tail -n 80 tmp/hardening_summary_stdout.txt
}
for entry in "shortcut_family_fit|$(probe family_fit $PRIOR)" \
             "better_topup|$(cmd topup $PRIOR)" \
             "better_fill_topup|$(cmd fill_topup $PRIOR)" \
             "better_fill|$(cmd fill $PRIOR)" \
             "alt_${OTHER}_prior|$(cmd reference $OTHER)" \
             "weaker_no_acquisition|$(cmd no_acquisition $PRIOR)" \
             "weaker_raw_gradients|$(cmd raw_gradients $PRIOR)" \
             "weaker_no_gradients|$(cmd no_gradients $PRIOR)" \
             "weaker_two_draws|$(cmd reference $PRIOR --draws 2)" \
             "weaker_learned_prior|$(cmd reference learned)"; do
  name=${entry%%|*}; command=${entry#*|}
  if [ "$(elapsed)" -gt "$AUDIT_DEADLINE" ]; then echo "time guard: skipping $name"; continue; fi
  mf tmp/audit_outputs "$name" "$command"
  echo "audit member $name started after $(elapsed) s"
  audit
done

# 5. Reference-only scoring documents, validation and final summary.
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python scripts/publish_reference.py
python validate.py --post-design || echo 'VALIDATION FAILED'
audit
echo "hardening job finished after $(elapsed) s"
