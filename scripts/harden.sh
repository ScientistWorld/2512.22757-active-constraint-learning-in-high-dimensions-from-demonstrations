#!/bin/bash
# One combined job: checks, both controls on both splits (published as soon as
# they finish), then the test audit cohort of variants and fixed shortcuts in
# priority order under a time guard. Source-digest-versioned raw records and
# memoized GP fits let a resubmission resume.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
START=$(date +%s)
# Stop starting new audit members after this many seconds (job limit 6 h).
AUDIT_DEADLINE=${AUDIT_DEADLINE:-16200}
elapsed() { echo $(( $(date +%s) - START )); }
mkdir -p outputs/previous tmp
# Recover an interrupted persistent-file probe before taking a new snapshot.
if [ -e tmp/hardening_public_edit_active ]; then
  cp tmp/constraints.before_hardening.py data/constraints.py
  rm tmp/hardening_public_edit_active
fi
python -c "import os; print('CPUs available:', len(os.sched_getaffinity(0)), flush=True)"
python scripts/check_tasks.py
python scripts/check_numerics.py
python eval/test/check_contract.py
python eval/test/check_oracle.py

# 1. Reference and random control, development then hidden test. These
#    bootstrap runs freeze the evaluator-owned raw comparison records.
find outputs -maxdepth 1 -type f -name '*.json' ! -name gp_acl.json ! -name random_baseline.json -exec mv -t outputs/previous {} +
bash scripts/method.sh
bash scripts/baseline.sh
bash scripts/evaluate_train.sh --bootstrap
cp scoring/scores_train.json eval/train/hardening_scores.json
bash scripts/evaluate_test.sh --bootstrap
python scripts/publish_reference.py || echo 'Reference publication refused: see failures above.'
echo "controls finished after $(elapsed) s"

# 2. Test audit cohort. Cheap probes first, then variants by priority; every
#    evaluator call reuses all saved records and runs only new members.
rm -rf tmp/audit_outputs
mkdir -p tmp/audit_outputs
manifest() { printf '{"protocol":"acl-v2","command":%s}\n' "$2" > "tmp/audit_outputs/$1.json"; }
manifest gp_acl '["python","method/gp_acl.py","--variant","reference"]'
# The reserved control name cannot replace the evaluator's frozen record.
manifest random_baseline '["python","scripts/shortcut_probe.py","--attack","impossible_predictions"]'
for name in oracle centered_disk contact_disk grid_order grid_resize false_count negative_count nan_count \
            negative_inf_count over_budget move_baseline replace_ground_truth impossible_predictions forge_flags \
            constant_unsafe empty_predictions null_predictions; do
  manifest "shortcut_$name" "[\"python\",\"scripts/shortcut_probe.py\",\"--attack\",\"$name\"]"
done
for name in shortcut_oracle shortcut_grid_order shortcut_grid_resize shortcut_false_demo_count \
            shortcut_negative_demo_count shortcut_replace_ground_truth shortcut_move_baseline; do
  if [ -e "redteam/$name/artifact.json" ]; then cp "redteam/$name/artifact.json" "tmp/audit_outputs/legacy_$name.json"; fi
done
# A producer's persistent replacement of data/ used to replace the answer key.
# The private simulator has no import or dependency on this file.
cp data/constraints.py tmp/constraints.before_hardening.py
touch tmp/hardening_public_edit_active
restore_audit_files() {
  cp tmp/constraints.before_hardening.py data/constraints.py
  rm -f tmp/hardening_public_edit_active
}
trap restore_audit_files EXIT
python - <<'PY'
from pathlib import Path
p=Path('data/constraints.py')
s=p.read_text();s+='\n# Red-team persistent public-truth replacement.\nCONSTRAINTS["wavy"]["function"] = lambda p: np.atleast_2d(p)[:,0]\n';p.write_text(s)
PY
audit() {
  bash scripts/evaluate_test.sh --outputs-dir tmp/audit_outputs --output eval/test/hardening_scores.json
  python scripts/summarize_hardening.py > tmp/hardening_summary_stdout.txt || true
  tail -n 40 tmp/hardening_summary_stdout.txt
}
audit
restore_audit_files
trap - EXIT
# Variants in priority order: name, then method variant or probe attack.
for entry in better_fill:fill weaker_legacy:legacy shortcut_family_fit:@family_fit \
             weaker_no_acquisition:no_acquisition weaker_raw_gradients:raw_gradients \
             better_periodic:periodic weaker_no_gradients:no_gradients better_periodic6:periodic6; do
  name=${entry%%:*}; variant=${entry#*:}
  if [ "$(elapsed)" -gt "$AUDIT_DEADLINE" ]; then echo "time guard: skipping $name"; continue; fi
  if [ "${variant:0:1}" = "@" ]; then
    manifest "$name" "[\"python\",\"scripts/shortcut_probe.py\",\"--attack\",\"${variant:1}\"]"
  else
    manifest "$name" "[\"python\",\"method/gp_acl.py\",\"--variant\",\"$variant\"]"
  fi
  echo "audit member $name started after $(elapsed) s"
  audit
done

# 3. Ship only the reference measurements in the scoring documents.
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python scripts/publish_reference.py
python validate.py --post-design
echo "hardening job finished after $(elapsed) s"
