#!/bin/bash
# One combined job: checks, both controls on both splits (published as soon as
# they finish), then the test audit cohort of variants and fixed shortcuts.
# Source-digest-versioned raw records let a resubmission resume.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p outputs/previous tmp
# Recover an interrupted persistent-file probe before taking a new snapshot.
if [ -e tmp/hardening_public_edit_active ]; then
  cp tmp/constraints.before_hardening.py data/constraints.py
  rm tmp/hardening_public_edit_active
fi
python scripts/check_numerics.py
python eval/test/check_contract.py
python eval/test/check_oracle.py

# 1. Reference and random control, development then hidden test. These
#    bootstrap runs freeze the evaluator-owned raw comparison records.
find outputs -maxdepth 1 -type f -name '*.json' ! -name gp_acl.json ! -name random_baseline.json -exec mv -t outputs/previous {} +
bash scripts/method.sh
bash scripts/baseline.sh
# Interleaved paired development instances first, for progress reports; the
# evaluator resumes from these saved records.
python scripts/dev_pilot.py 6
bash scripts/evaluate_train.sh --bootstrap
cp scoring/scores_train.json eval/train/hardening_scores.json
bash scripts/evaluate_test.sh --bootstrap
python scripts/publish_reference.py || echo 'Reference publication refused: see failures above.'

# 2. Test audit cohort. Cheap probes first, then the full cohort; all records
#    are source-versioned, so the second pass reuses everything already run.
rm -rf tmp/audit_outputs tmp/audit_probe_outputs
mkdir -p tmp/audit_outputs tmp/audit_probe_outputs
python - <<'PY'
import json
from pathlib import Path
def manifest(command): return json.dumps({'protocol':'acl-v2','command':command})
full=Path('tmp/audit_outputs');probe=Path('tmp/audit_probe_outputs')
for d in (full,probe):
    (d/'gp_acl.json').write_text(manifest(['python','method/gp_acl.py','--variant','reference']))
    # The reserved control name cannot replace the evaluator's frozen record.
    (d/'random_baseline.json').write_text(manifest(['python','scripts/shortcut_probe.py','--attack','impossible_predictions']))
for name in ['oracle','centered_disk','grid_order','grid_resize','false_count','negative_count','nan_count',
             'negative_inf_count','over_budget','move_baseline','replace_ground_truth','impossible_predictions','forge_flags']:
    for d in (full,probe):
        (d/f'shortcut_{name}.json').write_text(manifest(['python','scripts/shortcut_probe.py','--attack',name]))
for name in ['shortcut_oracle','shortcut_grid_order','shortcut_grid_resize','shortcut_false_demo_count',
             'shortcut_negative_demo_count','shortcut_replace_ground_truth','shortcut_move_baseline']:
    path=Path('redteam')/name/'artifact.json'
    if path.exists():
        for d in (full,probe): (d/f'legacy_{name}.json').write_text(path.read_text())
variants={'weaker_no_acquisition':'no_acquisition','better_boundary_fit_12':'periodic',
          'better_normalized':'normalized','weaker_no_gradients':'no_gradients'}
(probe/'weaker_no_acquisition.json').write_text(manifest(['python','method/gp_acl.py','--variant','no_acquisition']))
for name,variant in variants.items():
    (full/f'{name}.json').write_text(manifest(['python','method/gp_acl.py','--variant',variant]))
PY
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
bash scripts/evaluate_test.sh --outputs-dir tmp/audit_probe_outputs --output tmp/audit_probe_scores.json
bash scripts/evaluate_test.sh --outputs-dir tmp/audit_outputs --output eval/test/hardening_scores.json
restore_audit_files
trap - EXIT
python scripts/summarize_hardening.py

# 3. Ship only the reference measurements in the scoring documents.
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python scripts/publish_reference.py
python validate.py --post-design
