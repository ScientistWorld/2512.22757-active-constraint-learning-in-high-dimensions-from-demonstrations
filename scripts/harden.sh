#!/bin/bash
# One combined job: final algorithms, final evaluator, then all fixed shortcuts.
set -euo pipefail
cd /home/user
export PYTHONPATH=/home/user
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p outputs/previous outputs/audit
python scripts/check_numerics.py
python eval/test/check_contract.py
find outputs -maxdepth 1 -type f -name '*.json' -exec mv -t outputs/previous {} +
bash scripts/method.sh
bash scripts/baseline.sh
python - <<'PY'
import json
from pathlib import Path
for name,variant in [('better_boundary_fit','periodic6'),('better_boundary_fit_12','periodic'),('better_normalized','normalized'),('weaker_no_gradients','no_gradients'),('weaker_no_acquisition','no_acquisition')]:
    Path(f'outputs/{name}.json').write_text(json.dumps({'protocol':'acl-v2','command':['python','method/gp_acl.py','--variant',variant]}))
PY
# Method/simulator changed, so both comparisons are genuinely rerun.
rm -f eval/train/runs/*.json eval/test/runs/*.json
bash scripts/evaluate_train.sh --bootstrap --output eval/train/hardening_scores.json
bash scripts/evaluate_test.sh --bootstrap --output eval/test/hardening_scores.json
# The rest change only interface/evaluation behavior. Existing prediction
# artifacts are rejected/rescored directly; no old method is retrained.
python - <<'PY'
import json
from pathlib import Path
for name in ['oracle','grid_order','grid_resize','false_count','negative_count','nan_count','negative_inf_count','over_budget','move_baseline','replace_ground_truth','impossible_predictions','forge_flags']:
    Path(f'outputs/shortcut_{name}.json').write_text(json.dumps({'protocol':'acl-v2','command':['python','scripts/shortcut_probe.py','--attack',name]}))
for name in ['shortcut_oracle','shortcut_grid_order','shortcut_grid_resize','shortcut_false_demo_count','shortcut_negative_demo_count','shortcut_replace_ground_truth','shortcut_move_baseline']:
    path=Path('redteam')/name/'artifact.json'
    if path.exists():Path('outputs/legacy_'+name+'.json').write_text(path.read_text())
PY
# A producer's persistent replacement of data/ used to replace the answer key.
# The private simulator has no import or dependency on this file.
cp data/constraints.py tmp/constraints.before_hardening.py
restore_audit_files() {
  cp tmp/constraints.before_hardening.py data/constraints.py
  if [ -s tmp/baseline.before_hardening.json ]; then
    cp tmp/baseline.before_hardening.json outputs/random_baseline.json
  fi
}
trap restore_audit_files EXIT
python - <<'PY'
from pathlib import Path
p=Path('data/constraints.py')
s=p.read_text();s+='\n# Red-team persistent public-truth replacement.\nCONSTRAINTS["wavy"]["function"] = lambda p: np.atleast_2d(p)[:,0]\n';p.write_text(s)
PY
# Tamper with the displayed baseline's old output location as well.
cp outputs/random_baseline.json tmp/baseline.before_hardening.json
cat > outputs/random_baseline.json <<'JSON'
{"protocol":"acl-v2","command":["python","scripts/shortcut_probe.py","--attack","impossible_predictions"]}
JSON
bash scripts/evaluate_test.sh --output eval/test/hardening_scores.json
cp tmp/constraints.before_hardening.py data/constraints.py
cp tmp/baseline.before_hardening.json outputs/random_baseline.json
trap - EXIT
python scripts/summarize_hardening.py
# Ship only the reference measurements in the scoring documents.
find outputs -maxdepth 1 -type f -name '*.json' ! -name gp_acl.json ! -name random_baseline.json -exec mv -t outputs/audit {} +
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python scripts/publish_reference.py
python validate.py --post-design
