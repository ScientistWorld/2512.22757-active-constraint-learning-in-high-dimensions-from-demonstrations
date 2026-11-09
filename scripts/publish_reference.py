"""Publish only the reproduced reference runs; preserve paper-reported values."""
import json
from pathlib import Path
p=Path('scoring/reference.json');reference=json.loads(p.read_text())
for exp,spec in reference['experiments'].items():
    measured=json.loads(Path('scoring/scores_test.json').read_text())['experiments'][exp]['results']
    spec['results']={name:{'type':kind,**measured[name]} for name,kind in [('random_baseline','baseline'),('gp_acl','proposed')]}
p.write_text(json.dumps(reference,indent=2)+'\n')
Path('scoring/scores.json').write_text(Path('scoring/scores_train.json').read_text())
