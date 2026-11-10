#!/bin/bash
# Combined printed-equation verification; source-versioned records resume jobs.
set -euo pipefail
cd /home/user
# Report saved progress without changing source digests or recomputing science.
# evaluate.py independently checks each digest before reusing any record.
python - <<'CHECKPOINTS'
import json
from pathlib import Path
print('Saved checkpoint inventory (source digests verified by evaluator):',flush=True)
for side in ('train','test'):
    for path in sorted(Path(f'eval/{side}/runs').glob('*.json')):
        if path.stem.startswith(('shortcut_','legacy_')):
            continue
        saved=json.loads(path.read_text())
        counts={exp:len(records) for exp,records in saved.get('records',{}).items()}
        print(side,path.stem,counts,flush=True)
CHECKPOINTS
bash scripts/harden.sh
