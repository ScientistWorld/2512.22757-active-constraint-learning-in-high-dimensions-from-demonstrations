"""Write the reference's GP noise floor where it is defined (packager only).

Usage: python scripts/apply_noise_floor.py FLOOR

Edits NOISE_FLOOR in method/gp_acl.py (the code default) and the
--noise-floor of the reference and random-control manifests
(scripts/method.sh, scripts/baseline.sh). Each edit replaces exactly one
canonical line; anything else is an error. Idempotent.
"""
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def replace_one(path, pattern, text):
    source=path.read_text()
    if len(re.findall(pattern,source,flags=re.M))!=1:
        raise SystemExit(f'{path}: expected exactly one match of {pattern!r}')
    updated=re.sub(pattern,text,source,count=1,flags=re.M)
    if updated!=source:
        path.write_text(updated)
    return updated!=source


floor=float(sys.argv[1])
if not 0<floor<1:
    raise SystemExit('noise floor must lie in (0, 1)')
text=repr(floor)
changed=[]
if replace_one(ROOT/'method'/'gp_acl.py',r'^NOISE_FLOOR = [0-9.e-]+$',f'NOISE_FLOOR = {text}'):
    changed.append('method/gp_acl.py')
for script in ('method.sh','baseline.sh'):
    if replace_one(ROOT/'scripts'/script,r'"--prior","tied","--noise-floor","[0-9.e-]+"\]',f'"--prior","tied","--noise-floor","{text}"]'):
        changed.append(f'scripts/{script}')
print(f'applied noise floor {text}; changed {changed}',flush=True)
