"""Development-only diagnostic of where uniform-space errors fall (packager).

Usage: python scripts/dev_error_map.py NAME [NAME ...]

For each named development run (eval/train/runs/NAME.json, visible split
only): the mean fraction of uniform workspace states that are unsafe but
predicted safe ("false safe") and safe but predicted unsafe ("false
unsafe"), with false-safe states split by radial depth inside the boundary
and false-unsafe states by height outside it. Writes tmp/dev_error_map.json.
"""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dev_tasks',ROOT/'eval'/'train'/'tasks.py')
tasks=importlib.util.module_from_spec(spec);spec.loader.exec_module(tasks)
EDGES=[0.,.5,1.5,3.,np.inf]
LABELS=['0-0.5','0.5-1.5','1.5-3','>3']
out={}
for name in sys.argv[1:]:
    path=ROOT/'eval'/'train'/'runs'/f'{name}.json'
    if not path.exists():
        print(name,'missing',flush=True);continue
    records=json.loads(path.read_text())['records'];out[name]={}
    for exp,instances in records.items():
        rows=[]
        for index,record in enumerate(instances):
            p=np.asarray(record['predictions'])
            if record.get('failure') or not len(p):
                continue
            task=tasks.Task(exp,index);points,truth,uniform=task.evaluation()
            q=points[uniform]-task.center;t=truth[uniform];y=p[uniform]
            depth=task.radius(np.arctan2(q[:,1],q[:,0]))-np.linalg.norm(q,axis=1)
            false_safe=(t==0)&(y==1);false_unsafe=(t==1)&(y==0)
            row={'false_safe':float(false_safe.mean()),'false_unsafe':float(false_unsafe.mean()),
                 'unsafe_area_fraction':float((t==0).mean())}
            for lo,hi,label in zip(EDGES[:-1],EDGES[1:],LABELS):
                row[f'false_safe_depth_{label}']=float((false_safe&(depth>=lo)&(depth<hi)).mean())
                row[f'false_unsafe_height_{label}']=float((false_unsafe&(-depth>=lo)&(-depth<hi)).mean())
            rows.append(row)
        if rows:
            out[name][exp]={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
            print('dev error map',name,exp,{k:round(v,4) for k,v in out[name][exp].items()},flush=True)
Path(ROOT/'tmp').mkdir(exist_ok=True)
(ROOT/'tmp'/'dev_error_map.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
