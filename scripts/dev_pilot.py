"""Interleaved development pilot for job progress reports (train split only).

Runs the first K paired instances of the reference and the random control
through the development evaluator's own online protocol, and saves them in
its resumable, source-digest-versioned run files. The full evaluator then
continues from these records; nothing here is scored separately.
"""
import importlib.util
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'eval'/'train'))
spec=importlib.util.spec_from_file_location('train_evaluate',ROOT/'eval'/'train'/'evaluate.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)

K=int(sys.argv[1]) if len(sys.argv)>1 else 6
NAMES=('gp_acl','random_baseline')
manifests={n:ev.load(ROOT/'outputs'/f'{n}.json') for n in NAMES}
saved={}
for n in NAMES:
    digest=ev.command_digest(manifests[n])
    path=ev.HERE/'runs'/f'{n}.json'
    previous=ev.load(path) if path.exists() else {}
    saved[n]={'digest':digest,'records':previous.get('records',{}) if previous.get('digest')==digest else {}}


def quick(record,exp,index):
    if record.get('failure'):
        return None
    points,truth,uniform=ev.Task(exp,index).evaluation()
    p=np.asarray(record['predictions']);band=~uniform
    return (np.mean(p[uniform]==truth[uniform]),np.mean(p[band&(truth==1)]==1),np.mean(p[band&(truth==0)]==0))


for index in range(K):
    for exp in ev.EXPERIMENTS:
        for n in NAMES:
            records=saved[n]['records'].setdefault(exp,[])
            if len(records)!=index:
                continue
            record=ev.run_instance(manifests[n],ev.Task(exp,index))
            records.append(record)
            ev.write(ev.HERE/'runs'/f'{n}.json',saved[n])
            demos=len(record['initial_demos'])+sum(x['demo'] is not None for x in record['ledger'])
            print(f'pilot {n} {exp} {index+1}/{K}: '+str(record['failure'] or
                  f'demos={demos}, requests={len(record["ledger"])}'+ev.development_progress(record,exp,index)),flush=True)
for exp in ev.EXPERIMENTS:
    pairs=[(quick(saved['gp_acl']['records'][exp][i],exp,i),quick(saved['random_baseline']['records'][exp][i],exp,i))
           for i in range(min(K,len(saved['gp_acl']['records'].get(exp,[])),len(saved['random_baseline']['records'].get(exp,[]))))]
    pairs=[(a,b) for a,b in pairs if a is not None and b is not None]
    if pairs:
        a=np.mean([x for x,_ in pairs],axis=0);b=np.mean([y for _,y in pairs],axis=0)
        print(f'pilot {exp} n={len(pairs)} reference accuracy/safe/unsafe={np.round(a,4).tolist()} '
              f'random={np.round(b,4).tolist()}',flush=True)
