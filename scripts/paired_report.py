"""Paired development comparison of two methods in one evaluator score file.

Usage: python scripts/paired_report.py SCORES A B [SIDE]
Prints both methods' mean metrics, A minus B per-instance paired differences
with 95% bootstrap intervals, and both methods' acquisition statistics from
eval/<SIDE>/runs (default SIDE=train). Development use only.
"""
import json
import sys
from pathlib import Path
import numpy as np

scores=json.loads(Path(sys.argv[1]).read_text());a,b=sys.argv[2],sys.argv[3]
side=sys.argv[4] if len(sys.argv)>4 else 'train'
rng=np.random.default_rng(11)
print('failures',{k:v[:1] for k,v in scores['_failures'].items() if k in (a,b)},flush=True)
for exp,v in scores['experiments'].items():
    for name in (a,b):
        if v['results'].get(name):
            print(exp,name,{k:round(x,4) for k,x in v['results'][name].items()},flush=True)
    pa=scores['_paired_metrics'].get(a,{}).get(exp,[]);pb=scores['_paired_metrics'].get(b,{}).get(exp,[])
    if pa and len(pa)==len(pb):
        out={}
        for m in pa[0]:
            d=np.array([x[m]-y[m] for x,y in zip(pa,pb)])
            boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(axis=1)
            out[m]=[round(float(d.mean()),4)]+[round(float(x),4) for x in np.quantile(boot,[.025,.975])]
        print(exp,f'{a} - {b}',out,flush=True)
for name in (a,b):
    path=Path(f'eval/{side}/runs/{name}.json')
    if not path.exists(): continue
    for exp,records in json.loads(path.read_text())['records'].items():
        demos=[len(r['initial_demos'])+sum(x['demo'] is not None for x in r['ledger']) for r in records]
        print('acquisition',name,exp,{'demos_mean':round(float(np.mean(demos)),2),'demos_min':int(min(demos)),
              'requests_mean':round(float(np.mean([len(r['ledger']) for r in records])),2)},flush=True)
