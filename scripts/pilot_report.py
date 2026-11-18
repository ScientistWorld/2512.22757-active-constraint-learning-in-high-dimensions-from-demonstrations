"""Development pilot report: paired comparisons to the reference per
experiment (95% bootstrap intervals), unicycle per-constraint breakdown, and
acquisition statistics. Usage: python scripts/pilot_report.py SCORES [SIDE]"""
import json
import sys
from pathlib import Path
import numpy as np

scores=json.loads(Path(sys.argv[1]).read_text()); side=sys.argv[2] if len(sys.argv)>2 else 'train'
rng=np.random.default_rng(5)
print('failures',{k:v[:2] for k,v in scores['_failures'].items()})
print('rank',{k:(round(v,3) if v is not None else None) for k,v in scores['_rank']['values'].items()})
paired=scores['_paired_metrics']
for exp,block in scores['experiments'].items():
    ref=paired.get('gp_acl',{}).get(exp,[])
    print('==',exp)
    for name,values in block['results'].items():
        print('  %-26s'%name,{k:round(v,4) for k,v in values.items()})
    for name in paired:
        other=paired[name].get(exp,[])
        if name=='gp_acl' or not ref or len(other)!=len(ref): continue
        out={}
        for m in ('accuracy','balanced_accuracy','near_boundary_balanced_accuracy'):
            d=np.array([a[m]-b[m] for a,b in zip(other,ref)])
            boot=d[rng.integers(0,len(d),(5000,len(d)))].mean(axis=1)
            out[m]='%+.4f [%+.4f, %+.4f]'%(d.mean(),*np.quantile(boot,[.025,.975]))
        print('  %-26s minus reference'%name,out)
    if exp.startswith('unicycle') and ref:
        for fam in range(3):
            line={name:round(float(np.mean([r['accuracy'] for r in paired[name][exp][fam::3]])),4)
                  for name in paired if len(paired[name].get(exp,[]))==len(ref)}
            near={name:round(float(np.mean([r['near_boundary_balanced_accuracy'] for r in paired[name][exp][fam::3]])),4)
                  for name in ('gp_acl','random_baseline') if len(paired[name].get(exp,[]))==len(ref)}
            print('  family g%d accuracy'%(fam+1),line,'near',near)
for path in sorted(Path(f'eval/{side}/runs').glob('*.json')):
    run=json.loads(path.read_text())
    for exp,records in run['records'].items():
        if not records: continue
        demos=[len(r['initial_demos'])+sum(x['demo'] is not None for x in r['ledger']) for r in records]
        print('acquisition %-26s %-28s demos %.2f (min %d) requests %.1f'%(path.stem,exp,np.mean(demos),min(demos),
              np.mean([len(r['ledger']) for r in records])))
