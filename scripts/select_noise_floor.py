"""Development-split choice of the reference's GP noise floor (packager only).

The rule is fixed before the job runs. The previous package bounded the
fitted noise variance at 1e-10; the development pilot (eval/train/
hardening_pilot_summary.json) found that this makes the posterior mean carry
spurious unsafe regions more than four units from any observation, and that
a well-conditioned floor removes them. Candidates: 1e-4 (the inferred-noise
floor of GPyTorch, the toolkit of the cited training procedure), 1e-5, 1e-6.
Everything else is the previous reference.

On the development split only, each candidate's reference and its random
control (same GP settings) run on identical instances. A candidate
qualifies if both pass the success gate and its reference outranks its
random control (evaluator rank rule, scales from that pair). Among
qualifying candidates the one with the highest reference uniform-space
accuracy averaged over the two experiments (the paper's headline quantity)
is chosen; candidates within 0.0005 of the best are tied, and ties go to the
largest floor. If none qualifies, the gate-passing candidate with the
highest reference accuracy is chosen (same tie rule). Hidden-test data are
never read. Writes tmp/noise_floor_selection.json; the last printed line is
the chosen floor.
"""
import json
import sys
from pathlib import Path
import numpy as np

CANDIDATES=[('f4','0.0001'),('f5','1e-05'),('f6','1e-06')]
EXPS=('double_integrator_g3','complex_boundary_stress')
scores=json.loads(Path(sys.argv[1] if len(sys.argv)>1 else 'tmp/noise_floor_scores.json').read_text())
config=json.loads(Path('eval/train/scoring_config.json').read_text())
results={exp:v['results'] for exp,v in scores['experiments'].items()}
failed=set(scores['_failures'])
pm=scores['_paired_metrics']
rng=np.random.default_rng(20261113)


def interval(diff):
    diff=np.asarray(diff,float)
    boot=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(axis=1)
    return [float(diff.mean())]+np.quantile(boot,[.025,.975]).tolist()


def acquisition(name):
    path=Path(f'eval/train/runs/{name}.json')
    if not path.exists():
        return None
    out={}
    for exp,records in json.loads(path.read_text())['records'].items():
        demos=[len(r['initial_demos'])+sum(x['demo'] is not None for x in r['ledger']) for r in records]
        out[exp]={'demos_mean':float(np.mean(demos)),'demos_min':int(min(demos)),
                  'requests_mean':float(np.mean([len(r['ledger']) for r in records]))}
    return out


report={'rule':__doc__.strip(),'candidates':{}}
for cid,floor in CANDIDATES:
    ref,ran=f'{cid}_reference',f'{cid}_random'
    present=all(ref in results[e] and ran in results[e] for e in EXPS)
    if not present:
        report['candidates'][floor]={'missing':True};continue
    gate=ref not in failed and ran not in failed
    rank=0.
    for exp,spec in config['experiments'].items():
        for metric,definition in spec['metrics'].items():
            a,b=results[exp][ran][metric],results[exp][ref][metric]
            scale=float(np.std([a,b]))
            if scale>0:
                sign=1 if definition['higher_is_better'] else -1
                rank+=spec['weight']*abs(definition['coefficient'])*sign*(a-b)/scale
    paired={exp:{m:interval([x[m]-y[m] for x,y in zip(pm[ref][exp],pm[ran][exp])])
                 for m in pm[ref][exp][0]} for exp in EXPS if pm.get(ref,{}).get(exp)}
    report['candidates'][floor]={'gate_passed':gate,'random_rank_vs_reference':rank,
        'reference_outranks_random':bool(gate and rank<0),
        'reference_accuracy_mean':float(np.mean([results[e][ref]['accuracy'] for e in EXPS])),
        'metrics':{e:{'reference':results[e][ref],'random':results[e][ran]} for e in EXPS},
        'reference_minus_random':paired,'acquisition':{'reference':acquisition(ref),'random':acquisition(ran)}}


def choose(pool):
    if not pool: return None
    best=max(report['candidates'][f]['reference_accuracy_mean'] for f in pool)
    tied=[f for f in pool if report['candidates'][f]['reference_accuracy_mean']>=best-5e-4]
    return max(tied,key=float)


present=[f for _,f in CANDIDATES if not report['candidates'][f].get('missing')]
qualifying=[f for f in present if report['candidates'][f]['reference_outranks_random']]
chosen=choose(qualifying);basis='qualifying candidates'
if chosen is None:
    chosen=choose([f for f in present if report['candidates'][f]['gate_passed']]) or choose(present)
    basis='fallback: no candidate qualified'
report['chosen']={'noise_floor':chosen,'basis':basis}
Path('tmp').mkdir(exist_ok=True)
Path('tmp/noise_floor_selection.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
for floor,item in report['candidates'].items():
    if item.get('missing'):
        print('dev floor',floor,'missing',flush=True);continue
    for exp in EXPS:
        for who in ('reference','random'):
            print('dev floor',floor,exp,who,{k:round(v,4) for k,v in item['metrics'][exp][who].items()},flush=True)
        if exp in item['reference_minus_random']:
            print('dev floor',floor,exp,'reference-random',{m:[round(x,4) for x in v] for m,v in item['reference_minus_random'][exp].items()},flush=True)
    print('dev floor',floor,'gate',item['gate_passed'],'random rank vs reference',round(item['random_rank_vs_reference'],3),
          'reference accuracy mean',round(item['reference_accuracy_mean'],4),'acquisition',json.dumps(item['acquisition']),flush=True)
print(json.dumps(report['chosen']),flush=True)
print(chosen,flush=True)
