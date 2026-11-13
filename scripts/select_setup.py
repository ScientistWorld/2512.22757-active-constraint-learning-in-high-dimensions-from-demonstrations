"""Development-split choice of the shared initial demonstration set (packager only).

The rule is fixed before the job runs. The previous gym shared four initial
crossings, which already wrap each obstacle from several sides: with no
acquisition at all they gave the random control's uniform accuracy, so the
test could not show the paper's headline effect. The paper's own runs start
from a partial initial set (weak first models in Fig. 5; its random baseline
recovers only the parts of g3 its first demonstrations touched, Fig. 8).
Candidates, in order of increasing change from the previous gym:

  init2_rho010  two shared initial crossings; tied prior mean -0.1*signal
  init1_rho010  one shared initial crossing;  tied prior mean -0.1*signal
  init1_rho001  one shared initial crossing;  tied prior mean -0.01*signal,
                closer to the paper's zero-mean posterior (Eqs. 15-16)

The total budget (12 demonstrations, 40 requests) is unchanged. For each
candidate, on the development split only, the reference, the random-query
control and the no-acquisition ablation run on identical instances.

  Tier 1: reference and random both pass the success gate; the reference
          outranks random (evaluator rank rule, scales from these three
          members' mean metrics); and the paired 95% bootstrap interval of
          reference minus random uniform-space accuracy lies above zero in
          both experiments.
  Tier 2: as tier 1, but the complex-boundary experiment only needs a
          positive mean difference (g3, the paper's experiment, still needs
          its interval above zero).
The first candidate (in the order above) of the highest non-empty tier is
chosen. If no candidate qualifies, the gate-passing candidate with the
largest smaller-of-the-two lower interval bounds is chosen. Hidden-test data
are never read. Prints a report; the last line is "<id> <crossings> <rho>".
"""
import json
from pathlib import Path
import numpy as np

CANDIDATES=[('init2_rho010',2,'0.1'),('init1_rho010',1,'0.1'),('init1_rho001',1,'0.01')]
G3,STRESS='double_integrator_g3','complex_boundary_stress'
config=json.loads(Path('eval/train/scoring_config.json').read_text())
rng=np.random.default_rng(20261113)


def interval(diff):
    diff=np.asarray(diff,float)
    boot=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(axis=1)
    lo,hi=np.quantile(boot,[.025,.975])
    return {'mean':float(diff.mean()),'paired_95_interval':[float(lo),float(hi)],
            'wins':int((diff>0).sum()),'losses':int((diff<0).sum()),'n':len(diff)}


def acquisition(name):
    path=Path(f'eval/train/runs/{name}.json')
    if not path.exists():
        return None
    out={}
    for exp,records in json.loads(path.read_text())['records'].items():
        demos=[len(r['initial_demos'])+sum(x['demo'] is not None for x in r['ledger']) for r in records]
        contacts=[]
        for r in records:
            n=0
            for d in r['initial_demos']:
                p=np.asarray(d['positions']);n+=int(np.sum(np.sum(np.abs(2*(2*p[1:-1]-p[:-2]-p[2:])),axis=1)>1e-4))
            contacts.append(n)
        out[exp]={'demos_mean':float(np.mean(demos)),'demos_min':int(min(demos)),
                  'requests_mean':float(np.mean([len(r['ledger']) for r in records])),
                  'initial_contact_states_mean':float(np.mean(contacts)),
                  'instances_without_initial_contact':int(sum(c==0 for c in contacts))}
    return out


report={'rule':__doc__.strip(),'candidates':{}}
for cid,crossings,rho in CANDIDATES:
    path=Path(f'tmp/setup_select/{cid}_scores.json')
    if not path.exists():
        report['candidates'][cid]={'missing':True};continue
    scores=json.loads(path.read_text())
    names={k:f'{cid}_{k}' for k in ('reference','random','no_acquisition')}
    failed=set(scores['_failures'])
    results={exp:v['results'] for exp,v in scores['experiments'].items()}
    gate=names['reference'] not in failed and names['random'] not in failed
    cohort=[n for n in names.values() if n not in failed]
    rank={}
    if names['reference'] in cohort:
        for name in cohort:
            total=0.
            for exp,spec in config['experiments'].items():
                for metric,definition in spec['metrics'].items():
                    scale=float(np.std([results[exp][n][metric] for n in cohort]))
                    if scale==0:
                        continue
                    sign=1 if definition['higher_is_better'] else -1
                    total+=spec['weight']*abs(definition['coefficient'])*sign*(
                        results[exp][name][metric]-results[exp][names['reference']][metric])/scale
            rank[name]=total
    paired={}
    pm=scores['_paired_metrics']
    for exp in (G3,STRESS):
        ref=pm.get(names['reference'],{}).get(exp,[]);ran=pm.get(names['random'],{}).get(exp,[])
        if ref and len(ref)==len(ran):
            paired[exp]={m:interval([a[m]-b[m] for a,b in zip(ref,ran)]) for m in ref[0]}
    have=G3 in paired and STRESS in paired
    rank_ok=gate and rank.get(names['reference'],0.)>rank.get(names['random'],np.inf)
    low={exp:paired[exp]['accuracy']['paired_95_interval'][0] for exp in paired}
    tier1=bool(have and rank_ok and low[G3]>0 and low[STRESS]>0)
    tier2=bool(have and rank_ok and low[G3]>0 and paired[STRESS]['accuracy']['mean']>0)
    report['candidates'][cid]={'crossings':crossings,'rho':rho,'gate_passed':gate,
        'failures':{k:v for k,v in scores['_failures'].items() if k in names.values()},
        'rank_three_member_cohort':rank,'reference_outranks_random':bool(rank_ok),
        'reference_minus_random':paired,'tier1':tier1,'tier2':tier2,
        'metrics':{exp:{k:results[exp].get(n) for k,n in names.items()} for exp in results},
        'acquisition':{k:acquisition(n) for k,n in names.items()}}

entries=[(cid,c,r) for cid,c,r in CANDIDATES if not report['candidates'][cid].get('missing')]
chosen=next((e for e in entries if report['candidates'][e[0]]['tier1']),None)
basis='tier 1'
if chosen is None:
    chosen=next((e for e in entries if report['candidates'][e[0]]['tier2']),None);basis='tier 2'
if chosen is None:
    basis='fallback: largest smaller-of-two lower bound among gate-passing candidates'
    def worst(e):
        p=report['candidates'][e[0]]['reference_minus_random']
        return min(p[x]['accuracy']['paired_95_interval'][0] for x in (G3,STRESS)) if len(p)==2 else -np.inf
    passing=[e for e in entries if report['candidates'][e[0]]['gate_passed']]
    chosen=max(passing or entries,key=worst)
report['chosen']={'id':chosen[0],'crossings':chosen[1],'rho':chosen[2],'basis':basis}
Path('tmp').mkdir(exist_ok=True)
Path('tmp/setup_selection.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
for cid,item in report['candidates'].items():
    if item.get('missing'):
        print(cid,'missing',flush=True);continue
    for exp in (G3,STRESS):
        for k,v in item['metrics'][exp].items():
            if v: print('dev',cid,exp,k,{m:round(x,4) for m,x in v.items()},flush=True)
        acc=item['reference_minus_random'].get(exp,{}).get('accuracy')
        if acc: print('dev',cid,exp,'reference-random accuracy',round(acc['mean'],4),
                      [round(x,4) for x in acc['paired_95_interval']],flush=True)
    print('dev',cid,'gate',item['gate_passed'],'ranks',{k:round(v,3) for k,v in item['rank_three_member_cohort'].items()},
          'tier1',item['tier1'],'tier2',item['tier2'],'acquisition',json.dumps(item['acquisition']),flush=True)
print(json.dumps(report['chosen']),flush=True)
print(f'{chosen[0]} {chosen[1]} {chosen[2]}',flush=True)
