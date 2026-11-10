"""Summarize paired evidence without exporting private answers or parameters.

Inputs are evaluator score files only: the test audit cohort (reference,
fixed random control, variants and shortcut probes) and the development
control run. Every comparison pairs identical instances and initial data.
"""
import json
from pathlib import Path
import numpy as np

test=json.loads(Path('eval/test/hardening_scores.json').read_text())
train=json.loads(Path('eval/train/hardening_scores.json').read_text())
rng=np.random.default_rng(872)
CANDIDATES=['random_baseline','better_boundary_fit','better_boundary_fit_12',
            'weaker_no_acquisition','weaker_no_gradients','weaker_raw_gradients']


def paired_metrics(scores,candidate):
    out={}
    for exp in scores['experiments']:
        reference=scores['_paired_metrics']['gp_acl'][exp]
        values=scores['_paired_metrics'].get(candidate,{}).get(exp,[])
        if len(values)!=len(reference):
            continue
        out[exp]={}
        for metric in reference[0]:
            diff=np.array([a[metric]-b[metric] for a,b in zip(values,reference)])
            boot=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(axis=1)
            out[exp][metric]={'mean':float(diff.mean()),
                              'paired_95_interval':np.quantile(boot,[.025,.975]).tolist(),
                              'wins':int((diff>0).sum()),'losses':int((diff<0).sum()),'n':len(diff)}
    return out


def paired_rank(scores,side,candidate):
    """Rank change versus the reference with the cohort's own scales."""
    config=json.loads(Path(f'eval/{side}/scoring_config.json').read_text())
    total=0.;boot=np.zeros(10000)
    for exp,spec in config['experiments'].items():
        ref=scores['_paired_metrics']['gp_acl'][exp]
        val=scores['_paired_metrics'].get(candidate,{}).get(exp,[])
        if len(ref)!=len(val):
            return None
        per=np.zeros(len(ref))
        for metric,definition in spec['metrics'].items():
            scale=scores['_rank']['population_std'][f'{exp}.{metric}']
            if scale==0:
                continue
            sign=1 if definition['higher_is_better'] else -1
            per+=spec['weight']*abs(definition['coefficient'])*sign*np.array(
                [a[metric]-b[metric] for a,b in zip(val,ref)])/scale
        index=rng.integers(0,len(per),(10000,len(per)))
        boot+=per[index].mean(axis=1);total+=per.mean()
    return {'mean':float(total),'paired_95_interval':np.quantile(boot,[.025,.975]).tolist(),
            'passed_success_gate':candidate not in scores['_failures']}


report={'before':{
            'redteam_reference_accuracy':.9546448087431694,'redteam_random_accuracy':.9311475409836065,
            'redteam_better_rank':.473367,'redteam_better_simplified_rank':.636216,
            'redteam_weaker_no_gradients_rank':-4.462129,'redteam_seed_rank_spread':.579845,
            'fourth_audit_test_ranks':{'random_baseline':.603014,'better_boundary_fit_12':1.988915,
                                       'better_normalized_gradients':3.457279,'better_boundary_fit':.578652}},
        'test_rank':test['_rank'],
        'test_metrics':{e:v['results'] for e,v in test['experiments'].items()},
        'test_failures':test['_failures'],
        'test_raw_metrics_of_failed':test.get('_raw_metrics_of_failed',{}),
        'train_rank':train['_rank'],
        'train_metrics':{e:v['results'] for e,v in train['experiments'].items()},
        'train_failures':train['_failures'],
        'train_raw_metrics_of_failed':train.get('_raw_metrics_of_failed',{})}
report['paired']={c:paired_metrics(test,c) for c in CANDIDATES if c in test['_paired_metrics']}
report['paired_rank']={c:paired_rank(test,'test',c) for c in CANDIDATES if c in test['_paired_metrics']}
report['train_paired']={'random_baseline':paired_metrics(train,'random_baseline')}
report['train_paired_rank']={'random_baseline':paired_rank(train,'train','random_baseline')}

# Acquisition diagnostics from raw ledgers: evidence about the setup, never a
# scored reward. Contact-bearing demonstrations have a nonzero reduced
# stationarity residual somewhere along the path.
report['acquisition']={}
for side in ('test','train'):
    report['acquisition'][side]={}
    for name in ('gp_acl','random_baseline'):
        path=Path(f'eval/{side}/runs/{name}.json')
        if not path.exists():
            continue
        records=json.loads(path.read_text())['records']
        report['acquisition'][side][name]={}
        for exp,instances in records.items():
            totals=[];requests=[];informative=[]
            for record in instances:
                added=[r['demo'] for r in record['ledger'] if r['demo'] is not None]
                totals.append(len(record['initial_demos'])+len(added));requests.append(len(record['ledger']))
                for demo in added:
                    p=np.asarray(demo['positions'])
                    informative.append(bool(np.any(np.sum(np.abs(2*(2*p[1:-1]-p[:-2]-p[2:])),axis=1)>1e-4)))
            report['acquisition'][side][name][exp]={
                'demos_mean':float(np.mean(totals)),'demos_min':int(min(totals)),
                'requests_mean':float(np.mean(requests)),
                'added_contact_bearing_fraction':float(np.mean(informative)) if informative else None}
Path('eval/test/hardening_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'test_ranks':report['test_rank']['values'],'paired_rank':report['paired_rank'],
                  'train_paired_rank':report['train_paired_rank'],
                  'test_failures':report['test_failures'],'train_failures':report['train_failures'],
                  'acquisition':report['acquisition']},indent=1),flush=True)
for side,scores in (('train',train),('test',test)):
    for exp,v in scores['experiments'].items():
        for name in ('gp_acl','random_baseline'):
            print(side,exp,name,{k:round(x,4) for k,x in v['results'][name].items()},flush=True)
