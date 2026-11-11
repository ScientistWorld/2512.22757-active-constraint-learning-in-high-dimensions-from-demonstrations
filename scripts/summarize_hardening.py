"""Summarize paired evidence without exporting private answers or parameters.

Inputs are evaluator score files only: the hidden-test audit cohort
(reference, frozen random control, variants, shortcut probes), the
development audit cohort, and the development prior selection. Every
comparison pairs identical instances, initial data and query coordinates.
"""
import json
from pathlib import Path
import numpy as np

def read(path):
    p=Path(path)
    return json.loads(p.read_text()) if p.exists() else None

test=read('eval/test/hardening_scores.json')
train=read('eval/train/hardening_scores.json')
rng=np.random.default_rng(872)

# Previous final audit (iteration before this one), same rank convention.
BEFORE={'test_ranks_audit_cohort':{'random_baseline':-.905851,'better_fill':1.087569,'weaker_legacy':-.682791,
            'weaker_learned_prior':-.759296,'weaker_no_acquisition':-2.156079,'weaker_raw_gradients':-4.327973,
            'weaker_no_gradients':None,'weaker_two_draws':.151170,'alt_periodic':-1.418192,
            'shortcut_family_fit_g3_polynomial_only':-.184401},
        'shortcut_family_fit_undeformed_g3_three_member_cohort':2.583663,
        'reference_test':{'double_integrator_g3':{'accuracy':.963725,'balanced_accuracy':.808414,
            'near_boundary_balanced_accuracy':.778346,'safe_recall':.945263,'unsafe_recall':.671564},
            'complex_boundary_stress':{'accuracy':.936808,'balanced_accuracy':.766683,
            'near_boundary_balanced_accuracy':.749290,'safe_recall':.975122,'unsafe_recall':.558245}},
        'random_test':{'double_integrator_g3':{'accuracy':.967306,'balanced_accuracy':.760423,
            'near_boundary_balanced_accuracy':.733713,'safe_recall':.778159,'unsafe_recall':.742688},
            'complex_boundary_stress':{'accuracy':.944668,'balanced_accuracy':.737781,
            'near_boundary_balanced_accuracy':.702904,'safe_recall':.922407,'unsafe_recall':.553156}}}


def paired_metrics(scores,candidate,anchor='gp_acl'):
    out={}
    for exp in scores['experiments']:
        reference=scores['_paired_metrics'].get(anchor,{}).get(exp,[])
        values=scores['_paired_metrics'].get(candidate,{}).get(exp,[])
        if not reference or len(values)!=len(reference):
            continue
        out[exp]={}
        for metric in reference[0]:
            diff=np.array([a[metric]-b[metric] for a,b in zip(values,reference)])
            boot=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(axis=1)
            out[exp][metric]={'mean':float(diff.mean()),
                              'paired_95_interval':np.quantile(boot,[.025,.975]).tolist(),
                              'wins':int((diff>0).sum()),'losses':int((diff<0).sum()),'n':len(diff)}
    return out


def paired_rank(scores,side,candidate,cohort=None):
    """Rank change versus the reference, with paired bootstrap interval.

    cohort=None uses the evaluator's scales (all passing audit members);
    otherwise the population scales of the listed members' mean metrics.
    """
    config=json.loads(Path(f'eval/{side}/scoring_config.json').read_text())
    total=0.;boot=np.zeros(10000)
    for exp,spec in config['experiments'].items():
        ref=scores['_paired_metrics']['gp_acl'][exp]
        val=scores['_paired_metrics'].get(candidate,{}).get(exp,[])
        if len(ref)!=len(val):
            return None
        per=np.zeros(len(ref))
        for metric,definition in spec['metrics'].items():
            if cohort is None:
                scale=scores['_rank']['population_std'][f'{exp}.{metric}']
            else:
                scale=float(np.std([scores['experiments'][exp]['results'][n][metric] for n in cohort]))
            if scale==0:
                continue
            sign=1 if definition['higher_is_better'] else -1
            per+=spec['weight']*abs(definition['coefficient'])*sign*np.array(
                [a[metric]-b[metric] for a,b in zip(val,ref)])/scale
        index=rng.integers(0,len(per),(10000,len(per)))
        boot+=per[index].mean(axis=1);total+=per.mean()
    return {'mean':float(total),'paired_95_interval':np.quantile(boot,[.025,.975]).tolist(),
            'passed_success_gate':candidate not in scores['_failures']}


def cells(demos):
    found=set()
    for demo in demos:
        p=np.asarray(demo['positions'])
        tight=np.sum(np.abs(2*(2*p[1:-1]-p[:-2]-p[2:])),axis=1)>1e-4
        found|={tuple(c) for c in np.floor(p[1:-1][tight]).astype(int).tolist()}
    return found


def acquisition(side,names):
    out={}
    for name in names:
        path=Path(f'eval/{side}/runs/{name}.json')
        if not path.exists():
            continue
        records=json.loads(path.read_text())['records'];out[name]={}
        for exp,instances in records.items():
            totals=[];requests=[];informative=[];gained=[]
            for record in instances:
                added=[r['demo'] for r in record['ledger'] if r['demo'] is not None]
                totals.append(len(record['initial_demos'])+len(added));requests.append(len(record['ledger']))
                gained.append(len(cells(record['initial_demos']+added))-len(cells(record['initial_demos'])))
                for demo in added:
                    p=np.asarray(demo['positions'])
                    informative.append(bool(np.any(np.sum(np.abs(2*(2*p[1:-1]-p[:-2]-p[2:])),axis=1)>1e-4)))
            if totals:
                out[name][exp]={'demos_mean':float(np.mean(totals)),'demos_min':int(min(totals)),
                    'requests_mean':float(np.mean(requests)),
                    'added_contact_bearing_fraction':float(np.mean(informative)) if informative else None,
                    'new_unit_boundary_cells_mean':float(np.mean(gained))}
    return out


report={'before':BEFORE,'prior_selection':read('tmp/prior_selection.json')}
SMALL=['gp_acl','random_baseline']
for side,scores in (('test',test),('train',train)):
    if scores is None:
        continue
    candidates=sorted(n for n in scores['_paired_metrics'] if n!='gp_acl')
    passing=[n for n in scores['experiments'][next(iter(scores['experiments']))]['results'] if n not in scores['_failures']]
    report[side]={'rank':scores['_rank'],
        'metrics':{e:v['results'] for e,v in scores['experiments'].items()},
        'failures':scores['_failures'],'raw_metrics_of_failed':scores.get('_raw_metrics_of_failed',{}),
        'paired':{c:paired_metrics(scores,c) for c in candidates},
        'paired_rank':{c:paired_rank(scores,side,c) for c in candidates},
        # Shortcut before/after on the same small cohort: reference, frozen
        # random control and the probe itself.
        'three_member_cohort_rank':{c:paired_rank(scores,side,c,SMALL+[c]) for c in candidates
                                    if c.startswith('shortcut_') and c in passing},
        'acquisition':acquisition(side,['gp_acl','random_baseline']+[c for c in candidates if c.startswith(('better_','weaker_','alt_'))])}
Path('eval/test/hardening_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
for side in ('test','train'):
    if side not in report:
        continue
    print(json.dumps({side+'_ranks':report[side]['rank']['values'],'paired_rank':report[side]['paired_rank'],
                      'three_member_cohort_rank':report[side]['three_member_cohort_rank'],
                      'failures':{k:v[:2] for k,v in report[side]['failures'].items()}},indent=1),flush=True)
    for exp,v in report[side]['metrics'].items():
        for name,values in v.items():
            print(side,exp,name,{k:round(x,4) for k,x in values.items()},flush=True)
    for name,v in report[side]['acquisition'].items():
        print(side,'acquisition',name,v,flush=True)
