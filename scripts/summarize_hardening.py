"""Summarize paired evidence without exporting private answers or parameters.

Inputs are evaluator score files only: the hidden-test audit cohort
(reference, frozen random control, variants, shortcut probes), the
development audit cohort, and the development setup selection. Every
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

# Previous final audit (eval/test/hardening_eighth_summary.json): four shared
# initial crossings. Core-cohort ranks (every passing member except the
# alt_ prior), with paired 95% intervals, and the paired paper-direction
# gaps (random minus reference) on the hidden test.
BEFORE={'setup':{'initial_crossings':4,'rho':0.1,'budget':12,'request_limit':40},
        'core_cohort_rank':{'better_fill_topup':[1.3310,.9923,1.6726],'better_fill':[.8866,.6314,1.1425],
            'better_topup':[.3139,.1355,.5348],'weaker_two_draws':[.1937,-.3116,.6760],
            'random_baseline':[-.7770,-1.3677,-.1918],'weaker_learned_prior':[-.7803,-1.5073,-.0458],
            'weaker_no_acquisition':[-1.8370,-2.4768,-1.2480],'weaker_raw_gradients':[-4.2805,-5.0695,-3.5305],
            'weaker_no_gradients':None,'shortcut_family_fit':[-2.3091,-2.8285,-1.7905],
            'hybrid_stress_family_fit':[.5392,.2188,.8726]},
        'evaluator_cohort_rank':{'alt_zero_prior':-3.2754,'better_fill':.4081,'better_fill_topup':.6871,
            'better_topup':.2564,'random_baseline':-1.1103,'shortcut_family_fit':-1.649,
            'weaker_learned_prior':-1.4398,'weaker_no_acquisition':-2.0838,'weaker_no_gradients':-6.6687,
            'weaker_raw_gradients':-3.6251,'weaker_two_draws':.262},
        'random_minus_reference_test':{
            'double_integrator_g3':{'accuracy':[.0036,-.0007,.0077],'balanced_accuracy':[-.048,-.0741,-.0223],
                'near_boundary_balanced_accuracy':[-.0446,-.0641,-.0261]},
            'complex_boundary_stress':{'accuracy':[.0078,.0046,.011],'balanced_accuracy':[-.0318,-.0545,-.0091],
                'near_boundary_balanced_accuracy':[-.0454,-.0678,-.023]}},
        'reference_test':{'double_integrator_g3':{'accuracy':.963725,'balanced_accuracy':.808414,
            'near_boundary_balanced_accuracy':.778346,'safe_recall':.945263,'unsafe_recall':.671564},
            'complex_boundary_stress':{'accuracy':.937127,'balanced_accuracy':.765363,
            'near_boundary_balanced_accuracy':.746270,'safe_recall':.972407,'unsafe_recall':.558319}},
        'random_test':{'double_integrator_g3':{'accuracy':.967306,'balanced_accuracy':.760423,
            'near_boundary_balanced_accuracy':.733713,'safe_recall':.778159,'unsafe_recall':.742688},
            'complex_boundary_stress':{'accuracy':.944926,'balanced_accuracy':.733586,
            'near_boundary_balanced_accuracy':.700849,'safe_recall':.925623,'unsafe_recall':.541549}}}


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


def add_hybrid(scores):
    """Stress-only use of the family-fit probe: the reference on g3 and the
    probe's predictions on the complex-boundary experiment. Both component
    runs passed, so the combination is a passing attempt a submitter could
    make with the same code."""
    pm=scores['_paired_metrics']
    if 'shortcut_family_fit' not in pm or 'shortcut_family_fit' in scores['_failures']:
        return
    pm['hybrid_stress_family_fit']={'double_integrator_g3':pm['gp_acl']['double_integrator_g3'],
                                    'complex_boundary_stress':pm['shortcut_family_fit']['complex_boundary_stress']}
    for exp,rows in pm['hybrid_stress_family_fit'].items():
        scores['experiments'][exp]['results']['hybrid_stress_family_fit']={
            m:float(np.mean([r[m] for r in rows])) for m in rows[0]}


report={'before':BEFORE,'setup_selection':read('tmp/setup_selection.json'),
        # Development-only: where uniform-space errors fall, previous setup and final.
        'dev_error_map_before':read('tmp/dev_error_map_before.json'),
        'dev_error_map':read('tmp/dev_error_map.json')}
SMALL=['gp_acl','random_baseline']
for side,scores in (('test',test),('train',train)):
    if scores is None:
        continue
    evaluator_cohort=[n for n in scores['experiments'][next(iter(scores['experiments']))]['results']
                      if n not in scores['_failures']]
    add_hybrid(scores)
    candidates=sorted(n for n in scores['_paired_metrics'] if n!='gp_acl')
    passing=[n for n in scores['experiments'][next(iter(scores['experiments']))]['results'] if n not in scores['_failures']]
    # Core cohort: every passing attempt except the rejected alternative prior
    # mean, whose collapsed uniform accuracy (0.72) would otherwise set the
    # accuracy scale for everyone.
    core=[n for n in passing if not n.startswith('alt_')]
    report[side]={'rank':scores['_rank'],
        'metrics':{e:v['results'] for e,v in scores['experiments'].items()},
        'failures':scores['_failures'],'raw_metrics_of_failed':scores.get('_raw_metrics_of_failed',{}),
        'paired':{c:paired_metrics(scores,c) for c in candidates},
        'paired_rank':{c:paired_rank(scores,side,c) for c in candidates if c in evaluator_cohort or c in scores['_failures']},
        'core_cohort':core,
        'core_cohort_rank':{c:paired_rank(scores,side,c,core) for c in candidates if c in core},
        # Shortcut before/after on the same small cohort: reference, frozen
        # random control and the probe itself.
        'three_member_cohort_rank':{c:paired_rank(scores,side,c,SMALL+[c]) for c in candidates
                                    if c.startswith(('shortcut_','hybrid_')) and c in passing},
        'acquisition':acquisition(side,['gp_acl','random_baseline']+[c for c in candidates if c.startswith(('better_','weaker_','alt_','simple_','variant_'))])}
Path('eval/test/hardening_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
for side in ('test','train'):
    if side not in report:
        continue
    print(json.dumps({side+'_ranks':report[side]['rank']['values'],'paired_rank':report[side]['paired_rank'],
                      'core_cohort_rank':report[side]['core_cohort_rank'],
                      'three_member_cohort_rank':report[side]['three_member_cohort_rank'],
                      'failures':{k:v[:2] for k,v in report[side]['failures'].items()}},indent=1),flush=True)
    for exp,v in report[side]['metrics'].items():
        for name,values in v.items():
            print(side,exp,name,{k:round(x,4) for k,x in values.items()},flush=True)
    for name,v in report[side]['acquisition'].items():
        print(side,'acquisition',name,v,flush=True)
