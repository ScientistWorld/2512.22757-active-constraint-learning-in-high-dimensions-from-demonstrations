"""Summarize paired evidence without exporting private answers or parameters."""
import json
from pathlib import Path
import numpy as np
train=json.loads(Path('eval/train/hardening_scores.json').read_text())
test=json.loads(Path('eval/test/hardening_scores.json').read_text())
report={'before':{'reference_accuracy':.9546448087431694,'random_accuracy':.9311475409836065,
                  'reference_rank':0,'better_rank':.473367,'better_simplified_rank':.636216,
                  'weaker_no_gradients_rank':-4.462129,'seed_rank_spread':.579845},
        'test_rank':test['_rank'],'test_metrics':{e:v['results'] for e,v in test['experiments'].items()},
        'train_rank':train['_rank'],'train_metrics':{e:v['results'] for e,v in train['experiments'].items()},
        'paired':{},'test_failures':test['_failures'],'train_failures':train['_failures']}
rng=np.random.default_rng(872)
for candidate in ['random_baseline','better_boundary_fit','better_boundary_fit_12','better_normalized','weaker_no_acquisition','weaker_no_gradients']:
    differences={}
    for exp in test['experiments']:
        reference=test['_paired_metrics']['gp_acl'][exp]
        values=test['_paired_metrics'][candidate][exp]
        differences[exp]={}
        for metric in reference[0]:
            diff=np.array([a[metric]-b[metric] for a,b in zip(values,reference)])
            bootstrap=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(axis=1)
            differences[exp][metric]={'mean':float(diff.mean()),'paired_95_interval':np.quantile(bootstrap,[.025,.975]).tolist(),'wins':int((diff>0).sum()),'n':len(diff)}
    report['paired'][candidate]=differences
# Development uses the same paired comparison, independently of test. Failed
# runs retain raw metric diagnostics, but never acquire a successful rank.
report['train_paired']={}
for candidate in ['random_baseline','better_boundary_fit','better_boundary_fit_12','better_normalized','weaker_no_acquisition','weaker_no_gradients']:
    differences={}
    for exp in train['experiments']:
        reference=train['_paired_metrics']['gp_acl'][exp]
        values=train['_paired_metrics'][candidate][exp]
        differences[exp]={}
        for metric in reference[0]:
            diff=np.array([a[metric]-b[metric] for a,b in zip(values,reference)])
            bootstrap=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(axis=1)
            differences[exp][metric]={'mean':float(diff.mean()),'paired_95_interval':np.quantile(bootstrap,[.025,.975]).tolist(),'wins':int((diff>0).sum()),'n':len(diff)}
    report['train_paired'][candidate]=differences
# Normalized ranking gain: paired resampling uses the same task indices for
# candidate and reference, and fixed cohort scales reported by the ranker.
for side,scores,key in [('test',test,'paired_rank'),('train',train,'train_paired_rank')]:
    config=json.loads(Path(f'eval/{side}/scoring_config.json').read_text())
    report[key]={}
    for candidate in ['random_baseline','better_boundary_fit','better_boundary_fit_12','better_normalized','weaker_no_acquisition','weaker_no_gradients']:
        if candidate in scores['_failures']:continue
        bootstrap_total=np.zeros(10000);mean_total=0.
        for exp,spec in config['experiments'].items():
            ref=scores['_paired_metrics']['gp_acl'][exp];val=scores['_paired_metrics'][candidate][exp]
            if len(ref)!=len(val):raise ValueError('paired tasks do not match')
            per=np.zeros(len(ref))
            for metric,definition in spec['metrics'].items():
                scale=scores['_rank']['population_std'][f'{exp}.{metric}']
                if scale==0:continue
                sign=1 if definition['higher_is_better'] else -1
                per+=spec['weight']*abs(definition['coefficient'])*sign*np.array([a[metric]-b[metric] for a,b in zip(val,ref)])/scale
            bootstrap_total+=per[rng.integers(0,len(per),(10000,len(per)))].mean(axis=1)
            mean_total+=per.mean()
        report[key][candidate]={'mean':float(mean_total),'paired_95_interval':np.quantile(bootstrap_total,[.025,.975]).tolist()}
# Acquisition diagnostics are evidence about the setup, not scored rewards.
# Count contact-bearing demonstrations from raw positions using the reduced
# stationarity residual. No method-written verdict or count is read here.
report['acquisition']={}
for side in ('test','train'):
    config=json.loads(Path(f'eval/{side}/scoring_config.json').read_text())
    report['acquisition'][side]={}
    for name in ('gp_acl','random_baseline'):
        records=json.loads(Path(f'eval/{side}/runs/{name}.json').read_text())['records']
        report['acquisition'][side][name]={}
        for exp,instances in records.items():
            totals=[];requests=[];informative=[]
            initial_gradient=[];added_gradient=[];initial_edge=[];added_edge=[]
            for record in instances:
                added=[r['demo'] for r in record['ledger'] if r['demo'] is not None]
                totals.append(len(record['initial_demos'])+len(added));requests.append(len(record['ledger']))
                for demo in added:
                    p=np.asarray(demo['positions'])
                    informative.append(bool(np.any(np.sum(np.abs(2*(2*p[1:-1]-p[:-2]-p[2:])),axis=1)>1e-4)))
                for demos,gradient,edge in ((record['initial_demos'],initial_gradient,initial_edge),
                                            (added,added_gradient,added_edge)):
                    for demo in demos:
                        p=np.asarray(demo['positions'])
                        gradient.extend(np.linalg.norm(2*(2*p[1:-1]-p[:-2]-p[2:]),axis=1).tolist())
                        edge.append(float(np.max(np.linalg.norm(np.diff(p,axis=0),axis=1))))
            report['acquisition'][side][name][exp]={
                'demos_mean':float(np.mean(totals)),'demos_min':int(min(totals)),
                'requests_mean':float(np.mean(requests)),
                'added_contact_bearing_fraction':float(np.mean(informative)) if informative else None,
                'initial_gradient_quantiles':np.quantile(initial_gradient,[.5,.9,.99,1.]).tolist(),
                'added_gradient_quantiles':np.quantile(added_gradient,[.5,.9,.99,1.]).tolist() if added_gradient else None,
                'initial_longest_edge_quantiles':np.quantile(initial_edge,[.5,.9,1.]).tolist(),
                'added_longest_edge_quantiles':np.quantile(added_edge,[.5,.9,1.]).tolist() if added_edge else None}
Path('eval/test/hardening_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'paired_rank':report['paired_rank'],'test_ranks':report['test_rank']['values']},indent=2))
