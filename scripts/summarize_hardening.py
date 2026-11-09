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
        'paired':{},'test_failures':test['_failures']}
rng=np.random.default_rng(872)
for candidate in ['random_baseline','better_boundary_fit','better_boundary_fit_12','better_normalized','weaker_no_acquisition']:
    differences={}
    for exp in test['experiments']:
        if candidate in test['_failures']:continue
        reference=test['_paired_metrics']['gp_acl'][exp]
        values=test['_paired_metrics'][candidate][exp]
        differences[exp]={}
        for metric in reference[0]:
            diff=np.array([a[metric]-b[metric] for a,b in zip(values,reference)])
            bootstrap=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(axis=1)
            differences[exp][metric]={'mean':float(diff.mean()),'paired_95_interval':np.quantile(bootstrap,[.025,.975]).tolist(),'wins':int((diff>0).sum()),'n':len(diff)}
    report['paired'][candidate]=differences
# Normalized ranking gain: paired resampling uses the same task indices for
# candidate and reference, and fixed cohort scales reported by the ranker.
config=json.loads(Path('eval/test/scoring_config.json').read_text())
report['paired_rank']={}
for candidate in ['random_baseline','better_boundary_fit','better_boundary_fit_12','better_normalized','weaker_no_acquisition']:
    if candidate in test['_failures']:continue
    bootstrap_total=np.zeros(10000);mean_total=0.
    for exp,spec in config['experiments'].items():
        ref=test['_paired_metrics']['gp_acl'][exp];val=test['_paired_metrics'][candidate][exp]
        per=np.zeros(len(ref))
        for metric,definition in spec['metrics'].items():
            scale=test['_rank']['population_std'][f'{exp}.{metric}']
            if scale==0:continue
            sign=1 if definition['higher_is_better'] else -1
            per+=spec['weight']*abs(definition['coefficient'])*sign*np.array([a[metric]-b[metric] for a,b in zip(val,ref)])/scale
        bootstrap_total+=per[rng.integers(0,len(per),(10000,len(per)))].mean(axis=1)
        mean_total+=per.mean()
    report['paired_rank'][candidate]={'mean':float(mean_total),'paired_95_interval':np.quantile(bootstrap_total,[.025,.975]).tolist()}
Path('eval/test/hardening_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'paired_rank':report['paired_rank'],'test_ranks':report['test_rank']['values']},indent=2))
