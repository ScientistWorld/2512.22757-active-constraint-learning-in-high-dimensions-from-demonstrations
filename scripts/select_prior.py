"""Development-split choice of the reference GP prior mean (packager only).

Candidates: the paper's own zero prior mean (Eqs. 15-16) and the previous
package's mean tied to -0.1*signal. The rule was fixed before the job ran:
keep the paper's zero mean if, on the development split, its reference and
its random-query control both pass the success gate, its reference beats its
random control in uniform-space accuracy on the paper's g3 experiment (the
paper's headline quantity), and its reference outranks its random control in
the combined score. Otherwise keep the tied mean. Hidden-test data are never
read. Prints a report and, on the last line, the chosen prior.
"""
import json
import sys
from pathlib import Path
import numpy as np

scores=json.loads(Path(sys.argv[1]).read_text())
config=json.loads(Path('eval/train/scoring_config.json').read_text())
results={exp:v['results'] for exp,v in scores['experiments'].items()}
failed=set(scores['_failures'])
names=['zero_reference','zero_random','tied_reference','tied_random']
passing=[n for n in names if n not in failed and all(n in results[e] for e in results)]


def rank(name,anchor):
    total=0.
    for exp,spec in config['experiments'].items():
        for metric,definition in spec['metrics'].items():
            scale=float(np.std([results[exp][n][metric] for n in passing]))
            if scale==0:
                continue
            sign=1 if definition['higher_is_better'] else -1
            total+=spec['weight']*abs(definition['coefficient'])*sign*(
                results[exp][name][metric]-results[exp][anchor][metric])/scale
    return total


anchor=next((n for n in ('tied_reference','zero_reference') if n in passing),None)
ranks={n:(rank(n,anchor) if anchor else None) for n in passing}
paired={}
for prior in ('zero','tied'):
    ref=scores['_paired_metrics'].get(f'{prior}_reference',{})
    ran=scores['_paired_metrics'].get(f'{prior}_random',{})
    for exp in results:
        a=ref.get(exp,[]);b=ran.get(exp,[])
        if a and len(a)==len(b):
            d=np.array([x['accuracy']-y['accuracy'] for x,y in zip(a,b)])
            paired[f'{prior}.{exp}.accuracy_reference_minus_random']={
                'mean':float(d.mean()),'wins':int((d>0).sum()),'losses':int((d<0).sum())}
g3='double_integrator_g3'
keep_zero=('zero_reference' in passing and 'zero_random' in passing
           and results[g3]['zero_reference']['accuracy']>results[g3]['zero_random']['accuracy']
           and ranks['zero_reference']>ranks['zero_random'])
chosen='zero' if keep_zero else 'tied'
report={'rule':__doc__.strip(),'failed':sorted(failed&set(names)),'cohort':passing,'anchor':anchor,
        'ranks':ranks,'paired_uniform_accuracy':paired,
        'metrics':{exp:{n:results[exp][n] for n in names if n in results[exp]} for exp in results},
        'chosen':chosen}
Path('tmp').mkdir(exist_ok=True)
Path('tmp/prior_selection.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
for exp in results:
    for n in names:
        if n in results[exp]:
            print('dev',exp,n,{k:round(v,4) for k,v in results[exp][n].items()},flush=True)
print(json.dumps({'ranks':ranks,'paired_uniform_accuracy':paired,'failed':report['failed']},indent=1),flush=True)
print(chosen,flush=True)
