"""Development-only pilot (packager): what limits uniform-space accuracy.

Reads the frozen development records (eval/train/fixed_raw.json: reference
and random control, previous four-crossing setup) and the public development
generator only. For every record:
  1. Contact quality: are the tight states flagged by Problem 7 on the true
     boundary, and do their Problem 8 gradient estimates point along the true
     constraint gradient? (App. A.1 calls the estimated tight set a
     guaranteed under-approximation of the true one.)
  2. The final classifier is refitted to the record's demonstrations under
     alternative GP settings of the paper's training family; acquisition is
     unchanged (the record's demonstrations). Data sources: the reference's
     demonstrations, the random control's, and the four initial ones alone.
  3. Where uniform-space errors fall: false-unsafe states by distance to the
     nearest tight state, false-safe states by depth inside the obstacle.
Writes tmp/dev_pilot.json and prints a summary.
"""
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import method.gp_acl as M  # noqa: E402

spec=importlib.util.spec_from_file_location('dev_tasks',ROOT/'eval'/'train'/'tasks.py')
tasks=importlib.util.module_from_spec(spec);spec.loader.exec_module(tasks)

BASE=dict(prior='tied',rho=.1,noise_floor=1e-10,hinge=True,tight_tol=1e-4)
SETTINGS={
    'current':{},
    'tied_floor1e-4':dict(noise_floor=1e-4),
    'tied_floor1e-2':dict(noise_floor=1e-2),
    'tied_ml_floor1e-4':dict(noise_floor=1e-4,hinge=False),
    'learned':dict(prior='learned'),
    'learned_floor1e-4':dict(prior='learned',noise_floor=1e-4),
    'zero_ml_floor1e-4':dict(prior='zero',noise_floor=1e-4,hinge=False),
    'zero_floor1e-4':dict(prior='zero',noise_floor=1e-4),
    'current_tol1e-3':dict(tight_tol=1e-3),
}
DIST_EDGES=[0.,1.,2.,4.,np.inf];DIST_LABELS=['<1','1-2','2-4','>4']
DEPTH_EDGES=[0.,.5,1.5,3.,np.inf];DEPTH_LABELS=['0-0.5','0.5-1.5','1.5-3','>3']


def configure(setting):
    M.CONFIG.update(BASE);M.CONFIG.update(SETTINGS[setting])


def demos_of(record,source):
    demos=list(record['initial_demos'])
    if source!='initial_only':
        demos+=[x['demo'] for x in record['ledger'] if x['demo'] is not None]
    return demos


def contact_quality(task,demos):
    rows=[]
    for demo in demos:
        p=np.asarray(demo['positions'],float)
        grad=2*(2*p[1:-1]-p[:-2]-p[2:]);p7=np.abs(grad).sum(axis=1)
        for point,g,value in zip(p[1:-1],grad,p7):
            if value<=1e-6: continue
            v,dv=task.value_gradient(point[None])
            n=np.linalg.norm(dv[0]);w=-g
            rows.append((float(value),float(v[0]/max(n,1e-12)),float(w@dv[0]/max(n*np.linalg.norm(w),1e-12))))
    return rows


def evaluate(job):
    source,exp,index,record=job
    task=tasks.Task(exp,index);points,truth,uniform=task.evaluation()
    demos=demos_of(record,source)
    band=~uniform;near=band&(task.boundary_distance(points)<=tasks.NEAR_WIDTH)
    q=points[uniform]-task.center;t=truth[uniform]
    depth=task.radius(np.arctan2(q[:,1],q[:,0]))-np.linalg.norm(q,axis=1)
    out={'contacts':contact_quality(task,demos),'settings':{}}
    for setting in SETTINGS:
        configure(setting)
        try:
            contacts,gradients=M.observations(demos)
            gp=M.train_gp(contacts,gradients,M.demonstrated_states(demos))
            pred=np.ones(len(points),int) if gp is None else (gp.mean_var(points)[0]<=0).astype(int)
        except Exception as exc:  # recorded, never silently dropped
            out['settings'][setting]={'error':repr(exc)};continue
        safe=np.mean(pred[band&(truth==1)]==1);unsafe=np.mean(pred[band&(truth==0)]==0)
        y=pred[uniform]
        row={'accuracy':float(np.mean(y==t)),'balanced_accuracy':float(.5*(safe+unsafe)),
             'near_boundary_balanced_accuracy':float(.5*(np.mean(pred[near&(truth==1)]==1)+np.mean(pred[near&(truth==0)]==0))),
             'safe_recall':float(safe),'unsafe_recall':float(unsafe),
             'false_safe':float(np.mean((t==0)&(y==1))),'false_unsafe':float(np.mean((t==1)&(y==0))),
             'contacts':int(len(contacts))}
        if gp is not None:
            row.update(lengthscale=float(gp.ls),signal=float(gp.signal),noise=float(gp.noise),prior_mean=float(gp.constant))
        fu=(t==1)&(y==0);fs=(t==0)&(y==1)
        if len(contacts) and fu.any():
            d=M.nearest_distance(points[uniform][fu],contacts)
            for lo,hi,label in zip(DIST_EDGES[:-1],DIST_EDGES[1:],DIST_LABELS):
                row[f'false_unsafe_contact_dist_{label}']=float(np.sum((d>=lo)&(d<hi))/len(t))
        else:
            for label in DIST_LABELS: row[f'false_unsafe_contact_dist_{label}']=0.
        for lo,hi,label in zip(DEPTH_EDGES[:-1],DEPTH_EDGES[1:],DEPTH_LABELS):
            row[f'false_safe_depth_{label}']=float(np.mean(fs&(depth>=lo)&(depth<hi)))
        if setting=='current' and source!='initial_only':
            frozen=np.asarray(record['predictions'])
            row['agreement_with_frozen']=float(np.mean(frozen==pred)) if frozen.shape==pred.shape else None
        out['settings'][setting]=row
    return source,exp,index,out


def interval(diff,rng):
    diff=np.asarray(diff,float)
    boot=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(axis=1)
    return [float(diff.mean())]+np.quantile(boot,[.025,.975]).tolist()


def main():
    fixed=json.loads((ROOT/'eval'/'train'/'fixed_raw.json').read_text())
    sources={'reference':'gp_acl','random':'random_baseline','initial_only':'gp_acl'}
    jobs=[(source,exp,i,record) for source,name in sources.items()
          for exp,records in fixed[name]['records'].items() for i,record in enumerate(records)]
    workers=max(1,min(8,len(os.sched_getaffinity(0))))
    results={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for source,exp,index,out in pool.map(evaluate,jobs,chunksize=1):
            results.setdefault(source,{}).setdefault(exp,{})[index]=out
    rng=np.random.default_rng(5)
    summary={'settings':{k:{**BASE,**v} for k,v in SETTINGS.items()},'contact_quality':{},'mean':{},'paired_reference_minus_random':{}}
    for source,byexp in results.items():
        for exp,byindex in byexp.items():
            rows=[r for i in sorted(byindex) for r in byindex[i]['contacts']]
            arr=np.asarray(rows) if rows else np.zeros((0,3))
            if len(arr):
                flagged=arr[arr[:,0]>1e-4]
                good_mask=(np.abs(flagged[:,1])<=.05)&(flagged[:,2]>=.9)
                quant=lambda x,q:np.quantile(x,q).tolist() if len(x) else None
                summary['contact_quality'][f'{source}.{exp}']={
                    'flagged_tight_states':int(len(flagged)),
                    'fraction_offset_gt_0.05':float(np.mean(np.abs(flagged[:,1])>.05)),
                    'fraction_offset_gt_0.2':float(np.mean(np.abs(flagged[:,1])>.2)),
                    'fraction_cos_lt_0.9':float(np.mean(flagged[:,2]<.9)),
                    'fraction_cos_lt_0':float(np.mean(flagged[:,2]<0)),
                    'p7_quantiles_good_0_5_50':quant(flagged[good_mask,0],[0,.05,.5]),
                    'p7_quantiles_bad_50_95_100':quant(flagged[~good_mask,0],[.5,.95,1]),
                    'below_tolerance_states_1e-6_to_1e-4':int(np.sum(arr[:,0]<=1e-4))}
            for setting in SETTINGS:
                vals=[byindex[i]['settings'][setting] for i in sorted(byindex)]
                good=[v for v in vals if 'error' not in v]
                if not good: continue
                keys=sorted({k for v in good for k in v if v[k] is not None})
                summary['mean'].setdefault(setting,{}).setdefault(exp,{})[source]={
                    k:float((np.median if k in ('lengthscale','signal','noise','prior_mean','contacts') else np.mean)(
                        [v[k] for v in good if v.get(k) is not None])) for k in keys}
                summary['mean'][setting][exp][source]['errors']=len(vals)-len(good)
    for setting in SETTINGS:
        for exp in results['reference']:
            a=[results['reference'][exp][i]['settings'][setting] for i in sorted(results['reference'][exp])]
            b=[results['random'][exp][i]['settings'][setting] for i in sorted(results['random'][exp])]
            if any('error' in x for x in a+b): continue
            summary['paired_reference_minus_random'].setdefault(setting,{})[exp]={
                m:interval([x[m]-y[m] for x,y in zip(a,b)],rng) for m in ('accuracy','balanced_accuracy','near_boundary_balanced_accuracy')}
    Path(ROOT/'tmp').mkdir(exist_ok=True)
    (ROOT/'tmp'/'dev_pilot.json').write_text(json.dumps(summary,indent=1,allow_nan=False)+'\n')
    for key,value in summary['contact_quality'].items():
        print('contact quality',key,json.dumps(value),flush=True)
    for setting,byexp in summary['mean'].items():
        for exp,bysource in byexp.items():
            for source,v in bysource.items():
                print('pilot',setting,exp,source,{k:(round(x,4) if isinstance(x,float) else x) for k,x in v.items()},flush=True)
    for setting,byexp in summary['paired_reference_minus_random'].items():
        for exp,v in byexp.items():
            print('paired ref-random',setting,exp,{m:[round(x,4) for x in iv] for m,iv in v.items()},flush=True)


if __name__=='__main__':
    main()
