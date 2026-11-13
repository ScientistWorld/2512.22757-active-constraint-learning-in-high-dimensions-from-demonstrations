"""Development-only diagnostic of rejected demonstration requests (packager).

Usage: python scripts/dev_rejection_diag.py NAME [NAME ...]

For each named development run (eval/train/runs/NAME.json) every rejected
request is classified: start inside the obstacle, goal inside, endpoints too
close, or both endpoints feasible (the expert's solver failed). Solver
failures are retried with a more persistent solve of the same Problem 1
(more warm starts and iterations); a recovered trajectory is checked for
feasibility and KKT stationarity. Writes tmp/dev_rejections.json.
"""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dev_tasks',ROOT/'eval'/'train'/'tasks.py')
tasks=importlib.util.module_from_spec(spec);spec.loader.exec_module(tasks)
H=tasks.HORIZON


def persistent(task,start,goal):
    def unpack(z): return np.vstack([start,z.reshape(-1,2),goal])
    def cost(z): return np.sum(np.diff(unpack(z),axis=0)**2)
    def jac(z):
        p=unpack(z); return (2*(2*p[1:-1]-p[:-2]-p[2:])).ravel()
    def con(z): return -task.value_gradient(unpack(z)[1:-1])[0]
    def con_jac(z):
        grad=-task.value_gradient(unpack(z)[1:-1])[1]
        mat=np.zeros((H-2,2*(H-2)));k=np.arange(H-2);mat[k,2*k]=grad[:,0];mat[k,2*k+1]=grad[:,1]
        return mat
    ends=np.vstack([start,goal])-task.center
    a0,a1=np.arctan2(ends[:,1],ends[:,0])
    turn=(a1-a0+np.pi)%(2*np.pi)-np.pi
    frac=np.linspace(0,1,H);rad=np.linspace(*np.linalg.norm(ends,axis=1),H)
    for clearance in (.3,.6,1.2):
        for angular in (turn,turn-2*np.pi if turn>=0 else turn+2*np.pi):
            ang=a0+angular*frac
            r=np.maximum(rad,task.radius(ang)+clearance)
            warm=task.center+np.column_stack([r*np.cos(ang),r*np.sin(ang)]);warm[0]=start;warm[-1]=goal
            res=minimize(cost,warm[1:-1].ravel(),jac=jac,method='SLSQP',
                         constraints=[{'type':'ineq','fun':con,'jac':con_jac}],options={'maxiter':2000,'ftol':2e-12})
            if np.min(con(res.x))>=-1e-7:
                p=unpack(res.x);g=2*(2*p[1:-1]-p[:-2]-p[2:])
                v,dv=task.value_gradient(p[1:-1])
                lam=np.maximum(0,-np.sum(g*dv,axis=1)/np.sum(dv**2,axis=1))
                resid=float(np.max(np.abs(g+lam[:,None]*dv)))
                return {'success_flag':bool(res.success),'message':str(res.message),'stationarity':resid,
                        'complementarity':float(np.max(np.abs(lam*v))),'clearance':clearance}
    return None


def instance(job):
    exp,index,record=job
    task=tasks.Task(exp,index)
    counts={'requests':0,'accepted':0,'start_inside':0,'goal_inside':0,'too_close':0,'solver_failure':0,
            'recovered_feasible':0,'recovered_kkt_ok':0}
    examples=[]
    for entry in record['ledger']:
        counts['requests']+=1
        if entry['demo'] is not None:
            counts['accepted']+=1;continue
        s=np.asarray(entry['start'],float);g=np.asarray(entry['goal'],float)
        v=task.value_gradient(np.vstack([s,g]))[0]
        if v[0]>1e-8: counts['start_inside']+=1
        if v[1]>1e-8: counts['goal_inside']+=1
        if v[0]>1e-8 or v[1]>1e-8: continue
        if np.linalg.norm(s-g)<.15:
            counts['too_close']+=1;continue
        counts['solver_failure']+=1
        retry=persistent(task,s,g)
        if retry is not None:
            counts['recovered_feasible']+=1
            counts['recovered_kkt_ok']+=int(retry['stationarity']<1e-3 and retry['complementarity']<1e-5)
            examples.append({'index':index,**retry})
    return exp,counts,examples


def main():
    import os
    from concurrent.futures import ProcessPoolExecutor
    out={}
    workers=max(1,min(8,len(os.sched_getaffinity(0))))
    for name in sys.argv[1:]:
        path=ROOT/'eval'/'train'/'runs'/f'{name}.json'
        if not path.exists():
            print(name,'missing',flush=True);continue
        records=json.loads(path.read_text())['records'];out[name]={}
        jobs=[(exp,i,r) for exp,instances in records.items() for i,r in enumerate(instances)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for exp,counts,examples in pool.map(instance,jobs,chunksize=1):
                item=out[name].setdefault(exp,{'counts':{k:0 for k in counts},'examples':[]})
                for k,v in counts.items(): item['counts'][k]+=v
                item['examples']=(item['examples']+examples)[:5]
        for exp,item in out[name].items():
            print('rejections',name,exp,json.dumps(item['counts']),json.dumps(item['examples'])[:600],flush=True)
    Path(ROOT/'tmp').mkdir(exist_ok=True)
    (ROOT/'tmp'/'dev_rejections.json').write_text(json.dumps(out,indent=1)+'\n')


if __name__=='__main__':
    main()
