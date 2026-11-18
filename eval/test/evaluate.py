"""Private online evaluator. Methods supply commands, never truth or metrics.

The only fitting here is the submitted active learner's per-instance solve.
There is no offline training, backbone training, or baseline fitting at test
scoring time. Frozen evaluator-owned baseline/reference *raw* records are
rescored by exactly the same metric function as newly collected records.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from tasks import Task,EXPERIMENTS as PLANAR_EXPERIMENTS,TASKS_PER_EXPERIMENT,BUDGET,REQUEST_LIMIT,BOUND,NEAR_WIDTH,HORIZON
import system_tasks
from system_tasks import SystemTask

# The double-integrator experiments (tasks.py) and the paper's unicycle and
# quadrotor experiments (system_tasks.py), each with its own instance count.
EXPERIMENTS=PLANAR_EXPERIMENTS+system_tasks.EXPERIMENTS
COUNT={**{exp:TASKS_PER_EXPERIMENT for exp in PLANAR_EXPERIMENTS},
       **{exp:system_tasks.SETTINGS[exp]['tasks'] for exp in system_tasks.EXPERIMENTS}}
assert system_tasks.BUDGET==BUDGET and system_tasks.REQUEST_LIMIT==REQUEST_LIMIT


def make_task(exp,index):
    return Task(exp,index) if exp in PLANAR_EXPERIMENTS else SystemTask(exp,index)


def task_module(exp):
    return HERE/('tasks.py' if exp in PLANAR_EXPERIMENTS else 'system_tasks.py')


def header(task):
    if isinstance(task,SystemTask): return task.header()
    return {'system':'double_integrator','dimension':2,'state_dimension':4,'control_dimension':2,
            'model':{'dt':1},'bounds':[[-BOUND,BOUND],[-BOUND,BOUND]],'horizon':HORIZON,'dt':1}

SLICE='test'
# Whole-run success gate on the boundary band: mean balanced accuracy must
# beat chance (.5) by a margin and each class must be recalled at all. It
# rejects one-class, coin-flip and uninformed fixed-shape classifiers.
MIN_BALANCED=.55
MIN_RECALL=.10
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def load(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda v:(_ for _ in ()).throw(ValueError('nonfinite JSON '+v)))


def write(path,data,compact=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(data,indent=None if compact else 2,allow_nan=False)); temp.replace(path)


def command_digest(manifest):
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode())
    pending=[(ROOT/word).resolve() for word in manifest.get('command',[]) if (ROOT/word).is_file()]
    seen=set()
    while pending:
        path=pending.pop()
        if path in seen: continue
        seen.add(path)
        content=path.read_bytes();digest.update(str(path).encode());digest.update(content)
        if path.suffix!='.py': continue
        try: tree=ast.parse(content)
        except (SyntaxError,UnicodeDecodeError): continue
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): modules=[a.name for a in node.names];level=0
            elif isinstance(node,ast.ImportFrom):
                modules=[node.module or ''];level=node.level
                modules += [(node.module+'.' if node.module else '')+a.name for a in node.names]
            else: continue
            for module in modules:
                roots=[path.parent,ROOT] if not level else [path.parents[level-1]]
                for root in roots:
                    base=root.joinpath(*module.split('.')) if module else root
                    for dependency in (base.with_suffix('.py'),base/'__init__.py'):
                        try: dependency.resolve().relative_to(ROOT)
                        except ValueError: continue
                        if dependency.is_file(): pending.append(dependency.resolve())
    # Own task/evaluator versions invalidate caches when the setup changes.
    digest.update((HERE/'tasks.py').read_bytes())
    digest.update((HERE/'system_tasks.py').read_bytes())
    # Metric/rank changes rescore raw outputs. Bump this version only when
    # acquisition messages or their interpretation change.
    digest.update(b'acl-v2-online-protocol-5')
    return digest.hexdigest()


def instance_id(exp,index): return f'{exp}:{index}'


def compact(run):
    """Frozen comparison records keep what scoring reads: the predictions and
    the oracle ledger's accept/reject outcomes. Trajectories are dropped."""
    def one(record):
        return {**record,'initial_demos':[1]*len(record['initial_demos']),
                'ledger':[{'start':x['start'],'goal':x['goal'],'demo':None if x['demo'] is None else 1}
                          for x in record['ledger']],
                'producer_stderr':record.get('producer_stderr','')[-2000:]}
    return {**run,'records':{exp:[one(r) for r in records] for exp,records in run['records'].items()}}


def run_instance(manifest,task):
    version=hashlib.sha256(task_module(task.experiment).read_bytes()).hexdigest()[:12]
    initial_path=HERE/'oracle_cache'/version/f'initial_{task.experiment}_{task.index}.json'
    initial=load(initial_path) if initial_path.exists() else task.initial_demonstrations()
    if not initial_path.exists(): write(initial_path,initial)
    points,truth,uniform=task.evaluation()
    # The demonstration cache is private and keyed by task AND exact endpoints;
    # it accelerates paired variants without disclosing trajectories to them.
    cache_path=HERE/'oracle_cache'/version/f'{task.experiment}_{task.index}.json'
    oracle_cache=load(cache_path) if cache_path.exists() else {}
    ledger=[]; final=None
    process=subprocess.Popen(manifest['command'],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,text=True,bufsize=1)
    def tell(message):
        process.stdin.write(json.dumps(message,allow_nan=False)+'\n');process.stdin.flush()
    def receive():
        line=process.stdout.readline()
        if not line: raise ValueError('producer exited before completing protocol: '+process.stderr.read()[-1000:])
        if len(line)>1000000: raise ValueError('oversized response')
        return json.loads(line,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('nonfinite response')))
    failure=None
    try:
        tell({**header(task),'initial_demos':initial,'budget':BUDGET,'query_limit':REQUEST_LIMIT,'seed':7})
        dim=header(task)['dimension'];bound=header(task)['bounds'][0][1]
        while True:
            message=receive()
            if message.get('op')=='finish':
                tell({'points':points.tolist()})
                final=receive()
                break
            if message.get('op')!='query': raise ValueError('expected query or finish')
            if len(ledger)>=REQUEST_LIMIT: raise ValueError('query limit exceeded')
            if len(initial)+sum(d['demo'] is not None for d in ledger)>=BUDGET:
                raise ValueError('demonstration budget exceeded')
            endpoints=np.asarray([message['start'],message['goal']],float)
            if endpoints.shape!=(2,dim) or not np.all(np.isfinite(endpoints)) or np.max(np.abs(endpoints))>bound:
                raise ValueError('invalid endpoint coordinates')
            key=hashlib.sha256(endpoints.tobytes()).hexdigest()
            if key not in oracle_cache:
                oracle_cache[key]=task.demonstrate(*endpoints)
                write(cache_path,oracle_cache)
            demo=oracle_cache[key]
            ledger.append({'start':endpoints[0].tolist(),'goal':endpoints[1].tolist(),'demo':demo})
            tell({'demo':demo,'requests_remaining':REQUEST_LIMIT-len(ledger)})
        if not isinstance(final,dict) or set(final)-{'predictions','n_demos'}:
            raise ValueError('final output may contain only predictions and optional authenticated n_demos')
        pred=np.asarray(final['predictions'])
        if pred.shape!=(len(points),) or pred.dtype.kind not in 'biuf' or not np.all(np.isfinite(pred)) or not np.all((pred==0)|(pred==1)):
            raise ValueError('predictions must be finite binary labels of evaluator-owned size')
        # Optional self-reported counts are never used as evidence, and lies /
        # impossible declarations are rejected rather than given a count bonus.
        if 'n_demos' in final:
            count=final['n_demos']
            actual=len(initial)+sum(d['demo'] is not None for d in ledger)
            if type(count) not in (int,float) or not np.isfinite(count) or count!=actual:
                raise ValueError('declared demonstration count disagrees with oracle ledger')
        pred=pred.astype(int).tolist()
    except Exception as exc:
        failure=str(exc);pred=[]
    finally:
        process.kill()
        _,producer_stderr=process.communicate(timeout=10)
    # Retain diagnostics as opaque text. No scored calculation reads this field.
    return {'instance':instance_id(task.experiment,task.index),'predictions':pred,
            'initial_demos':initial,'ledger':ledger,'failure':failure,
            'producer_stderr':producer_stderr[-16384:]}


# Independent instances may run concurrently; records are kept in index order.
WORKERS=max(1,min(8,int(os.environ.get('ACL_EVAL_WORKERS',len(os.sched_getaffinity(0))))))


def _instance(job):
    manifest,exp,index=job
    return index,run_instance(manifest,make_task(exp,index))


def collect(manifest,exp,records,save,report):
    """Run the missing instances; stop scheduling more after a failure.

    Deterministic protocol violations need no more expensive solves: such an
    attempt has already failed the whole task.
    """
    pending=list(range(len(records),COUNT[exp]))
    if WORKERS<=1 or len(pending)<=1:
        for index in pending:
            record=run_instance(manifest,make_task(exp,index));records.append(record);save();report(index,record)
            if record['failure']: break
        return records
    from concurrent.futures import ProcessPoolExecutor,as_completed
    done={};failed=False
    with ProcessPoolExecutor(max_workers=min(WORKERS,len(pending))) as pool:
        futures=[pool.submit(_instance,(manifest,exp,index)) for index in pending]
        for future in as_completed(futures):
            if future.cancelled(): continue
            index,record=future.result()
            report(index,record);done[index]=record
            if record['failure'] and not failed:
                failed=True
                for other in futures: other.cancel()
            grew=False
            while len(records) in done:
                records.append(done.pop(len(records)));grew=True
            if grew: save()
    return records


def metrics(records,exp):
    values=[]; errors=[]; per_instance=[]
    if len(records)!=COUNT[exp]:
        return None,['wrong number of instances']+[r['failure'] for r in records if r.get('failure')],[]
    for index,record in enumerate(records):
        if record['instance']!=instance_id(exp,index): errors.append('instance mismatch');continue
        if record.get('failure'): errors.append(record['failure']);continue
        if len(record['initial_demos'])+sum(x['demo'] is not None for x in record['ledger'])>BUDGET or len(record['ledger'])>REQUEST_LIMIT:
            errors.append('budget violation');continue
        task=make_task(exp,index);points,truth,uniform=task.evaluation()
        p=np.asarray(record['predictions'])
        if p.shape!=truth.shape or not np.all(np.isfinite(p)) or not np.all((p==0)|(p==1)):
            errors.append('invalid predictions');continue
        band=~uniform
        safe=np.mean(p[band&(truth==1)]==1);unsafe=np.mean(p[band&(truth==0)]==0)
        one={'accuracy':float(np.mean(p[uniform]==truth[uniform])),
             'balanced_accuracy':float(.5*(safe+unsafe)),
             'near_boundary_balanced_accuracy':float(.5*(safe+unsafe)),
             'safe_recall':float(safe),'unsafe_recall':float(unsafe)}
        # The narrow inner band resolves the boundary closest to the surface.
        near=band&(task.boundary_distance(points)<=getattr(task,'near_width',NEAR_WIDTH))
        one['near_boundary_balanced_accuracy']=float(.5*(np.mean(p[near&(truth==1)]==1)+np.mean(p[near&(truth==0)]==0)))
        per_instance.append(one);values.append(one)
    if errors: return None,errors,per_instance
    result={k:float(np.mean([v[k] for v in values])) for k in values[0]}
    if result['balanced_accuracy']<MIN_BALANCED or min(result['safe_recall'],result['unsafe_recall'])<MIN_RECALL:
        return result,[f'boundary balanced accuracy below {MIN_BALANCED:.2f} or a class recall below {MIN_RECALL:.2f}'],per_instance
    return result,[],per_instance


def rank(all_metrics,anchors,failures,reference):
    valid=[name for name in all_metrics if name not in failures]
    ranks={name:0. for name in valid};scales={}
    for exp,spec in reference['experiments'].items():
        for metric,definition in spec['metrics'].items():
            population=[all_metrics[name][exp][metric] for name in valid]
            scale=float(np.std(population)) if population else 0.;scales[f'{exp}.{metric}']=scale
            if scale==0: continue  # zero variation implies zero improvements.
            direction=1 if definition['higher_is_better'] else -1
            for name in valid:
                ranks[name]+=spec['weight']*abs(definition['coefficient'])*direction*(all_metrics[name][exp][metric]-anchors[exp][metric])/scale
    return {'values':{name:(ranks[name] if name in ranks else None) for name in all_metrics},
            'order':sorted(valid,key=lambda n:-ranks[n])+sorted(failures),
            'population_std':scales,'failed':failures,
            'rule':'passed runs first; sum weight * abs(coefficient) * better-direction improvement / population std; zero-variance terms contribute zero'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--outputs-dir',type=Path,default=ROOT/'outputs')
    parser.add_argument('--output',type=Path,default=ROOT/'scoring'/f'scores_{SLICE}.json')
    parser.add_argument('--bootstrap',action='store_true',help='Packager only: freeze independently rerun baseline/reference raw records.')
    args=parser.parse_args()
    manifests={};manifest_errors={}
    paths=sorted(args.outputs_dir.glob('*.json'))
    for path in paths:
        name=path.stem
        if name=='random_baseline' and not args.bootstrap: continue
        try:
            manifest=load(path)
            if manifest.get('protocol')!='acl-v2' or not isinstance(manifest.get('command'),list) or not manifest['command'] or not all(isinstance(w,str) for w in manifest['command']):
                raise ValueError('expected acl-v2 command manifest; static predictions cannot answer unseen instances')
            manifests[name]=manifest
        except Exception as exc: manifest_errors[name]=[str(exc)]
    if not paths: raise SystemExit('No methods found on disk')
    fixed_path=HERE/'fixed_raw.json'
    if not args.bootstrap and not fixed_path.exists(): raise SystemExit('Missing evaluator-owned fixed_raw.json; run reference reproduction first')
    raw={};all_metrics={};failures=dict(manifest_errors);details={};raw_metrics={}
    for name,manifest in manifests.items():
        # The fixed comparison cannot be replaced through outputs/.
        if name=='random_baseline' and not args.bootstrap: continue
        digest=command_digest(manifest)
        path=HERE/'runs'/f'{name}.json'
        previous=load(path) if path.exists() else {}
        old=previous.get('records',{}) if previous.get('digest')==digest else {}
        raw[name]={'digest':digest,'records':{}}
        for exp in EXPERIMENTS:
            records=old.get(exp,[])
            raw[name]['records'][exp]=records
            def save(): write(path,raw[name])
            def report(index,record,name=name,exp=exp):
                print(f'[{time.strftime("%H:%M:%S")}] {SLICE} {name} {exp} {index+1}/{COUNT[exp]}: '+str(record['failure'] or ('raw output saved; demos='+str(len(record['initial_demos'])+sum(x['demo'] is not None for x in record['ledger']))+', requests='+str(len(record['ledger'])))),flush=True)
            collect(manifest,exp,records,save,report)
            raw[name]['records'][exp]=records
            value,error,instances=metrics(records,exp)
            all_metrics.setdefault(name,{})[exp]=value
            details.setdefault(name,{})[exp]=instances
            if error: failures.setdefault(name,[]).extend(error)
            if any(r.get('failure') for r in records): break
        write(path,raw[name])
    if args.bootstrap:
        for name in ('gp_acl','random_baseline'):
            if name not in raw: raise ValueError('Bootstrap requires reference and random baseline manifests')
        write(fixed_path,{name:compact(raw[name]) for name in ('gp_acl','random_baseline')},compact=True)
    fixed=load(fixed_path)
    anchors={}
    for name in ('gp_acl','random_baseline'):
        fixed_metrics={}
        for exp in EXPERIMENTS:
            value,error,instances=metrics(fixed[name]['records'][exp],exp)
            if value is None: raise ValueError('Invalid fixed comparison '+str(error))
            if error and name=='random_baseline': failures.setdefault(name,[]).extend(error)
            fixed_metrics[exp]=value
            if name=='gp_acl': anchors[exp]=value
        if name=='random_baseline':
            all_metrics[name]=fixed_metrics
            for exp in EXPERIMENTS:
                details.setdefault(name,{})[exp]=metrics(fixed[name]['records'][exp],exp)[2]
    reference=load(HERE/'scoring_config.json')
    # One violation fails the run across every scored experiment, preventing
    # success in one task from paying for failure in another.
    for name in set(all_metrics)|set(failures):
        if name in failures:
            raw_metrics[name]=all_metrics.get(name,{})
            all_metrics[name]={exp:{m:0. for m in reference['experiments'][exp]['metrics']} for exp in EXPERIMENTS}
    scores={'_slice':SLICE,'experiments':{exp:{'results':{name:vals[exp] for name,vals in all_metrics.items()}} for exp in EXPERIMENTS},
            '_failures':failures,'_raw_metrics_of_failed':raw_metrics,
            '_rank':rank(all_metrics,anchors,failures,reference),'_paired_metrics':details}
    write(args.output,scores)
    print(json.dumps({'output':str(args.output),'rank':scores['_rank']['values'],'failures':failures},allow_nan=False),flush=True)


if __name__=='__main__': main()
